#!/usr/bin/env python3
"""
Sequential Gipformer ONNX inference for very large audio folders.

- Process one file at a time
- Stable for huge directories
- Output JSONL only: wav, text
- Supports resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    print("Error: sherpa-onnx is not installed. Install with: pip install sherpa-onnx", file=sys.stderr)
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed. Install with: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.signal import resample_poly as scipy_resample_poly
except Exception:
    scipy_resample_poly = None


REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"
SAMPLE_RATE = 16000
FEATURE_DIM = 80

ONNX_FILES = {
    "fp32": {
        "encoder": "encoder-epoch-35-avg-6.onnx",
        "decoder": "decoder-epoch-35-avg-6.onnx",
        "joiner": "joiner-epoch-35-avg-6.onnx",
    },
    "int8": {
        "encoder": "encoder-epoch-35-avg-6.int8.onnx",
        "decoder": "decoder-epoch-35-avg-6.int8.onnx",
        "joiner": "joiner-epoch-35-avg-6.int8.onnx",
    },
}


def positive_int(value: str) -> int:
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value}")
    return v


def non_negative_int(value: str) -> int:
    v = int(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"Expected non-negative integer, got {value}")
    return v


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Sequential Gipformer ONNX inference for large audio folders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_argument_group("Input")
    src.add_argument("--audio", type=str, nargs="*", default=[], help="Explicit audio files")
    src.add_argument("--audio-dir", type=str, default="", help="Directory containing audio")
    src.add_argument("--manifest", type=str, default="", help="Text file: one audio path per line")
    src.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"],
        help="Extensions used with --audio-dir",
    )
    src.add_argument("--recursive", action="store_true", help="Recursively scan --audio-dir")
    src.add_argument(
        "--sort",
        type=str,
        choices=["none", "size_asc", "size_desc"],
        default="none",
        help="Sort files by size",
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--quantize", choices=["fp32", "int8"], default="fp32")
    model.add_argument("--cache-dir", type=str, default="")
    model.add_argument("--download-mode", choices=["snapshot", "single"], default="snapshot")
    model.add_argument("--local-files-only", action="store_true")

    infer = parser.add_argument_group("Inference")
    infer.add_argument(
        "--num-threads",
        type=non_negative_int,
        default=max(1, cpu_count - 1),
        help="Recognizer CPU threads",
    )
    infer.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
    )
    infer.add_argument("--max-active-paths", type=positive_int, default=4)
    infer.add_argument("--allow-resample", action="store_true")

    out = parser.add_argument_group("Output")
    out.add_argument("--output", type=str, default="results.jsonl")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--flush-every", type=positive_int, default=1)
    out.add_argument("--progress-every", type=positive_int, default=10)
    out.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.audio and not args.audio_dir and not args.manifest:
        parser.error("Provide at least one of --audio, --audio-dir, or --manifest")

    return args


def normalize_extensions(exts: Iterable[str]) -> tuple[str, ...]:
    out = []
    for ext in exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        out.append(ext)
    return tuple(dict.fromkeys(out))


def iter_files_scandir(root: str, recursive: bool, exts: set[str]):
    stack = [os.path.abspath(os.path.expanduser(root))]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            if os.path.splitext(entry.name)[1].lower() in exts:
                                yield entry.path
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def discover_files(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []

    for p in args.audio:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isfile(p):
            paths.append(p)

    if args.manifest:
        with open(os.path.expanduser(args.manifest), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = os.path.abspath(os.path.expanduser(line))
                if os.path.isfile(p):
                    paths.append(p)

    if args.audio_dir:
        exts = set(normalize_extensions(args.extensions))
        paths.extend(iter_files_scandir(args.audio_dir, args.recursive, exts))

    paths = list(dict.fromkeys(paths))

    if args.sort != "none":
        def safe_size(p: str) -> int:
            try:
                return os.path.getsize(p)
            except OSError:
                return 0

        reverse = args.sort == "size_desc"
        paths.sort(key=safe_size, reverse=reverse)

    return paths


def load_done_set(output_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isfile(output_path):
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav = obj.get("wav")
            if isinstance(wav, str):
                done.add(wav)
    return done


def download_model(quantize: str, cache_dir: str = "", mode: str = "snapshot", local_files_only: bool = False) -> dict[str, str]:
    files = ONNX_FILES[quantize]
    wanted = list(files.values()) + ["tokens.txt"]
    cache_dir = cache_dir or None

    if mode == "snapshot":
        repo_dir = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=wanted,
            cache_dir=cache_dir,
            max_workers=min(8, len(wanted)),
            local_files_only=local_files_only,
        )
        repo_dir = str(repo_dir)
        out = {k: str(Path(repo_dir) / v) for k, v in files.items()}
        out["tokens"] = str(Path(repo_dir) / "tokens.txt")
        return out

    out: dict[str, str] = {}
    for k, filename in files.items():
        out[k] = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    out["tokens"] = hf_hub_download(
        repo_id=REPO_ID,
        filename="tokens.txt",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return out


def resample_audio(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples

    if scipy_resample_poly is not None:
        g = math.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        resampled = scipy_resample_poly(samples, up, down)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    old_len = len(samples)
    if old_len == 0:
        return samples.astype(np.float32, copy=False)

    new_len = int(round(old_len * dst_sr / src_sr))
    if new_len <= 1:
        return np.ascontiguousarray(samples[:1], dtype=np.float32)

    old_idx = np.linspace(0.0, 1.0, num=old_len, endpoint=True)
    new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    resampled = np.interp(new_idx, old_idx, samples)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def read_audio(path: str, allow_resample: bool) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)

    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)

    samples = np.asarray(samples, dtype=np.float32)

    if sample_rate != SAMPLE_RATE:
        if not allow_resample:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
        samples = resample_audio(samples, sample_rate, SAMPLE_RATE)

    return np.ascontiguousarray(samples, dtype=np.float32)


def create_recognizer(args: argparse.Namespace, model_paths: dict[str, str]) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_paths["encoder"],
        decoder=model_paths["decoder"],
        joiner=model_paths["joiner"],
        tokens=model_paths["tokens"],
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        provider="cpu",
    )


def make_wav_key(path: str, audio_dir: str) -> str:
    path = os.path.abspath(path)
    if audio_dir:
        root = os.path.abspath(os.path.expanduser(audio_dir))
        try:
            return os.path.relpath(path, root)
        except ValueError:
            return os.path.basename(path)
    return os.path.basename(path)


def write_jsonl_line(f, wav_key: str, text: str) -> None:
    line = json.dumps(
        {"wav": wav_key, "text": text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    f.write(line + "\n")


def main() -> int:
    args = parse_args()

    files = discover_files(args)
    if not files:
        print("No input files found.", file=sys.stderr)
        return 2

    done = set()
    if args.resume:
        done = load_done_set(args.output)

    filtered_files = []
    for p in files:
        wav_key = make_wav_key(p, args.audio_dir)
        if wav_key not in done:
            filtered_files.append(p)

    files = filtered_files

    if not files:
        print("Nothing to do.")
        return 0

    if not args.quiet:
        print(f"Discovered: {len(files)} file(s)", flush=True)
        print(f"Recognizer threads: {args.num_threads}", flush=True)
        print(f"Output: {args.output}", flush=True)

    t0 = time.perf_counter()
    model_paths = download_model(
        quantize=args.quantize,
        cache_dir=args.cache_dir,
        mode=args.download_mode,
        local_files_only=args.local_files_only,
    )
    recognizer = create_recognizer(args, model_paths)
    if not args.quiet:
        print(f"Model ready in {time.perf_counter() - t0:.2f}s", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = len(files)
    ok_count = 0
    err_count = 0
    started = time.perf_counter()

    with open(args.output, "a", encoding="utf-8") as fout:
        for idx, path in enumerate(files, 1):
            wav_key = make_wav_key(path, args.audio_dir)
            file_start = time.perf_counter()

            try:
                samples = read_audio(path, args.allow_resample)

                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, samples)
                recognizer.decode_streams([stream])

                text = stream.result.text.strip()
                write_jsonl_line(fout, wav_key, text)
                ok_count += 1
                status = "ok"

            except Exception as e:
                write_jsonl_line(fout, wav_key, "")
                err_count += 1
                status = f"err: {type(e).__name__}: {e}"

            if idx % args.flush_every == 0:
                fout.flush()
                os.fsync(fout.fileno())

            if not args.quiet:
                elapsed_file = time.perf_counter() - file_start
                print(
                    f"[{idx}/{total}] {wav_key} | {status} | {elapsed_file:.2f}s",
                    flush=True,
                )

            elif idx % args.progress_every == 0:
                print(f"[progress] {idx}/{total}", flush=True)

    total_elapsed = time.perf_counter() - started
    print(
        f"Finished | total={total} ok={ok_count} err={err_count} time={total_elapsed:.2f}s",
        flush=True,
    )
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

# python infer_onnx_bulk.py --audio-dir data --recursive --extensions .wav --quantize int8 --allow-resample --output results.jsonl --quiet