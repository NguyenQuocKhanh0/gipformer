#!/usr/bin/env python3
"""
Fast bulk Gipformer ONNX inference for very large audio directories.

Output JSONL format:
{"wav":"file1.wav","text":"..."}
{"wav":"file2.wav","text":"..."}

Optimizations:
- Minimal output fields
- Faster directory scan with os.scandir
- Lightweight queues/items
- Keep one recognizer hot
- Batched decode_streams()
- Optional resume from existing JSONL
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

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

_SENTINEL = object()


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
    default_read_workers = min(8, max(2, cpu_count // 4))
    default_batch_size = 16
    default_loaded_queue = max(default_batch_size * 8, default_read_workers * 16)

    parser = argparse.ArgumentParser(
        description="Fast bulk Gipformer ONNX inference for huge audio collections",
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
        default="size_asc",
        help="Sort files by size to improve batching",
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--quantize", choices=["fp32", "int8"], default="fp32")
    model.add_argument("--cache-dir", type=str, default="")
    model.add_argument("--download-mode", choices=["snapshot", "single"], default="snapshot")
    model.add_argument("--local-files-only", action="store_true")

    infer = parser.add_argument_group("Inference")
    infer.add_argument("--batch-size", type=positive_int, default=default_batch_size)
    infer.add_argument("--max-wait-ms", type=float, default=20.0)
    infer.add_argument("--num-threads", type=non_negative_int, default=0)
    infer.add_argument("--read-workers", type=non_negative_int, default=default_read_workers)
    infer.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
    )
    infer.add_argument("--max-active-paths", type=positive_int, default=4)
    infer.add_argument("--allow-resample", action="store_true")
    infer.add_argument("--loaded-queue-size", type=positive_int, default=default_loaded_queue)

    out = parser.add_argument_group("Output")
    out.add_argument("--output", type=str, default="results.jsonl")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--flush-every", type=positive_int, default=100)
    out.add_argument("--progress-every", type=positive_int, default=100)
    out.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.audio and not args.audio_dir and not args.manifest:
        parser.error("Provide at least one of --audio, --audio-dir, or --manifest")

    if args.read_workers == 0:
        args.read_workers = default_read_workers

    if args.num_threads == 0:
        reserve = 1
        args.num_threads = max(1, cpu_count - args.read_workers - reserve)

    if args.max_wait_ms < 0:
        parser.error("--max-wait-ms must be >= 0")

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

    # deduplicate while preserving order
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


def read_audio(path: str, allow_resample: bool):
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)

        samples = np.asarray(samples, dtype=np.float32)

        if sample_rate != SAMPLE_RATE:
            if not allow_resample:
                return (path, False, None, 0, f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
            samples = resample_audio(samples, sample_rate, SAMPLE_RATE)
            sample_rate = SAMPLE_RATE

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        return (path, True, samples, sample_rate, "")
    except Exception as e:
        return (path, False, None, 0, f"{type(e).__name__}: {e}")


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


class JsonlWriter(threading.Thread):
    def __init__(self, output_path: str, result_queue: queue.Queue, flush_every: int = 100):
        super().__init__(name="jsonl-writer", daemon=True)
        self.output_path = output_path
        self.result_queue = result_queue
        self.flush_every = flush_every
        self.written = 0

    def run(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            while True:
                item = self.result_queue.get()
                try:
                    if item is _SENTINEL:
                        f.flush()
                        os.fsync(f.fileno())
                        return
                    f.write(item)
                    self.written += 1
                    if self.written % self.flush_every == 0:
                        f.flush()
                finally:
                    self.result_queue.task_done()


def producer_loop(file_queue: queue.Queue, loaded_queue: queue.Queue, allow_resample: bool) -> None:
    while True:
        path = file_queue.get()
        try:
            if path is _SENTINEL:
                loaded_queue.put(_SENTINEL)
                return
            loaded_queue.put(read_audio(path, allow_resample))
        finally:
            file_queue.task_done()


def encode_jsonl_line(wav_path: str, text: str) -> str:
    return json.dumps(
        {"wav": os.path.basename(wav_path), "text": text},
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def decode_batch(recognizer: sherpa_onnx.OfflineRecognizer, batch: list[tuple]) -> list[str]:
    results: list[Optional[str]] = [None] * len(batch)
    streams = []
    positions = []
    paths = []

    for i, item in enumerate(batch):
        path, ok, samples, sample_rate, error = item
        if not ok or samples is None:
            results[i] = encode_jsonl_line(path, "")
            continue

        s = recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)
        streams.append(s)
        positions.append(i)
        paths.append(path)

    if streams:
        recognizer.decode_streams(streams)
        for i, path, stream in zip(positions, paths, streams):
            text = stream.result.text.strip()
            results[i] = encode_jsonl_line(path, text)

    return [x for x in results if x is not None]


def inference_loop(
    recognizer: sherpa_onnx.OfflineRecognizer,
    loaded_queue: queue.Queue,
    result_queue: queue.Queue,
    num_producers: int,
    batch_size: int,
    max_wait_ms: float,
    progress_every: int,
    quiet: bool,
) -> int:
    alive_producers = num_producers
    max_wait_s = max_wait_ms / 1000.0
    finished = 0

    while True:
        batch = []

        while not batch and alive_producers > 0:
            item = loaded_queue.get()
            loaded_queue.task_done()
            if item is _SENTINEL:
                alive_producers -= 1
                continue
            batch.append(item)

        if not batch:
            break

        deadline = time.perf_counter() + max_wait_s
        while len(batch) < batch_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                item = loaded_queue.get(timeout=timeout)
                loaded_queue.task_done()
            except queue.Empty:
                break

            if item is _SENTINEL:
                alive_producers -= 1
                if alive_producers <= 0:
                    break
                continue
            batch.append(item)

        lines = decode_batch(recognizer, batch)
        for line in lines:
            result_queue.put(line)

        finished += len(lines)
        if (not quiet) and finished % progress_every == 0:
            print(f"[progress] finished={finished}", flush=True)

    return finished


def main() -> int:
    args = parse_args()

    files = discover_files(args)
    if not files:
        print("No input files found.", file=sys.stderr)
        return 2

    if args.resume:
        done = load_done_set(args.output)
        if done:
            files = [p for p in files if os.path.basename(p) not in done]

    if not files:
        print("Nothing to do.")
        return 0

    if not args.quiet:
        print(
            f"Discovered {len(files)} files | read_workers={args.read_workers} | "
            f"recognizer_threads={args.num_threads} | batch_size={args.batch_size}",
            flush=True,
        )

    model_paths = download_model(
        quantize=args.quantize,
        cache_dir=args.cache_dir,
        mode=args.download_mode,
        local_files_only=args.local_files_only,
    )
    recognizer = create_recognizer(args, model_paths)

    file_queue: queue.Queue = queue.Queue(maxsize=max(args.loaded_queue_size, args.read_workers * 2))
    loaded_queue: queue.Queue = queue.Queue(maxsize=args.loaded_queue_size)
    result_queue: queue.Queue = queue.Queue(maxsize=max(args.batch_size * 16, 256))

    writer = JsonlWriter(args.output, result_queue, flush_every=args.flush_every)
    writer.start()

    producers = []
    for idx in range(args.read_workers):
        t = threading.Thread(
            target=producer_loop,
            name=f"reader-{idx}",
            args=(file_queue, loaded_queue, args.allow_resample),
            daemon=True,
        )
        t.start()
        producers.append(t)

    for path in files:
        file_queue.put(path)
    for _ in producers:
        file_queue.put(_SENTINEL)

    started = time.perf_counter()

    finished = inference_loop(
        recognizer=recognizer,
        loaded_queue=loaded_queue,
        result_queue=result_queue,
        num_producers=len(producers),
        batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        progress_every=args.progress_every,
        quiet=args.quiet,
    )

    file_queue.join()
    loaded_queue.join()

    result_queue.put(_SENTINEL)
    result_queue.join()
    writer.join()

    for t in producers:
        t.join(timeout=0.1)

    elapsed = time.perf_counter() - started
    if not args.quiet:
        print(f"Finished {finished} files in {elapsed:.2f}s", flush=True)
        print(f"Output: {args.output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




# python infer_onnx_bulk_min.py \
#   --audio-dir data \
#   --recursive \
#   --extensions .wav \
#   --quantize int8 \
#   --batch-size 16 \
#   --allow-resample \
#   --output results.jsonl \
#   --resume