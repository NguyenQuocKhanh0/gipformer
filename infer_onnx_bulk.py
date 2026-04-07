#!/usr/bin/env python3
"""
High-throughput Gipformer ONNX inference for large audio collections.

Design goals:
- Keep the recognizer hot: one dedicated inference worker owns the recognizer
- Overlap I/O + preprocessing with NN compute using producer/consumer queues
- Use micro-batching with recognizer.decode_streams(streams)
- Resume safely from an existing JSONL output
- Download model files once using Hugging Face snapshot cache

Example:
    python infer_onnx_bulk.py \
        --audio-dir data \
        --extensions .wav .flac \
        --quantize int8 \
        --batch-size 16 \
        --max-wait-ms 20 \
        --output results.jsonl
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
from dataclasses import dataclass, asdict
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
    from scipy.signal import resample_poly as scipy_resample_poly  # type: ignore
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


@dataclass(slots=True)
class LoadedAudio:
    path: str
    ok: bool
    samples: Optional[np.ndarray] = None
    orig_sample_rate: int = 0
    num_samples: int = 0
    duration_sec: float = 0.0
    load_sec: float = 0.0
    error: str = ""


@dataclass(slots=True)
class ResultRecord:
    path: str
    ok: bool
    text: str = ""
    error: str = ""
    duration_sec: float = 0.0
    load_sec: float = 0.0
    infer_sec: float = 0.0
    total_sec: float = 0.0
    rtf: float = 0.0
    sample_rate: int = SAMPLE_RATE
    num_samples: int = 0


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
    return ivalue


def non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value}")
    return ivalue


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 4
    default_read_workers = min(8, max(2, cpu_count // 4))
    default_batch_size = 16
    default_loaded_queue = max(default_batch_size * 8, default_read_workers * 16)

    parser = argparse.ArgumentParser(
        description="High-throughput Gipformer ONNX inference for large batches of audio files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_argument_group("Input sources")
    src.add_argument("--audio", type=str, nargs="*", default=[], help="Explicit audio file path(s)")
    src.add_argument("--audio-dir", type=str, default="", help="Directory containing audio files")
    src.add_argument("--manifest", type=str, default="", help="Text file: one audio path per line")
    src.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".flac", ".ogg", ".mp3", ".m4a"],
        help="Extensions used with --audio-dir",
    )
    src.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan --audio-dir",
    )
    src.add_argument(
        "--sort",
        type=str,
        choices=["none", "size_asc", "size_desc"],
        default="size_asc",
        help="Sort discovered files by byte size to improve batching of similar durations",
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--quantize", choices=["fp32", "int8"], default="fp32")
    model.add_argument("--cache-dir", type=str, default="", help="Optional Hugging Face cache dir")
    model.add_argument(
        "--download-mode",
        choices=["snapshot", "single"],
        default="snapshot",
        help="snapshot downloads all required files concurrently; single uses hf_hub_download per file",
    )
    model.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access network; only use existing Hugging Face cache",
    )

    infer = parser.add_argument_group("Inference")
    infer.add_argument("--batch-size", type=positive_int, default=default_batch_size)
    infer.add_argument(
        "--max-wait-ms",
        type=float,
        default=20.0,
        help="Wait this long to fill a batch before decoding what is available",
    )
    infer.add_argument(
        "--num-threads",
        type=non_negative_int,
        default=0,
        help="Threads used by the recognizer. 0 = auto (all remaining CPU cores)",
    )
    infer.add_argument(
        "--read-workers",
        type=non_negative_int,
        default=default_read_workers,
        help="Audio loading/preprocessing workers. 0 = auto",
    )
    infer.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="modified_beam_search",
    )
    infer.add_argument(
        "--max-active-paths",
        type=positive_int,
        default=4,
        help="Used by modified_beam_search",
    )
    infer.add_argument(
        "--allow-resample",
        action="store_true",
        help="Resample non-16k audio to 16k. Requires scipy for best speed/quality; falls back to numpy interpolation",
    )
    infer.add_argument(
        "--loaded-queue-size",
        type=positive_int,
        default=default_loaded_queue,
        help="Prefetched decoded audio objects kept in RAM",
    )

    out = parser.add_argument_group("Output")
    out.add_argument("--output", type=str, default="results.jsonl")
    out.add_argument("--resume", action="store_true", help="Skip files already present in --output")
    out.add_argument(
        "--progress-every",
        type=positive_int,
        default=100,
        help="Print progress every N completed files",
    )
    out.add_argument(
        "--flush-every",
        type=positive_int,
        default=100,
        help="Flush JSONL every N records",
    )
    out.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-progress logging",
    )

    args = parser.parse_args()

    if args.read_workers == 0:
        args.read_workers = default_read_workers

    if args.num_threads == 0:
        reserve = 1
        args.num_threads = max(1, cpu_count - args.read_workers - reserve)

    if args.max_wait_ms < 0:
        parser.error("--max-wait-ms must be >= 0")

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


def discover_files(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []

    for p in args.audio:
        p = str(Path(p).expanduser())
        if Path(p).is_file():
            paths.append(p)

    if args.manifest:
        manifest = Path(args.manifest).expanduser()
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = str(Path(line).expanduser())
                if Path(p).is_file():
                    paths.append(p)

    if args.audio_dir:
        exts = normalize_extensions(args.extensions)
        root = Path(args.audio_dir).expanduser()
        iterator = root.rglob("*") if args.recursive else root.glob("*")
        for p in iterator:
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))

    # de-duplicate while preserving order
    paths = list(dict.fromkeys(paths))

    if args.sort != "none":
        def safe_size(path: str) -> int:
            try:
                return os.path.getsize(path)
            except OSError:
                return 0

        reverse = args.sort == "size_desc"
        paths.sort(key=safe_size, reverse=reverse)

    return paths


def load_done_set(output_path: str) -> set[str]:
    done: set[str] = set()
    path = Path(output_path)
    if not path.is_file():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = obj.get("path")
            if isinstance(p, str):
                done.add(p)
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
        paths = {k: str(Path(repo_dir) / v) for k, v in files.items()}
        paths["tokens"] = str(Path(repo_dir) / "tokens.txt")
        return paths

    paths: dict[str, str] = {}
    for key, filename in files.items():
        paths[key] = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    paths["tokens"] = hf_hub_download(
        repo_id=REPO_ID,
        filename="tokens.txt",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return paths


def resample_audio(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples

    if scipy_resample_poly is not None:
        g = math.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        resampled = scipy_resample_poly(samples, up, down)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    # Fallback: linear interpolation. Fast, dependency-free, but lower quality than scipy.signal.resample_poly.
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


def read_audio(path: str, allow_resample: bool) -> LoadedAudio:
    start = time.perf_counter()
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)

        samples = np.asarray(samples, dtype=np.float32)

        if sample_rate != SAMPLE_RATE:
            if not allow_resample:
                raise ValueError(
                    f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz. Re-run with --allow-resample to convert automatically."
                )
            samples = resample_audio(samples, sample_rate, SAMPLE_RATE)
            sample_rate = SAMPLE_RATE

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        duration = len(samples) / sample_rate if sample_rate > 0 else 0.0

        return LoadedAudio(
            path=path,
            ok=True,
            samples=samples,
            orig_sample_rate=sample_rate,
            num_samples=len(samples),
            duration_sec=duration,
            load_sec=time.perf_counter() - start,
        )
    except Exception as e:
        return LoadedAudio(
            path=path,
            ok=False,
            samples=None,
            orig_sample_rate=0,
            num_samples=0,
            duration_sec=0.0,
            load_sec=time.perf_counter() - start,
            error=f"{type(e).__name__}: {e}",
        )


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
        self.output_path = Path(output_path)
        self.result_queue = result_queue
        self.flush_every = flush_every
        self.written = 0
        self._fh = None

    def run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("a", encoding="utf-8")
        try:
            while True:
                item = self.result_queue.get()
                try:
                    if item is _SENTINEL:
                        self._fh.flush()
                        os.fsync(self._fh.fileno())
                        return
                    line = json.dumps(asdict(item), ensure_ascii=False)
                    self._fh.write(line + "\n")
                    self.written += 1
                    if self.written % self.flush_every == 0:
                        self._fh.flush()
                finally:
                    self.result_queue.task_done()
        finally:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()


def producer_loop(file_queue: queue.Queue, loaded_queue: queue.Queue, allow_resample: bool) -> None:
    while True:
        path = file_queue.get()
        try:
            if path is _SENTINEL:
                loaded_queue.put(_SENTINEL)
                return
            loaded_queue.put(read_audio(str(path), allow_resample=allow_resample))
        finally:
            file_queue.task_done()


class ThroughputStats:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.finished_files = 0
        self.ok_files = 0
        self.err_files = 0
        self.audio_sec = 0.0
        self.infer_sec = 0.0
        self.load_sec = 0.0
        self.lock = threading.Lock()

    def update(self, rec: ResultRecord) -> None:
        with self.lock:
            self.finished_files += 1
            self.audio_sec += rec.duration_sec
            self.infer_sec += rec.infer_sec
            self.load_sec += rec.load_sec
            if rec.ok:
                self.ok_files += 1
            else:
                self.err_files += 1

    def snapshot(self) -> dict[str, float | int]:
        with self.lock:
            wall = time.perf_counter() - self.started_at
            throughput = self.audio_sec / wall if wall > 0 else 0.0
            overall_rtf = wall / self.audio_sec if self.audio_sec > 0 else 0.0
            infer_rtf = self.infer_sec / self.audio_sec if self.audio_sec > 0 else 0.0
            return {
                "wall_sec": wall,
                "finished_files": self.finished_files,
                "ok_files": self.ok_files,
                "err_files": self.err_files,
                "audio_sec": self.audio_sec,
                "throughput_x": throughput,
                "overall_rtf": overall_rtf,
                "infer_rtf": infer_rtf,
                "load_sec_sum": self.load_sec,
                "infer_sec_sum": self.infer_sec,
            }


def decode_batch(
    recognizer: sherpa_onnx.OfflineRecognizer,
    batch: list[LoadedAudio],
) -> list[ResultRecord]:
    ready_items: list[LoadedAudio] = []
    ready_positions: list[int] = []
    out: list[Optional[ResultRecord]] = [None] * len(batch)

    for idx, item in enumerate(batch):
        if not item.ok or item.samples is None:
            out[idx] = ResultRecord(
                path=item.path,
                ok=False,
                error=item.error,
                duration_sec=item.duration_sec,
                load_sec=item.load_sec,
                infer_sec=0.0,
                total_sec=item.load_sec,
                rtf=0.0,
                sample_rate=item.orig_sample_rate,
                num_samples=item.num_samples,
            )
        else:
            ready_items.append(item)
            ready_positions.append(idx)

    if not ready_items:
        return [x for x in out if x is not None]

    streams = []
    for item in ready_items:
        s = recognizer.create_stream()
        s.accept_waveform(item.orig_sample_rate, item.samples)
        streams.append(s)

    infer_start = time.perf_counter()
    recognizer.decode_streams(streams)
    infer_elapsed = time.perf_counter() - infer_start

    batch_total_audio = sum(i.duration_sec for i in ready_items)
    batch_rtf = infer_elapsed / batch_total_audio if batch_total_audio > 0 else 0.0

    for pos, item, stream in zip(ready_positions, ready_items, streams):
        text = stream.result.text.strip()
        infer_share = infer_elapsed * (item.duration_sec / batch_total_audio) if batch_total_audio > 0 else 0.0
        total_sec = item.load_sec + infer_share
        out[pos] = ResultRecord(
            path=item.path,
            ok=True,
            text=text,
            duration_sec=item.duration_sec,
            load_sec=item.load_sec,
            infer_sec=infer_share,
            total_sec=total_sec,
            rtf=batch_rtf,
            sample_rate=item.orig_sample_rate,
            num_samples=item.num_samples,
        )

    return [x for x in out if x is not None]


def inference_loop(
    recognizer: sherpa_onnx.OfflineRecognizer,
    loaded_queue: queue.Queue,
    result_queue: queue.Queue,
    num_producers: int,
    batch_size: int,
    max_wait_ms: float,
    stats: ThroughputStats,
    progress_every: int,
    quiet: bool,
) -> None:
    alive_producers = num_producers
    max_wait_s = max_wait_ms / 1000.0

    while True:
        batch: list[LoadedAudio] = []

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

        records = decode_batch(recognizer, batch)
        for rec in records:
            stats.update(rec)
            result_queue.put(rec)

        snap = stats.snapshot()
        if (not quiet) and snap["finished_files"] % progress_every == 0:
            print(
                f"[progress] files={snap['finished_files']} ok={snap['ok_files']} err={snap['err_files']} "
                f"audio={snap['audio_sec']:.1f}s wall={snap['wall_sec']:.1f}s "
                f"xRT={snap['throughput_x']:.2f} overall_RTF={snap['overall_rtf']:.4f} infer_RTF={snap['infer_rtf']:.4f}",
                flush=True,
            )


def main() -> int:
    args = parse_args()

    files = discover_files(args)
    if not files:
        print("No input files found.", file=sys.stderr)
        return 2

    if args.resume:
        done = load_done_set(args.output)
        if done:
            files = [p for p in files if p not in done]

    if not files:
        print("Nothing to do: all discovered files are already present in the output.")
        return 0

    if not args.quiet:
        print(
            f"Discovered {len(files)} file(s) | read_workers={args.read_workers} | recognizer_threads={args.num_threads} | "
            f"batch_size={args.batch_size} | max_wait_ms={args.max_wait_ms} | loaded_queue_size={args.loaded_queue_size}",
            flush=True,
        )

    model_start = time.perf_counter()
    model_paths = download_model(
        quantize=args.quantize,
        cache_dir=args.cache_dir,
        mode=args.download_mode,
        local_files_only=args.local_files_only,
    )
    recognizer = create_recognizer(args, model_paths)
    model_elapsed = time.perf_counter() - model_start

    if not args.quiet:
        print(f"Model ready in {model_elapsed:.2f}s", flush=True)

    file_queue: queue.Queue = queue.Queue(maxsize=max(args.loaded_queue_size, args.read_workers * 2))
    loaded_queue: queue.Queue = queue.Queue(maxsize=args.loaded_queue_size)
    result_queue: queue.Queue = queue.Queue(maxsize=max(args.batch_size * 16, 256))

    writer = JsonlWriter(args.output, result_queue, flush_every=args.flush_every)
    writer.start()

    producers: list[threading.Thread] = []
    for idx in range(args.read_workers):
        t = threading.Thread(
            target=producer_loop,
            name=f"reader-{idx}",
            args=(file_queue, loaded_queue, args.allow_resample),
            daemon=True,
        )
        t.start()
        producers.append(t)

    enqueue_start = time.perf_counter()
    for path in files:
        file_queue.put(path)
    for _ in producers:
        file_queue.put(_SENTINEL)
    enqueue_elapsed = time.perf_counter() - enqueue_start

    stats = ThroughputStats()
    inference_loop(
        recognizer=recognizer,
        loaded_queue=loaded_queue,
        result_queue=result_queue,
        num_producers=len(producers),
        batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        stats=stats,
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

    snap = stats.snapshot()
    if not args.quiet:
        print(
            f"Finished | files={snap['finished_files']} ok={snap['ok_files']} err={snap['err_files']} "
            f"audio={snap['audio_sec']:.1f}s wall={snap['wall_sec']:.1f}s xRT={snap['throughput_x']:.2f} "
            f"overall_RTF={snap['overall_rtf']:.4f} infer_RTF={snap['infer_rtf']:.4f} "
            f"enqueue={enqueue_elapsed:.2f}s",
            flush=True,
        )
        print(f"Output written to: {args.output}", flush=True)

    return 0 if snap["ok_files"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
# python infer_onnx_bulk.py --audio-dir data --recursive --extensions .wav
#  --output results.jsonl --allow-resample