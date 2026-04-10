#!/usr/bin/env python3
"""
GPU-oriented Gipformer ONNX inference for very large audio folders.

Pipeline:
- 1 producer thread: discovers/enqueues files
- N preprocess workers: read audio + mono mix + resample to 16 kHz
- 1 ASR consumer (main thread): batch decode on GPU with sherpa-onnx
- JSONL output only: {"wav": ..., "text": ...}
- Supports resume

Notes:
- Keeps explicit resample to 16 kHz as requested
- Preprocess workers push finished items into a queue; whichever finishes first
  can be consumed first by the ASR thread
- Output order is completion order, not original input order
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    print(
        "Error: sherpa-onnx is not installed. Install with: pip install sherpa-onnx",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print(
        "Error: huggingface_hub is not installed. Install with: pip install huggingface_hub",
        file=sys.stderr,
    )
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


# ---------------------------
# Data classes
# ---------------------------

@dataclass
class TaskItem:
    index: int
    path: str
    wav_key: str


@dataclass
class ReadyItem:
    kind: str  # "ok" | "err" | "worker_done"
    index: int = 0
    path: str = ""
    wav_key: str = ""
    samples: Optional[np.ndarray] = None
    duration_sec: float = 0.0
    prep_time_sec: float = 0.0
    error: str = ""


# ---------------------------
# CLI helpers
# ---------------------------

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


def positive_float(value: str) -> float:
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive float, got {value}")
    return v


def non_negative_float(value: str) -> float:
    v = float(value)
    if v < 0:
        raise argparse.ArgumentTypeError(f"Expected non-negative float, got {value}")
    return v


def parse_args() -> argparse.Namespace:
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Queue-based Gipformer ONNX inference with parallel resampling and batched GPU ASR",
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
    model.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")

    infer = parser.add_argument_group("Inference")
    infer.add_argument(
        "--num-threads",
        type=non_negative_int,
        default=max(1, cpu_count - 1),
        help="Recognizer CPU threads used internally by sherpa-onnx / ONNX Runtime",
    )
    infer.add_argument(
        "--decoding-method",
        choices=["greedy_search", "modified_beam_search"],
        default="greedy_search",
    )
    infer.add_argument("--max-active-paths", type=positive_int, default=4)
    infer.add_argument(
        "--batch-size",
        type=positive_int,
        default=16,
        help="Max number of utterances per GPU decode batch",
    )
    infer.add_argument(
        "--max-batch-wait-ms",
        type=non_negative_int,
        default=60,
        help="If pending batch is not full, wait at most this many ms before decoding it",
    )
    infer.add_argument(
        "--max-batch-seconds",
        type=positive_float,
        default=120.0,
        help="Upper bound of summed audio seconds per GPU batch for stability",
    )

    pipe = parser.add_argument_group("Pipeline")
    pipe.add_argument(
        "--resample-workers",
        type=positive_int,
        default=min(8, max(2, cpu_count)),
        help="Parallel workers for file read + mono mix + resample to 16 kHz",
    )
    pipe.add_argument(
        "--task-queue-size",
        type=positive_int,
        default=256,
        help="Bounded queue size for paths waiting to preprocess",
    )
    pipe.add_argument(
        "--ready-queue-size",
        type=positive_int,
        default=128,
        help="Bounded queue size for resampled audios waiting for ASR",
    )

    out = parser.add_argument_group("Output")
    out.add_argument("--output", type=str, default="results.jsonl")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--flush-every", type=positive_int, default=50)
    out.add_argument("--fsync-every", type=non_negative_int, default=0)
    out.add_argument("--progress-every", type=positive_int, default=100)
    out.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.audio and not args.audio_dir and not args.manifest:
        parser.error("Provide at least one of --audio, --audio-dir, or --manifest")

    return args


# ---------------------------
# File discovery / resume
# ---------------------------

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


def make_wav_key(path: str, audio_dir: str) -> str:
    path = os.path.abspath(path)
    if audio_dir:
        root = os.path.abspath(os.path.expanduser(audio_dir))
        try:
            return os.path.relpath(path, root)
        except ValueError:
            return os.path.basename(path)
    return os.path.basename(path)


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


# ---------------------------
# Model download / recognizer
# ---------------------------

def download_model(
    quantize: str,
    cache_dir: str = "",
    mode: str = "snapshot",
    local_files_only: bool = False,
) -> dict[str, str]:
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


def create_recognizer(
    args: argparse.Namespace,
    model_paths: dict[str, str],
) -> sherpa_onnx.OfflineRecognizer:
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
        provider=args.provider,
    )


# ---------------------------
# Audio preprocessing
# ---------------------------

def resample_audio(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.ascontiguousarray(samples, dtype=np.float32)

    if scipy_resample_poly is not None:
        g = math.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        resampled = scipy_resample_poly(samples, up, down)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    old_len = len(samples)
    if old_len == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)

    new_len = int(round(old_len * dst_sr / src_sr))
    if new_len <= 1:
        return np.ascontiguousarray(samples[:1], dtype=np.float32)

    old_idx = np.linspace(0.0, 1.0, num=old_len, endpoint=True)
    new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    resampled = np.interp(new_idx, old_idx, samples)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def read_audio_resample_16k(path: str) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)

    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)

    samples = np.asarray(samples, dtype=np.float32)
    samples = resample_audio(samples, sample_rate, SAMPLE_RATE)
    return np.ascontiguousarray(samples, dtype=np.float32)


# ---------------------------
# Queues / threads
# ---------------------------

_TASK_STOP = object()


def preprocess_worker(
    worker_id: int,
    task_queue: "queue.Queue[object]",
    ready_queue: "queue.Queue[ReadyItem]",
) -> None:
    del worker_id

    while True:
        task = task_queue.get()
        if task is _TASK_STOP:
            task_queue.task_done()
            break

        assert isinstance(task, TaskItem)
        t0 = time.perf_counter()

        try:
            samples = read_audio_resample_16k(task.path)
            prep_elapsed = time.perf_counter() - t0

            ready_queue.put(
                ReadyItem(
                    kind="ok",
                    index=task.index,
                    path=task.path,
                    wav_key=task.wav_key,
                    samples=samples,
                    duration_sec=(len(samples) / SAMPLE_RATE) if len(samples) > 0 else 0.0,
                    prep_time_sec=prep_elapsed,
                )
            )

        except Exception as e:
            prep_elapsed = time.perf_counter() - t0
            ready_queue.put(
                ReadyItem(
                    kind="err",
                    index=task.index,
                    path=task.path,
                    wav_key=task.wav_key,
                    prep_time_sec=prep_elapsed,
                    error=f"{type(e).__name__}: {e}",
                )
            )

        finally:
            task_queue.task_done()

    ready_queue.put(ReadyItem(kind="worker_done"))


def producer_thread_fn(
    files: list[str],
    audio_dir: str,
    task_queue: "queue.Queue[object]",
    num_workers: int,
) -> None:
    try:
        for idx, path in enumerate(files, 1):
            wav_key = make_wav_key(path, audio_dir)
            task_queue.put(TaskItem(index=idx, path=path, wav_key=wav_key))
    finally:
        for _ in range(num_workers):
            task_queue.put(_TASK_STOP)


# ---------------------------
# Output / logging
# ---------------------------

def write_jsonl_line(f, wav_key: str, text: str) -> None:
    line = json.dumps(
        {"wav": wav_key, "text": text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    f.write(line + "\n")


def maybe_flush(f, written_count: int, flush_every: int, fsync_every: int) -> None:
    if written_count % flush_every == 0:
        f.flush()
    if fsync_every > 0 and written_count % fsync_every == 0:
        f.flush()
        os.fsync(f.fileno())


def format_status_line(
    idx: int,
    total: int,
    wav_key: str,
    status: str,
    prep_time: float,
    asr_time: float,
    duration: float,
) -> str:
    rtf = (asr_time / duration) if duration > 0 else 0.0
    return (
        f"[{idx}/{total}] {wav_key} | {status} "
        f"| prep={prep_time:.2f}s asr={asr_time:.2f}s dur={duration:.2f}s rtf={rtf:.3f}"
    )


# ---------------------------
# Batch decode
# ---------------------------

def build_decode_batch(
    pending: list[ReadyItem],
    batch_size: int,
    max_batch_seconds: float,
) -> list[ReadyItem]:
    if not pending:
        return []

    batch: list[ReadyItem] = []
    total_sec = 0.0

    for item in pending:
        if item.kind != "ok":
            break

        would_count = len(batch) + 1
        would_sec = total_sec + item.duration_sec

        if batch and would_count > batch_size:
            break

        if batch and would_sec > max_batch_seconds:
            break

        batch.append(item)
        total_sec = would_sec

        if len(batch) >= batch_size:
            break

    if not batch:
        batch.append(pending[0])

    del pending[: len(batch)]
    return batch


def decode_batch(
    recognizer: sherpa_onnx.OfflineRecognizer,
    batch: list[ReadyItem],
) -> tuple[list[tuple[ReadyItem, str, float]], list[tuple[ReadyItem, str, float]]]:
    """
    Returns:
      ok_results: [(item, text, asr_time_sec), ...]
      err_results: [(item, error_message, asr_time_sec), ...]
    """
    if not batch:
        return [], []

    streams = []
    t0 = time.perf_counter()

    try:
        for item in batch:
            assert item.samples is not None
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, item.samples)
            streams.append(s)

        recognizer.decode_streams(streams)
        elapsed = time.perf_counter() - t0

        ok_results = []
        per_item_asr = elapsed / max(1, len(batch))
        for item, stream in zip(batch, streams):
            text = stream.result.text.strip()
            ok_results.append((item, text, per_item_asr))

        return ok_results, []

    except Exception as batch_error:
        # Fallback to single-item decode so one bad item does not ruin the whole batch.
        err_results: list[tuple[ReadyItem, str, float]] = []
        ok_results: list[tuple[ReadyItem, str, float]] = []

        for item in batch:
            t1 = time.perf_counter()
            try:
                assert item.samples is not None
                s = recognizer.create_stream()
                s.accept_waveform(SAMPLE_RATE, item.samples)
                recognizer.decode_streams([s])
                elapsed = time.perf_counter() - t1
                ok_results.append((item, s.result.text.strip(), elapsed))
            except Exception as e:
                elapsed = time.perf_counter() - t1
                err_results.append(
                    (
                        item,
                        f"batch={type(batch_error).__name__}: {batch_error}; "
                        f"single={type(e).__name__}: {e}",
                        elapsed,
                    )
                )

        return ok_results, err_results


# ---------------------------
# Main
# ---------------------------

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
    total = len(files)

    if total == 0:
        print("Nothing to do.")
        return 0

    if not args.quiet:
        print(f"Discovered: {total} file(s)", flush=True)
        print(f"Resample workers: {args.resample_workers}", flush=True)
        print(f"Recognizer threads: {args.num_threads}", flush=True)
        print(f"Batch size: {args.batch_size}", flush=True)
        print(f"Max batch wait: {args.max_batch_wait_ms} ms", flush=True)
        print(f"Max batch seconds: {args.max_batch_seconds}", flush=True)
        print(f"Provider: {args.provider}", flush=True)
        print(f"Output: {args.output}", flush=True)

    t_model = time.perf_counter()
    model_paths = download_model(
        quantize=args.quantize,
        cache_dir=args.cache_dir,
        mode=args.download_mode,
        local_files_only=args.local_files_only,
    )
    recognizer = create_recognizer(args, model_paths)
    if not args.quiet:
        print(f"Model ready in {time.perf_counter() - t_model:.2f}s", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    task_queue: "queue.Queue[object]" = queue.Queue(maxsize=args.task_queue_size)
    ready_queue: "queue.Queue[ReadyItem]" = queue.Queue(maxsize=args.ready_queue_size)

    producer = threading.Thread(
        target=producer_thread_fn,
        args=(files, args.audio_dir, task_queue, args.resample_workers),
        daemon=True,
        name="producer",
    )
    producer.start()

    workers: list[threading.Thread] = []
    for i in range(args.resample_workers):
        t = threading.Thread(
            target=preprocess_worker,
            args=(i, task_queue, ready_queue),
            daemon=True,
            name=f"prep-{i}",
        )
        t.start()
        workers.append(t)

    max_wait_sec = args.max_batch_wait_ms / 1000.0
    pending: list[ReadyItem] = []

    started = time.perf_counter()
    written_count = 0
    processed_count = 0
    ok_count = 0
    err_count = 0
    active_workers = args.resample_workers

    total_audio_sec = 0.0
    total_prep_sec = 0.0
    total_asr_sec = 0.0

    with open(args.output, "a", encoding="utf-8") as fout:
        while active_workers > 0 or pending:
            item: Optional[ReadyItem] = None

            try:
                timeout = max_wait_sec if pending else 0.5
                item = ready_queue.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is not None:
                if item.kind == "worker_done":
                    active_workers -= 1

                elif item.kind == "err":
                    processed_count += 1
                    err_count += 1
                    written_count += 1

                    write_jsonl_line(fout, item.wav_key, "")
                    maybe_flush(fout, written_count, args.flush_every, args.fsync_every)

                    if not args.quiet:
                        print(
                            format_status_line(
                                idx=processed_count,
                                total=total,
                                wav_key=item.wav_key,
                                status=f"prep_err: {item.error}",
                                prep_time=item.prep_time_sec,
                                asr_time=0.0,
                                duration=0.0,
                            ),
                            flush=True,
                        )
                    elif processed_count % args.progress_every == 0:
                        print(f"[progress] {processed_count}/{total}", flush=True)

                elif item.kind == "ok":
                    pending.append(item)

            should_decode = False
            if pending:
                if len(pending) >= args.batch_size:
                    should_decode = True
                elif item is None:
                    should_decode = True
                elif active_workers == 0 and ready_queue.empty():
                    should_decode = True

            if should_decode:
                batch = build_decode_batch(
                    pending=pending,
                    batch_size=args.batch_size,
                    max_batch_seconds=args.max_batch_seconds,
                )

                ok_results, err_results = decode_batch(recognizer, batch)

                for ready, text, asr_elapsed in ok_results:
                    processed_count += 1
                    ok_count += 1
                    written_count += 1

                    total_audio_sec += ready.duration_sec
                    total_prep_sec += ready.prep_time_sec
                    total_asr_sec += asr_elapsed

                    write_jsonl_line(fout, ready.wav_key, text)
                    maybe_flush(fout, written_count, args.flush_every, args.fsync_every)

                    if not args.quiet:
                        print(
                            format_status_line(
                                idx=processed_count,
                                total=total,
                                wav_key=ready.wav_key,
                                status="ok",
                                prep_time=ready.prep_time_sec,
                                asr_time=asr_elapsed,
                                duration=ready.duration_sec,
                            ),
                            flush=True,
                        )
                    elif processed_count % args.progress_every == 0:
                        print(f"[progress] {processed_count}/{total}", flush=True)

                for ready, error_msg, asr_elapsed in err_results:
                    processed_count += 1
                    err_count += 1
                    written_count += 1

                    total_audio_sec += ready.duration_sec
                    total_prep_sec += ready.prep_time_sec
                    total_asr_sec += asr_elapsed

                    write_jsonl_line(fout, ready.wav_key, "")
                    maybe_flush(fout, written_count, args.flush_every, args.fsync_every)

                    if not args.quiet:
                        print(
                            format_status_line(
                                idx=processed_count,
                                total=total,
                                wav_key=ready.wav_key,
                                status=f"asr_err: {error_msg}",
                                prep_time=ready.prep_time_sec,
                                asr_time=asr_elapsed,
                                duration=ready.duration_sec,
                            ),
                            flush=True,
                        )
                    elif processed_count % args.progress_every == 0:
                        print(f"[progress] {processed_count}/{total}", flush=True)

        fout.flush()
        if args.fsync_every > 0:
            os.fsync(fout.fileno())

    producer.join()
    for t in workers:
        t.join()

    total_elapsed = time.perf_counter() - started
    overall_rtf = (total_elapsed / total_audio_sec) if total_audio_sec > 0 else 0.0
    asr_rtf = (total_asr_sec / total_audio_sec) if total_audio_sec > 0 else 0.0

    print(
        (
            f"Finished | total={total} ok={ok_count} err={err_count} "
            f"time={total_elapsed:.2f}s audio={total_audio_sec:.2f}s "
            f"prep={total_prep_sec:.2f}s asr={total_asr_sec:.2f}s "
            f"overall_rtf={overall_rtf:.4f} asr_rtf={asr_rtf:.4f}"
        ),
        flush=True,
    )

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
