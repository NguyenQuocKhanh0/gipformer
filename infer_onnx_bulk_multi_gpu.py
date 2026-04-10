#!/usr/bin/env python3
"""
Multi-GPU Gipformer ONNX inference with bounded memory and worker recycling.

Key ideas to reduce long-running VRAM growth:
- Parent only sends lightweight file tasks across processes.
- Each GPU process does its own read + resample + ASR locally.
- Local queues are bounded to limit in-flight numpy arrays.
- Each GPU worker can be recycled periodically to release CUDA/ORT memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
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


# ---------------------------
# Data models
# ---------------------------

@dataclass
class FileTask:
    index: int
    path: str
    wav_key: str


@dataclass
class LocalReadyItem:
    kind: str  # ok | err | worker_done
    index: int = 0
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
        description="Bounded-memory multi-GPU Gipformer ONNX inference",
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
    src.add_argument("--sort", choices=["none", "size_asc", "size_desc"], default="none")

    model = parser.add_argument_group("Model")
    model.add_argument("--quantize", choices=["fp32", "int8"], default="fp32")
    model.add_argument("--cache-dir", type=str, default="")
    model.add_argument("--download-mode", choices=["snapshot", "single"], default="snapshot")
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")

    infer = parser.add_argument_group("Inference")
    infer.add_argument("--num-threads", type=non_negative_int, default=max(1, cpu_count - 1))
    infer.add_argument("--decoding-method", choices=["greedy_search", "modified_beam_search"], default="greedy_search")
    infer.add_argument("--max-active-paths", type=positive_int, default=4)
    infer.add_argument("--batch-size", type=positive_int, default=8)
    infer.add_argument("--max-batch-wait-ms", type=non_negative_int, default=40)
    infer.add_argument("--max-batch-seconds", type=positive_float, default=45.0)

    mgpu = parser.add_argument_group("Multi-GPU")
    mgpu.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1], help="Physical GPU ids to use")

    pipe = parser.add_argument_group("Pipeline")
    pipe.add_argument("--shared-task-queue-size", type=positive_int, default=128)
    pipe.add_argument("--local-task-queue-size", type=positive_int, default=16)
    pipe.add_argument("--local-ready-queue-size", type=positive_int, default=8)
    pipe.add_argument(
        "--resample-workers-per-gpu",
        type=positive_int,
        default=max(1, min(4, cpu_count // max(1, 2))),
        help="Read/resample threads inside each GPU process",
    )

    recycle = parser.add_argument_group("Worker recycle")
    recycle.add_argument(
        "--recycle-every-files",
        type=non_negative_int,
        default=1500,
        help="Recycle each GPU worker after this many decoded files; 0 disables",
    )
    recycle.add_argument(
        "--recycle-every-audio-seconds",
        type=non_negative_float,
        default=0.0,
        help="Recycle each GPU worker after this many decoded audio seconds; 0 disables",
    )
    recycle.add_argument(
        "--gc-every-batches",
        type=positive_int,
        default=20,
        help="Run gc.collect() every N ASR batches in each worker",
    )

    out = parser.add_argument_group("Output")
    out.add_argument("--output", type=str, default="results.jsonl")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--flush-every", type=positive_int, default=100)
    out.add_argument("--fsync-every", type=non_negative_int, default=0)
    out.add_argument("--progress-every", type=positive_int, default=100)
    out.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.audio and not args.audio_dir and not args.manifest:
        parser.error("Provide at least one of --audio, --audio-dir, or --manifest")

    if args.provider == "cpu":
        args.gpu_ids = [0]

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
# Model
# ---------------------------

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


# ---------------------------
# Audio
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
# Child worker internals
# ---------------------------

_LOCAL_STOP = object()


def local_preprocess_worker(
    in_q: "queue.Queue[object]",
    ready_q: "queue.Queue[LocalReadyItem]",
) -> None:
    while True:
        item = in_q.get()
        if item is _LOCAL_STOP:
            in_q.task_done()
            break

        assert isinstance(item, FileTask)
        t0 = time.perf_counter()
        try:
            samples = read_audio_resample_16k(item.path)
            dt = time.perf_counter() - t0
            ready_q.put(
                LocalReadyItem(
                    kind="ok",
                    index=item.index,
                    wav_key=item.wav_key,
                    samples=samples,
                    duration_sec=(len(samples) / SAMPLE_RATE) if len(samples) > 0 else 0.0,
                    prep_time_sec=dt,
                )
            )
        except Exception as e:
            dt = time.perf_counter() - t0
            ready_q.put(
                LocalReadyItem(
                    kind="err",
                    index=item.index,
                    wav_key=item.wav_key,
                    samples=None,
                    duration_sec=0.0,
                    prep_time_sec=dt,
                    error=f"{type(e).__name__}: {e}",
                )
            )
        finally:
            in_q.task_done()

    ready_q.put(LocalReadyItem(kind="worker_done"))


def create_recognizer_in_child(args_dict: dict, model_paths: dict[str, str]):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=model_paths["encoder"],
        decoder=model_paths["decoder"],
        joiner=model_paths["joiner"],
        tokens=model_paths["tokens"],
        num_threads=args_dict["num_threads"],
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method=args_dict["decoding_method"],
        max_active_paths=args_dict["max_active_paths"],
        provider=args_dict["provider"],
    )


def build_batch(pending: list[LocalReadyItem], batch_size: int, max_batch_seconds: float) -> list[LocalReadyItem]:
    batch: list[LocalReadyItem] = []
    total_sec = 0.0

    while pending and len(batch) < batch_size:
        item = pending[0]
        if item.kind != "ok":
            break

        if batch and (total_sec + item.duration_sec > max_batch_seconds):
            break

        batch.append(item)
        total_sec += item.duration_sec
        pending.pop(0)

    if not batch and pending:
        batch.append(pending.pop(0))

    return batch


def send_result(
    result_q: "mp.Queue[dict]",
    gpu_id: int,
    item: LocalReadyItem,
    text: str,
    asr_time_sec: float,
    error: str = "",
) -> None:
    result_q.put(
        {
            "kind": "result",
            "gpu_id": gpu_id,
            "index": item.index,
            "wav_key": item.wav_key,
            "text": text,
            "duration_sec": item.duration_sec,
            "prep_time_sec": item.prep_time_sec,
            "asr_time_sec": asr_time_sec,
            "error": error,
        }
    )


def gpu_worker_main(
    gpu_id: int,
    shared_task_q: "mp.Queue[FileTask]",
    producer_done: "mp.Event",
    result_q: "mp.Queue[dict]",
    model_paths: dict[str, str],
    args_dict: dict,
) -> None:
    # Must happen before importing/loading CUDA-dependent runtime in this process.
    if args_dict["provider"] == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    local_task_q: "queue.Queue[object]" = queue.Queue(maxsize=args_dict["local_task_queue_size"])
    local_ready_q: "queue.Queue[LocalReadyItem]" = queue.Queue(maxsize=args_dict["local_ready_queue_size"])

    stop_fetch = threading.Event()
    fetch_done = threading.Event()

    decoded_files = 0
    decoded_audio_sec = 0.0
    recycled = False

    def feeder() -> None:
        try:
            while not stop_fetch.is_set():
                try:
                    task = shared_task_q.get(timeout=0.2)
                except queue.Empty:
                    if producer_done.is_set():
                        break
                    continue
                local_task_q.put(task)
        finally:
            for _ in range(args_dict["resample_workers_per_gpu"]):
                local_task_q.put(_LOCAL_STOP)
            fetch_done.set()

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    prep_threads: list[threading.Thread] = []
    for _ in range(args_dict["resample_workers_per_gpu"]):
        t = threading.Thread(target=local_preprocess_worker, args=(local_task_q, local_ready_q), daemon=True)
        t.start()
        prep_threads.append(t)

    try:
        recognizer = create_recognizer_in_child(args_dict, model_paths)
    except Exception as e:
        result_q.put(
            {
                "kind": "worker_exit",
                "gpu_id": gpu_id,
                "recycled": False,
                "fatal_error": f"init_err: {type(e).__name__}: {e}",
            }
        )
        return

    pending: list[LocalReadyItem] = []
    prep_workers_alive = args_dict["resample_workers_per_gpu"]
    batch_counter = 0

    try:
        while True:
            item = None
            try:
                timeout = (args_dict["max_batch_wait_ms"] / 1000.0) if pending else 0.5
                item = local_ready_q.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is not None:
                if item.kind == "worker_done":
                    prep_workers_alive -= 1
                elif item.kind == "err":
                    send_result(result_q, gpu_id, item, text="", asr_time_sec=0.0, error=f"prep_err: {item.error}")
                else:
                    pending.append(item)

            should_decode = False
            if pending:
                if len(pending) >= args_dict["batch_size"]:
                    should_decode = True
                elif item is None:
                    should_decode = True
                elif prep_workers_alive == 0 and local_ready_q.empty():
                    should_decode = True

            if should_decode:
                batch = build_batch(
                    pending=pending,
                    batch_size=args_dict["batch_size"],
                    max_batch_seconds=args_dict["max_batch_seconds"],
                )

                if batch:
                    try:
                        streams = []
                        t0 = time.perf_counter()

                        for b in batch:
                            s = recognizer.create_stream()
                            s.accept_waveform(SAMPLE_RATE, b.samples)
                            streams.append(s)

                        recognizer.decode_streams(streams)
                        elapsed = time.perf_counter() - t0
                        per_item = elapsed / max(1, len(batch))

                        for b, s in zip(batch, streams):
                            send_result(result_q, gpu_id, b, text=s.result.text.strip(), asr_time_sec=per_item)
                            decoded_files += 1
                            decoded_audio_sec += b.duration_sec

                    except Exception as batch_error:
                        for b in batch:
                            t1 = time.perf_counter()
                            try:
                                s = recognizer.create_stream()
                                s.accept_waveform(SAMPLE_RATE, b.samples)
                                recognizer.decode_streams([s])
                                elapsed = time.perf_counter() - t1
                                send_result(result_q, gpu_id, b, text=s.result.text.strip(), asr_time_sec=elapsed)
                                decoded_files += 1
                                decoded_audio_sec += b.duration_sec
                            except Exception as e:
                                elapsed = time.perf_counter() - t1
                                send_result(
                                    result_q,
                                    gpu_id,
                                    b,
                                    text="",
                                    asr_time_sec=elapsed,
                                    error=f"asr_err: batch={type(batch_error).__name__}: {batch_error}; "
                                          f"single={type(e).__name__}: {e}",
                                )

                    finally:
                        for b in batch:
                            b.samples = None
                        del batch
                        if "streams" in locals():
                            del streams
                        batch_counter += 1
                        if batch_counter % args_dict["gc_every_batches"] == 0:
                            gc.collect()

            if args_dict["recycle_every_files"] > 0 and decoded_files >= args_dict["recycle_every_files"]:
                recycled = True
                stop_fetch.set()

            if (
                args_dict["recycle_every_audio_seconds"] > 0
                and decoded_audio_sec >= args_dict["recycle_every_audio_seconds"]
            ):
                recycled = True
                stop_fetch.set()

            if fetch_done.is_set() and prep_workers_alive == 0 and not pending:
                break

    finally:
        stop_fetch.set()
        feeder_thread.join()
        local_task_q.join()
        for t in prep_threads:
            t.join()

        # Drop recognizer/session so process exit can return memory.
        try:
            del recognizer
        except Exception:
            pass
        gc.collect()

        result_q.put(
            {
                "kind": "worker_exit",
                "gpu_id": gpu_id,
                "recycled": recycled,
                "fatal_error": "",
            }
        )


# ---------------------------
# Main output helpers
# ---------------------------

def write_jsonl_line(f, wav_key: str, text: str) -> None:
    line = json.dumps({"wav": wav_key, "text": text}, ensure_ascii=False, separators=(",", ":"))
    f.write(line + "\n")


def maybe_flush(f, written_count: int, flush_every: int, fsync_every: int) -> None:
    if written_count % flush_every == 0:
        f.flush()
    if fsync_every > 0 and written_count % fsync_every == 0:
        f.flush()
        os.fsync(f.fileno())


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    args = parse_args()

    files = discover_files(args)
    if not files:
        print("No input files found.", file=sys.stderr)
        return 2

    done = load_done_set(args.output) if args.resume else set()
    filtered: list[str] = []
    for p in files:
        wav_key = make_wav_key(p, args.audio_dir)
        if wav_key not in done:
            filtered.append(p)

    files = filtered
    total = len(files)

    if total == 0:
        print("Nothing to do.")
        return 0

    if not args.quiet:
        print(f"Discovered: {total} file(s)", flush=True)
        print(f"GPU ids: {args.gpu_ids}", flush=True)
        print(f"Batch size: {args.batch_size}", flush=True)
        print(f"Max batch seconds: {args.max_batch_seconds}", flush=True)
        print(f"Resample workers / GPU: {args.resample_workers_per_gpu}", flush=True)
        print(f"Recycle every files: {args.recycle_every_files}", flush=True)
        print(f"Recycle every audio seconds: {args.recycle_every_audio_seconds}", flush=True)
        print(f"Output: {args.output}", flush=True)

    t0 = time.perf_counter()
    model_paths = download_model(
        quantize=args.quantize,
        cache_dir=args.cache_dir,
        mode=args.download_mode,
        local_files_only=args.local_files_only,
    )
    if not args.quiet:
        print(f"Model files ready in {time.perf_counter() - t0:.2f}s", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ctx = mp.get_context("spawn")
    shared_task_q: "mp.Queue[FileTask]" = ctx.Queue(maxsize=args.shared_task_queue_size)
    result_q: "mp.Queue[dict]" = ctx.Queue(maxsize=256)
    producer_done = ctx.Event()

    def producer() -> None:
        try:
            for idx, path in enumerate(files, 1):
                shared_task_q.put(FileTask(index=idx, path=path, wav_key=make_wav_key(path, args.audio_dir)))
        finally:
            producer_done.set()

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    args_dict = {
        "provider": args.provider,
        "num_threads": args.num_threads,
        "decoding_method": args.decoding_method,
        "max_active_paths": args.max_active_paths,
        "batch_size": args.batch_size,
        "max_batch_wait_ms": args.max_batch_wait_ms,
        "max_batch_seconds": args.max_batch_seconds,
        "local_task_queue_size": args.local_task_queue_size,
        "local_ready_queue_size": args.local_ready_queue_size,
        "resample_workers_per_gpu": args.resample_workers_per_gpu,
        "recycle_every_files": args.recycle_every_files,
        "recycle_every_audio_seconds": args.recycle_every_audio_seconds,
        "gc_every_batches": args.gc_every_batches,
    }

    active_workers: dict[int, mp.Process] = {}

    def spawn_worker(gpu_id: int) -> None:
        p = ctx.Process(
            target=gpu_worker_main,
            args=(gpu_id, shared_task_q, producer_done, result_q, model_paths, args_dict),
            daemon=True,
        )
        p.start()
        active_workers[gpu_id] = p

    gpu_ids = args.gpu_ids if args.provider == "cuda" else [0]
    for gpu_id in gpu_ids:
        spawn_worker(gpu_id)

    processed_count = 0
    written_count = 0
    ok_count = 0
    err_count = 0

    total_audio_sec = 0.0
    total_prep_sec = 0.0
    total_asr_sec = 0.0

    worker_restarts = {gpu_id: 0 for gpu_id in gpu_ids}
    started = time.perf_counter()

    with open(args.output, "a", encoding="utf-8") as fout:
        while processed_count < total or active_workers:
            try:
                msg = result_q.get(timeout=0.5)
            except queue.Empty:
                msg = None

            if msg is not None:
                if msg["kind"] == "result":
                    processed_count += 1
                    written_count += 1
                    total_audio_sec += msg["duration_sec"]
                    total_prep_sec += msg["prep_time_sec"]
                    total_asr_sec += msg["asr_time_sec"]

                    if msg["error"]:
                        err_count += 1
                        write_jsonl_line(fout, msg["wav_key"], "")
                        status = msg["error"]
                    else:
                        ok_count += 1
                        write_jsonl_line(fout, msg["wav_key"], msg["text"])
                        status = "ok"

                    maybe_flush(fout, written_count, args.flush_every, args.fsync_every)

                    if not args.quiet:
                        rtf = (msg["asr_time_sec"] / msg["duration_sec"]) if msg["duration_sec"] > 0 else 0.0
                        print(
                            f"[{processed_count}/{total}] gpu={msg['gpu_id']} {msg['wav_key']} | {status} "
                            f"| prep={msg['prep_time_sec']:.2f}s asr={msg['asr_time_sec']:.2f}s "
                            f"dur={msg['duration_sec']:.2f}s rtf={rtf:.3f}",
                            flush=True,
                        )
                    elif processed_count % args.progress_every == 0:
                        print(f"[progress] {processed_count}/{total}", flush=True)

                elif msg["kind"] == "worker_exit":
                    gpu_id = msg["gpu_id"]
                    proc = active_workers.pop(gpu_id, None)
                    if proc is not None:
                        proc.join(timeout=1)

                    if msg["fatal_error"] and not args.quiet:
                        print(f"[worker-exit] gpu={gpu_id} fatal={msg['fatal_error']}", flush=True)
                    elif msg["recycled"] and not args.quiet:
                        print(f"[worker-exit] gpu={gpu_id} recycled", flush=True)

                    if processed_count < total:
                        worker_restarts[gpu_id] += 1
                        spawn_worker(gpu_id)

            # Clean up unexpectedly dead workers too.
            dead_gpu_ids = []
            for gpu_id, proc in active_workers.items():
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    dead_gpu_ids.append(gpu_id)

            for gpu_id in dead_gpu_ids:
                active_workers.pop(gpu_id, None)
                if processed_count < total:
                    worker_restarts[gpu_id] += 1
                    spawn_worker(gpu_id)

            if processed_count >= total:
                break

        fout.flush()
        if args.fsync_every > 0:
            os.fsync(fout.fileno())

    prod_thread.join()

    for proc in active_workers.values():
        proc.join(timeout=2)

    total_elapsed = time.perf_counter() - started
    overall_rtf = (total_elapsed / total_audio_sec) if total_audio_sec > 0 else 0.0
    asr_rtf = (total_asr_sec / total_audio_sec) if total_audio_sec > 0 else 0.0

    print(
        f"Finished | total={total} ok={ok_count} err={err_count} "
        f"time={total_elapsed:.2f}s audio={total_audio_sec:.2f}s "
        f"prep={total_prep_sec:.2f}s asr={total_asr_sec:.2f}s "
        f"overall_rtf={overall_rtf:.4f} asr_rtf={asr_rtf:.4f} "
        f"restarts={worker_restarts}",
        flush=True,
    )

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
