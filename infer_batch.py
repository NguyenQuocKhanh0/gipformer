#!/usr/bin/env python3
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from fractions import Fraction
from typing import Dict, List, Optional
from chunkformer import ChunkFormerModel
import numpy as np
import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    raise ImportError("Cần cài sherpa-onnx: pip install sherpa-onnx")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("Cần cài huggingface_hub: pip install huggingface_hub")

# scipy cho resample chất lượng tốt hơn; không có thì fallback nội suy tuyến tính
try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SAMPLE_RATE = 16000
FEATURE_DIM = 80
DEFAULT_REPO_ID = "g-group-ai-lab/gipformer-65M-rnnt"

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


def read_audio(filename: str):
    samples, sample_rate = sf.read(filename, dtype="float32", always_2d=False)

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def resample_to_target_sr(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    if orig_sr == target_sr:
        return np.ascontiguousarray(samples, dtype=np.float32)

    if samples.size == 0:
        return np.asarray(samples, dtype=np.float32)

    if HAS_SCIPY:
        ratio = Fraction(target_sr, orig_sr).limit_denominator()
        out = resample_poly(samples, ratio.numerator, ratio.denominator)
        return np.ascontiguousarray(out, dtype=np.float32)

    # fallback: nội suy tuyến tính
    old_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    new_len = max(1, int(round(len(samples) * target_sr / orig_sr)))
    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    out = np.interp(new_x, old_x, samples)
    return np.ascontiguousarray(out, dtype=np.float32)


def preprocess_audio(audio_path: str) -> Dict:
    samples, sample_rate = read_audio(audio_path)
    samples = resample_to_target_sr(samples, sample_rate, SAMPLE_RATE)

    return {
        "path": audio_path,
        "name": os.path.basename(audio_path),
        "samples": samples,
        "sample_rate": SAMPLE_RATE,
        "duration": len(samples) / SAMPLE_RATE if len(samples) > 0 else 0.0,
    }


def download_model(repo_id: str, quantize: str = "fp32") -> dict:
    if quantize not in ONNX_FILES:
        raise ValueError(f"quantize phải là một trong {list(ONNX_FILES.keys())}")

    files = ONNX_FILES[quantize]
    print(f"Downloading {quantize} model from {repo_id}...")

    paths = {}
    for key, filename in files.items():
        paths[key] = hf_hub_download(repo_id=repo_id, filename=filename)

    paths["tokens"] = hf_hub_download(repo_id=repo_id, filename="tokens.txt")

    print("Model downloaded successfully.")
    return paths


class GipformerOnnxModel:
    """
    Wrapper cho sherpa-onnx.
    """

    def __init__(
        self,
        recognizer: sherpa_onnx.OfflineRecognizer,
        repo_id: str,
        quantize: str,
        num_threads: int,
        decoding_method: str,
        provider: str,
    ):
        self.recognizer = recognizer
        self.repo_id = repo_id
        self.quantize = quantize
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.provider = provider

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        quantize: str = "fp32",
        num_threads: int = 4,
        decoding_method: str = "modified_beam_search",
        provider: str = "cpu",
    ):
        model_paths = download_model(repo_id=repo_id, quantize=quantize)

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=model_paths["encoder"],
            decoder=model_paths["decoder"],
            joiner=model_paths["joiner"],
            tokens=model_paths["tokens"],
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method=decoding_method,
            provider=provider,  # quan trọng: dùng cuda thật sự nếu có
        )

        return cls(
            recognizer=recognizer,
            repo_id=repo_id,
            quantize=quantize,
            num_threads=num_threads,
            decoding_method=decoding_method,
            provider=provider,
        )

    def to(self, device: str):
        """
        Giữ tương thích interface, nhưng không đổi provider sau khi model đã tạo.
        """
        return self

    def decode_preloaded_batch(self, batch_items: List[Dict]) -> List[str]:
        streams = []
        for item in batch_items:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(item["sample_rate"], item["samples"])
            streams.append(stream)

        self.recognizer.decode_streams(streams)
        return [s.result.text.strip() for s in streams]

    def decode(self, audio_path: str) -> str:
        item = preprocess_audio(audio_path)
        return self.decode_preloaded_batch([item])[0]


def flush_consistency_batch(
    batch_items: List[Dict],
    model_a,
    model_b,
    output_dir: str,
    save_txt: bool,
) -> int:
    if not batch_items:
        return 0

    texts_a = model_a.decode_preloaded_batch(batch_items)
    texts_b = model_b.batch_decode(batch_items,
                                    chunk_size=64,
                                    left_context_size=128,
                                    right_context_size=128,
                                    total_batch_duration=600,)

    kept = 0
    for item, t_a, t_b in zip(batch_items, texts_a, texts_b):
        if t_a.strip() == t_b.strip():
            kept += 1
            src_audio = item["path"]
            dst_audio = os.path.join(output_dir, item["name"])
            shutil.copy2(src_audio, dst_audio)

            if save_txt:
                txt_path = os.path.join(
                    output_dir,
                    os.path.splitext(item["name"])[0] + ".txt"
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(t_a.strip())

    return kept


def keep_consistent_asr_parallel(
    audio_dir: str,
    output_dir: str,
    save_txt: bool = True,
    provider: str = "cpu",          # "cuda" nếu sherpa-onnx GPU hoạt động
    preprocess_workers: int = 8,    # số worker đọc + resample
    model_num_threads: int = 4,     # num_threads của mỗi recognizer
    infer_batch_size: int = 8,      # số mẫu / batch infer
    infer_max_batch_seconds: float = 120.0,  # tổng thời lượng tối đa / batch
    max_in_flight: int = 32,        # số file preprocess song song tối đa
    flush_timeout: float = 0.3,     # batch chưa đầy nhưng chờ quá lâu thì infer luôn
):
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    )
    if not audio_files:
        raise ValueError("Không có file wav")

    audio_paths = [os.path.join(audio_dir, f) for f in audio_files]
    total = len(audio_paths)

    print(f"📦 Tổng số file: {total}")
    print(f"⚙️  preprocess_workers={preprocess_workers}, max_in_flight={max_in_flight}")
    print(f"🧠 infer_batch_size={infer_batch_size}, infer_max_batch_seconds={infer_max_batch_seconds}")
    print(f"🚀 provider={provider}, model_num_threads={model_num_threads}")

    # Tạo 2 model để so sánh consistency
    model_a = GipformerOnnxModel.from_pretrained(
        repo_id=DEFAULT_REPO_ID,
        quantize="fp32",
        num_threads=model_num_threads,
        decoding_method="modified_beam_search",
        provider=provider,
    )

    model_b = ChunkFormerModel.from_pretrained(
        "khanhld/chunkformer-rnnt-large-vie"
    ).to(provider)

    kept = 0
    processed = 0
    next_submit_idx = 0

    ready_batch: List[Dict] = []
    ready_batch_duration = 0.0
    ready_batch_open_time: Optional[float] = None

    start_time = time.time()

    def should_flush() -> bool:
        if not ready_batch:
            return False
        if len(ready_batch) >= infer_batch_size:
            return True
        if ready_batch_duration >= infer_max_batch_seconds:
            return True
        if ready_batch_open_time is not None and (time.time() - ready_batch_open_time) >= flush_timeout:
            return True
        return False

    def flush_ready_batch():
        nonlocal kept, processed, ready_batch, ready_batch_duration, ready_batch_open_time
        if not ready_batch:
            return

        batch_kept = flush_consistency_batch(
            batch_items=ready_batch,
            model_a=model_a,
            model_b=model_b,
            output_dir=output_dir,
            save_txt=save_txt,
        )
        kept += batch_kept
        processed += len(ready_batch)

        print(
            f"Processed {processed}/{total} | "
            f"batch={len(ready_batch)} | "
            f"kept={kept}"
        )

        ready_batch = []
        ready_batch_duration = 0.0
        ready_batch_open_time = None

    with ThreadPoolExecutor(max_workers=preprocess_workers) as ex:
        pending = {}

        # nạp trước một lượng task
        initial = min(max_in_flight, total)
        for _ in range(initial):
            path = audio_paths[next_submit_idx]
            fut = ex.submit(preprocess_audio, path)
            pending[fut] = path
            next_submit_idx += 1

        while pending:
            done, _ = wait(
                pending.keys(),
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                if should_flush():
                    flush_ready_batch()
                continue

            for fut in done:
                src_path = pending.pop(fut)

                try:
                    item = fut.result()
                except Exception as e:
                    processed += 1
                    print(f"❌ Lỗi preprocess {src_path}: {e}")
                    # vẫn nạp tiếp task mới
                    if next_submit_idx < total:
                        path = audio_paths[next_submit_idx]
                        nf = ex.submit(preprocess_audio, path)
                        pending[nf] = path
                        next_submit_idx += 1
                    continue

                if not ready_batch:
                    ready_batch_open_time = time.time()

                ready_batch.append(item)
                ready_batch_duration += item["duration"]

                # nạp tiếp để luôn giữ pipeline đầy
                if next_submit_idx < total:
                    path = audio_paths[next_submit_idx]
                    nf = ex.submit(preprocess_audio, path)
                    pending[nf] = path
                    next_submit_idx += 1

            if should_flush():
                flush_ready_batch()

        # flush phần còn lại
        flush_ready_batch()

    elapsed = time.time() - start_time
    kept_ratio = (kept / total * 100.0) if total > 0 else 0.0

    print(f"\n📊 Tổng audio: {total}")
    print(f"✅ Audio đồng nhất: {kept}")
    print(f"📉 Tỉ lệ giữ lại: {kept_ratio:.2f}%")
    print(f"⏱️  Tổng thời gian: {elapsed:.2f}s")

    return kept


if __name__ == "__main__":
    keep_consistent_asr_parallel(
        audio_dir="data",
        output_dir="audio_kept",
        save_txt=True,
        provider="cpu",          # đổi thành "cuda" nếu bản sherpa-onnx GPU đã setup đúng
        preprocess_workers=8,
        model_num_threads=4,
        infer_batch_size=8,
        infer_max_batch_seconds=120.0,
        max_in_flight=32,
        flush_timeout=0.3,
    )