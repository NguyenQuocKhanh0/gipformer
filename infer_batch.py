#!/usr/bin/env python3
import os
import shutil
import time
from typing import List, Union

import soundfile as sf

try:
    import sherpa_onnx
except ImportError:
    raise ImportError("Cần cài sherpa-onnx: pip install sherpa-onnx")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("Cần cài huggingface_hub: pip install huggingface_hub")


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
    samples, sample_rate = sf.read(filename, dtype="float32")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    return samples, sample_rate


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
    Wrapper cho sherpa-onnx để có interface gần giống ChunkFormerModel.

    Hỗ trợ:
        model = GipformerOnnxModel.from_pretrained(...).to("cuda")
        texts = model.batch_decode(audio_paths=[...], chunk_size=64, ...)
    """

    def __init__(
        self,
        recognizer: sherpa_onnx.OfflineRecognizer,
        repo_id: str,
        quantize: str,
        num_threads: int,
        decoding_method: str,
    ):
        self.recognizer = recognizer
        self.repo_id = repo_id
        self.quantize = quantize
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        quantize: str = "fp32",
        num_threads: int = 4,
        decoding_method: str = "modified_beam_search",
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
        )

        return cls(
            recognizer=recognizer,
            repo_id=repo_id,
            quantize=quantize,
            num_threads=num_threads,
            decoding_method=decoding_method,
        )

    def to(self, device: str):
        """
        Giữ API giống torch model.
        sherpa-onnx không dùng .to("cuda") kiểu PyTorch như ChunkFormer,
        nên hàm này chỉ để tương thích interface.
        """
        self.device = device
        return self

    def decode(self, audio_path: str) -> str:
        samples, sample_rate = read_audio(audio_path)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_streams([stream])

        return stream.result.text.strip()

    def batch_decode(
        self,
        audio_paths: List[str],
        chunk_size: int = 64,
        left_context_size: int = 128,
        right_context_size: int = 128,
        total_batch_duration: int = 600,
    ) -> List[str]:
        """
        Giữ chữ ký hàm gần giống ChunkFormerModel.batch_decode().

        Lưu ý:
        - Các tham số chunk/context/batch_duration hiện chỉ để tương thích interface.
        - sherpa-onnx OfflineRecognizer ở đây đang decode tuần tự từng file.
        """
        _ = chunk_size
        _ = left_context_size
        _ = right_context_size
        _ = total_batch_duration

        results = []
        for path in audio_paths:
            text = self.decode(path)
            results.append(text)
        return results


def keep_consistent_asr(
    audio_dir: str,
    output_dir: str,
    save_txt: bool = True,
):
    """
    Bản tương thích với format ChunkFormer:
    - dùng 2 model ONNX để decode
    - chỉ giữ các file có kết quả giống nhau

    Ở đây mình dùng:
    - model 1: fp32 + modified_beam_search
    - model 2: int8 + modified_beam_search

    Bạn cũng có thể đổi sang:
    - fp32 + greedy_search
    - fp32 + modified_beam_search
    nếu muốn so sánh theo decoding method thay vì quantization.
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted(
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    )

    if not audio_files:
        raise ValueError("Không có file wav")

    model_a = GipformerOnnxModel.from_pretrained(
        repo_id=DEFAULT_REPO_ID,
        quantize="fp32",
        num_threads=4,
        decoding_method="modified_beam_search",
    ).to("cuda")

    model_b = GipformerOnnxModel.from_pretrained(
        repo_id=DEFAULT_REPO_ID,
        quantize="int8",
        num_threads=4,
        decoding_method="modified_beam_search",
    ).to("cuda")

    audio_paths = [os.path.join(audio_dir, f) for f in audio_files]

    texts_a = model_a.batch_decode(
        audio_paths=audio_paths,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=600,
    )

    texts_b = model_b.batch_decode(
        audio_paths=audio_paths,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=600,
    )

    kept = 0

    for fname, t_a, t_b in zip(audio_files, texts_a, texts_b):
        if t_a.strip() == t_b.strip():
            kept += 1
            src_audio = os.path.join(audio_dir, fname)
            dst_audio = os.path.join(output_dir, fname)

            shutil.copy2(src_audio, dst_audio)

            if save_txt:
                txt_path = os.path.join(
                    output_dir,
                    os.path.splitext(fname)[0] + ".txt"
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(t_a.strip())

    total = len(audio_files)

    print(f"📊 Tổng audio: {total}")
    print(f"✅ Audio đồng nhất: {kept}")
    print(f"📉 Tỉ lệ giữ lại: {kept / total * 100:.2f}%")

    return kept


if __name__ == "__main__":
    # ví dụ chạy thử
    keep_consistent_asr(
        audio_dir="data",
        output_dir="audio_kept",
        save_txt=True,
    )