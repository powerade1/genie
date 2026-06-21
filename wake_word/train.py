import collections
import sys
from pathlib import Path

import librosa
import numpy as np
import openwakeword
import torch
from numpy.lib.format import open_memmap
import yaml
from torch import nn, optim
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from audio.speech_trim import trim_and_pad  # noqa: E402

CONFIG_PATH = BASE_DIR / "genie.yaml"
USER_SUBDIR = "user"
# 본인 녹음: pitch augmentation + loss 가중치로 TTS 편향 보정
USER_PITCH_SHIFTS = (-2, -1, 0, 1, 2)
USER_POSITIVE_LOSS_WEIGHT = 2.0
TTS_POSITIVE_LOSS_WEIGHT = 0.1
NEGATIVE_LOSS_WEIGHT = 1.0


def stack_clips(audio_data, clip_size=16000 * 2):
    combined_data = np.hstack(audio_data)
    chunks = []
    for start in range(0, combined_data.shape[0], clip_size):
        chunk = combined_data[start : start + clip_size]
        if chunk.shape[0] != clip_size:
            chunk = np.hstack((chunk, np.zeros(clip_size - chunk.shape[0], dtype=chunk.dtype)))
        chunks.append(chunk)
    return np.array(chunks)


def trim_mmap(mmap_path):
    mmap_file = np.load(mmap_path, mmap_mode="r")
    last_row = mmap_file.shape[0]
    for index in range(mmap_file.shape[0] - 1, -1, -1):
        if not np.all(mmap_file[index, :, :] == 0):
            last_row = index + 1
            break

    if last_row == mmap_file.shape[0]:
        return

    trimmed_path = mmap_path.replace(".npy", "_trimmed.npy")
    trimmed = open_memmap(
        trimmed_path,
        mode="w+",
        dtype=np.float32,
        shape=(last_row, mmap_file.shape[1], mmap_file.shape[2]),
    )
    trimmed[:] = mmap_file[:last_row, :, :]
    trimmed.flush()
    Path(mmap_path).replace(trimmed_path)


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_audio_paths(directory: Path):
    extensions = {".wav", ".mp3", ".flac", ".m4a"}
    return sorted(
        str(path)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def load_audio(path: str, target_sr: int = 16000, pitch_semitones: int = 0, trim: bool = False):
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    if trim:
        audio = trim_and_pad(audio)
    if pitch_semitones:
        audio = librosa.effects.pitch_shift(
            y=audio, sr=target_sr, n_steps=pitch_semitones
        )
        if trim:
            audio = trim_and_pad(audio)
    return (audio * 32767).astype(np.int16)


def collect_positive_paths(positive_dir: Path) -> tuple[list[str], list[str]]:
    """TTS positive(루트)와 본인 녹음(user/)을 분리해 수집한다."""
    user_dir = positive_dir / USER_SUBDIR
    tts_paths = collect_audio_paths(positive_dir)
    user_paths = collect_audio_paths(user_dir) if user_dir.is_dir() else []
    return tts_paths, user_paths


def compute_features(audio_paths, clip_size_seconds, output_file, feature_extractor):
    clip_samples = 16000 * clip_size_seconds
    durations = []
    for path in audio_paths:
        audio = load_audio(path)
        durations.append(len(audio) / 16000)

    n_total = max(1, int(sum(durations) // clip_size_seconds))
    feature_shape = feature_extractor.get_embedding_shape(clip_size_seconds)
    output_shape = (n_total, feature_shape[0], feature_shape[1])

    fp = open_memmap(output_file, mode="w+", dtype=np.float32, shape=output_shape)
    row_counter = 0
    batch_size = 64

    for start in tqdm(range(0, len(audio_paths), batch_size), desc="특징 추출"):
        batch_paths = audio_paths[start : start + batch_size]
        wav_data = [load_audio(path) for path in batch_paths]
        wav_data = stack_clips(wav_data, clip_size=clip_samples).astype(np.int16)
        features = feature_extractor.embed_clips(x=wav_data, batch_size=1024, ncpu=4)

        if row_counter + features.shape[0] > n_total:
            fp[row_counter:n_total, :, :] = features[0 : n_total - row_counter, :, :]
            fp.flush()
            break

        fp[row_counter : row_counter + features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()

    trim_mmap(output_file)
    return np.load(output_file, mmap_mode="r")


def fixed_clips(audio_arrays, clip_size: int) -> np.ndarray:
    """각 오디오를 정확히 clip_size 샘플 1개로 맞춘다 (파일 1개 = 클립 1개)."""
    clips = []
    for audio in audio_arrays:
        if len(audio) >= clip_size:
            clips.append(audio[:clip_size])
        else:
            clips.append(
                np.hstack(
                    (audio, np.zeros(clip_size - len(audio), dtype=audio.dtype))
                )
            )
    return np.array(clips)


def compute_user_features(
    user_paths, clip_size_seconds, output_file, feature_extractor
):
    """본인 녹음에 pitch shift augmentation을 적용해 특징을 추출한다."""
    if not user_paths:
        return None

    clip_samples = 16000 * clip_size_seconds
    variants = [(path, shift) for path in user_paths for shift in USER_PITCH_SHIFTS]
    n_total = len(variants)
    feature_shape = feature_extractor.get_embedding_shape(clip_size_seconds)
    output_shape = (n_total, feature_shape[0], feature_shape[1])

    fp = open_memmap(output_file, mode="w+", dtype=np.float32, shape=output_shape)
    batch_size = 64

    for start in tqdm(
        range(0, len(variants), batch_size), desc="본인 목소리 특징 추출"
    ):
        batch = variants[start : start + batch_size]
        wav_data = [
            load_audio(path, pitch_semitones=shift, trim=True) for path, shift in batch
        ]
        wav_data = fixed_clips(wav_data, clip_samples).astype(np.int16)
        features = feature_extractor.embed_clips(x=wav_data, batch_size=1024, ncpu=4)
        if features.shape[0] != len(batch):
            raise RuntimeError(
                f"특징 수 불일치: 입력 {len(batch)}개, 출력 {features.shape[0]}개"
            )
        end = start + len(batch)
        fp[start:end, :, :] = features
        fp.flush()

    trim_mmap(output_file)
    return np.load(output_file, mmap_mode="r")


def build_model(input_shape, layer_dim):
    flat_dim = input_shape[0] * input_shape[1]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat_dim, layer_dim),
        nn.LayerNorm(layer_dim),
        nn.ReLU(),
        nn.Linear(layer_dim, layer_dim),
        nn.LayerNorm(layer_dim),
        nn.ReLU(),
        nn.Linear(layer_dim, 1),
        nn.Sigmoid(),
    )


def train_model(config):
    positive_dir = (BASE_DIR / config["positive_dir"]).resolve()
    negative_dir = (BASE_DIR / config["negative_dir"]).resolve()
    output_dir = (BASE_DIR / config["output_dir"]).resolve()
    features_dir = (BASE_DIR / config["features_dir"]).resolve()
    features_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts_positive_paths, user_positive_paths = collect_positive_paths(positive_dir)
    negative_paths = collect_audio_paths(negative_dir)

    if not tts_positive_paths and not user_positive_paths:
        raise FileNotFoundError(
            f"positive 샘플이 없습니다. `python wake_word/data_gen.py` 또는 "
            f"`python wake_word/record_user.py`를 실행하세요.\n경로: {positive_dir}"
        )
    if not negative_paths:
        raise FileNotFoundError(
            "negative 샘플이 없습니다. Common Voice 한국어 클립을 "
            f"{negative_dir} 에 배치하세요. (README.md 참고)"
        )

    user_aug_count = len(user_positive_paths) * len(USER_PITCH_SHIFTS)
    print(
        f"positive: TTS {len(tts_positive_paths)}개, "
        f"본인 {len(user_positive_paths)}개 (×{len(USER_PITCH_SHIFTS)} pitch = {user_aug_count}), "
        f"negative: {len(negative_paths)}개"
    )

    feature_extractor = openwakeword.utils.AudioFeatures()
    clip_size_seconds = config["clip_size_seconds"]

    negative_features_path = features_dir / "negative_features.npy"
    tts_positive_features_path = features_dir / "positive_tts_features.npy"
    user_positive_features_path = features_dir / "positive_user_features.npy"

    print("negative 특징 추출 중...")
    negative_features = np.array(
        compute_features(
            negative_paths,
            clip_size_seconds,
            str(negative_features_path),
            feature_extractor,
        )
    )
    tts_positive_features = np.empty((0, *negative_features.shape[1:]), dtype=np.float32)
    if tts_positive_paths:
        print("TTS positive 특징 추출 중...")
        tts_positive_features = np.array(
            compute_features(
                tts_positive_paths,
                clip_size_seconds,
                str(tts_positive_features_path),
                feature_extractor,
            )
        )

    user_positive_features = None
    if user_positive_paths:
        print("본인 목소리 positive 특징 추출 중...")
        user_positive_features = np.array(
            compute_user_features(
                user_positive_paths,
                clip_size_seconds,
                str(user_positive_features_path),
                feature_extractor,
            )
        )

    positive_parts = [part for part in (tts_positive_features, user_positive_features) if part is not None and len(part)]
    positive_features = np.vstack(positive_parts) if positive_parts else np.empty((0, *negative_features.shape[1:]))

    x = np.vstack((negative_features, positive_features))
    y = np.array([0] * len(negative_features) + [1] * len(positive_features)).astype(
        np.float32
    )[..., None]

    sample_weights = np.array(
        [NEGATIVE_LOSS_WEIGHT] * len(negative_features)
        + [TTS_POSITIVE_LOSS_WEIGHT] * len(tts_positive_features)
        + [USER_POSITIVE_LOSS_WEIGHT] * (len(user_positive_features) if user_positive_features is not None else 0),
        dtype=np.float32,
    )

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(sample_weights),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = build_model(x.shape[1:], config["layer_dim"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = collections.defaultdict(list)

    print("모델 학습 중...")
    for epoch in range(config["epochs"]):
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy(
                predictions, batch_y, weight=batch_w[..., None]
            )
            loss.backward()
            optimizer.step()

            history["loss"].append(float(loss.detach().numpy()))
            tp = sum(predictions.flatten()[batch_y.flatten() == 1] >= 0.5)
            fn = sum(predictions.flatten()[batch_y.flatten() == 1] < 0.5)
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            history["recall"].append(recall)

        print(f"epoch {epoch + 1}/{config['epochs']} - loss: {history['loss'][-1]:.4f}, recall: {recall:.4f}")

    model_name = config["model_name"]
    onnx_path = output_dir / f"{model_name}.onnx"
    tflite_path = output_dir / f"{model_name}.tflite"

    model.eval()
    torch.onnx.export(model, args=torch.zeros((1, *x.shape[1:])), f=str(onnx_path))

    try:
        from openwakeword.train import convert_onnx_to_tflite

        convert_onnx_to_tflite(str(onnx_path), str(tflite_path))
        final_model = tflite_path
    except Exception as error:
        print(f"tflite 변환 실패, onnx 모델 사용: {error}")
        final_model = onnx_path

    print(f"\n학습 완료: {final_model}")
    return str(final_model)


if __name__ == "__main__":
    train_model(load_config())
