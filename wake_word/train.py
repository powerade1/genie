import collections
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
CONFIG_PATH = BASE_DIR / "genie.yaml"


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


def load_audio(path: str, target_sr: int = 16000):
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return (audio * 32767).astype(np.int16)


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

    positive_paths = collect_audio_paths(positive_dir)
    negative_paths = collect_audio_paths(negative_dir)

    if not positive_paths:
        raise FileNotFoundError(
            f"positive 샘플이 없습니다. 먼저 `python wake_word/data_gen.py`를 실행하세요.\n"
            f"경로: {positive_dir}"
        )
    if not negative_paths:
        raise FileNotFoundError(
            "negative 샘플이 없습니다. Common Voice 한국어 클립을 "
            f"{negative_dir} 에 배치하세요. (README.md 참고)"
        )

    print(f"positive: {len(positive_paths)}개, negative: {len(negative_paths)}개")

    feature_extractor = openwakeword.utils.AudioFeatures()
    clip_size_seconds = config["clip_size_seconds"]

    negative_features_path = features_dir / "negative_features.npy"
    positive_features_path = features_dir / "positive_features.npy"

    print("negative 특징 추출 중...")
    negative_features = compute_features(
        negative_paths, clip_size_seconds, str(negative_features_path), feature_extractor
    )
    print("positive 특징 추출 중...")
    positive_features = compute_features(
        positive_paths, clip_size_seconds, str(positive_features_path), feature_extractor
    )

    negative_features = np.array(negative_features)
    positive_features = np.array(positive_features)

    x = np.vstack((negative_features, positive_features))
    y = np.array([0] * len(negative_features) + [1] * len(positive_features)).astype(np.float32)[..., None]

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
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
        for batch_x, batch_y in loader:
            weights = torch.ones(batch_y.shape[0])
            weights[batch_y.flatten() == 1] = 0.1

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy(predictions, batch_y, weights[..., None])
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
