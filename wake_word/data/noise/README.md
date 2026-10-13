# Noise 샘플 (선택)

Wake word positive 샘플 augmentation 시 배경 노이즈로 사용됩니다.

## 사용법

1. 환경 소음, ESC-50, MUSAN 등의 `.wav` / `.mp3` / `.flac` 파일을 이 폴더에 넣습니다.
2. `python wake_word/data_gen.py`를 실행하면 50% 확률로 노이즈가 믹스됩니다.

## 폴더가 비어 있는 경우

`wake_word/data/negative/`의 클립 일부(최대 200개)를 자동으로 노이즈 소스로 사용합니다.
negative도 없으면 볼륨 변조(-20 ~ -3dB)만 적용됩니다.
