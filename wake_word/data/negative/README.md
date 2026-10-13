# Negative 샘플 준비

Wake word 모델 학습을 위해 "지니" / "지니야"가 포함되지 않은 음성 샘플이 필요합니다.

## 다운로드

1. [Mozilla Common Voice 한국어](https://commonvoice.mozilla.org/ko/datasets)에서 데이터셋을 다운로드합니다.
2. 압축을 풀고 `clips/` 폴더의 `.mp3` 또는 `.wav` 파일을 이 디렉터리에 복사합니다.

## 권장 수량

- 최소: 수백 개
- 권장: 수천 개 이상

## 확인

학습 전에 아래 명령으로 파일 수를 확인할 수 있습니다.

```bash
ls wake_word/data/negative/*.{wav,mp3} 2>/dev/null | wc -l
```
