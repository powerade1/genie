#!/bin/bash

cd /Users/woojin/Desktop/DEV/2_active/genie

# 가상환경
source /Users/woojin/anaconda3/etc/profile.d/conda.sh
conda activate genie

# launchd 로그 즉시 flush
export PYTHONUNBUFFERED=1

# 실행 (-u: unbuffered stdout/stderr)
exec python -u main.py

# 추후 추가 가능 예시
# npm run dev
# uvicorn app:app --host 0.0.0.0 --port 8000
