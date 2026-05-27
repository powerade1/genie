#!/bin/bash

cd /Users/woojin/Desktop/DEV/2_active/genie

# 가상환경
source /Users/woojin/anaconda3/etc/profile.d/conda.sh
conda activate genie

# 실행
python main.py

# 추후 추가 가능 예시
# npm run dev
# uvicorn app:app --host 0.0.0.0 --port 8000
