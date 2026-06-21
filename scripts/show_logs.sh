#!/bin/bash
# 개발용 라이브 로그 뷰어. Terminal 창에서 tail -F 로 genie 로그를 따라간다.
# 잠자기 후 main.py 가 재시작되어도 같은 로그 파일에 append 되므로
# -F (대문자) 옵션 덕분에 한 창에서 계속 흐름이 보인다.

set -u

cd /Users/woojin/Desktop/DEV/2_active/genie

mkdir -p logs
# tail 이 기다리지 않도록 파일을 미리 보장
touch logs/genie.out.log logs/genie.err.log logs/wake.out.log logs/wake.err.log

# Terminal 창 제목
printf '\e]0;genie logs\a'

clear
echo "============================================================"
echo "  genie live logs   (Ctrl+C 로 종료, 창은 닫아도 됨)"
echo "  cwd:  $(pwd)"
echo "  files:"
echo "    out  → logs/genie.out.log"
echo "    err  → logs/genie.err.log"
echo "    wake → logs/wake.out.log"
echo "============================================================"
echo

exec tail -n 80 -F \
    logs/genie.out.log \
    logs/genie.err.log \
    logs/wake.out.log
