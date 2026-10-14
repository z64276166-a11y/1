#!/usr/bin/env bash
# Mac / Linux 启动脚本：在"终端"里运行 ./start.sh
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "第一次运行，正在安装依赖，需要几分钟…"
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ || exit 1
fi
if [ ! -f .env ]; then
  cp config.example.env .env
  echo "请先填写 .env 里的 DEEPSEEK_API_KEY（已为你打开），保存后重新运行 ./start.sh"
  open -e .env 2>/dev/null || ${EDITOR:-nano} .env
  exit 1
fi
(sleep 3; open http://127.0.0.1:8765 2>/dev/null || xdg-open http://127.0.0.1:8765) >/dev/null 2>&1 &
.venv/bin/python server.py
