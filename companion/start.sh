#!/usr/bin/env bash
# Mac / Linux 启动脚本：在"终端"里运行 ./start.sh
cd "$(dirname "$0")"
PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
# 依赖没装完整、或者程序更新后依赖有变化时（重新）安装
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "正在安装依赖，第一次需要几分钟…"
  [ -d .venv ] || python3 -m venv .venv || exit 1
  .venv/bin/pip install -q --upgrade pip -i "$PIP_MIRROR" || true
  .venv/bin/pip install -r requirements.txt -i "$PIP_MIRROR" || .venv/bin/pip install -r requirements.txt || exit 1
  touch .venv/.installed
fi
if [ ! -f .env ]; then
  cp config.example.env .env
  echo "请先填写 .env 里的 DEEPSEEK_API_KEY（已为你打开），保存后重新运行 ./start.sh"
  open -e .env 2>/dev/null || ${EDITOR:-nano} .env
  exit 1
fi
# 语音识别模型从 Hugging Face 下载，国内用镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PORT="$(grep -E '^PORT=' .env | tail -1 | cut -d= -f2)"
PORT="${PORT:-8765}"
URL="http://127.0.0.1:$PORT"
echo "启动中…第一次会下载语音识别模型，准备好后浏览器会自动打开 $URL"
# 等服务真正就绪再打开浏览器
(for _ in $(seq 1 900); do curl -s -o /dev/null "$URL/config" && break; sleep 1; done
 open "$URL" 2>/dev/null || xdg-open "$URL") >/dev/null 2>&1 &
.venv/bin/python server.py
