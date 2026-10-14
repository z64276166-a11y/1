@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv (
  echo 第一次运行，正在安装依赖，需要几分钟…
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
)
if not exist .env (
  copy config.example.env .env >nul
  echo 请先用记事本打开 .env 填写 DEEPSEEK_API_KEY，然后重新运行本脚本
  notepad .env
  pause
  exit /b
)
start "" http://127.0.0.1:8765
.venv\Scripts\python server.py
pause
