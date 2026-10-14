#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
[ -f .env ] || { cp config.example.env .env; echo "请先编辑 .env 填写 DEEPSEEK_API_KEY"; exit 1; }
.venv/bin/python server.py
