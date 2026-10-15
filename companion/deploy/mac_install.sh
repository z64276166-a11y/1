#!/usr/bin/env bash
# 在【Mac】上一键安装 / 更新"和小雪视频通话"，然后启动。
# 打开"终端"，粘贴这一行回车：
#   curl -fsSL https://raw.githubusercontent.com/z64276166-a11y/1/claude/digital-companion-difficulty-fu7ztw/companion/deploy/mac_install.sh | bash
# 显卡服务器的 IP 变了的话：
#   curl -fsSL ...同上... | GPU_IP=新的IP bash
set -e
BRANCH="claude/digital-companion-difficulty-fu7ztw"
ZIP="https://github.com/z64276166-a11y/1/archive/refs/heads/$BRANCH.zip"
DIR="$HOME/xiaoxue"
GPU_IP="${GPU_IP:-117.50.216.222}"

echo "==> 安装位置：$DIR"
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
  echo "❌ 需要 Python 3.9 以上。会弹出安装「命令行开发者工具」的窗口，装好后重新运行这一行命令。"
  xcode-select --install 2>/dev/null || true
  exit 1
fi

TMP="$(mktemp -d)"
echo "==> 下载程序…"
ok=""
for p in "" "https://ghfast.top/" "https://gh-proxy.com/"; do
  if curl -fsSL --max-time 300 -o "$TMP/app.zip" "${p}${ZIP}"; then ok=1; break; fi
done
[ -n "$ok" ] || { echo "❌ 下载失败，请检查网络后重试"; exit 1; }
unzip -q "$TMP/app.zip" -d "$TMP"
SRC="$(find "$TMP" -maxdepth 2 -type d -name companion | head -1)"
mkdir -p "$DIR"
# 更新时保留你的配置（.env）、她的记忆（data/）和已安装的环境（.venv/）
rsync -a --exclude ".env" --exclude "data/" --exclude ".venv/" "$SRC/" "$DIR/"
rm -rf "$TMP"
chmod +x "$DIR/start.sh"

if [ ! -f "$DIR/.env" ]; then
  echo
  echo "==> 需要 DeepSeek 的 API Key（在 https://platform.deepseek.com 的 API Keys 页面创建，sk- 开头）"
  KEY=""
  while [ -z "$KEY" ]; do
    printf "请粘贴 DeepSeek API Key 后回车："
    read -r KEY < /dev/tty
  done
  cat > "$DIR/.env" <<EOF
# 完整的可选项见 config.example.env
DEEPSEEK_API_KEY=$KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
COMPANION_NAME=小雪
USER_NAME=你
PERSONA_FILE=persona.md

# 真人视频：画面由优云智算的显卡服务器生成
AVATAR_MODE=realistic
AVATAR_TRANSPORT=webrtc
LIVETALKING_URL=http://$GPU_IP:8010
CAMERA_GRAIN=1

IDLE_SECONDS=60
PORT=8765
EOF
  chmod 600 "$DIR/.env"
elif [ "$GPU_IP" != "117.50.216.222" ]; then
  # 显卡服务器换了 IP：只改这一行
  python3 - "$DIR/.env" "$GPU_IP" <<'PY'
import re, sys
p, ip = sys.argv[1], sys.argv[2]
s = open(p, encoding="utf-8").read()
s = re.sub(r"^LIVETALKING_URL=.*$", f"LIVETALKING_URL=http://{ip}:8010", s, flags=re.M)
open(p, "w", encoding="utf-8").write(s)
PY
  echo "==> 已把显卡服务器地址改成 $GPU_IP"
fi

# 桌面上放一个双击就能启动的图标
LAUNCHER="$HOME/Desktop/和小雪视频.command"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
cd "$DIR" && ./start.sh
EOF
chmod +x "$LAUNCHER"

echo
echo "==> 安装完成。以后双击桌面上的「和小雪视频」就能启动（记得先把显卡服务器开机）。"
echo "==> 现在启动…（第一次要装依赖、下载语音识别模型，需要几分钟；关闭这个窗口或按 Ctrl+C 就是退出）"
echo
cd "$DIR" && exec ./start.sh
