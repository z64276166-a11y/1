#!/usr/bin/env bash
# 在【优云智算 compshare.cn】的 "LiveTalking一键部署" 镜像实例上运行（地域选 华北二A）
# 这台机器有公网 IP，画面直接用 WebRTC 推给浏览器，不需要阿里云中转、也不需要 SSH 隧道
#
# 用法：把 her.mp4 和本脚本上传到同一个文件夹，然后在 JupyterLab 的 Terminal 里：
#   bash compshare_start.sh                 # 默认用同目录下的 her.mp4，wav2lip 模型
#   MODEL=musetalk bash compshare_start.sh  # 换成更清晰的 MuseTalk（第一次生成形象较慢）
#   VOICE=zh-CN-XiaoxiaoNeural bash compshare_start.sh   # 换声音
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VIDEO="${1:-$HERE/her.mp4}"
MODEL="${MODEL:-wav2lip}"
VOICE="${VOICE:-zh-CN-XiaoyiNeural}"
PORT="${PORT:-8010}"
# 形象 ID 跟着视频文件名和模型走：换了视频或模型就会重新生成
AVATAR="${AVATAR:-$(basename "${VIDEO%.*}")_${MODEL}}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate livetalking
fi
cd /root/LiveTalking

# 镜像开机会自动启动一个默认形象的 LiveTalking，先停掉它，腾出 8010 端口
pkill -f "python.*app.py" 2>/dev/null || true
sleep 2

if [ ! -d "data/avatars/$AVATAR" ]; then
  [ -f "$VIDEO" ] || { echo "❌ 没找到视频 $VIDEO，请把 her.mp4 上传到和本脚本同一个文件夹"; exit 1; }
  if [ "$MODEL" = "wav2lip" ]; then
    [ -f models/wav2lip.pth ] || { echo "❌ 镜像里没有 models/wav2lip.pth，改用：MODEL=musetalk bash $0"; exit 1; }
    echo ">>> 用 $VIDEO 生成 wav2lip 形象 $AVATAR（几分钟）…"
    python -m avatars.wav2lip.genavatar --video_path "$VIDEO" --img_size 256 --avatar_id "$AVATAR"
  else
    if ! python -c "import face_recognition" 2>/dev/null; then
      echo ">>> 安装 MuseTalk 生成形象需要的依赖…"
      (apt-get update -y && apt-get install -y build-essential cmake) >/dev/null 2>&1 || true
      pip install -q "setuptools<81" face_recognition
    fi
    echo ">>> 用 $VIDEO 生成 MuseTalk 形象 $AVATAR（可能要十几分钟）…"
    python -m avatars.musetalk.genavatar --avatar_id "$AVATAR" --file "$VIDEO"
  fi
fi

IP=$(curl -s --max-time 5 ifconfig.me || echo "你的服务器公网IP")
echo
echo ">>> 启动：模型 $MODEL，形象 $AVATAR，声音 $VOICE"
echo "    Mac 上 .env 里填：LIVETALKING_URL=http://$IP:$PORT"
echo "    （先确认控制台防火墙已放行 TCP $PORT 和 UDP 50000-51000）"
echo
exec python app.py --transport webrtc --model "$MODEL" --avatar_id "$AVATAR" \
  --tts edgetts --REF_FILE "$VOICE" --listenport "$PORT"
