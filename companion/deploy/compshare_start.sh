#!/usr/bin/env bash
# 在【优云智算 compshare.cn】的 "LiveTalking一键部署" 镜像实例上运行（地域选 华北二A）
# 这台机器有公网 IP，画面直接用 WebRTC 推给浏览器，不需要阿里云中转、也不需要 SSH 隧道
#
# 用法：把 her.mp4 和本脚本上传到同一个文件夹，然后在 JupyterLab 的 Terminal 里：
#   bash compshare_start.sh                 # 默认用同目录下的 her.mp4，wav2lip 模型
#   MODEL=musetalk bash compshare_start.sh  # 换成更清晰的 MuseTalk（第一次生成形象较慢）
#   DASHSCOPE_API_KEY=sk-xxx bash compshare_start.sh  # 用阿里云通义千问语音（推荐，国内稳定）
#   VOICE=Cherry bash compshare_start.sh    # 换声音（千问语音的音色名，或 Edge 语音名）
#   SRS_IP=139.196.104.146 bash compshare_start.sh      # 没有独立IP的地域：画面推到阿里云 SRS 中转
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VIDEO="${1:-$HERE/her.mp4}"
MODEL="${MODEL:-wav2lip}"
# 语音合成：微软 Edge 语音经常拒绝云服务器的 IP（403），有阿里云百炼 Key 就用通义千问语音
KEY_FILE="$HERE/.dashscope_key"
if [ -n "$DASHSCOPE_API_KEY" ]; then echo "$DASHSCOPE_API_KEY" > "$KEY_FILE"; chmod 600 "$KEY_FILE"
elif [ -f "$KEY_FILE" ]; then DASHSCOPE_API_KEY="$(cat "$KEY_FILE")"; fi
export DASHSCOPE_API_KEY
if [ -z "$TTS" ]; then
  if [ -n "$DASHSCOPE_API_KEY" ]; then TTS=qwentts; else TTS=edgetts; fi
fi
if [ "$TTS" = "qwentts" ]; then VOICE="${VOICE:-Cherry}"; else VOICE="${VOICE:-zh-CN-XiaoyiNeural}"; fi
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

echo
echo ">>> 启动：模型 $MODEL，形象 $AVATAR，语音 $TTS / $VOICE"
if [ -n "$SRS_IP" ]; then
  # 没有独立 IP：画面主动推到阿里云 SRS，Mac 从 SRS 拉流；控制接口用控制台的"实例级端口配置"暴露 $PORT
  echo "    画面推流到 SRS：$SRS_IP"
  echo "    Mac 上 .env 里填：AVATAR_TRANSPORT=rtcpush"
  echo "                     SRS_WHEP_URL=http://$SRS_IP:1985/rtc/v1/whep/?app=live&stream=livestream"
  echo "                     LIVETALKING_URL=控制台给 $PORT 端口分配的外网地址"
  echo
  exec python app.py --transport rtcpush --model "$MODEL" --avatar_id "$AVATAR" \
    --tts "$TTS" --REF_FILE "$VOICE" --listenport "$PORT" \
    --push_url "http://$SRS_IP:1985/rtc/v1/whip/?app=live&stream=livestream"
fi
IP=$(curl -s --max-time 5 ifconfig.me || echo "你的服务器公网IP")
echo "    Mac 上 .env 里填：AVATAR_TRANSPORT=webrtc"
echo "                     LIVETALKING_URL=http://$IP:$PORT"
echo "    （先确认控制台防火墙已放行 TCP $PORT 和 UDP 50000-51000）"
echo
exec python app.py --transport webrtc --model "$MODEL" --avatar_id "$AVATAR" \
  --tts "$TTS" --REF_FILE "$VOICE" --listenport "$PORT"
