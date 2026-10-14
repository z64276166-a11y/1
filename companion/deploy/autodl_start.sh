#!/usr/bin/env bash
# 在【AutoDL GPU 机器】上运行（镜像选社区镜像 lipku/livetalking）
# 用法：bash autodl_start.sh 阿里云服务器公网IP [她的视频路径]
# 第一次运行会用视频生成形象（几分钟），之后再运行就直接启动
set -e
SRS_IP="$1"
VIDEO="${2:-/root/autodl-tmp/her.mp4}"
AVATAR="${AVATAR:-her_wav2lip}"
VOICE="${VOICE:-zh-CN-XiaoyiNeural}"
[ -n "$SRS_IP" ] || { echo "用法：bash autodl_start.sh 阿里云服务器公网IP [视频路径]"; exit 1; }

cd /root/livetalking
if [ ! -f /root/autodl-tmp/.lt_installed ]; then
  echo ">>> 更新 LiveTalking…"
  source /etc/network_turbo 2>/dev/null || true   # AutoDL 学术加速，访问 GitHub 更快
  git pull || echo "（更新失败，用镜像自带版本继续）"
  pip install -r requirements.txt -q
  unset http_proxy https_proxy
  touch /root/autodl-tmp/.lt_installed
fi

[ -f models/wav2lip.pth ] || { echo "❌ 缺少 models/wav2lip.pth，确认创建实例时选的是 lipku/livetalking 镜像"; exit 1; }

if [ ! -d "data/avatars/$AVATAR" ]; then
  if [ -f "$VIDEO" ]; then
    echo ">>> 用 $VIDEO 生成形象 $AVATAR（几分钟）…"
    python -m avatars.wav2lip.genavatar --video_path "$VIDEO" --img_size 256 --avatar_id "$AVATAR"
  else
    echo "⚠️  没找到 $VIDEO，先用自带的示例形象（把她的视频上传到这个路径后重新运行即可）"
    AVATAR=wav2lip256_avatar1
  fi
fi

echo ">>> 启动：形象 $AVATAR，声音 $VOICE，推流到 $SRS_IP"
unset http_proxy https_proxy
exec python app.py --listenport 6006 --transport rtcpush --model wav2lip \
  --avatar_id "$AVATAR" --tts edgetts --REF_FILE "$VOICE" \
  --push_url "http://$SRS_IP:1985/rtc/v1/whip/?app=live&stream=livestream"
