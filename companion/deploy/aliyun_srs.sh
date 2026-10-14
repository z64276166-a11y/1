#!/usr/bin/env bash
# 在【阿里云轻量应用服务器】上运行：搭建 SRS 视频中转
# 用法：bash aliyun_srs.sh 你的服务器公网IP
set -e
IP="$1"
if [ -z "$IP" ]; then
  IP=$(curl -s --max-time 5 ifconfig.me || true)
  [ -n "$IP" ] || { echo "请带上公网IP：bash aliyun_srs.sh 1.2.3.4"; exit 1; }
fi
echo ">>> 公网 IP：$IP"

if ! command -v docker >/dev/null 2>&1; then
  echo ">>> 安装 Docker…"
  if command -v apt-get >/dev/null; then apt-get update -y && apt-get install -y docker.io
  elif command -v dnf >/dev/null; then dnf install -y docker || dnf install -y podman-docker
  elif command -v yum >/dev/null; then yum install -y docker
  fi
  command -v docker >/dev/null || curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
  systemctl enable --now docker 2>/dev/null || true
fi

docker rm -f srs >/dev/null 2>&1 || true
docker run -d --restart always --name srs \
  -p 1985:1985 -p 8000:8000/udp \
  -e CANDIDATE="$IP" \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 ./objs/srs -c conf/rtc.conf

# 系统防火墙（如果开着）
command -v ufw >/dev/null && ufw status | grep -q active && { ufw allow 1985/tcp; ufw allow 8000/udp; }
command -v firewall-cmd >/dev/null && firewall-cmd --state >/dev/null 2>&1 && {
  firewall-cmd --permanent --add-port=1985/tcp --add-port=8000/udp && firewall-cmd --reload; }

sleep 3
if curl -s --max-time 5 http://127.0.0.1:1985/api/v1/versions | grep -q '"code":0'; then
  echo
  echo "✅ SRS 已启动"
  echo "   别忘了在阿里云控制台 → 轻量应用服务器 → 防火墙 里添加两条规则：TCP 1985、UDP 8000"
  echo "   推流地址（填到 AutoDL 脚本）：$IP"
  echo "   拉流地址（填到 Mac 的 .env 的 SRS_WHEP_URL）：http://$IP:1985/rtc/v1/whep/?app=live&stream=livestream"
else
  echo "❌ SRS 好像没起来，运行 docker logs srs 看看报错"
fi
