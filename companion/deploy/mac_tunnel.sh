#!/usr/bin/env bash
# 在【Mac】上运行：把 AutoDL 的控制接口（6006）接到本机，保持这个窗口开着
# 用法：bash mac_tunnel.sh "AutoDL 控制台复制的登录指令"
# 例如：bash mac_tunnel.sh "ssh -p 12345 root@connect.westb.seetacloud.com"
CMD="$1"
PORT=$(echo "$CMD" | grep -oE '\-p ?[0-9]+' | grep -oE '[0-9]+')
HOST=$(echo "$CMD" | grep -oE '[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+')
[ -n "$PORT" ] && [ -n "$HOST" ] || { echo "用法：bash mac_tunnel.sh \"ssh -p 端口 root@地址\""; exit 1; }
echo "连接中，提示输入密码时粘贴 AutoDL 控制台里的密码（输入时不显示）。连上后保持窗口开着。"
ssh -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -N -L 6006:127.0.0.1:6006 -p "$PORT" "$HOST"
