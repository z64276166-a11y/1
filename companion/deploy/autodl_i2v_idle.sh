#!/usr/bin/env bash
# =============================================================================
#  autodl_i2v_idle.sh   【可选 · 最高写实】一张照片 → 真人"视频通话"待机视频
#  模型：通义万相 Wan2.2-TI2V-5B（官方仓库 Wan-Video/Wan2.2，Apache-2.0）
# -----------------------------------------------------------------------------
#  在哪运行：AutoDL 上租的 RTX 4090（24GB）实例。可以就是跑 LiveTalking 的那台
#           （lipku/livetalking 镜像）。本脚本装在 /root/autodl-tmp 下的【独立 Python
#           环境】里，不会改动 LiveTalking 的环境。
#
#  它会做什么：
#    1) 第一次运行：安装 Wan2.2 官方代码 + 独立环境（torch 2.6 / CUDA 12.4），
#       从 ModelScope（魔搭，国内快）下载 TI2V-5B 权重（约 34GB，只下一次）
#    2) 把照片按"手机视频通话"构图自动裁成 1280×704（Wan2.2-5B 横屏 720P 的固定尺寸），
#       眼睛在画面高度约 40% 处、脸宽约占画面 23%（照片里脸太小会自动放大裁剪）
#    3) 用"安静听你说话、嘴巴一直闭着"的提示词生成约 5 秒的自然待机视频：
#       呼吸起伏、自然眨眼、头部极轻微的晃动、发丝/衣服的细微飘动
#       默认一次生成 3 个不同随机种子的候选，由你挑最自然、嘴巴全程闭合的那个
#    4) ffmpeg 后处理：25fps、1280×720、正放 + 倒放拼成无缝循环（约 9.6 秒）、H.264 yuv420p
#
#  用时（RTX 4090）：
#    首次安装 + 下载权重 约 20~40 分钟（看网速）；
#    之后每个候选约 10 分钟（官方实测：4090 单卡 TI2V-5B 720P 图生视频 ≈ 525 秒，峰值显存 22.8GB；
#    另加 T5 文本编码在 CPU 上约 0.5~1.5 分钟/个，第一次加载模型约 1~3 分钟）
#    （RTX 3090 24GB 也能跑，大约慢一倍；显存小于 24GB 的卡不行）
#
#  用法（在 AutoDL 的 JupyterLab → Terminal 里）：
#    1. 先停掉 LiveTalking（在它的终端里按 Ctrl+C），否则显存不够
#    2. 把本脚本和照片拖进 /root/autodl-tmp/
#    3. 推荐放后台跑（关掉网页也不会中断）：
#         nohup bash /root/autodl-tmp/autodl_i2v_idle.sh /root/autodl-tmp/照片.png > /root/autodl-tmp/wan_idle.log 2>&1 &
#         tail -f /root/autodl-tmp/wan_idle.log        # 看进度，Ctrl+C 只是退出查看，不会停止任务
#       或者直接前台跑：
#         bash /root/autodl-tmp/autodl_i2v_idle.sh /root/autodl-tmp/照片.png
#
#  常用可选设置（写在 bash 前面，例如  SEEDS="7" bash autodl_i2v_idle.sh 照片.png）：
#    SEEDS="101 202 303"  生成哪几个随机种子（每个约 10 分钟）；想多几个候选就多写几个
#    ONLY=prep            只做第 2 步裁剪，先打开 input_1280x704.png 看构图满不满意（不占显卡；需要已经完整装过一次）
#    ONLY=post            不重新生成，只重做第 4 步后处理（比如改了 GRAIN 之后）
#    GRAIN=4              给成片加一点"手机摄像头"的细微噪点（0=不加，默认 0，原因见下面注释）
#    FACE_FRAC=0.23       希望脸宽占画面宽度的比例（照片里脸太小会自动放大裁剪；脸已经很大时不会缩小）
#    EYE_Y=0.40           眼睛放在画面高度的哪个位置（0=最上 1=最下）
#    FRAMES=121  STEPS=50  GUIDE=5.0  SHIFT=5.0   官方默认值，一般不用改（FRAMES 必须是 4n+1，4090 上不要超过 121）
#    FRAMES=81            想快一点：每个约 6 分钟，循环成片约 6.4 秒
#    PROMPT="..."  NEG="..."                      完全替换正/反向提示词（一般不用改）
#    DL_SOURCE=hf-mirror  魔搭下载失败时改用 hf-mirror.com 下载
#
#  生成完以后：
#    结果在 /root/autodl-tmp/wan_idle_out/<照片名>/ 里：
#      <照片名>_wan_s101.mp4         ← 成片（1280×720，25fps，无缝循环）
#      <照片名>_wan_s101_sheet.jpg   ← 12 帧缩略图，快速看整体动作
#      <照片名>_wan_s101_mouth.jpg   ← 24 帧嘴部特写，确认嘴巴全程闭合（有张嘴的就别用）
#    挑一个最好的，用它生成 LiveTalking 形象（换一个新的形象名，避免用到旧的缓存）：
#      AVATAR=her_wan bash /root/autodl-tmp/autodl_start.sh 阿里云公网IP /root/autodl-tmp/wan_idle_out/<照片名>/<照片名>_wan_s101.mp4
#    也可以把 mp4 下载回电脑，替换 companion/avatar/her.mp4。
#
#  对照片的要求（决定成片有多真）：上半身、正脸看镜头、平视、嘴巴闭合、头发和手不挡嘴和下巴；
#  最好本身就像"手机前置摄像头拍的"室内照片。照片越像真人，视频越像真人。
# =============================================================================
set -Eeuo pipefail

die()  { echo "【错误】$*" >&2; exit 1; }
warn() { echo "【注意】$*" >&2; }
step() { echo; echo "========== $* =========="; }
trap 'echo "【错误】脚本第 $LINENO 行出错：$BASH_COMMAND —— 把上面的报错截图发给我。" >&2' ERR

# ----------------------------- 参数 ------------------------------------------
PHOTO="${1:-}"
[ -n "$PHOTO" ] && [ -f "$PHOTO" ] || die "用法：bash autodl_i2v_idle.sh 照片路径 [输出目录]   （照片不存在：'$PHOTO'）"
PHOTO="$(readlink -f "$PHOTO")"
NAME="$(basename "${PHOTO%.*}")"
NAME="${NAME// /_}"

BASE="${BASE:-/root/autodl-tmp}"                 # AutoDL 数据盘（关机后数据保留）
WORK="${WORK:-$BASE/wan22_idle}"                 # 代码、环境、权重都放这里
OUT="${2:-$BASE/wan_idle_out/$NAME}"             # 输出目录
REPO="$WORK/Wan2.2"
VENV="$WORK/venv"
CKPT="$WORK/Wan2.2-TI2V-5B"
TOOLS="$WORK/tools"
VPY="$VENV/bin/python"

# 固定到写这个脚本时核对过的官方提交（2026-09-21），保证下面的调用方式和代码一致。
# 想用最新版：WAN_COMMIT=latest
WAN_COMMIT="${WAN_COMMIT:-1ea34ff48f87168174e12956e200b1d908b1c5ff}"
WAN_REPO_URL="${WAN_REPO_URL:-https://github.com/Wan-Video/Wan2.2.git}"
MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"                 # ModelScope 和 HuggingFace 上同名
DL_SOURCE="${DL_SOURCE:-modelscope}"             # modelscope | hf-mirror

SEEDS="${SEEDS:-101 202 303}"
FRAMES="${FRAMES:-121}"     # 121 帧 = 官方默认，24fps 下约 5 秒；必须是 4n+1
STEPS="${STEPS:-50}"        # 官方默认 50 步
GUIDE="${GUIDE:-5.0}"       # 官方默认 CFG 5.0
SHIFT="${SHIFT:-5.0}"       # 官方默认 shift 5.0（720P）
ATTN="${ATTN:-auto}"        # auto：有 flash_attn 用 flash_attn，没有就用 PyTorch SDPA
GRAIN="${GRAIN:-0}"
FACE_FRAC="${FACE_FRAC:-0.23}"
EYE_Y="${EYE_Y:-0.40}"
ONLY="${ONLY:-all}"         # all | prep | post

[[ "$FRAMES" =~ ^[0-9]+$ ]] && (( FRAMES >= 9 && (FRAMES - 1) % 4 == 0 )) || die "FRAMES 必须是 4n+1（例如 81、121），现在是 $FRAMES"
for s in $SEEDS; do [[ "$s" =~ ^[0-9]+$ ]] || die "SEEDS 只能是数字，现在是 '$SEEDS'"; done
[[ "$GRAIN" =~ ^[0-9]+$ ]] || die "GRAIN 只能是 0~20 的整数"
case "$ONLY" in all|prep|post) ;; *) die "ONLY 只能是 all / prep / post";; esac
(( FRAMES <= 121 )) || warn "FRAMES=$FRAMES 超过官方默认 121：4090 的 24GB 显存可能不够（会报 CUDA out of memory），画面也更容易跑偏"

# ------------------------ 默认提示词（正向，英文） ---------------------------
# 要点：真实手机视频通话、平视看镜头、安静地听、嘴唇全程闭合、只有呼吸/眨眼/发丝等微动、机位固定。
DEFAULT_PROMPT="A real, unedited smartphone front-camera video call recording. The young woman from the photo sits still and faces the camera at eye level, looking straight into the lens with a calm, gentle, attentive expression, as if quietly listening to the person on the other end of the call. Her lips stay softly closed the entire time; she does not speak, does not smile widely and does not move her mouth. Only natural, subtle idle micro-motion: slow relaxed breathing that gently lifts her shoulders and chest, a few natural blinks, tiny involuntary head micro-movements, a couple of loose hair strands and the fabric of her clothes shifting very slightly. The phone stands on a fixed stand: no camera movement, no zoom, no cuts. Soft natural indoor light, realistic skin texture with fine pores, faint camera sensor noise, true-to-life colors, documentary realism."

# ------------------------ 默认反向提示词（中文） -----------------------------
# 在官方默认反向词的基础上：加了"说话/张嘴/露牙/转头/镜头运动/头发被吹起/磨皮/CG"等；
# 去掉了官方的"静态、静止"（官方用它们逼模型多动，待机视频反而要少动），
# 但保留"画面完全静止不动"，防止生成一段一动不动的视频。
DEFAULT_NEG="说话，张嘴，张开嘴巴，嘴巴开合，嘴唇蠕动，口型变化，露出牙齿，咬嘴唇，舔嘴唇，大笑，咧嘴笑，打哈欠，唱歌，咀嚼，吐舌头，表情夸张，皱眉，哭泣，转头，扭头，侧脸，低头，仰头，点头，摇头，大幅度动作，快速动作，身体晃动，手出现在画面中，手挡住脸，头发挡住嘴巴，头发被风吹起，镜头移动，镜头推拉，变焦，摇镜头，手持抖动，场景切换，画面闪烁，光线突变，人物变脸，五官变形，脸部扭曲，妆容变化，色调艳丽，过曝，细节模糊不清，字幕，文字，水印，风格，作品，画作，画面，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，画面完全静止不动，杂乱的背景，背景人很多，倒着走，塑料感皮肤，过度磨皮，CG感，动漫，卡通，3D渲染，游戏画面"

PROMPT="${PROMPT:-$DEFAULT_PROMPT}"
NEG="${NEG:-$DEFAULT_NEG}"

# ------------------- AutoDL 学术加速（只在访问 GitHub/HF 时打开） -----------
_OLD_HTTP_PROXY=""; _OLD_HTTPS_PROXY=""
turbo_on() {
  _OLD_HTTP_PROXY="${http_proxy-}"; _OLD_HTTPS_PROXY="${https_proxy-}"
  if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo >/dev/null 2>&1 || true
  fi
}
turbo_off() {   # 用完立刻关掉（AutoDL 官方建议），否则会拖慢 pip / 魔搭 下载
  if [ -n "$_OLD_HTTP_PROXY" ];  then export http_proxy="$_OLD_HTTP_PROXY";   else unset http_proxy  || true; fi
  if [ -n "$_OLD_HTTPS_PROXY" ]; then export https_proxy="$_OLD_HTTPS_PROXY"; else unset https_proxy || true; fi
}

# ======================= 写出两个 Python 小工具 ===============================
mkdir -p "$WORK" "$TOOLS" "$OUT"

# ---- 工具 1：按"视频通话"构图裁剪照片 ----------------------------------------
cat > "$TOOLS/prep_image.py" <<'PYEOF'
# -*- coding: utf-8 -*-
"""把任意照片裁成 Wan2.2-TI2V-5B 的横屏尺寸 1280x704，构图像手机/电脑视频通话。
用法: python prep_image.py 输入照片 输出png 输出json [YuNet模型.onnx]
环境变量: FACE_FRAC 脸宽/画面宽（默认 0.23）  EYE_Y 眼睛所在高度（默认 0.40，0=顶 1=底）
注意: Wan2.2-TI2V 会按输入图片的宽高比自动决定输出尺寸，所以必须先裁成正好 1280x704
      (16:9 的 1920x1080 直接喂进去会变成 1248x704)。
人脸检测: 优先用 OpenCV 的 YuNet（准，带眼睛/嘴角关键点）；没有模型文件时退回 Haar（可能误检）。
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

TW, TH = 1280, 704            # Wan2.2-TI2V-5B 720P 横屏（官方只支持 1280*704 / 704*1280）
FW, FH = 1280, 720            # 最终给 LiveTalking 的尺寸

src, dst, meta_path = sys.argv[1:4]
yunet_path = sys.argv[4] if len(sys.argv) > 4 else ""
face_frac = float(os.environ.get("FACE_FRAC", "0.23"))
eye_y = float(os.environ.get("EYE_Y", "0.40"))

img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
W, H = img.size
bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
warnings = []
face = None        # dict: box=[x,y,w,h], eyes=(x,y), mouth=(x,y,width)
detector = "none"

if yunet_path and os.path.isfile(yunet_path):
    try:
        det = cv2.FaceDetectorYN.create(yunet_path, "", (W, H), 0.6, 0.3, 5000)
        _, faces = det.detect(bgr)
        if faces is not None and len(faces):
            f = max(faces, key=lambda r: r[2] * r[3])
            reye, leye, mr, ml = f[4:6], f[6:8], f[10:12], f[12:14]
            face = {"box": [float(v) for v in f[:4]],
                    "eyes": (float(reye[0] + leye[0]) / 2, float(reye[1] + leye[1]) / 2),
                    "mouth": (float(mr[0] + ml[0]) / 2, float(mr[1] + ml[1]) / 2,
                              float(np.hypot(*(ml - mr)))),
                    "score": float(f[14])}
            detector = "yunet"
            if len(faces) > 1:
                warnings.append("照片里检测到 %d 张脸，按最大的那张裁剪" % len(faces))
    except Exception as e:  # noqa: BLE001
        warnings.append("YuNet 人脸检测出错（%s），改用 Haar" % e)
if face is None:
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    min_side = max(40, min(W, H) // 12)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=8, minSize=(min_side, min_side))
    if len(faces):
        x, y, w, h = [float(v) for v in max(faces, key=lambda r: r[2] * r[3])]
        # Haar 的框比 YuNet 宽一些（约 1.25 倍），换算成可比的脸宽；眼睛约在框高 40% 处
        face = {"box": [x + 0.1 * w, y, 0.8 * w, h], "eyes": (x + w / 2, y + 0.40 * h),
                "mouth": (x + w / 2, y + 0.80 * h, 0.35 * w)}
        detector = "haar"
        warnings.append("用的是 Haar 粗略人脸检测，可能框错：请打开 input_1280x704.png 确认构图")
if face is None:
    warnings.append("没有检测到正脸：已按'居中偏上'裁剪，请打开 input_1280x704.png 确认（最好换一张正脸照片）")

ar = TW / TH
max_cw = min(float(W), H * ar)          # 能裁出的最大 20:11 区域宽度
cw = max_cw
if face is not None and face["box"][2] / cw < face_frac:
    # 脸太小（人离镜头太远）→ 放大裁剪；裁剪区域至少 640 像素宽，避免放大超过 2 倍变糊
    cw = min(max_cw, max(face["box"][2] / face_frac, min(max_cw, TW * 0.5)))
ch = cw / ar
if face is not None:
    cx, ey = face["box"][0] + face["box"][2] / 2.0, face["eyes"][1]
else:
    cx, ey = W / 2.0, H * (0.28 if H > W else 0.36)
x0 = min(max(cx - cw / 2.0, 0.0), W - cw)
y0 = min(max(ey - eye_y * ch, 0.0), H - ch)
box = (int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch)))
crop = img.crop(box)
up = TW / crop.width
crop.resize((TW, TH), Image.LANCZOS).save(dst)

meta = {"source": os.path.abspath(src), "source_size": [W, H], "crop_box": list(box),
        "upscale": round(up, 3), "detector": detector}
if face is not None:
    s = TW / (box[2] - box[0])
    k = FH / TH                          # ffmpeg：1280x704 等比放大到 1309x720，再居中裁成 1280x720
    off = (round(TW * k) - FW) // 2

    def to720(x, y):
        return [round(((x - box[0]) * s) * k - off, 1), round(((y - box[1]) * s) * k, 1)]

    bx, by, bw, bh = face["box"]
    fb = to720(bx, by) + [round(bw * s * k, 1), round(bh * s * k, 1)]
    mx, my, mw = face["mouth"]
    meta["face_box_1280x720"] = fb
    meta["eyes_1280x720"] = to720(*face["eyes"])
    meta["mouth_1280x720"] = to720(mx, my) + [round(mw * s * k, 1)]
    meta["face_width_ratio_720p"] = round(fb[2] / FW, 3)
    if fb[2] / FW < 0.20:
        warnings.append("脸宽只占画面 %.0f%%（目标约 22~30%%）：照片里人离镜头太远或分辨率不够，建议换一张脸更大的照片" % (100 * fb[2] / FW))
    if fb[2] / FW > 0.40:
        warnings.append("脸宽占画面 %.0f%%，偏大（离镜头太近），可以换一张离镜头远一点的照片" % (100 * fb[2] / FW))
if up > 2.05:
    warnings.append("照片分辨率偏低，被放大了 %.1f 倍，成片会偏糊" % up)
meta["warnings"] = warnings
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print("原图 %dx%d → 裁剪区域 %s（放大 %.2f 倍）→ %s" % (W, H, box, up, dst))
if face is not None:
    print("人脸检测：%s；成片中脸宽约占画面 %.0f%%，眼睛在画面高度 %.0f%% 处"
          % (detector, 100 * meta["face_width_ratio_720p"], 100 * meta["eyes_1280x720"][1] / FH))
for w in warnings:
    print("【注意】" + w)
PYEOF

# ---- 工具 2：调用 Wan2.2 官方 WanTI2V 管线生成（可自定义反向提示词、多种子） ----
cat > "$TOOLS/wan_idle_gen.py" <<'PYEOF'
# -*- coding: utf-8 -*-
"""Wan2.2-TI2V-5B 图生视频（待机视频专用的薄封装）。

调用方式与官方命令完全一致：
  python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ... --offload_model True \
                     --convert_model_dtype --t5_cpu --image ... --prompt ...
即 WanTI2V(config=WAN_CONFIGS['ti2v-5B'], t5_cpu=True, convert_model_dtype=True)
   .generate(..., size=SIZE_CONFIGS['1280*704'], max_area=MAX_AREA_CONFIGS['1280*704'], offload_model=True)
额外功能（官方 generate.py 没有的）：
  * 自定义反向提示词（WanTI2V.generate 的 n_prompt 参数；官方命令行没暴露）
  * 一次加载模型，连续生成多个随机种子
  * 逐帧无损 PNG 输出（官方 save_video 是有损 mp4），后处理不再二次压缩
  * 没装上 flash_attn 时，自动换成 PyTorch 自带的 scaled_dot_product_attention
"""
import argparse
import json
import logging
import math
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Wan2.2 官方仓库目录")
    p.add_argument("--ckpt", help="Wan2.2-TI2V-5B 权重目录")
    p.add_argument("--image", help="已裁好的 1280x704 输入图")
    p.add_argument("--out_root", help="输出目录（每个种子一个 seedXXX 子目录）")
    p.add_argument("--seeds", type=int, nargs="+", default=[101])
    p.add_argument("--prompt_file")
    p.add_argument("--neg_file")
    p.add_argument("--frames", type=int, default=121)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guide", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--size", default="1280*704", choices=["1280*704", "704*1280"])
    p.add_argument("--attn", default="auto", choices=["auto", "flash", "sdpa"])
    p.add_argument("--run_key", default="", help="照片+参数的指纹，变了就重新生成")
    p.add_argument("--selftest", action="store_true", help="只做注意力实现的自检，不加载模型")
    return p.parse_args()


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                   q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                   dtype=None, version=None):
    """与 wan.modules.attention.flash_attention 同签名、同语义的 SDPA 实现。
    q/k/v: [B, L, N, C]。k_lens 之后的 key 被屏蔽（与 flash_attn_varlen_func 一致）。"""
    import torch
    import torch.nn.functional as F
    if dtype is None:
        dtype = torch.bfloat16
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("SDPA 替代实现不支持滑动窗口注意力（TI2V-5B 用不到）")
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    if q_lens is not None and int(q_lens.min()) != lq:
        raise NotImplementedError("SDPA 替代实现不支持 q_lens 填充（TI2V-5B 用不到）")
    v = half(v)
    q = half(q).to(v.dtype)
    k = half(k).to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    attn_mask = None
    if k_lens is not None and int(k_lens.min()) < lk:
        idx = torch.arange(lk, device=q.device)
        attn_mask = (idx[None, :] < k_lens.to(q.device)[:, None]).view(b, 1, 1, lk)
    x = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return x.transpose(1, 2).contiguous().type(out_dtype)


def selftest(wan_attn, using_sdpa):
    """随机数据对比：SDPA 替代实现 / flash_attn 实现 vs. 朴素 fp32 注意力。"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    b, lq, lk, n, d = 1, 257, 96, 4, 128
    q = torch.randn(b, lq, n, d, device=dev)
    k = torch.randn(b, lk, n, d, device=dev)
    v = torch.randn(b, lk, n, d, device=dev)

    def naive(klen):
        s = torch.einsum("blnd,bmnd->bnlm", q.double(), k.double()) / math.sqrt(d)
        if klen is not None:
            s[..., klen:] = float("-inf")
        return torch.einsum("bnlm,bmnd->blnd", s.softmax(-1), v.double()).float()

    worst = 0.0
    impls = [("sdpa", sdpa_attention)]
    if dev == "cuda" and (wan_attn.FLASH_ATTN_2_AVAILABLE or wan_attn.FLASH_ATTN_3_AVAILABLE):
        impls.append(("flash_attn", wan_attn.flash_attention))
    for name, fn in impls:
        for klen in (None, lk, 60):
            kl = None if klen is None else torch.tensor([klen], dtype=torch.long)
            out = fn(q, k, v, k_lens=kl)
            err = (out.float() - naive(None if klen in (None, lk) else klen)).abs().max().item()
            worst = max(worst, err)
            print("  自检 %-10s k_lens=%-4s 最大误差 %.4f" % (name, klen, err))
    print("  自检设备=%s，生成时使用：%s" % (dev, "PyTorch SDPA" if using_sdpa else "flash_attn"))
    if worst > 0.06:
        raise SystemExit("注意力实现自检失败（误差 %.3f 过大）" % worst)
    print("  自检通过")


def save_frames(video, out_dir):
    """video: [C, N, H, W]，取值 [-1, 1] → out_dir/000000.png ...（无损）"""
    import torch
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    arr = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).cpu().numpy()
    for i, fr in enumerate(arr):
        Image.fromarray(fr).save(os.path.join(out_dir, "%06d.png" % i), compress_level=1)
    return arr.shape


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.repo))
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])
    import torch
    import wan
    import wan.modules.attention as wan_attn
    import wan.modules.model as wan_model
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    have_flash = bool(wan_attn.FLASH_ATTN_2_AVAILABLE or wan_attn.FLASH_ATTN_3_AVAILABLE)
    if args.attn == "flash" and not have_flash:
        raise SystemExit("要求使用 flash_attn，但它没有装好")
    using_sdpa = args.attn == "sdpa" or not have_flash
    if using_sdpa:
        # WanModel 的自注意力/交叉注意力都通过 wan.modules.model 里的 flash_attention 调用
        wan_model.flash_attention = sdpa_attention
        logging.info("flash_attn 不可用或未启用 → 使用 PyTorch SDPA（画质相同）")

    if args.selftest:
        selftest(wan_attn, using_sdpa)
        return

    assert (args.frames - 1) % 4 == 0, "frames 必须是 4n+1"
    from PIL import Image
    cfg = WAN_CONFIGS["ti2v-5B"]
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read().strip()
    with open(args.neg_file, encoding="utf-8") as f:
        neg = f.read().strip()
    img = Image.open(args.image).convert("RGB")
    if img.size != SIZE_CONFIGS[args.size]:
        logging.warning("输入图是 %s，不是 %s；Wan 会按比例自动缩放/裁剪", img.size, SIZE_CONFIGS[args.size])

    def done_key(seed):
        path = os.path.join(args.out_root, "seed%d" % seed, "DONE")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        return None

    todo = [s for s in args.seeds if done_key(s) != args.run_key]
    skipped = [s for s in args.seeds if s not in todo]
    if skipped:
        logging.info("这些种子已经生成过（照片和参数都没变），跳过：%s", skipped)
    if not todo:
        return

    logging.info("加载模型（第一次从磁盘读 30+GB，约 1~3 分钟）…")
    t0 = time.time()
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,                 # = 官方 --t5_cpu
        convert_model_dtype=True,    # = 官方 --convert_model_dtype（DiT 转 bf16）
    )
    logging.info("模型加载完成，用时 %.0f 秒", time.time() - t0)

    for seed in todo:
        d = os.path.join(args.out_root, "seed%d" % seed)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "DONE")):
            os.remove(os.path.join(d, "DONE"))
        logging.info("==== 生成 seed=%d（%d 帧，%d 步）====", seed, args.frames, args.steps)
        t1 = time.time()
        video = pipe.generate(
            prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frames,
            shift=args.shift,
            sample_solver=args.solver,
            sampling_steps=args.steps,
            guide_scale=args.guide,
            n_prompt=neg,
            seed=seed,
            offload_model=True,          # = 官方 --offload_model True
        )
        shape = save_frames(video, os.path.join(d, "frames"))
        # 官方同款预览 mp4（24fps，有损），只用来快速预览
        save_video(tensor=video[None], save_file=os.path.join(d, "raw_24fps_preview.mp4"),
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
        del video
        torch.cuda.empty_cache()
        meta = {"seed": seed, "frames": int(shape[0]), "height": int(shape[1]), "width": int(shape[2]),
                "steps": args.steps, "guide": args.guide, "shift": args.shift, "solver": args.solver,
                "attention": "sdpa" if using_sdpa else "flash_attn",
                "seconds": round(time.time() - t1, 1), "prompt": prompt, "negative_prompt": neg}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        with open(os.path.join(d, "DONE"), "w", encoding="utf-8") as f:
            f.write(args.run_key)
        logging.info("seed=%d 完成：%d 帧 %dx%d，用时 %.0f 秒", seed, shape[0], shape[2], shape[1], time.time() - t1)


if __name__ == "__main__":
    main()
PYEOF

[ "${ONLY_WRITE_TOOLS:-0}" = "1" ] && { echo "工具已写到 $TOOLS"; exit 0; }

# ============================ 0. 检查机器 ====================================
cgroup_mem_gb() {   # 容器的内存上限（AutoDL 里 free 看到的是宿主机内存，不准）
  local lim=""
  if [ -r /sys/fs/cgroup/memory.max ]; then lim=$(cat /sys/fs/cgroup/memory.max)
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes); fi
  if [[ "$lim" =~ ^[0-9]+$ ]] && (( lim < 1099511627776 )); then echo $(( lim / 1073741824 ))
  else free -g | awk '/^Mem:/{print $2}'; fi
}
cgroup_cpus() {     # 容器实际能用的 CPU 核数（给 CPU 上跑的 T5 文本编码器用）
  local q p n
  n=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max || true
    if [[ "$q" =~ ^[0-9]+$ ]] && [[ "$p" =~ ^[0-9]+$ ]] && (( p > 0 )); then n=$(( q / p )); fi
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [[ "$q" =~ ^[0-9]+$ ]] && (( q > 0 && p > 0 )); then n=$(( q / p )); fi
  fi
  (( n >= 1 )) || n=1
  (( n <= $(nproc) )) || n=$(nproc)
  echo "$n"
}

if [ "$ONLY" = "all" ]; then
  step "0. 检查机器"
  command -v nvidia-smi >/dev/null 2>&1 || die "没有检测到 NVIDIA 显卡（nvidia-smi 不存在）。请确认租的是 GPU 实例，且不是用「无卡模式」开机的"
  GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -n1)
  GPU_NAME=$(echo "$GPU_LINE" | cut -d, -f1 | xargs)
  GPU_TOTAL=$(echo "$GPU_LINE" | cut -d, -f2 | tr -dc 0-9)
  GPU_USED=$(echo "$GPU_LINE" | cut -d, -f3 | tr -dc 0-9)
  GPU_TOTAL=${GPU_TOTAL:-0}; GPU_USED=${GPU_USED:-0}
  CUDA_DRV=$(nvidia-smi | grep -o 'CUDA Version: *[0-9.]*' | grep -o '[0-9.]*$' || echo "?")
  echo "显卡：$GPU_NAME，显存 ${GPU_TOTAL}MiB（已用 ${GPU_USED}MiB），驱动支持的最高 CUDA：$CUDA_DRV"
  (( GPU_TOTAL >= 22000 )) || die "显存不够：需要 24GB 显卡（RTX 4090 / 3090 等），当前 ${GPU_TOTAL}MiB"
  (( GPU_USED <= 2500 )) || warn "显卡已被占用 ${GPU_USED}MiB——LiveTalking 可能还开着。请先停掉它，否则生成时会爆显存（峰值约 23GB）"
  MEM_GB=$(cgroup_mem_gb)
  echo "内存：约 ${MEM_GB}GB；CPU：$(cgroup_cpus) 核"
  (( MEM_GB >= 40 )) || warn "内存只有 ${MEM_GB}GB，加载模型时峰值约 35GB，可能被系统杀掉。建议选内存 ≥ 60GB 的实例"
  NEED_GB=3
  [ -f "$CKPT/.download_ok" ] || NEED_GB=$(( NEED_GB + 36 ))
  [ -f "$WORK/.install_ok" ]  || NEED_GB=$(( NEED_GB + 9 ))
  FREE_GB=$(df -BG --output=avail "$BASE" | tail -n1 | tr -dc 0-9)
  echo "数据盘 $BASE 剩余 ${FREE_GB}GB，本次还需要约 ${NEED_GB}GB"
  (( FREE_GB >= NEED_GB )) || die "数据盘空间不够。AutoDL 控制台 → 容器实例 → 更多 → 扩容数据盘（建议扩到 100GB），或删掉不用的文件后重试"
fi

# ============================ 1. 安装（只做一次） =============================
install_flash_attn() {
  # flash_attn 官方预编译 wheel（不需要现场编译）。下载失败也没关系：会自动改用 PyTorch SDPA。
  local pyt abi whl url
  pyt=$("$VPY" -c 'import sys;print("cp%d%d"%sys.version_info[:2])')
  abi=$("$VPY" -c 'import torch;print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')
  whl="flash_attn-2.7.4.post1+cu12torch2.6cxx11abi${abi}-${pyt}-${pyt}-linux_x86_64.whl"
  url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${whl}"
  mkdir -p "$WORK/wheels"
  if [ ! -f "$WORK/wheels/$whl" ]; then
    turbo_on
    if curl -fL --retry 3 --connect-timeout 20 -o "$WORK/wheels/$whl.part" "$url"; then
      mv "$WORK/wheels/$whl.part" "$WORK/wheels/$whl"
    fi
    turbo_off
  fi
  if [ -f "$WORK/wheels/$whl" ] && "$VPY" -m pip install --no-cache-dir --no-deps "$WORK/wheels/$whl" \
     && "$VPY" -c "import flash_attn; from flash_attn import flash_attn_varlen_func" 2>/dev/null; then
    echo "flash_attn 安装成功"
  else
    "$VPY" -m pip uninstall -y flash_attn >/dev/null 2>&1 || true   # 装坏了的要删掉，否则 import wan 会报错
    warn "flash_attn 没装上（不影响效果），生成时会自动改用 PyTorch SDPA"
  fi
}

pick_python() {   # Wan2.2 要求 Python >= 3.10；numpy<2 最高支持到 3.12
  local c v
  for c in python3.10 python3.11 python3.12 python3 python /root/miniconda3/bin/python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
    case "$v" in 3.10|3.11|3.12) command -v "$c"; return 0 ;; esac
  done
  return 1
}

if [ "$ONLY" = "all" ] && [ ! -f "$WORK/.install_ok" ]; then
  step "1. 安装 Wan2.2 + 独立 Python 环境（只做一次，约 10~20 分钟）"

  # 1a. 官方代码（GitHub，用 AutoDL 学术加速）
  if [ ! -f "$REPO/generate.py" ]; then
    rm -rf "$REPO.tmp"; mkdir -p "$REPO.tmp"
    turbo_on
    if [ "$WAN_COMMIT" = "latest" ]; then
      git clone --depth 1 "$WAN_REPO_URL" "$REPO.tmp"
    else
      ( cd "$REPO.tmp" && git init -q && git remote add origin "$WAN_REPO_URL" \
        && git fetch -q --depth 1 origin "$WAN_COMMIT" && git checkout -q FETCH_HEAD ) \
      || { rm -rf "$REPO.tmp"; git clone "$WAN_REPO_URL" "$REPO.tmp" && git -C "$REPO.tmp" checkout -q "$WAN_COMMIT"; }
    fi
    turbo_off
    rm -rf "$REPO"
    mv "$REPO.tmp" "$REPO"
  fi
  echo "Wan2.2 代码版本：$(git -C "$REPO" rev-parse --short HEAD)"

  # 1b. 独立虚拟环境（放在数据盘，不碰 LiveTalking 的环境）
  if [ ! -x "$VPY" ]; then
    rm -rf "$VENV"
    if BASE_PY=$(pick_python) && "$BASE_PY" -m venv "$VENV" && [ -x "$VPY" ] && "$VPY" -m pip --version >/dev/null 2>&1; then
      echo "用 $BASE_PY 创建了虚拟环境"
    elif command -v conda >/dev/null 2>&1; then
      rm -rf "$VENV"
      conda create -y -p "$VENV" python=3.10 pip
    else
      die "找不到 Python 3.10~3.12，也没有 conda。请换用 AutoDL 的 PyTorch 基础镜像，或告诉我报错"
    fi
  fi
  echo "Python：$("$VPY" -V)"
  if ! "$VPY" -m pip config list 2>/dev/null | grep -q 'index-url'; then
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"   # 没配置镜像源时用阿里云
  fi
  PIPI=("$VPY" -m pip install --no-cache-dir --disable-pip-version-check)
  "${PIPI[@]}" -U pip setuptools wheel

  # 1c. PyTorch 2.6.0（PyPI 上的 Linux 版默认就是 CUDA 12.4 版本，自带 CUDA 运行库）
  #     如需指定源：TORCH_PIP_ARGS="--index-url https://download.pytorch.org/whl/cu124"
  # shellcheck disable=SC2086
  "${PIPI[@]}" torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 ${TORCH_PIP_ARGS:-}
  "$VPY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, '显卡可用' if ok else '显卡不可用'); sys.exit(0 if ok else 1)" \
    || die "PyTorch 用不了显卡：多半是机器驱动太旧（需要支持 CUDA 12.x 的驱动）。换一台 AutoDL 机器（选'最高支持 CUDA ≥ 12.4'的）"

  # 1d. 官方依赖（requirements.txt 去掉 flash_attn，单独装预编译版）
  #     额外：einops / peft / decord / librosa —— Wan2.2 的 `import wan` 会顺带导入
  #     Animate / S2V 模块，它们需要这几个包，不装会直接报 ModuleNotFoundError。
  #     diffusers / peft 固定到和 torch 2.6 + transformers 4.51 验证过能一起用的版本。
  printf 'torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\nnumpy<2\n' > "$WORK/constraints.txt"
  grep -v -i -E '^[[:space:]]*flash[_-]attn' "$REPO/requirements.txt" > "$WORK/requirements_noflash.txt"
  "${PIPI[@]}" -r "$WORK/requirements_noflash.txt" -c "$WORK/constraints.txt" \
    "diffusers==0.35.2" "peft==0.17.1" einops decord librosa "modelscope>=1.20"
  # 无桌面的服务器上 opencv-python 可能缺 libGL.so.1 → 换成功能相同的 headless 版本
  if ! "$VPY" -c "import cv2" >/dev/null 2>&1; then
    OCV=$("$VPY" -c "import importlib.metadata as m; print(m.version('opencv-python'))" 2>/dev/null || echo "4.11.0.86")
    "$VPY" -m pip uninstall -y opencv-python >/dev/null 2>&1 || true
    "${PIPI[@]}" "opencv-python-headless==$OCV" -c "$WORK/constraints.txt"
    "$VPY" -c "import cv2" || die "OpenCV 装不上，把上面的报错截图发给我"
  fi
  install_flash_attn

  # 1e. 自检：能 import wan，注意力实现数值正确
  ( cd "$WORK" && "$VPY" "$TOOLS/wan_idle_gen.py" --repo "$REPO" --attn "$ATTN" --selftest ) \
    || die "环境自检失败，把上面的报错截图发给我"
  touch "$WORK/.install_ok"
fi
[ -x "$VPY" ] || die "环境还没装好：先不带 ONLY 参数完整运行一次"

# ============================ 2. 下载权重（只做一次） ==========================
verify_ckpt() {   # 检查关键文件是否完整（大小和官方一致）
  "$VPY" - "$CKPT" <<'PYEOF'
import os, sys
d = sys.argv[1]
need = {
    "Wan2.2_VAE.pth": 2818839170,
    "models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
    "diffusion_pytorch_model-00001-of-00003.safetensors": 9825014472,
    "diffusion_pytorch_model-00002-of-00003.safetensors": 9995661736,
    "diffusion_pytorch_model-00003-of-00003.safetensors": 178558176,
    "diffusion_pytorch_model.safetensors.index.json": None,
    "config.json": None,
    "google/umt5-xxl/spiece.model": None,
    "google/umt5-xxl/tokenizer.json": None,
    "google/umt5-xxl/tokenizer_config.json": None,
    "google/umt5-xxl/special_tokens_map.json": None,
}
bad = []
for f, size in need.items():
    p = os.path.join(d, f)
    if not os.path.isfile(p):
        bad.append(f + "（缺失）")
    elif size is not None and os.path.getsize(p) != size:
        bad.append(f + "（大小不对：%d != %d）" % (os.path.getsize(p), size))
if bad:
    print("权重不完整：" + "；".join(bad))
    sys.exit(1)
print("权重完整")
PYEOF
}

if [ "$ONLY" = "all" ] && [ ! -f "$CKPT/.download_ok" ]; then
  step "2. 下载 Wan2.2-TI2V-5B 权重（约 34GB，只下一次；断了重新运行会续传）"
  mkdir -p "$CKPT"
  export MODELSCOPE_CACHE="$WORK/.modelscope_cache"
  download_ms() {
    "$VPY" -c "
import inspect
from modelscope import snapshot_download
kw = {'local_dir': '$CKPT'}
if 'max_workers' in inspect.signature(snapshot_download).parameters:
    kw['max_workers'] = 8
snapshot_download('$MODEL_ID', **kw)
"
  }
  download_hf() {   # hf-mirror.com 是国内的 HuggingFace 镜像
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_ENABLE_HF_TRANSFER=0 "$VPY" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL_ID', local_dir='$CKPT', max_workers=8)
"
  }
  ok=0
  for attempt in 1 2 3; do
    if [ "$DL_SOURCE" = "hf-mirror" ]; then download_hf && ok=1 || true
    else download_ms && ok=1 || true; fi
    if [ "$ok" = "1" ] && verify_ckpt; then break; fi
    ok=0; warn "第 $attempt 次下载没成功，10 秒后重试…"; sleep 10
  done
  if [ "$ok" != "1" ] && [ "$DL_SOURCE" != "hf-mirror" ]; then
    warn "魔搭下载失败，改用 hf-mirror.com"
    download_hf && verify_ckpt && ok=1 || true
  fi
  [ "$ok" = "1" ] || die "权重下载失败。可以重新运行脚本继续下载（已下载的部分会保留）"
  rm -rf "$WORK/.modelscope_cache"
  touch "$CKPT/.download_ok"
fi

# ============================ 3. 裁剪照片 =====================================
step "3. 按视频通话构图裁剪照片"
# OpenCV 的 YuNet 人脸检测模型（230KB，带眼睛/嘴角关键点）。下载失败也能继续（退回 Haar 粗检测）
YUNET="$TOOLS/face_detection_yunet_2023mar.onnx"
YUNET_SHA=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
if [ ! -f "$YUNET" ]; then
  for u in "https://hf-mirror.com/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx" \
           "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx" \
           "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"; do
    case "$u" in https://hf-mirror.com/*) ;; *) turbo_on ;; esac
    curl -fsSL --retry 2 --connect-timeout 15 -o "$YUNET.part" "$u" || true
    turbo_off
    if [ -f "$YUNET.part" ] && [ "$(sha256sum "$YUNET.part" | cut -d' ' -f1)" = "$YUNET_SHA" ]; then
      mv "$YUNET.part" "$YUNET"; break
    fi
    rm -f "$YUNET.part"
  done
  [ -f "$YUNET" ] || warn "YuNet 人脸模型下载失败，改用 OpenCV 自带的 Haar 粗检测（构图可能不准，请检查裁剪结果）"
fi
FACE_FRAC="$FACE_FRAC" EYE_Y="$EYE_Y" \
  "$VPY" "$TOOLS/prep_image.py" "$PHOTO" "$OUT/input_1280x704.png" "$OUT/input_meta.json" "$YUNET"
printf '%s\n' "$PROMPT" > "$OUT/prompt.txt"
printf '%s\n' "$NEG"    > "$OUT/negative_prompt.txt"
if [ "$ONLY" = "prep" ]; then
  echo "只裁剪：请打开 $OUT/input_1280x704.png 看构图。满意的话去掉 ONLY=prep 再运行一次。"
  exit 0
fi

# ============================ 4. 生成 =========================================
# 照片 + 提示词 + 参数 的指纹：都没变的种子会跳过，避免重复花时间
RUN_KEY=$( { sha256sum "$OUT/input_1280x704.png" | cut -d' ' -f1; cat "$OUT/prompt.txt" "$OUT/negative_prompt.txt"; \
             echo "$FRAMES $STEPS $GUIDE $SHIFT unipc 1280*704"; } | sha256sum | cut -d' ' -f1 )
if [ "$ONLY" = "all" ]; then
  step "4. 生成待机视频：种子 [$SEEDS]，每个约 10 分钟（4090）"
  CPUS=$(cgroup_cpus)
  # shellcheck disable=SC2086
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" \
    "$VPY" "$TOOLS/wan_idle_gen.py" --repo "$REPO" --ckpt "$CKPT" \
      --image "$OUT/input_1280x704.png" --out_root "$OUT" --seeds $SEEDS \
      --prompt_file "$OUT/prompt.txt" --neg_file "$OUT/negative_prompt.txt" \
      --frames "$FRAMES" --steps "$STEPS" --guide "$GUIDE" --shift "$SHIFT" \
      --attn "$ATTN" --run_key "$RUN_KEY" 2>&1 | tee -a "$OUT/generate.log" \
    || die "生成失败（日志：$OUT/generate.log）。常见原因：显存不够（先停掉 LiveTalking）、内存不够、权重不完整。把报错截图发给我"
fi

# ============================ 5. 后处理 =======================================
# - 帧率：模型按 24fps 生成，这里直接按 25fps 播放（快 4%，待机微动看不出来），不插帧不重复帧
# - 尺寸：1280×704 等比放大到 1309×720，再居中裁成 1280×720（左右各裁约 14 像素，不加黑边）
# - 无缝循环：正放 0…N-1，再倒放 N-2…1（不重复首尾帧）→ 2N-2 帧，最后一帧自然接回第一帧。
#   （LiveTalking 播放底片时本身也是"正放→倒放"来回循环的，两者叠加依然无缝）
# - GRAIN（噪点）默认 0：wav2lip/MuseTalk 会把整张脸换成它生成的、没有噪点的脸，
#   底片上有噪点反而会让脸部出现一块"干净的方框"。更好的做法是在口型合成之后（或网页播放时）
#   再统一加噪点。想先试试效果可以 GRAIN=3~5。
step "5. 后处理：25fps / 1280x720 / 正放+倒放无缝循环"
if command -v ffmpeg >/dev/null 2>&1 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264; then
  FFMPEG=$(command -v ffmpeg)
else
  FFMPEG=$("$VPY" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')   # 环境里自带的 ffmpeg
fi

MOUTH_CROP=$("$VPY" - "$OUT/input_meta.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
mo = m.get("mouth_1280x720")
if not mo:
    print("crop=640:360:320:300")          # 没检测到脸：取画面中下部
else:
    mx, my, mw = mo                        # 嘴巴中心 + 两个嘴角的距离
    mw = max(mw, 24.0)
    cw, ch = min(2.4 * mw, 1280.0), min(1.7 * mw, 720.0)    # 嘴 + 下巴
    x0 = max(0.0, min(mx - cw / 2, 1280 - cw)); y0 = max(0.0, min(my - 0.45 * ch, 720 - ch))
    print("crop=%d:%d:%d:%d" % (int(cw) // 2 * 2, int(ch) // 2 * 2, int(x0), int(y0)))
PYEOF
)

made=()
for seed in $SEEDS; do
  d="$OUT/seed$seed"
  if [ ! -f "$d/DONE" ]; then warn "seed $seed 还没有生成结果，跳过"; continue; fi
  n=$(find "$d/frames" -maxdepth 1 -name '*.png' | wc -l)
  (( n >= 9 )) || { warn "seed $seed 的帧数不对（$n），跳过"; continue; }
  rm -rf "$d/seq"; mkdir -p "$d/seq"
  j=0
  for (( i = 0; i < n; i++ ));    do ln -s "../frames/$(printf '%06d' "$i").png" "$d/seq/$(printf '%06d' "$j").png"; j=$((j + 1)); done
  for (( i = n - 2; i >= 1; i-- )); do ln -s "../frames/$(printf '%06d' "$i").png" "$d/seq/$(printf '%06d' "$j").png"; j=$((j + 1)); done

  final="$OUT/${NAME}_wan_s${seed}.mp4"
  VF="scale=1280:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1280:720"
  VF="$VF,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
  if (( GRAIN > 0 )); then VF="$VF,noise=c0s=${GRAIN}:c0f=t"; fi   # 只加在亮度上、逐帧变化的细噪点
  "$FFMPEG" -hide_banner -loglevel error -y -framerate 25 -start_number 0 -i "$d/seq/%06d.png" \
    -vf "$VF" -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
    -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
    -g 50 -movflags +faststart -an "$final"

  # 检查图：整体 12 帧 + 嘴部 24 帧（确认嘴巴全程闭合）
  total=$(( 2 * n - 2 ))
  e1=$(( total / 12 )); (( e1 >= 1 )) || e1=1
  e2=$(( total / 24 )); (( e2 >= 1 )) || e2=1
  "$FFMPEG" -hide_banner -loglevel error -y -i "$final" \
    -vf "select='not(mod(n\,$e1))',scale=426:-2,tile=4x3" -frames:v 1 "${final%.mp4}_sheet.jpg"
  "$FFMPEG" -hide_banner -loglevel error -y -i "$final" \
    -vf "select='not(mod(n\,$e2))',$MOUTH_CROP,scale=240:-2,tile=6x4" -frames:v 1 "${final%.mp4}_mouth.jpg"
  echo "seed $seed → $final （$total 帧，$(awk "BEGIN{printf \"%.1f\", $total/25}") 秒）"
  made+=("$final")
done

echo
if (( ${#made[@]} == 0 )); then
  warn "没有生成任何成片。看看上面的报错，或者 $OUT/generate.log"
  exit 1
fi
echo "【完成】成片（1280×720 / 25fps / H.264 yuv420p / 无缝循环）："
for f in "${made[@]}"; do echo "  $f"; done
cat <<EOF

下一步：
  1) 在 JupyterLab 左侧打开 $OUT ，双击 *_mouth.jpg 看嘴巴是否全程闭合、*_sheet.jpg 看整体，
     再下载 mp4 看一遍（重点看：嘴、眼睛、有没有变脸、背景有没有乱动）
  2) 挑最好的一个，生成 LiveTalking 形象并启动（换个新形象名，避免用到旧缓存）：
       AVATAR=her_wan bash $BASE/autodl_start.sh 阿里云公网IP ${made[0]}
  3) 都不满意：换几个种子再来，例如  SEEDS="404 505 606" bash $0 $PHOTO
EOF
