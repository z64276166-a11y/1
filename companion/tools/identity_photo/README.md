# 锁定长相、重新生成"视频通话"真实照片

`avatar/her_photo.jpg` 和 `avatar/her_hanfu_photo.jpg` 就是用这个工具生成的。

它不在原图上修修补补，而是用原图里**她的脸**作为身份参考，**从头生成**一张新照片：手机前置摄像头视角、正脸平视、居家环境、自然光。这样能彻底摆脱原图的 CG 感。

技术方案：
- 底模：RealVisXL V5.0 Lightning（SDXL 写实模型）
- 锁脸：InstantID（ArcFace 身份向量 + 正脸关键点）
- 人脸检测和识别：onnxruntime，不需要安装 insightface

## 安装

```bash
python3 -m venv .idvenv && source .idvenv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 有 N 卡就装 CUDA 版
pip install diffusers transformers accelerate safetensors onnxruntime opencv-python pillow huggingface_hub
python tools/identity_photo/download_models.py
```

## 使用

```bash
python tools/identity_photo/gen_identity.py --out her.png --seed 7 21 33 44
```

- 一次跑多个种子，挑最自然的一张。每张图旁边会生成一个 `.json`，里面有和原图的相似度、脸的大小和角度。
- 4 核 CPU 上每张约 3 分钟，峰值内存约 10 GB。
- `--prompt` 改衣服和场景，`--face` 换身份参考图（默认是 `avatar/original.webp`）。
- 选好照片后，用 `tools/idle_avatar/` 把它做成待机视频。

查看与原图的相似度（大于 0.6 基本可以认为是同一个人）：

```bash
python tools/identity_photo/face_sim.py avatar/original.webp her_s7.png
```
