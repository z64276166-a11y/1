# 用一张照片生成待机视频（LivePortrait）

`avatar/her.mp4` 就是用这个工具做的。以后想换形象或换一张照片时，可以用它重新生成。

它会把**一张照片**做成一段可以无缝循环的待机视频，给 LiveTalking 当底片：
- 自然眨眼，间隔随机，偶尔连眨两下
- 头部轻微晃动（±1~2°）和呼吸起伏
- **嘴巴全程闭合**：表情参数完全不动，嘴型只交给 wav2lip / MuseTalk 去生成
- 首尾无缝衔接

## 安装（Python 3.10，CPU 即可，有 GPU 也行）

```bash
git clone https://github.com/KwaiVGI/LivePortrait tools/idle_avatar/LivePortrait
python3.10 -m venv .lpvenv && source .lpvenv/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r tools/idle_avatar/LivePortrait/requirements_base.txt onnxruntime==1.18.0 openvino==2024.4.0
huggingface-cli download KwaiVGI/LivePortrait --local-dir tools/idle_avatar/LivePortrait/pretrained_weights --exclude "*.git*" "README.md" "docs"
```

## 使用

```bash
python tools/idle_avatar/idle_animate.py --source 照片.png --out her.mp4 --seconds 12 --fps 25 --check
```

在 4 核 CPU 上，12 秒的视频大约需要 10 分钟，渲染到一半中断了可以接着跑。

常用参数：

| 参数 | 作用 |
|---|---|
| `--width 1280` | 先把照片缩放到这个宽度（建议 720p，速度和画质比较平衡） |
| `--crop x0,y0,x1,y1` | 先裁剪照片 |
| `--yaw-amp / --pitch-amp / --roll-amp` | 头部晃动幅度（度） |
| `--base-yaw -8` | 让头稍微转正一点 |
| `--breath-amp 0` | 关掉呼吸起伏 |
| `--breath-mask 人物蒙版.png` | 呼吸只作用在人身上，背景完全不动（横屏照片推荐使用） |
| `--seed N` | 换一种眨眼和晃动节奏 |
| `--check` | 渲染完测量眼睛和嘴巴的张开程度，用来自检 |

照片要求：上半身，脸完整露出，嘴巴闭合，头发不要挡住嘴和下巴。
