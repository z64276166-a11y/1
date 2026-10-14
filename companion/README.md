# 和她视频通话 · 数字人女友（自用版）

在自己电脑上运行的"视频通话女友"：
- **大脑**：DeepSeek，人设可自定义（`persona.md`）
- **长期记忆**：自动记住关于你的事、你们聊过的事，下次通话时会自然提起（存在本地 `data/memory.db`）
- **实时语音**：直接说话就行，自动判断你说完了；她说话时你开口可以**打断**她
- **主动性**：接通时根据时间、多久没聊、待跟进的事主动打招呼；你沉默太久她会找话题
- **两种形象**
  - `live2d`：二次元形象，任何电脑都能跑（默认）
  - `realistic`：**真人照片实感**的视频画面，需要 N 卡（见下文）

---

## 一、先跑起来（10 分钟）

1. 安装 [Python 3.10~3.12](https://www.python.org/downloads/)（安装时勾选 *Add Python to PATH*）
2. 在 [DeepSeek 开放平台](https://platform.deepseek.com/) 充值几块钱，创建一个 API Key
3. **Windows**：双击 `start.bat`；**Mac**：打开"终端"，把 `start.sh` 拖进终端窗口后回车（如果提示没有权限，先运行 `chmod +x` 再把文件拖进来）
   - 第一次会自动安装依赖，然后弹出记事本让你填 `DEEPSEEK_API_KEY`，填好保存后再双击一次
4. 浏览器（推荐 Chrome / Edge）会自动打开 `http://127.0.0.1:8765`，点 **接通视频**，允许麦克风

> 建议戴耳机，否则她的声音可能被麦克风收进去，误以为你在说话而打断她自己。

第一次启动会自动下载语音识别模型（几百 MB 到 3 GB），需要等一会儿。

## 二、调教她

| 想改什么 | 改哪里 |
|---|---|
| 名字、怎么称呼你 | `.env` 里的 `COMPANION_NAME` / `USER_NAME` |
| 性格、背景、说话方式 | `persona.md`（写得越细越像真人） |
| 声音 | `.env` 里的 `EDGE_VOICE`，可选：`zh-CN-XiaoyiNeural`（活泼）、`zh-CN-XiaoxiaoNeural`（温柔）、`zh-CN-liaoning-XiaobeiNeural`（东北话）、`zh-TW-HsiaoChenNeural`（台湾腔） |
| 语音识别更准（Mac M 系列芯片 / 内存 16 GB 以上） | `.env` 里设 `WHISPER_MODEL=medium` |
| 想要独一无二的声音 | 部署 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，用 1 分钟左右的参考音频克隆声音，`.env` 里设 `TTS_ENGINE=gptsovits` |
| 她多久不说话会主动找你 | `.env` 里的 `IDLE_SECONDS` |
| 看她记住了什么 | 页面上的"查看她记得的事"，或访问 `http://127.0.0.1:8765/memory` |
| 清空记忆重新开始 | 删除 `data/memory.db` |

## 三、真人写实形象（像真正打视频）

写实画面由开源项目 **[LiveTalking](https://github.com/lipku/LiveTalking)** 渲染：它用一段"她"的真人视频作底，根据声音实时生成口型，再通过 WebRTC 把画面推给浏览器。本程序负责大脑、记忆和打断，LiveTalking 负责画面和声音。

### 1. 显卡要求

| 口型模型 | 画质 | 最低显卡（实时 25fps） |
|---|---|---|
| `wav2lip` | 能用，嘴部略糊 | RTX 3060 |
| `musetalk` | **好，推荐** | RTX 3080 Ti / 3090 / 4090 |

没有 N 卡（或者是 A 卡、核显、Mac）就跑不了写实模式，请用 Live2D 模式。阿里云的普通服务器没有 GPU，帮不上忙；租云 GPU 能跑，但按小时计费、延迟也更高，不建议用来长期陪伴。

### 2. 准备"她"的样子

需要一段 **10~30 秒、正脸、上半身、自然坐着、偶尔眨眼、嘴巴闭着** 的视频（越稳越好，背景就是之后视频通话的背景）。

- **推荐用 AI 生成一个不存在的人**：先用文生图（Flux / SD / 即梦 / 可灵等）生成一张满意的写实人像，再用图生视频（可灵、即梦，或本地的 Wan 2.x）生成一段"看着镜头、轻微动作、闭嘴"的视频
- 也可以拍一个**本人同意**的真人朋友
- 不要用明星、网红或任何没授权的真人的照片或视频

### 3. 部署 LiveTalking

按它的 README 安装（Python 3.10 + CUDA 版 PyTorch），下载对应的模型权重放进 `models/`。启动后打开 `http://127.0.0.1:8010/avatar.html`，上传上一步的视频生成形象，记下形象 ID。然后启动：

```bat
python app.py --transport webrtc --model musetalk --avatar_id 你的形象ID --tts edgetts --REF_FILE zh-CN-XiaoyiNeural
```

（声音在这里设置：`--REF_FILE` 填 edge 的声音名；想用克隆声音就设 `--tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE 参考音频路径 --REF_TEXT 参考音频文字`。）

### 4. 切换到写实模式

`.env` 里改成：

```
AVATAR_MODE=realistic
LIVETALKING_URL=http://127.0.0.1:8010
```

先启动 LiveTalking，再双击 `start.bat`，点"接通视频"即可。

## 四、常见问题

- **说话没反应**：看页面右上角的状态。一直是"正在听…"说明麦克风音量太小，检查系统的输入设备
- **她老是自己打断自己**：戴耳机
- **语音识别不准**：有 N 卡的话 `.env` 里设 `WHISPER_DEVICE=cuda`，模型会自动用 `large-v3`
- **形象加载失败**：默认 Live2D 模型从网上 CDN 加载，网络不好时可以把模型下载到 `static/models/` 里再改 `LIVE2D_MODEL`

## 文件说明

```
server.py     服务端：通话、打断、主动开口
brain.py      DeepSeek 对话 + 记忆整理
memory.py     长期记忆（SQLite）
voice.py      语音识别（faster-whisper）和语音合成（edge-tts / GPT-SoVITS）
persona.md    人设
static/       通话页面（Live2D / 写实视频、语音检测、字幕）
```
