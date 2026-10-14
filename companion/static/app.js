// 视频通话前端：Live2D 形象 + 口型同步 + 语音活动检测（可打断）+ 字幕
const $ = (id) => document.getElementById(id);
let CFG = {};
let ws = null;
let audioCtx = null;
let analyser = null;
let model = null;
let mouthOpen = 0;

// ---------------- 形象 ----------------
async function initAvatar() {
  const app = new PIXI.Application({
    view: $("avatar"), resizeTo: $("stage"), backgroundAlpha: 0, antialias: true,
    resolution: window.devicePixelRatio || 1, autoDensity: true,
  });
  try {
    model = await PIXI.live2d.Live2DModel.from(CFG.model, { autoInteract: false });
  } catch (e) {
    console.error(e);
    setStatus("形象加载失败，检查 LIVE2D_MODEL");
    return;
  }
  app.stage.addChild(model);
  model.anchor.set(0.5, 0.5);
  const layout = () => {
    const w = app.screen.width, h = app.screen.height;
    // 像视频通话一样取上半身特写
    const s = Math.min(h / model.internalModel.height * 1.9, w / model.internalModel.width * 2.4);
    model.scale.set(s);
    // 头顶留一点空，让脸在画面上方三分之一处
    model.position.set(w / 2, h * 0.04 + model.height / 2);
  };
  layout();
  window.addEventListener("resize", layout);
  // 眼睛跟随鼠标，像在看着你
  window.addEventListener("pointermove", (e) => model.focus(e.clientX, e.clientY));
  // 口型同步：每帧根据她声音的响度设置嘴巴张开
  const buf = new Uint8Array(1024);
  model.internalModel.on("beforeModelUpdate", () => {
    let target = 0;
    if (analyser && speaking) {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += ((v - 128) / 128) ** 2;
      target = Math.min(1, Math.sqrt(sum / buf.length) * 6);
    }
    mouthOpen += (target - mouthOpen) * 0.5;
    model.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", mouthOpen);
  });
}

// ---------------- 真人写实模式：LiveTalking 通过 WebRTC 推真人视频 ----------------
let pc = null;
let avatarSession = null;
let speakPoll = null;

async function connectRealistic() {
  pc = new RTCPeerConnection({ sdpSemantics: "unified-plan" });
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  const stream = new MediaStream();
  pc.ontrack = (e) => { stream.addTrack(e.track); $("herview").srcObject = stream; };
  await pc.setLocalDescription(await pc.createOffer());
  await new Promise((ok) => {
    if (pc.iceGatheringState === "complete") return ok();
    pc.addEventListener("icegatheringstatechange", () => pc.iceGatheringState === "complete" && ok());
    setTimeout(ok, 3000);
  });
  if (CFG.avatarTransport === "rtcpush") {
    // 画面由 GPU 机器推到云端 SRS，这里用 WHEP 从 SRS 拉流；LiveTalking 的会话固定为 0
    const r = await fetch(CFG.whepUrl, {
      method: "POST", headers: { "Content-Type": "application/sdp" }, body: pc.localDescription.sdp,
    });
    if (!r.ok) throw new Error(`拉流失败 ${r.status}`);
    await pc.setRemoteDescription({ type: "answer", sdp: await r.text() });
    avatarSession = "0";
  } else {
    const r = await fetch(`${CFG.liveTalkingUrl}/offer`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    const ans = await r.json();
    const a = ans.data || ans;
    await pc.setRemoteDescription({ sdp: a.sdp, type: a.type });
    avatarSession = String(a.sessionid);
  }
  // 轮询她是否在说话（用于打断判断和沉默计时）
  speakPoll = setInterval(async () => {
    try {
      const res = await (await fetch(`${CFG.liveTalkingUrl}/is_speaking`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionid: avatarSession }),
      })).json();
      const now = !!res.data;
      if (speaking && !now) { setStatus("正在听…"); resetIdle(); }
      if (now) { setStatus("她在说话"); clearIdle(); }
      speaking = now;
    } catch {}
  }, 400);
}

// ---------------- 播放她的声音 ----------------
let playQueue = [];
let currentSource = null;
let speaking = false;
let currentTurn = 0;
let lastServerTurn = 0;

function enqueueAudio(msg) {
  if (msg.turn < currentTurn) return;
  playQueue.push(msg);
  if (!currentSource) playNext();
}

async function playNext() {
  const msg = playQueue.shift();
  if (!msg) {
    speaking = false;
    currentSource = null;
    if (inCall) setStatus("正在听…");
    resetIdle();
    return;
  }
  const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
  let buffer;
  try {
    buffer = await audioCtx.decodeAudioData(bytes.buffer);
  } catch { return playNext(); }
  if (msg.turn < currentTurn) return playNext();
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(analyser);
  currentSource = src;
  speaking = true;
  setStatus("她在说话");
  $("sub-her").textContent = msg.text;
  src.onended = () => { if (currentSource === src) { currentSource = null; playNext(); } };
  src.start();
  // 偶尔做个动作，更生动
  if (model && Math.random() < 0.25) model.motion("Idle", undefined, 1);
}

function stopSpeaking() {
  playQueue = [];
  if (currentSource) { const s = currentSource; currentSource = null; try { s.stop(); } catch {} }
  speaking = false;
}

// ---------------- 麦克风 + 语音活动检测 ----------------
let micOn = true;
let micStream = null;

async function initMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const src = audioCtx.createMediaStreamSource(micStream);
  const proc = audioCtx.createScriptProcessor(2048, 1, 1);
  const rate = audioCtx.sampleRate;
  const frameMs = 2048 / rate * 1000;
  let noise = 0.005, inSpeech = false, loud = 0, quiet = 0;
  let pre = [], chunks = [];
  const PRE_FRAMES = Math.ceil(300 / frameMs);
  const END_FRAMES = Math.ceil(750 / frameMs);
  const mute = audioCtx.createGain(); mute.gain.value = 0;
  src.connect(proc); proc.connect(mute); mute.connect(audioCtx.destination);

  proc.onaudioprocess = (e) => {
    if (!inCall || !micOn) return;
    const data = new Float32Array(e.inputBuffer.getChannelData(0));
    let sum = 0;
    for (const v of data) sum += v * v;
    const rms = Math.sqrt(sum / data.length);
    // 她说话时提高门槛，减少把她自己的声音当成你在说话
    const startTh = Math.max(noise * 3, 0.012) * (speaking ? 2.5 : 1);
    const needFrames = speaking ? Math.ceil(280 / frameMs) : 2;

    if (!inSpeech) {
      pre.push(data); if (pre.length > PRE_FRAMES) pre.shift();
      if (rms > startTh) {
        if (++loud >= needFrames) {
          inSpeech = true; quiet = 0; chunks = [...pre]; pre = [];
          if (speaking || playQueue.length) {
            stopSpeaking(); currentTurn = lastServerTurn + 1;
            ws?.send(JSON.stringify({ type: "interrupt" }));
          }
          setStatus("在听你说…"); clearIdle();
        }
      } else {
        loud = 0;
        noise = noise * 0.95 + rms * 0.05;
      }
    } else {
      chunks.push(data);
      quiet = rms < Math.max(noise * 2, 0.008) ? quiet + 1 : 0;
      if (quiet >= END_FRAMES || chunks.length * frameMs > 30000) {
        inSpeech = false; loud = 0;
        const total = chunks.length * frameMs;
        if (total > 450) sendUtterance(chunks, rate);
        chunks = [];
      }
    }
  };
}

function sendUtterance(chunks, rate) {
  const len = chunks.reduce((n, c) => n + c.length, 0);
  const all = new Float32Array(len);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }
  // 降采样到 16kHz
  const ratio = rate / 16000;
  const out = new Float32Array(Math.floor(len / ratio));
  for (let i = 0; i < out.length; i++) {
    const p = i * ratio, j = Math.floor(p), f = p - j;
    out[i] = all[j] * (1 - f) + (all[j + 1] ?? all[j]) * f;
  }
  setStatus("思考中…");
  ws?.send(out.buffer);
}

// ---------------- 摄像头小窗 ----------------
let camStream = null;
async function toggleCam() {
  const v = $("selfview");
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop()); camStream = null;
    v.style.display = "none"; $("btn-cam").classList.add("off");
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    v.srcObject = camStream; v.style.display = "block"; $("btn-cam").classList.remove("off");
  } catch { $("btn-cam").classList.add("off"); }
}

// ---------------- 沉默时她主动说话 ----------------
let idleTimer = null;
function clearIdle() { clearTimeout(idleTimer); }
function resetIdle() {
  clearIdle();
  if (!CFG.idleSeconds || !inCall) return;
  idleTimer = setTimeout(() => {
    if (!speaking) ws?.send(JSON.stringify({ type: "idle" }));
  }, CFG.idleSeconds * 1000);
}

// ---------------- 通话 ----------------
let inCall = false;
let callStart = 0;
let timerInt = null;
let herLine = "";

function setStatus(t) { $("status").textContent = t; }

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case "reply_start":
        lastServerTurn = msg.turn; currentTurn = Math.max(currentTurn, msg.turn);
        herLine = ""; setStatus("思考中…"); clearIdle();
        break;
      case "reply_delta":
        herLine += msg.text;
        break;
      case "audio":
        enqueueAudio(msg);
        break;
      case "say":  // 写实模式：声音和画面由 LiveTalking 播放，这里只显示字幕
        if (msg.turn >= currentTurn) { $("sub-her").textContent = msg.text; speaking = true; }
        break;
      case "reply_end":
        if (!speaking && !playQueue.length) { setStatus("正在听…"); resetIdle(); }
        break;
      case "user_text":
        $("sub-user").textContent = "你：" + msg.text;
        break;
      case "error":
        setStatus(msg.text);
        break;
    }
  };
  ws.onopen = () => ws.send(JSON.stringify({ type: "hello", avatarSession }));
  ws.onclose = () => { if (inCall) { setStatus("连接断开，重连中…"); setTimeout(connect, 1500); } };
}

async function startCall() {
  audioCtx = audioCtx || new AudioContext();
  await audioCtx.resume();
  if (!analyser) {
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.connect(audioCtx.destination);
  }
  inCall = true;
  document.body.classList.add("in-call");
  if (!micStream) {
    try { await initMic(); } catch { setStatus("没有麦克风权限，可以打字聊"); micOn = false; $("btn-mic").classList.add("off"); }
  }
  if (CFG.avatarMode === "realistic" && !pc) {
    setStatus("正在接通画面…");
    try { await connectRealistic(); }
    catch (e) { console.error(e); setStatus("画面没接通：" + e.message); }
  }
  connect();
  callStart = Date.now();
  timerInt = setInterval(() => {
    const s = Math.floor((Date.now() - callStart) / 1000);
    $("timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
}

function hangUp() {
  inCall = false;
  stopSpeaking(); clearIdle(); clearInterval(timerInt);
  ws?.close(); ws = null;
  clearInterval(speakPoll); pc?.close(); pc = null; avatarSession = null;
  $("herview").srcObject = null;
  $("sub-her").textContent = ""; $("sub-user").textContent = "";
  document.body.classList.remove("in-call");
  setStatus("已挂断");
}

$("btn-call").onclick = startCall;
$("btn-hang").onclick = hangUp;
$("btn-cam").onclick = toggleCam;
$("btn-mic").onclick = () => { micOn = !micOn; $("btn-mic").classList.toggle("off", !micOn); };
$("textform").onsubmit = (e) => {
  e.preventDefault();
  const t = $("textinput").value.trim();
  if (!t || !ws) return;
  stopSpeaking(); currentTurn = lastServerTurn + 1;
  ws.send(JSON.stringify({ type: "text", text: t }));
  $("textinput").value = "";
};

(async () => {
  CFG = await (await fetch("/config")).json();
  $("name").textContent = CFG.name;
  $("lobby-name").textContent = CFG.name;
  document.title = `和${CFG.name}视频通话`;
  if (!CFG.asr) $("lobby-tip").textContent = "语音识别没加载成功，暂时只能打字（看服务端窗口报错）";
  $("btn-cam").classList.add("off");
  if (CFG.avatarMode === "realistic") document.body.classList.add("realistic");
  else await initAvatar();
})();
