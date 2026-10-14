#!/usr/bin/env python
# coding: utf-8
"""
idle_animate.py - make a natural, seamlessly looping IDLE video of a person from ONE photo
using LivePortrait (human model) on CPU.

The idle video is meant as the base video for LiveTalking (wav2lip / MuseTalk):
  * front camera static, upper body, mouth CLOSED the whole time
  * natural blinks (random, seeded, occasional double blink)
  * very subtle low-frequency head motion (yaw / pitch / roll)
  * subtle breathing (whole upper body + head rises/falls by ~1-2 px)
  * all motion curves are periodic over the clip -> loops seamlessly (forward or ping-pong)

How it works
  1. The face is cropped/aligned once (LivePortrait cropper), the appearance feature volume
     f_s and the source implicit keypoints x_s are computed once.
  2. For every *keyframe* the driving keypoints are built directly:
        x_d = s * (x_c @ R(pitch+dp, yaw+dy, roll+dr) + exp_s) + t
            + eye_delta(blink) - eye_delta(open)   (eye retargeting module)
            + lip_delta(closed)                     (only if the source mouth is open)
     then stitched, warped and decoded (warp_decode) -> 512x512 face crop.
     The expression (exp_s) is never changed, so the mouth never moves.
  3. Frames between keyframes (no blink there, head moves < 1 px/frame) are interpolated from
     the neighbouring rendered crops (cross-fade, or optical flow if the motion is large) - big CPU
     saving, ~50 dB PSNR vs direct render at --key-step 4. Every blink frame is rendered by the network.
     The decoder runs in OpenVINO (lp_fast.py, ~3x faster than torch on CPU, same output).
     Rendered keyframes are cached in <out>.cache/<hash>/ so a killed run resumes where it stopped.
  4. Each crop is pasted back into the full original frame (LivePortrait soft mask),
     a smooth breathing displacement is applied to the full frame, and frames are piped to
     ffmpeg (H.264, yuv420p, even dimensions).

Usage (run with the LivePortrait venv python; lp_fast.py must sit next to this file):
  python idle_animate.py --source face.png --out idle.mp4 --seconds 12 --fps 25
  optional: --crop x0,y0,x1,y1  --width 720  --seed N  --yaw-amp 2 --pitch-amp 1.5 --roll-amp 1
            --base-yaw -8 (re-pose head more frontal)  --breath-amp 1.5  --key-step 4  --check
"""

import argparse
import math
import os
import subprocess
import sys
import time

LP_DIR = os.environ.get("LIVEPORTRAIT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "LivePortrait"))
sys.path.insert(0, LP_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # lp_fast.py lives next to this script

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCHEDULE_VERSION = 2  # bump when motion/blink curve code changes (invalidates keyframe caches)


def parse_args():
    p = argparse.ArgumentParser(description="LivePortrait idle-video generator (CPU)")
    p.add_argument("--source", required=True, help="source portrait image (upper body, face visible)")
    p.add_argument("--out", required=True, help="output mp4 path")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    # optional pre-processing of the source
    p.add_argument("--crop", type=str, default=None, help="x0,y0,x1,y1 crop of the source (pixels) before animating")
    p.add_argument("--width", type=int, default=0, help="resize source to this width (keeps aspect); 0 = keep")
    # head motion (degrees)
    p.add_argument("--yaw-amp", type=float, default=2.0)
    p.add_argument("--pitch-amp", type=float, default=1.5)
    p.add_argument("--roll-amp", type=float, default=1.0)
    p.add_argument("--base-yaw", type=float, default=0.0, help="constant yaw offset (deg) to re-pose the head (e.g. more frontal)")
    p.add_argument("--base-pitch", type=float, default=0.0)
    p.add_argument("--base-roll", type=float, default=0.0)
    # blinks
    p.add_argument("--blink-min", type=float, default=2.5, help="min seconds between blinks")
    p.add_argument("--blink-max", type=float, default=5.0, help="max seconds between blinks")
    p.add_argument("--double-blink-prob", type=float, default=0.2)
    p.add_argument("--closed-eye-ratio", type=float, default=0.0, help="eye-open ratio used at full blink")
    p.add_argument("--no-blink", action="store_true")
    # breathing
    p.add_argument("--breath-amp", type=float, default=1.5, help="breathing rise in pixels per 1280 px of frame height (0=off)")
    p.add_argument("--breath-period", type=float, default=4.0, help="approx breathing period (s); snapped to divide the clip")
    p.add_argument("--breath-mask", type=str, default=None,
                   help="optional person mask (grayscale image, white = person, same framing as the source after --crop/--width). "
                        "If given, breathing moves only the person and the background stays perfectly static "
                        "(instead of the default broad gaussian around the face, which also moves nearby background)")
    p.add_argument("--breath-mask-grow", type=int, default=6, help="dilate the person mask by this many px before feathering")
    p.add_argument("--breath-mask-blur", type=float, default=10.0, help="gaussian sigma (px) used to feather the person mask")
    # mouth
    p.add_argument("--force-close-lip", action="store_true",
                   help="always apply lip-close retargeting (default: only if source mouth ratio >= 0.03)")
    # speed / quality
    p.add_argument("--key-step", type=int, default=4,
                   help="render every Nth frame with the network, interpolate the rest (1 = all). "
                        "Measured vs direct renders: step 3 ~53 dB, 4 ~50 dB, 6 ~45 dB PSNR")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 4))
    p.add_argument("--backend", choices=["ov", "torch"], default="ov",
                   help="ov = OpenVINO for the decoder (~2x faster per frame on CPU, same output); torch = pure LivePortrait")
    p.add_argument("--crf", type=int, default=16)
    p.add_argument("--preset", type=str, default="slow")
    p.add_argument("--crop-scale", type=float, default=2.3, help="LivePortrait face crop scale")
    p.add_argument("--vy-ratio", type=float, default=-0.125)
    p.add_argument("--no-stitch", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="where rendered keyframes are cached (resume after a crash/kill). default: <out>.cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--debug-dir", type=str, default=None, help="save rendered keyframe crops + curves here")
    p.add_argument("--check", action="store_true", help="measure eye/lip open ratios on rendered crops (QA)")
    return p.parse_args()


# ----------------------------------------------------------------------------------------------
# motion curves
# ----------------------------------------------------------------------------------------------
def periodic_noise(n_frames, fps, amp, rng, periods=(12.0, 6.0, 4.0, 2.4), weights=(1.0, 0.6, 0.35, 0.15)):
    """Smooth, non-robotic curve = sum of sines whose frequencies are integer multiples of 1/T,
    so curve[0] continues seamlessly after curve[n-1] (periodic over the clip)."""
    T = n_frames / fps
    t = np.arange(n_frames) / fps
    y = np.zeros(n_frames, dtype=np.float64)
    used = set()
    for P, w in zip(periods, weights):
        k = max(1, int(round(T / P)))
        while k in used:  # keep components distinct
            k += 1
        used.add(k)
        phase = rng.uniform(0, 2 * np.pi)
        ww = w * rng.uniform(0.7, 1.3)
        y += ww * np.sin(2 * np.pi * k * t / T + phase)
    m = np.abs(y).max()
    if m > 1e-9:
        y = y / m * amp
    return y


def blink_schedule(n_frames, fps, rng, bmin, bmax, double_prob):
    """Return per-frame blink closure b in [0,1] (1 = fully closed) and the list of blink frames."""
    b = np.zeros(n_frames, dtype=np.float64)
    margin = int(round(0.35 * fps))  # keep blinks away from the loop boundary
    t = rng.uniform(0.6, 1.6)  # first blink
    peaks = []
    while True:
        p = int(round(t * fps))
        if p > n_frames - 1 - margin:
            break
        if p >= margin:
            peaks.append(p)
            if rng.random() < double_prob:
                p2 = p + int(round(rng.uniform(0.28, 0.42) * fps))
                if p2 <= n_frames - 1 - margin:
                    peaks.append(p2)
                    t = p2 / fps
        t += rng.uniform(bmin, bmax)
    for p in peaks:
        # 25 fps: close 1-2 frames, (rarely) hold 1, open 2-3 frames -> 0.16-0.24 s visible blink
        close_n = 1 if rng.random() < 0.5 else 2
        hold = 1 if rng.random() < 0.15 else 0
        open_n = 2 if rng.random() < 0.4 else 3
        scale_f = fps / 25.0
        close_n = max(1, int(round(close_n * scale_f)))
        open_n = max(1, int(round(open_n * scale_f)))
        hold = int(round(hold * scale_f))
        # closing (fast)
        for k in range(1, close_n + 1):
            f = p - k
            u = (close_n + 1 - k) / (close_n + 1)
            if 0 <= f < n_frames:
                b[f] = max(b[f], math.sin(math.pi / 2 * u) ** 2)
        for h in range(hold + 1):
            if 0 <= p + h < n_frames:
                b[p + h] = 1.0
        # opening (slower)
        for k in range(1, open_n + 1):
            f = p + hold + k
            if 0 <= f < n_frames:
                b[f] = max(b[f], math.cos(math.pi / 2 * k / (open_n + 1)) ** 2)
    return b, peaks


# ----------------------------------------------------------------------------------------------
# optical-flow interpolation between two rendered crops
# ----------------------------------------------------------------------------------------------
_dis = None


def interp_crop(A, B, t, flow_thresh=3.0):
    """In-between of two rendered crops. Idle motion is sub-pixel per frame, so a plain cross-fade is the
    most accurate (measured: 50 dB PSNR vs a direct network render at key-step 4); only if the
    displacement between the keys is large (p99 > flow_thresh px) a flow-based interpolation is used."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ga = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    F = _dis.calc(ga, gb, None)  # A(x) ~ B(x + F(x))
    if np.percentile(np.linalg.norm(F, axis=2), 99) < flow_thresh:
        out = (1 - t) * A.astype(np.float32) + t * B.astype(np.float32)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Aw = cv2.remap(A, gx - t * F[..., 0], gy - t * F[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    Bw = cv2.remap(B, gx + (1 - t) * F[..., 0], gy + (1 - t) * F[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    out = (1 - t) * Aw.astype(np.float32) + t * Bw.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)

    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper
    from src.utils.camera import get_rotation_matrix
    from src.utils.crop import prepare_paste_back, paste_back
    from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio

    t_start = time.time()
    fps = args.fps
    n = int(round(args.seconds * fps))
    rng = np.random.default_rng(args.seed)

    # ---------------- source ----------------
    img = cv2.imread(args.source, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(args.source).convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if args.crop:
        x0, y0, x1, y1 = [int(v) for v in args.crop.split(",")]
        img = img[y0:y1, x0:x1]
    if args.width and args.width != img.shape[1]:
        h = int(round(img.shape[0] * args.width / img.shape[1]))
        img = cv2.resize(img, (args.width, h), interpolation=cv2.INTER_AREA if args.width < img.shape[1] else cv2.INTER_LANCZOS4)
    H, W = img.shape[:2]
    H2, W2 = H - (H % 2), W - (W % 2)
    img = np.ascontiguousarray(img[:H2, :W2])
    H, W = H2, W2
    print(f"[idle] source {W}x{H}, {n} frames @ {fps} fps", flush=True)

    inf_cfg = InferenceConfig(flag_force_cpu=True, flag_use_half_precision=False, flag_do_torch_compile=False)
    crop_cfg = CropConfig(flag_force_cpu=True, scale=args.crop_scale, vy_ratio=args.vy_ratio)
    wrapper = LivePortraitWrapper(inf_cfg)
    cropper = Cropper(crop_cfg=crop_cfg, flag_force_cpu=True)

    crop_info = cropper.crop_source_image(img, crop_cfg)
    if crop_info is None:
        raise SystemExit("No face detected in the source image")
    src_lmk = crop_info["lmk_crop"]
    I_s = wrapper.prepare_source(crop_info["img_crop_256x256"])
    x_s_info = wrapper.get_kp_info(I_s)
    f_s = wrapper.extract_feature_3d(I_s)
    x_s = wrapper.transform_keypoint(x_s_info)
    x_c_s = x_s_info["kp"]
    exp_s = x_s_info["exp"].clone()
    scale_s = x_s_info["scale"]
    t_s = x_s_info["t"]
    p0, y0_, r0 = x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]
    mask_ori = prepare_paste_back(inf_cfg.mask_crop, crop_info["M_c2o"], dsize=(W, H))
    M_c2o = crop_info["M_c2o"]

    c_s_eyes = calc_eye_close_ratio(src_lmk[None])  # 1x2
    c_s_lip = calc_lip_close_ratio(src_lmk[None])  # 1x1
    open_ratio = float(c_s_eyes.mean())
    print(f"[idle] source pose pitch={p0.item():.1f} yaw={y0_.item():.1f} roll={r0.item():.1f}  "
          f"eye ratio={c_s_eyes.round(3).tolist()} lip ratio={float(c_s_lip.ravel()[0]):.3f}", flush=True)

    # lip: close the mouth (constant over the whole clip) if the source mouth is open
    lip_delta = None
    comb_lip = wrapper.calc_combined_lip_ratio([0.0], src_lmk)
    if args.force_close_lip or float(c_s_lip.ravel()[0]) >= inf_cfg.lip_normalize_threshold:
        lip_delta = wrapper.retarget_lip(x_s, comb_lip)
        print("[idle] applying lip-close retargeting", flush=True)

    warp_decode = wrapper.warp_decode
    if args.backend == "ov":
        try:
            from lp_fast import FastWarpDecode
            t_ov = time.time()
            warp_decode = FastWarpDecode(wrapper, threads=args.threads)
            print(f"[idle] OpenVINO decoder ready ({time.time() - t_ov:.1f}s)", flush=True)
        except Exception as e:  # fall back to torch
            print(f"[idle] OpenVINO backend unavailable ({e!r}); using torch", flush=True)

    eye_cache = {}

    def eye_delta(ratio):
        key = round(float(ratio), 4)
        if key not in eye_cache:
            comb = wrapper.calc_combined_eye_ratio([[key]], src_lmk)
            eye_cache[key] = wrapper.retarget_eye(x_s, comb)
        return eye_cache[key]

    eye_open_delta = eye_delta(open_ratio)

    # ---------------- motion curves ----------------
    yaw = periodic_noise(n, fps, args.yaw_amp, rng)
    pitch = periodic_noise(n, fps, args.pitch_amp, rng)
    roll = periodic_noise(n, fps, args.roll_amp, rng)
    if args.no_blink:
        blink, peaks = np.zeros(n), []
    else:
        blink, peaks = blink_schedule(n, fps, rng, args.blink_min, args.blink_max, args.double_blink_prob)
    T = n / fps
    kb = max(1, int(round(T / args.breath_period)))
    breath = np.sin(2 * np.pi * kb * np.arange(n) / n - np.pi / 2)  # starts at exhale bottom
    print(f"[idle] blink peaks (s): {[round(p / fps, 2) for p in peaks]}", flush=True)

    # ---------------- keyframe selection ----------------
    step = max(1, args.key_step)
    keys = set(range(0, n, step))
    for i in range(n):
        if blink[i] > 0:
            keys.update([i - 1, i, i + 1])
    keys = sorted(k for k in keys if 0 <= k < n)

    @torch.no_grad()
    def render(i):
        R_d = get_rotation_matrix(p0 + args.base_pitch + float(pitch[i]),
                                  y0_ + args.base_yaw + float(yaw[i]),
                                  r0 + args.base_roll + float(roll[i]))
        x_d = scale_s * (x_c_s @ R_d + exp_s) + t_s  # expression kept = source (mouth stays as is)
        if blink[i] > 0:
            ratio = open_ratio * (1 - blink[i]) + args.closed_eye_ratio * blink[i]
            x_d = x_d + (eye_delta(ratio) - eye_open_delta)
        if lip_delta is not None:
            x_d = x_d + lip_delta
        if not args.no_stitch:
            x_d = wrapper.stitching(x_s, x_d)
        out = warp_decode(f_s, x_s, x_d)
        return wrapper.parse_output(out["out"])[0]  # 512x512x3 RGB uint8

    # ---------------- keyframe cache (resume after crash / OOM kill) ----------------
    cache = None
    if not args.no_cache:
        import hashlib
        hsh = hashlib.sha1()
        hsh.update(img.tobytes())
        hsh.update(str((SCHEDULE_VERSION, img.shape, n, fps, args.seed, args.yaw_amp, args.pitch_amp, args.roll_amp, args.base_yaw,
                        args.base_pitch, args.base_roll, 0.0, args.blink_min, args.blink_max,
                        args.double_blink_prob, args.closed_eye_ratio, args.no_blink, args.force_close_lip,
                        args.crop_scale, args.vy_ratio, args.no_stitch)).encode())
        cache = os.path.join(args.cache_dir or (os.path.abspath(args.out) + ".cache"), hsh.hexdigest()[:16])
        os.makedirs(cache, exist_ok=True)

    crops = {}
    t0 = time.time()
    n_rendered = 0
    for j, k in enumerate(keys):
        cpath = os.path.join(cache, f"key_{k:05d}.png") if cache else None
        if cpath and os.path.exists(cpath):
            im = cv2.imread(cpath, cv2.IMREAD_COLOR)
            if im is not None:
                crops[k] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                continue
        tk = time.time()
        crops[k] = render(k)
        n_rendered += 1
        if cpath:
            cv2.imwrite(cpath + ".tmp.png", cv2.cvtColor(crops[k], cv2.COLOR_RGB2BGR))
            os.replace(cpath + ".tmp.png", cpath)
        el = time.time() - t0
        eta = el / n_rendered * (len(keys) - j - 1)
        print(f"[idle] render key {j + 1}/{len(keys)} (frame {k}) {time.time() - tk:.2f}s  elapsed {el:.0f}s eta {eta:.0f}s", flush=True)
    render_time = time.time() - t0
    spf_render = render_time / max(1, n_rendered)
    if n_rendered < len(keys):
        print(f"[idle] {len(keys) - n_rendered} keyframes loaded from cache {cache}", flush=True)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        np.savez(os.path.join(args.debug_dir, "curves.npz"), yaw=yaw, pitch=pitch, roll=roll, blink=blink, breath=breath)

    if args.check:
        eyes, lips = [], []
        for k in keys:
            lm = cropper.calc_lmk_from_cropped_image(crops[k])
            if lm is None:
                continue
            eyes.append((k, float(calc_eye_close_ratio(lm[None]).mean())))
            lips.append((k, float(calc_lip_close_ratio(lm[None]).ravel()[0])))
        if lips:
            print(f"[check] lip-open ratio: max={max(v for _, v in lips):.4f} mean={np.mean([v for _, v in lips]):.4f} (source {float(c_s_lip.ravel()[0]):.4f})")
            e_open = [v for k, v in eyes if blink[k] == 0]
            e_closed = [v for k, v in eyes if blink[k] >= 0.99]
            print(f"[check] eye ratio open frames: mean={np.mean(e_open):.3f}  full-blink frames: "
                  f"{np.round(e_closed, 3).tolist() if e_closed else 'n/a'} (source {open_ratio:.3f})")

    # ---------------- compose + encode ----------------
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # breathing weight: 0 at the bottom edge, ramps to 1 at ~55% height (chest) and stays 1 above;
    # horizontally a broad gaussian centred on the face so far background barely moves
    face_cx = float(M_c2o[0, 2] + M_c2o[0, 0] * 256 + M_c2o[0, 1] * 256)
    yr = np.clip((H - gy) / (0.45 * H), 0, 1)
    wy = yr * yr * (3 - 2 * yr)
    if args.breath_mask:
        pm = cv2.imread(args.breath_mask, cv2.IMREAD_GRAYSCALE)
        if pm is None:
            raise SystemExit(f"cannot read --breath-mask {args.breath_mask}")
        pm =cv2.resize(pm, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        if args.breath_mask_grow > 0:
            k = 2 * args.breath_mask_grow + 1
            pm = cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if args.breath_mask_blur > 0:
            pm = cv2.GaussianBlur(pm, (0, 0), args.breath_mask_blur)
        wx = np.clip(pm, 0, 1)
        print(f"[idle] breathing restricted to person mask {args.breath_mask}", flush=True)
    else:
        wx = np.exp(-0.5 * ((gx - face_cx) / (0.45 * W)) ** 2)
    breath_w = (wy * wx).astype(np.float32)
    breath_px = args.breath_amp * H / 1280.0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    keys_ext = keys + [n]  # frame n == frame 0 (periodic)
    t1 = time.time()
    for i in range(n):
        if i in crops:
            crop = crops[i]
        else:
            a = max(k for k in keys if k < i)
            b = min(k for k in keys_ext if k > i)
            B = crops[b % n] if b < n else crops[0]
            crop = interp_crop(crops[a], B, (i - a) / (b - a))
        if args.debug_dir and i in crops:
            cv2.imwrite(os.path.join(args.debug_dir, f"crop_{i:04d}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        frame = paste_back(crop, M_c2o, img, mask_ori)
        if breath_px > 0:
            d = float(breath[i]) * breath_px
            frame = cv2.remap(frame, gx, gy + d * breath_w, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        ff.stdin.write(np.ascontiguousarray(frame).tobytes())
    ff.stdin.close()
    rc = ff.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed with code {rc}")
    total = time.time() - t_start
    print(f"[idle] done -> {args.out}", flush=True)
    print(f"[idle] keyframes rendered: {len(keys)}/{n}  network s/keyframe: {spf_render:.2f}  "
          f"compose+encode: {time.time() - t1:.1f}s  total: {total:.1f}s  effective s/output-frame: {total / n:.2f}", flush=True)


if __name__ == "__main__":
    main()
