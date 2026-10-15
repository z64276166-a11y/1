#!/usr/bin/env python
"""Identity-locked photoreal generator: RealVisXL V5.0 Lightning + InstantID (IdentityNet ControlNet + face IP-Adapter).

Creates a NEW photo of the person in --face (text-to-image, not img2img), with a frontal 5-point face layout so the
head is straight-on, eye-level and large in a 16:9 frame. CPU-only, bf16 UNet/ControlNet (AMX), fp32 VAE.

Run with:
  python gen_identity.py --out out.png [options]

Identity embedding: antelopev2 glintr100 ArcFace via onnxruntime (face_lib.py), exactly the embedding InstantID was
trained on (raw, un-normalised). Several --face images are averaged.
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 模型路径都可以用环境变量覆盖，见同目录 README.md

import numpy as np
import torch
from PIL import Image

SRC_IMG = os.environ.get("REF_FACE", os.path.join(HERE, "..", "..", "avatar", "original.webp"))
INSTANTID = os.environ.get("INSTANTID_DIR", os.path.join(HERE, "models", "instantid"))
_hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = os.environ.get("SDXL_BASE") or (glob.glob(os.path.join(
    _hf, "hub", "models--SG161222--RealVisXL_V5.0_Lightning", "snapshots", "*")) or [""])[0]

DEFAULT_PROMPT = (
    "webcam photo during a video call, a 23 year old Chinese woman sitting at home in the evening, "
    "facing the camera straight on, looking into the lens, head level, calm relaxed face, lips gently closed, "
    "long black hair pulled back into a neat low ponytail, forehead and both ears visible, "
    "natural skin with visible pores, bare face without makeup, plain light grey crew-neck sweatshirt, "
    "small bedroom behind her, white wall, warm table lamp light, bookshelf and bed in the background, "
    "slight grain, soft focus, amateur snapshot"
)
DEFAULT_NEG = (
    "cgi, 3d render, illustration, painting, anime, doll, wax figure, plastic skin, airbrushed, retouched, "
    "beauty filter, heavy makeup, glossy lips, red lipstick, blush, rosy cheeks, red nose, eyeliner, "
    "cinematic lighting, dramatic rim light, studio portrait, professional photography, fashion editorial, bokeh, "
    "hanfu, costume, jewelry, earrings, forehead mark, head tilt, looking away, side view, open mouth, teeth, "
    "hands, hair over face, loose strands on face, bangs, messy flyaway hair, wind blown hair, wet hair, "
    "laptop, computer screen, "
    "monitor, watermark, text, logo, user interface, deformed, asymmetrical eyes, cross-eyed"
)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--face", nargs="+", default=[SRC_IMG],
                    help="reference image(s); largest face is used, embeddings averaged")
    ap.add_argument("--width", type=int, default=1280, help="final output width")
    ap.add_argument("--height", type=int, default=720, help="final output height")
    ap.add_argument("--gen-size", default="auto",
                    help="WxH actually sampled (multiple of 64). auto: ~1MP SDXL bucket with the output aspect "
                         "(1280x720 -> 1344x768, then centre-crop to 16:9 and resize)")
    ap.add_argument("--seed", type=int, nargs="+", default=[7], help="one or more seeds (one model load); "
                    "with several seeds outputs are <out>_s<seed>.png")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.5, help="<=1 disables CFG (2x faster, negative ignored)")
    ap.add_argument("--identity-strength", type=float, default=0.8, help="IP-Adapter (face embedding) scale")
    ap.add_argument("--kps-strength", type=float, default=0.8,
                    help="IdentityNet ControlNet scale (pose/layout + identity)")
    ap.add_argument("--kps-end", type=float, default=1.0, help="fraction of steps the ControlNet stays on")
    ap.add_argument("--sampler", default="dpmpp_sde_karras", choices=["dpmpp_sde_karras", "dpmpp_2m_karras", "euler"])
    # face layout (fractions of the FINAL frame): eye midpoint position and inter-eye distance
    ap.add_argument("--face-cx", type=float, default=0.5)
    ap.add_argument("--face-eye-y", type=float, default=0.41)
    ap.add_argument("--face-ipd", type=float, default=0.108,
                    help="inter-eye distance / frame width (0.108 -> face box ~23-24%% of width)")
    ap.add_argument("--face-roll", type=float, default=0.0)
    ap.add_argument("--kps-from", default=None, help="take the 5-point layout from this image instead (its face)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def pick_gen_size(a):
    if a.gen_size != "auto":
        w, h = (int(v) for v in a.gen_size.lower().split("x"))
        return w, h
    ar = a.width / a.height
    if a.width * a.height <= 640 * 640:  # small test sizes: sample directly
        return (a.width // 64) * 64 or 64, (a.height // 64) * 64 or 64
    best = None
    for h in range(512, 1601, 64):
        for w in range(512, 2049, 64):
            if abs(w * h - 1024 * 1024) > 0.12 * 1024 * 1024:
                continue
            score = abs(w / h - ar)
            if best is None or score < best[0]:
                best = (score, w, h)
    return best[1], best[2]


def to_final(img, W, H):
    """centre-crop to the target aspect, then resize."""
    w, h = img.size
    ar = W / H
    if abs(w / h - ar) > 1e-3:
        if w / h > ar:
            nw = round(h * ar); x0 = (w - nw) // 2; img = img.crop((x0, 0, x0 + nw, h))
        else:
            nh = round(w / ar); y0 = (h - nh) // 2; img = img.crop((0, y0, w, y0 + nh))
    return img.resize((W, H), Image.LANCZOS) if img.size != (W, H) else img


def trim():
    """return freed heap to the OS (glibc keeps fp32 text-encoder pages otherwise)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@torch.no_grad()
def encode_text(prompts):
    """SDXL dual-CLIP encoding with >77-token support (75-token chunks concatenated, A1111-style)."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    tks = [CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer"),
           CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer_2")]
    tes = [CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32),
           CLIPTextModelWithProjection.from_pretrained(BASE, subfolder="text_encoder_2", variant="fp16",
                                                       torch_dtype=torch.float32)]
    ids_all = [[tk(p, add_special_tokens=False, truncation=False).input_ids for p in prompts] for tk in tks]
    n_chunks = max(1, max((len(i) + 74) // 75 for i in ids_all[0] + ids_all[1]))
    out = []
    for pi in range(len(prompts)):
        hs, pooled = [], None
        for k in range(2):
            tk, te, ids = tks[k], tes[k], ids_all[k][pi]
            chunks = []
            for c in range(n_chunks):
                piece = ids[c * 75:(c + 1) * 75]
                row = [tk.bos_token_id] + piece + [tk.eos_token_id]
                row += [tk.pad_token_id] * (77 - len(row))
                chunks.append(row)
            o = te(torch.tensor(chunks), output_hidden_states=True)
            hs.append(o.hidden_states[-2].reshape(1, -1, o.hidden_states[-2].shape[-1]))
            if k == 1:
                pooled = o.text_embeds[:1]
        out.append((torch.cat(hs, -1), pooled))
    del tes, o
    gc.collect()
    trim()
    return out


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    from face_lib import FaceAnalyzer, load_bgr, frontal_kps, draw_kps, head_pose_from_kps

    T0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(a.out)[0]

    # ---- 1. identity embedding ----
    fa = FaceAnalyzer(rec="glintr100")
    embs = []
    for p in a.face:
        fs = fa.get(load_bgr(p))
        if not fs:
            sys.exit(f"no face found in {p}")
        embs.append(fs[0]["embedding"])
        print(f"face ref {os.path.basename(p)}: det={fs[0]['det_score']:.2f} |emb|={np.linalg.norm(fs[0]['embedding']):.1f}",
              flush=True)
    if len(embs) == 1:
        face_emb = embs[0]
    else:  # average directions, keep typical raw norm (InstantID expects un-normalised embeddings)
        nrm = np.mean([np.linalg.norm(e) for e in embs])
        d = np.mean([e / np.linalg.norm(e) for e in embs], 0)
        face_emb = d / np.linalg.norm(d) * nrm

    # ---- 2. layout (keypoint image) ----
    GW, GH = pick_gen_size(a)
    if a.kps_from:
        img = load_bgr(a.kps_from)
        fs = fa.get(img, with_embedding=False)
        kps = fs[0]["kps"] * np.array([GW / img.shape[1], GH / img.shape[0]])
    else:
        # layout is specified in final-frame fractions; map through the centre crop used by to_final()
        ar = a.width / a.height
        cw, ch = (GH * ar, GH) if GW / GH > ar else (GW, GW / ar)
        x0, y0 = (GW - cw) / 2, (GH - ch) / 2
        kps = frontal_kps(cw, ch, a.face_cx, a.face_eye_y, a.face_ipd, a.face_roll) + np.array([x0, y0])
    kps_img = draw_kps(GW, GH, kps)
    Image.fromarray(kps_img).save(stem + ".kps.png")
    print(f"gen size {GW}x{GH} -> final {a.width}x{a.height}", flush=True)

    # ---- 3. text ----
    use_cfg = a.cfg > 1.0
    t = time.time()
    enc = encode_text([a.prompt] + ([a.negative] if use_cfg and a.negative else []))
    pe, ppe = enc[0]
    if use_cfg:
        if a.negative:
            npe, nppe = enc[1]
        else:
            npe, nppe = torch.zeros_like(pe), torch.zeros_like(ppe)
        # equalise chunk counts already done in encode_text (shared n_chunks)
        pe = torch.cat([npe, pe]); ppe = torch.cat([nppe, ppe])
    print(f"text encoded in {time.time() - t:.1f}s  tokens={pe.shape[1]}", flush=True)

    # ---- 4. models ----
    from diffusers import (UNet2DConditionModel, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler,
                           EulerDiscreteScheduler)
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, IPAdapterAttnProcessor
    dt = torch.bfloat16
    t = time.time()
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", variant="fp16", torch_dtype=dt).eval()
    sd = torch.load(os.path.join(INSTANTID, "ip-adapter.bin"), map_location="cpu", weights_only=True)
    unet._load_ip_adapter_weights([sd], low_cpu_mem_usage=True)
    del sd
    for p in unet.attn_processors.values():
        if isinstance(p, (IPAdapterAttnProcessor2_0, IPAdapterAttnProcessor)):
            p.scale = [a.identity_strength]
    unet.to(dt)
    cn = ControlNetModel.from_pretrained(os.path.join(INSTANTID, "ControlNetModel"), torch_dtype=dt).eval()
    gc.collect()
    print(f"unet+ipadapter+controlnet loaded in {time.time() - t:.1f}s", flush=True)

    scfg = DPMSolverMultistepScheduler.load_config(BASE, subfolder="scheduler")
    if a.sampler == "euler":
        sched = EulerDiscreteScheduler.from_config(scfg, timestep_spacing="trailing")
    else:
        sched = DPMSolverMultistepScheduler.from_config(
            scfg, algorithm_type="sde-dpmsolver++" if a.sampler == "dpmpp_sde_karras" else "dpmsolver++",
            use_karras_sigmas=True, final_sigmas_type="zero")
    sched.set_timesteps(a.steps)

    B = 2 if use_cfg else 1
    fe = torch.tensor(face_emb, dtype=torch.float32).reshape(1, 1, 512)
    fe = torch.cat([torch.zeros_like(fe), fe]) if use_cfg else fe  # InstantID: zero embedding = uncond
    with torch.no_grad():
        id_tokens = unet.encoder_hid_proj.image_projection_layers[0](fe.to(dt))  # (B,16,2048) for IdentityNet
    unet_img_embeds = [fe.to(dt).unsqueeze(1)]  # (B, num_images=1, seq=1, 512)
    time_ids = torch.tensor([[GH, GW, 0, 0, GH, GW]] * B, dtype=dt)
    added = {"text_embeds": ppe.to(dt), "time_ids": time_ids}
    cond = torch.from_numpy(kps_img).permute(2, 0, 1).float().div(255).unsqueeze(0).repeat(B, 1, 1, 1).to(dt)
    text = pe.to(dt)

    lats, sec_steps = [], []
    for seed in a.seed:
        sched.set_timesteps(a.steps)  # resets multistep solver state
        ts = sched.timesteps
        g = torch.Generator("cpu").manual_seed(seed)
        lat = torch.randn((1, 4, GH // 8, GW // 8), generator=g, dtype=torch.float32) * sched.init_noise_sigma
        t_loop = time.time()
        for i, tt in enumerate(ts):
            ts0 = time.time()
            li = torch.cat([lat] * B)
            li = sched.scale_model_input(li, tt).to(dt)
            keep = not ((i + 1) / len(ts) > a.kps_end)
            with torch.no_grad():
                down = mid = None
                if keep and a.kps_strength > 0:
                    down, mid = cn(li, tt, encoder_hidden_states=id_tokens, controlnet_cond=cond,
                                   conditioning_scale=a.kps_strength, added_cond_kwargs=added, return_dict=False)
                noise = unet(li, tt, encoder_hidden_states=text,
                             added_cond_kwargs={**added, "image_embeds": unet_img_embeds},
                             down_block_additional_residuals=down, mid_block_additional_residual=mid,
                             return_dict=False)[0].float()
            if use_cfg:
                u, c = noise.chunk(2)
                noise = u + a.cfg * (c - u)
            if a.sampler == "euler":
                lat = sched.step(noise, tt, lat, return_dict=False)[0]
            else:
                lat = sched.step(noise, tt, lat, generator=g, return_dict=False)[0]
            print(f"seed {seed} step {i + 1}/{len(ts)} t={int(tt)} {time.time() - ts0:.1f}s "
                  f"cn={'on' if keep else 'off'}", flush=True)
        lats.append(lat)
        sec_steps.append(time.time() - t_loop)
    del unet, cn, id_tokens, text
    gc.collect()
    trim()

    # ---- 5. decode (fp32 VAE, tiled) + 6. identity / framing check ----
    import cv2
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", variant="fp16", torch_dtype=torch.float32).eval()
    vae.enable_tiling()
    anchor = fa.get(load_bgr(SRC_IMG))[0]["normed_embedding"]
    ref_n = face_emb / np.linalg.norm(face_emb)
    for seed, lat, ss in zip(a.seed, lats, sec_steps):
        t = time.time()
        with torch.no_grad():
            x = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
        x = (x / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        img = Image.fromarray((x * 255).round().astype(np.uint8))
        print(f"vae decode {time.time() - t:.1f}s", flush=True)
        out = a.out if len(a.seed) == 1 else f"{stem}_s{seed}.png"
        ostem = os.path.splitext(out)[0]
        img.save(ostem + ".raw.png")
        final = to_final(img, a.width, a.height)
        final.save(out)
        bgr = cv2.cvtColor(np.asarray(final), cv2.COLOR_RGB2BGR)
        fs = fa.get(bgr)
        rep = {"out": out, "seed": seed, "gen_size": [GW, GH], "seconds_total": round(time.time() - T0, 1),
               "seconds_sampling": round(ss, 1), "args": vars(a)}
        if fs:
            f = fs[0]
            x1, y1, x2, y2 = f["bbox"]
            yaw, pitch, roll = head_pose_from_kps(f["kps"])
            rep.update(sim_vs_original=round(float(anchor @ f["normed_embedding"]), 4),
                       sim_vs_reference_embedding=round(float(ref_n @ f["normed_embedding"]), 4),
                       face_w_frac=round(float(x2 - x1) / a.width, 3), yaw=round(yaw, 1), pitch=round(pitch, 1),
                       roll=round(roll, 1))
        else:
            rep["warning"] = "no face detected in output"
        with open(ostem + ".json", "w") as fh:
            json.dump(rep, fh, indent=1, default=str)
        print("RESULT " + json.dumps({k: v for k, v in rep.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
