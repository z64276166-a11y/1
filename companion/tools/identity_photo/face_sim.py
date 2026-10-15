#!/usr/bin/env python
"""Face-identity similarity between images (cosine of ArcFace embeddings, onnxruntime only).

usage: face_sim.py REF.png CAND1.png [CAND2.png ...] [--model glintr100|w600k_r50|both] [--json]

Prints, per candidate: cosine similarity to the reference's largest face, face box size relative to frame,
and a rough head-pose indicator (yaw/pitch/roll from the 5 keypoints).
Rule of thumb for ArcFace cosine: >0.6 very likely same person, 0.4-0.6 resemblance / same-ish,
<0.3 different person. (Stylised CG vs photo usually caps out around 0.5-0.7.)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from face_lib import FaceAnalyzer, load_bgr, head_pose_from_kps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--model", default="both", choices=["glintr100", "w600k_r50", "both"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    models = ["glintr100", "w600k_r50"] if a.model == "both" else [a.model]
    res = []
    analyzers = {m: FaceAnalyzer(rec=m) for m in models}
    ref_img = load_bgr(a.ref)
    ref = {m: analyzers[m].get(ref_img) for m in models}
    if not ref[models[0]]:
        sys.exit("no face in reference " + a.ref)
    for c in a.cands:
        img = load_bgr(c)
        H, W = img.shape[:2]
        row = {"image": c}
        for m in models:
            fs = analyzers[m].get(img)
            if not fs:
                row[m] = None
                continue
            row[m] = round(float(np.dot(ref[m][0]["normed_embedding"], fs[0]["normed_embedding"])), 4)
            if m == models[0]:
                x1, y1, x2, y2 = fs[0]["bbox"]
                yaw, pitch, roll = head_pose_from_kps(fs[0]["kps"])
                row.update(face_w_frac=round(float(x2 - x1) / W, 3), face_cx=round(float(x1 + x2) / 2 / W, 3),
                           face_cy=round(float(y1 + y2) / 2 / H, 3), yaw=round(yaw, 1), pitch=round(pitch, 1),
                           roll=round(roll, 1), n_faces=len(fs))
        res.append(row)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res:
            sims = "  ".join(f"{m}={r.get(m)}" for m in models)
            extra = (f"  face_w={r['face_w_frac']:.3f} center=({r['face_cx']:.2f},{r['face_cy']:.2f}) "
                     f"yaw~{r['yaw']} pitch~{r['pitch']} roll~{r['roll']}") if "face_w_frac" in r else "  (no face)"
            print(f"{os.path.basename(r['image'])}: {sims}{extra}")


if __name__ == "__main__":
    main()
