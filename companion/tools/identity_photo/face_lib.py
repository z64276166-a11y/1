"""Minimal insightface-compatible face toolkit built on onnxruntime only (no insightface pip).

- SCRFD-10G (antelopev2 scrfd_10g_bnkps.onnx == buffalo_l det_10g.onnx): face boxes + 5 keypoints
- ArcFace recognition (antelopev2 glintr100.onnx, or buffalo_l w600k_r50.onnx): 512-d embedding
- InstantID helpers: draw_kps(), frontal keypoint layout

Logic ported from insightface/python-package (model_zoo/scrfd.py, arcface_onnx.py, utils/face_align.py).
"""
import os
import math
import numpy as np
import cv2
import onnxruntime as ort

MODELS = os.environ.get("FACE_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
DET_PATH = os.path.join(MODELS, "antelopev2", "scrfd_10g_bnkps.onnx")
REC_PATHS = {
    "glintr100": os.path.join(MODELS, "antelopev2", "glintr100.onnx"),  # what InstantID uses
    "w600k_r50": os.path.join(MODELS, "buffalo_l", "models", "buffalo_l", "w600k_r50.onnx"),
}

# canonical ArcFace 112x112 five-point template (frontal face)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _sess(path, threads=None):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class SCRFD:
    def __init__(self, path=DET_PATH, det_thresh=0.5, nms_thresh=0.4):
        self.s = _sess(path)
        self.inp = self.s.get_inputs()[0].name
        self.outs = [o.name for o in self.s.get_outputs()]
        assert len(self.outs) == 9, "expected scrfd *_bnkps model with keypoints"
        self.strides = [8, 16, 32]
        self.na = 2
        self.det_thresh, self.nms_thresh = det_thresh, nms_thresh
        self.cache = {}

    def _forward(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, (w, h), (127.5, 127.5, 127.5), swapRB=True)
        net = self.s.run(self.outs, {self.inp: blob})
        scores_l, boxes_l, kps_l = [], [], []
        batched = net[0].ndim == 3
        for i, st in enumerate(self.strides):
            sc = net[i][0] if batched else net[i]
            bb = (net[i + 3][0] if batched else net[i + 3]) * st
            kp = (net[i + 6][0] if batched else net[i + 6]) * st
            fh, fw = h // st, w // st
            key = (fh, fw, st)
            if key not in self.cache:
                c = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
                c = (c * st).reshape(-1, 2)
                c = np.stack([c] * self.na, axis=1).reshape(-1, 2)
                self.cache[key] = c
            c = self.cache[key]
            pos = np.where(sc >= self.det_thresh)[0]
            boxes = np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], -1)
            kps = np.stack([c[:, i % 2] + kp[:, i] for i in range(10)], -1).reshape(-1, 5, 2)
            scores_l.append(sc[pos]); boxes_l.append(boxes[pos]); kps_l.append(kps[pos])
        return np.vstack(scores_l), np.vstack(boxes_l), np.vstack(kps_l)

    def _nms(self, dets):
        x1, y1, x2, y2, s = dets.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.nms_thresh)[0] + 1]
        return keep

    def detect(self, img_bgr, size=640):
        h, w = img_bgr.shape[:2]
        if h / w > 1.0:
            nh, nw = size, int(size * w / h)
        else:
            nw, nh = size, int(size * h / w)
        scale = nh / h
        det_img = np.zeros((size, size, 3), np.uint8)
        det_img[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
        sc, bx, kp = self._forward(det_img)
        if len(sc) == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        bx = bx / scale; kp = kp / scale
        dets = np.hstack([bx, sc]).astype(np.float32)
        order = sc.ravel().argsort()[::-1]
        dets, kp = dets[order], kp[order]
        keep = self._nms(dets)
        return dets[keep], kp[keep]


def _umeyama(src, dst):
    """Similarity transform (scale+rot+trans) src->dst, like skimage SimilarityTransform.estimate."""
    num, dim = src.shape
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    A = dd.T @ sd / num
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[-1] = -1
    U, S, Vt = np.linalg.svd(A)
    T = np.eye(dim + 1)
    T[:dim, :dim] = U @ np.diag(d) @ Vt
    scale = 1.0 / sd.var(0).sum() * (S @ d)
    T[:dim, dim] = dm - scale * (T[:dim, :dim] @ sm)
    T[:dim, :dim] *= scale
    return T[:2]


def norm_crop(img_bgr, kps, size=112):
    M = _umeyama(np.asarray(kps, np.float64), ARCFACE_DST.astype(np.float64) * (size / 112.0))
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)


class ArcFace:
    def __init__(self, name="glintr100"):
        self.name = name
        self.s = _sess(REC_PATHS[name])
        self.inp = self.s.get_inputs()[0].name
        # both glintr100 and w600k_r50 are non-mxnet exports -> mean=std=127.5, RGB
        self.mean, self.std = 127.5, 127.5

    def embed_aligned(self, crop_bgr):
        blob = cv2.dnn.blobFromImage(crop_bgr, 1.0 / self.std, (112, 112), (self.mean,) * 3, swapRB=True)
        return self.s.run(None, {self.inp: blob})[0][0].astype(np.float32)


class FaceAnalyzer:
    """get(img) -> list of dicts sorted by face area (largest first):
    bbox, kps, det_score, embedding (raw, as InstantID uses), normed_embedding."""

    def __init__(self, rec="glintr100", det_size=640):
        self.det = SCRFD()
        self.rec = ArcFace(rec) if rec else None
        self.det_size = det_size

    def get(self, img_bgr, with_embedding=True):
        dets, kpss = self.det.detect(img_bgr, self.det_size)
        faces = []
        for d, k in zip(dets, kpss):
            f = {"bbox": d[:4], "det_score": float(d[4]), "kps": k}
            if with_embedding and self.rec is not None:
                e = self.rec.embed_aligned(norm_crop(img_bgr, k))
                f["embedding"] = e
                f["normed_embedding"] = e / (np.linalg.norm(e) + 1e-8)
            faces.append(f)
        faces.sort(key=lambda f: -(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        return faces


def load_bgr(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def head_pose_from_kps(kps):
    """Rough yaw/pitch/roll (deg) from 5 points. yaw from nose offset vs eye/mouth midline, roll from eye line.
    Only an indicator (0 = frontal); not a calibrated estimator."""
    le, re, n, lm, rm = [np.asarray(p, np.float64) for p in kps]
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    eye_c, mouth_c = (le + re) / 2, (lm + rm) / 2
    ipd = np.linalg.norm(re - le)
    mid = (eye_c + mouth_c) / 2
    # horizontal nose offset relative to face midline, normalised by inter-eye distance
    axis = mouth_c - eye_c
    axis /= np.linalg.norm(axis) + 1e-8
    perp = np.array([axis[1], -axis[0]])
    off = np.dot(n - mid, perp) / ipd
    # template: nose sits 0.0 off-midline; ~0.5 ipd offset ~= 30-35 deg yaw
    yaw = math.degrees(math.asin(max(-1, min(1, off * 1.1))))
    # pitch: nose position along the eye->mouth axis (template ratio ~0.49)
    t = np.dot(n - eye_c, axis) / (np.linalg.norm(mouth_c - eye_c) + 1e-8)
    pitch = (t - 0.49) * 120.0
    return yaw, pitch, roll


# ---------------- InstantID helpers ----------------

def frontal_kps(W, H, cx=0.5, eye_y=0.40, ipd=0.105, roll=0.0):
    """Five keypoints for a perfectly frontal face (ArcFace template geometry),
    eye midpoint at (cx*W, eye_y*H), inter-eye distance ipd*W pixels."""
    t = ARCFACE_DST.astype(np.float64).copy()
    eye_mid = (t[0] + t[1]) / 2
    s = ipd * W / np.linalg.norm(t[1] - t[0])
    t = (t - eye_mid) * s
    if roll:
        a = math.radians(roll)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        t = t @ R.T
    t += np.array([cx * W, eye_y * H])
    return t


def draw_kps(W, H, kps, colors=((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255))):
    """Exactly InstantID's draw_kps (RGB uint8 HxWx3)."""
    stick = 4
    limbs = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.asarray(kps)
    out = np.zeros([H, W, 3])
    for a, b in limbs:
        color = colors[a]
        x = kps[[a, b], 0]; y = kps[[a, b], 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        poly = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stick), int(angle), 0, 360, 1)
        out = cv2.fillConvexPoly(out.copy(), poly, color)
    out = (out * 0.6).astype(np.uint8)
    for i, (x, y) in enumerate(kps):
        out = cv2.circle(out.copy(), (int(x), int(y)), 10, colors[i], -1)
    return out.astype(np.uint8)
