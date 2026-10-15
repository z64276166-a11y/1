"""下载 gen_identity.py 需要的模型到本目录的 models/（约 6 GB，只需一次）。"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")
jobs = [
    ("DIAMONIK7777/antelopev2", "glintr100.onnx", "antelopev2"),
    ("DIAMONIK7777/antelopev2", "scrfd_10g_bnkps.onnx", "antelopev2"),
    ("public-data/insightface", "models/buffalo_l/w600k_r50.onnx", "buffalo_l"),
    ("InstantX/InstantID", "ControlNetModel/config.json", "instantid"),
    ("InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors", "instantid"),
    ("InstantX/InstantID", "ip-adapter.bin", "instantid"),
]
for repo, fn, sub in jobs:
    print("下载", repo, fn, flush=True)
    hf_hub_download(repo, fn, local_dir=os.path.join(M, sub))
print("下载 RealVisXL V5.0 Lightning（SDXL 写实底模）…", flush=True)
snapshot_download("SG161222/RealVisXL_V5.0_Lightning",
                  allow_patterns=["*.json", "*.txt", "tokenizer*/*", "*fp16*.safetensors"])
print("完成")
