# coding: utf-8
"""
Faster CPU warp_decode for LivePortrait: OpenVINO runs the 2D decoder (warping third/fourth convs +
SPADE generator, ~3x faster than torch eager on CPU); optionally also the dense-motion hourglass
(no gain measured on this CPU, so off by default),
torch runs the cheap keypoint / 5D grid_sample parts (OpenVINO has no 5D GridSample).
Numerically equivalent to LivePortraitWrapper.warp_decode (fp32).
"""
import os
import torch
import torch.nn.functional as F
import numpy as np


class _DMCore(torch.nn.Module):
    """hourglass -> mask(softmax) + occlusion map"""

    def __init__(self, dm):
        super().__init__()
        self.hourglass = dm.hourglass
        self.mask = dm.mask
        self.occlusion = dm.occlusion

    def forward(self, inp):
        prediction = self.hourglass(inp)
        mask = F.softmax(self.mask(prediction), dim=1)
        bs, _, d, h, w = prediction.shape
        occ = torch.sigmoid(self.occlusion(prediction.reshape(bs, -1, h, w)))
        return mask, occ


class _Decoder(torch.nn.Module):
    """third -> fourth -> * occlusion -> SPADE generator"""

    def __init__(self, wm, gen):
        super().__init__()
        self.third = wm.third
        self.fourth = wm.fourth
        self.gen = gen

    def forward(self, warped, occ):
        out = self.fourth(self.third(warped))
        out = out * occ
        return self.gen(feature=out)


class FastWarpDecode:
    def __init__(self, wrapper, cache_dir=None, threads=4, ov_dense_motion=False):
        import openvino as ov
        self.w = wrapper
        wm = wrapper.warping_module
        self.wm = wm
        self.dm = wm.dense_motion_network
        assert self.dm is not None and self.dm.occlusion is not None and wm.flag_use_occlusion_map
        cache_dir = cache_dir or os.path.join(os.path.dirname(wrapper.inference_cfg.checkpoint_W), "..", "ov_fp32")
        os.makedirs(cache_dir, exist_ok=True)
        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": int(threads)}
        num_kp = self.dm.num_kp
        comp = self.dm.compress.out_channels
        dm_xml = os.path.join(cache_dir, "dm_core.xml")
        dec_xml = os.path.join(cache_dir, "decoder.xml")
        with torch.no_grad():
            if ov_dense_motion and not os.path.exists(dm_xml):
                ex = torch.randn(1, (num_kp + 1) * (comp + 1), 16, 64, 64)
                ov.save_model(ov.convert_model(_DMCore(self.dm).eval(), example_input=ex), dm_xml, compress_to_fp16=False)
            if not os.path.exists(dec_xml):
                ex = (torch.randn(1, 512, 64, 64), torch.rand(1, 1, 64, 64))
                ov.save_model(ov.convert_model(_Decoder(wm, wrapper.spade_generator).eval(), example_input=ex), dec_xml,
                              compress_to_fp16=False)
        # OpenVINO is not faster than torch/oneDNN for the 3D hourglass on this CPU -> torch by default
        self.dm_core = core.compile_model(dm_xml, "CPU", cfg) if ov_dense_motion else None
        self.dm_core_torch = _DMCore(self.dm).eval()
        self.decoder = core.compile_model(dec_xml, "CPU", cfg)
        self._feat_c = None
        self._feat_key = None

    @torch.no_grad()
    def __call__(self, feature_3d, kp_source, kp_driving):
        dm = self.dm
        # compressed feature only depends on the source -> cache it
        if self._feat_key is not feature_3d:
            self._feat_c = F.relu(dm.norm(dm.compress(feature_3d)))
            self._feat_key = feature_3d
        feature = self._feat_c
        bs, _, d, h, w = feature.shape
        sparse_motion = dm.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = dm.create_deformed_feature(feature, sparse_motion)
        heatmap = dm.create_heatmap_representations(deformed_feature, kp_driving, kp_source)
        inp = torch.cat([heatmap, deformed_feature], dim=2).view(bs, -1, d, h, w)
        if self.dm_core is not None:
            res = self.dm_core(inp.numpy())
            mask = torch.from_numpy(res[0])
            occ = res[1]
        else:
            mask, occ = self.dm_core_torch(inp)
            occ = occ.numpy()
        mask = mask.unsqueeze(2)
        sm = sparse_motion.permute(0, 1, 5, 2, 3, 4)
        deformation = (sm * mask).sum(dim=1).permute(0, 2, 3, 4, 1)
        out = F.grid_sample(feature_3d, deformation, align_corners=False)
        b, c, dd, hh, ww = out.shape
        out = out.view(b, c * dd, hh, ww)
        img = self.decoder((out.numpy(), occ))[0]
        return {"out": torch.from_numpy(img)}
