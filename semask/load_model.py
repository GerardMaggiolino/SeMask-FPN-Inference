import argparse
import os
import gdown

import torch
import time

from .mmseg.models import build_segmentor
from mmcv import Config


__MODELS = {
    "ade20k_tiny": {
        "config_path": "configs/semask_swin/ade20k/semfpn_semask_swin_tiny_patch4_window7_512x512_80k_ade20k.py",
        "gdrive_id": "1L0daUHWQGNGCXHF-cKWEauPSyBV0GLOR",
    },
    "ade20k_small": {
        "config_path": "configs/semask_swin/ade20k/semfpn_semask_swin_small_patch4_window7_512x512_80k_ade20k.py",
        "gdrive_id": "1QhDG4SyGFtWL5kP9BbBoyPqTuFu7fH_y",
    },
    "ade20k_base": {
        "config_path": "configs/semask_swin/ade20k/semfpn_semask_swin_base_patch4_window12_512x512_80k_ade20k.py",
        "gdrive_id": "1PXCEhrrUy5TJC4dUp7YDQvaapnMzGT6C",
    },
    "ade20k_large": {
        "config_path": "configs/semask_swin/ade20k/semfpn_semask_swin_large_patch4_window12_640x640_80k_ade20k.py",
        "gdrive_id": "1u5flfAQCiQJbMZbZPIlGUGTYBz9Ca7rE",
    },
    "cityscapes_tiny": {
        "config_path": "configs/semask_swin/cityscapes/semfpn_semask_swin_tiny_patch4_window7_768x768_80k_cityscapes.py",
        "gdrive_id": "1_JBOJQSUVes-CWs075XyPnuNfG5psELr",
    },
    "cityscapes_small": {
        "config_path": "configs/semask_swin/cityscapes/semfpn_semask_swin_small_patch4_window7_768x768_80k_cityscapes.py",
        "gdrive_id": "1WyT207dZmdwETBUR6aeiqOVfQdUIV_fN",
    },
    "cityscapes_base": {
        "config_path": "configs/semask_swin/cityscapes/semfpn_semask_swin_base_patch4_window12_768x768_80k_cityscapes.py",
        "gdrive_id": "1-LzVB6XzD7IR0zzE5qmE0EM4ZTv429b4",
    },
    "cityscapes_large": {
        "config_path": "configs/semask_swin/cityscapes/semfpn_semask_swin_large_patch4_window12_768x768_80k_cityscapes.py",
        "gdrive_id": "1R9DDCmucQ_a_6ZkMGufEZCzJ-_qVMqCB",
    },
    "coco10k_tiny": {
        "config_path": "configs/semask_swin/coco_stuff10k/semfpn_semask_swin_tiny_patch4_window7_512x512_80k_coco10k.py",
        "gdrive_id": "1qhXsJ8H64JPI_DW7CNzhxpHSEG2sKaIl",
    },
    "coco10k_small": {
        "config_path": "configs/semask_swin/coco_stuff10k/semfpn_semask_swin_small_patch4_window7_512x512_80k_coco10k.py",
        "gdrive_id": "1ddXSMQu5ClkbLNMyQdyT0ATaOr86vIkL",
    },
    "coco10k_base": {
        "config_path": "configs/semask_swin/coco_stuff10k/semfpn_semask_swin_base_patch4_window12_512x512_80k_coco10k.py",
        "gdrive_id": "1pGWI7U9bZJoe4ZaDx7ktWELx-uVN7rL0",
    },
    "coco10k_large": {
        "config_path": "configs/semask_swin/coco_stuff1/semfpn_semask_swin_large_patch4_window12_640x640_80k_coco10k.py",
        "gdrive_id": "1F6B9x9pX-SYEth7hdtxeNUeQ3XncOH7G",
    },
}


def download_with_gdown(gid, download_path):
    gdown.download("https://drive.google.com/uc?id={}".format(gid), download_path)


def load_model(model_name):
    if model_name not in __MODELS:
        raise TypeError(
            "{} not in list of available models.\nSupported list: {}".format(
                model_name, list(__MODELS.keys())
            )
        )

    weight_path = os.path.join(
        os.path.dirname(__file__), "weights/{}".format(model_name + ".pth")
    )
    if not os.path.exists(weight_path):
        download_with_gdown(__MODELS[model_name]["gdrive_id"], weight_path)

    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        classes = ckpt["meta"]["CLASSES"]
        palette = ckpt["meta"]["PALETTE"]
    except:
        raise RuntimeError(
            "Failed to load checkpoint at {}.\n".format(weight_path),
            "Try manually downloading the checkpoint and placing it at the above path"
        )

    config_path = os.path.join(
        os.path.dirname(__file__), __MODELS[model_name]["config_path"]
    )
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)
    for k in model.state_dict():
        if k not in state_dict:
            raise RuntimeWarning("Loading weights, missing {} weight".format(k))
    model.load_state_dict(state_dict, strict=False)
    model.classes = classes
    model.palette = torch.tensor(palette)

    """
    out = model.inference(
        x, [{"ori_shape": [512, 512], "flip": False, "scale_factor": 1}], False
    )
    """

    return model




