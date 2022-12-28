import os
from copy import deepcopy
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from loftr_tools.src.loftr import LoFTR, default_cfg

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
def match_single_pair(img0_raw, img1_raw, matcher_state_dict):
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(matcher_state_dict)
    #matcher.load_state_dict(torch.load("./weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    default_cfg['coarse']
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    return mkpts0, mkpts1, mconf
