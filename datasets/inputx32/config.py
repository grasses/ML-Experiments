#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/22, homeway'

"""Basic configure of Dataset"""

import torch
import argparse


def load_cfg(dataset_id):
    parser = argparse.ArgumentParser(description="default model config")
    parser.add_argument("-device", action="store", default=1, type=int, help="GPU device id")
    cfg, unknown = parser.parse_known_args()
    cfg.seed = 100
    cfg.input_size = 32
    cfg.resize_size = 32
    cfg.input_shape = (3, 32, 32)
    cfg.mean = (0.4914, 0.4822, 0.4465)
    cfg.std = (0.2471, 0.2435, 0.2616)
    cfg.lr = 0.01
    cfg.momentum = 0.9
    cfg.weight_decay = 5e-4
    cfg.batch_size = 128
    cfg.subsize = 1000

    if dataset_id == "CIFAR10":
        cfg.lr = 0.01
        cfg.subsize = 5000
        cfg.TRAIN_ITERS = 80000
    elif dataset_id == "CINIC10":
        cfg.subsize = 5000
        cfg.TRAIN_ITERS = 20000
    elif dataset_id == "LFW":
        cfg.subsize = 1000
        cfg.TRAIN_ITERS = 20000
    elif dataset_id == "VGGFace2":
        cfg.subsize = 5000
        cfg.TRAIN_ITERS = 20000
    elif "CelebA" in dataset_id:
        cfg.lr = 8e-3
        cfg.subsize = 16277
        cfg.TRAIN_ITERS = 20000
    elif dataset_id == "HAM10000":
        cfg.subsize = 3754
        cfg.TRAIN_ITERS = 20000
    elif dataset_id == "GTSRB":
        cfg.subsize = 2664
        cfg.TRAIN_ITERS = 50000
    cfg.device = torch.device(f"cuda:{cfg.device}") if torch.cuda.is_available() else torch.device("cpu")
    return cfg






