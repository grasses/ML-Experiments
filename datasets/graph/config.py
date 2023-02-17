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
    cfg.lr = 0.01
    cfg.momentum = 0.9
    cfg.weight_decay = 5e-4
    cfg.lr_decay_steps = [25, 35]
    cfg.batch_size = 128
    cfg.subsize = 1000
    cfg.data_verbose = 0
    cfg.log_every = 1
    cfg.eval_every = 5
    cfg.train_verbose = 1
    cfg.dropout = 0.5
    cfg.train_epochs = 50

    cfg.train_ratio = 0.5
    cfg.val_ratio = 0.2
    cfg.use_nlabel_asfeat = 0
    cfg.use_org_node_attr = 0
    cfg.use_degree_asfeat = 0

    if dataset_id == "AIDS":
        cfg.hidden_dim = [64, 32, 16]
        cfg.num_head = 2
        cfg.use_org_node_attr = 1
    elif "Tox21" in dataset_id:
        cfg.lr = 0.001
        cfg.hidden_dim = [128, 64, 32, 16]
        cfg.use_degree_asfeat = 1
        cfg.use_nlabel_asfeat = 1

    elif dataset_id == "Yeast":
        cfg.hidden_dim = [128, 64, 32, 16]
        cfg.use_degree_asfeat = 1
        cfg.use_nlabel_asfeat = 1
    elif dataset_id == "COLLAB":
        cfg.hidden_dim = [128, 64, 32, 16]
        cfg.use_degree_asfeat = 1
        cfg.use_nlabel_asfeat = 0

    cfg.device = torch.device(f"cuda:{cfg.device}") if torch.cuda.is_available() else torch.device("cpu")
    return cfg






