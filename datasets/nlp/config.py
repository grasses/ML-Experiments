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
    return cfg