#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/09/23, homeway'


import torch
import os.path as osp
import logging
from datasets import DLoader
logger = logging.getLogger("ModelLoader")
MODEL_ROOT = osp.join(osp.abspath(osp.dirname(__file__)), "ckpt")


class MLoader(object):
    def __init__(self, dataset_id, arch_id):
        self.dataset_id = dataset_id
        self.arch_id = arch_id
        self.dloader = DLoader(dataset_id)
        self.dtype = self.dloader.get_dtype(dataset_id)
        self.cfg = self.dloader.cfg
        self.task = f"train({dataset_id},{arch_id})-"
        self.model_root = osp.join(MODEL_ROOT, self.task)

    def __call__(self, *args, **kwargs):
        if self.dtype == "image":
            return self.get_image_model(*args, **kwargs)
        if self.dtype == "nlp":
            return self.get_nlp_model(*args, **kwargs)
        if self.dtype == "graph":
            return self.get_graph_model(*args, **kwargs)

    def get_image_model(self, pretrained=False, *args, **kwargs):
        num_classes = self.dloader.get_num_classess(self.dataset_id)
        if self.dataset_id in self.dloader.task_dataset["inputx32"]:
            from models.inputx32 import vgg19_bn, vgg13_bn, vgg11_bn, vgg16_bn
            from models.inputx32 import resnet18, resnet34, resnet50, densenet121, densenet169, mobilenet_v2, googlenet
        elif self.dataset_id in self.dloader.task_dataset["inputx224"]:
            from models.inputx224 import vgg19_bn, vgg13_bn, vgg11_bn, vgg16_bn
            from models.inputx224 import resnet50, alexnet, densenet121, densenet169, mobilenet_v2
        else:
            raise NotImplementedError()
        self.model = eval(f'{self.arch_id}')(
            pretrained=pretrained,
            num_classes=num_classes,
            *args,
            **kwargs
        )
        self.model.task = self.task
        return self.model

    def get_graph_model(self, *args, **kwargs):
        from models.graph import GCN, GAT, GraphSAGE
        data_loader = self.dloader(split="train")
        in_dim = data_loader.num_features
        out_dim = data_loader.num_classes
        if self.arch_id == 'gcn':
            self.model = GCN(in_dim, out_dim, hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout)
        elif self.arch_id == 'gat':
            self.model = GAT(in_dim, out_dim, hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout, num_head=self.cfg.num_head)
        elif self.arch_id == 'sage':
            self.model = GraphSAGE(in_dim, out_dim, hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout)
        else:
            raise NotImplementedError(self.arch_id)
        self.model.task = self.task
        return self.model

    def get_nlp_model(self, *args, **kwargs):
        pass

    def load_weights(self, model, seed, device=None):
        path = osp.join(MODEL_ROOT, self.task, f"final_s{seed}.pth")
        if not osp.exists(path):
            raise FileNotFoundError(f"-> ckpt:{path} not found!")
        cache = torch.load(path, map_location="cpu")
        model.load_state_dict(cache["state_dict"])
        if device is not None:
            model.to(device)
        return model


def get_model(dataset_id, arch_id, *args, **kwargs):
    return MLoader(dataset_id, arch_id)(*args, **kwargs)


if __name__ == "__main__":
    print(get_model(dataset_id="CIFAR10", arch_id="resnet50"))













