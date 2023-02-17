#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'

import os
import os.path as osp
import numpy as np
import torch
import torchvision
import logging
from torchvision import transforms
from utils import ops
from . import inputx32, inputx224
from datasets.inputx32 import CIFAR10, CINIC10, CelebA, LFW, VGGFace2, BCN20000, HAM10000, GTSRB
from datasets.inputx224 import ImageNet
from datasets.graph import loader as graphloader

DATA_ROOT = osp.join(osp.abspath(osp.dirname(__file__)), "data")
logger = logging.getLogger('DataLoader')


task_dataset = {
    "inputx32": ["CIFAR10", "CINIC10", "CelebA", "LFW", "VGGFace2", "BCN20000", "HAM10000", "GTSRB"],
    "inputx24": ["ImageNet"],
    "graph": ["Citation", "AIDS", "Tox21_AR", "COLLAB"],
    "nlp": ["ImageNet"],
    "audio": ["SpeechCommands"],
}
for i in range(40):
    task_dataset["inputx32"].append(f"CelebA+{i}")


class DLoader(object):
    def __init__(self, dataset_id, batch_size=128):
        self.data_root = os.path.join(DATA_ROOT, dataset_id)
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"-> datapath: {self.data_root}")

        self.task_dataset = task_dataset
        self.batch_size = batch_size
        self.dataset_id = dataset_id
        self.dtype = self.get_dtype(dataset_id)
        self.cfg = self.get_config(dataset_id)
        self.cfg.data_root = os.path.join(DATA_ROOT, dataset_id)
        ops.set_default_seed(self.cfg.seed)

    def get_config(self, dataset_id):
        dtype = self.get_dtype(dataset_id)
        if dtype == "image":
            cfg = eval(f"inputx{self.get_size(dataset_id)}.load_cfg(dataset_id=dataset_id)")
        elif dtype == "graph":
            from datasets.graph import config
            cfg = eval(f"config.load_cfg(dataset_id=dataset_id)")
        elif dtype == "nlp":
            cfg = eval(f"nlp.load_cfg(dataset_id=dataset_id)")
        else:
            raise NotImplementedError(f"-> dtype:{dtype} not implemented!!")
        return cfg

    def __call__(self, *args, **kwargs):
        if self.dtype == "image":
            return self.get_image_loader(*args, **kwargs)
        elif self.dtype == "nlp":
            return self.get_nlp_loader(*args, **kwargs)
        elif self.dtype == "graph":
            return self.get_graph_loader(*args, **kwargs)
        else:
            raise NotImplementedError(f"-> dtype:{self.dtype} not implemented!!")

    def get_graph_loader(self, split, shuffle=False):
        from datasets.graph import loader as graph
        graph_dr = graph.DataReader(self.cfg, self.dataset_id)
        if split == 'train':
            gids = graph_dr.data['splits']['train']
        elif split == 'val':
            gids = graph_dr.data['splits']['val']
        else:
            gids = graph_dr.data['splits']['test']
        graph_data = graph.GraphData(graph_dr, gids)
        data_loader = graph.DataLoader(graph_data, batch_size=self.batch_size, shuffle=shuffle, collate_fn=graph.collate_batch)
        data_loader.dataset_id = self.dataset_id
        data_loader.num_classes = graph_data.num_classes
        data_loader.num_features = graph_data.num_features
        data_loader.n_node_max = graph_data.n_node_max
        data_loader.dtype = "graph"
        return data_loader

    def get_nlp_loader(self, split, shuffle=False):
        pass

    def get_image_loader(self, split, shuffle=False):
        shots = -1
        dataset_id = self.dataset_id
        if "CelebA" in self.dataset_id:
            dataset_id, shots = self.dataset_id.split("+")

        cfg = self.get_config(dataset_id=self.dataset_id)
        normalize = torchvision.transforms.Normalize(mean=cfg.mean, std=cfg.std)
        ops.set_default_seed(cfg.seed)

        if split == 'train' or split == 'val':
            split = 'train'
            dataset = eval(dataset_id)(
                self.data_root, split, transform=transforms.Compose([
                    transforms.Resize(cfg.resize_size),
                    transforms.CenterCrop(cfg.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                shots=int(shots), seed=cfg.seed, preload=False
            )
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            num_classes = dataset.num_classes
            if split == "train":
                idxs = indices[cfg.subsize:]
            else:
                idxs = indices[:cfg.subsize]
            dataset = torch.utils.data.Subset(dataset, idxs)
            dataset.num_classes = num_classes
            print(f"-> split dataset into train:{len(indices[cfg.subsize:])} and val:{len(indices[:cfg.subsize])}")
        elif split == 'test':
            dataset = eval(dataset_id)(
                self.data_root, split, transform=transforms.Compose([
                    transforms.Resize(cfg.resize_size),
                    transforms.CenterCrop(cfg.input_size),
                    transforms.ToTensor(),
                    normalize,
                ]),
                shots=int(shots), seed=cfg.seed, preload=False
            )
        else:
            raise NotImplementedError()

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=4, pin_memory=False
        )
        data_loader.dtype = "image"
        data_loader.dataset_id = self.dataset_id
        data_loader.mean = cfg.mean
        data_loader.std = cfg.std
        data_loader.input_size = cfg.input_size
        data_loader.num_classes = dataset.num_classes
        data_loader.bounds = self.get_bounds(cfg.mean, cfg.std)
        data_loader.unnormalize = self.unnormalize
        logger.info(
            f'-> get_dataloader success: {self.dataset_id}_{split}, iter_size:{len(data_loader)} '
            f'batch_size:{self.batch_size} num_classes:{data_loader.num_classes}')
        return data_loader

    @staticmethod
    def get_num_classess(dataset_id):
        NUM_CLASSES = {
            "CIFAR10": 10,
            "CINIC10": 10,
            "CelebA": 2,
            "LFW": 2,
            "VGGFace2": 2,
            "BCN20000": 7,
            "HAM10000": 7,
            "GTSRB": 43,
            "ImageNet": 1000,
        }
        for i in range(40):
            NUM_CLASSES[f"CelebA+{i}"] = 2
            NUM_CLASSES[f"CelebA32+{i}"] = 2
        return NUM_CLASSES[dataset_id]

    @staticmethod
    def get_dtype(dataset_id):
        for k, v in task_dataset.items():
            if dataset_id in v:
                if "input" in k: return "image"
                return k

    @staticmethod
    def unnormalize(tensor, mean, std, clamp=False):
        tmp = tensor.clone()
        for t, m, s in zip(tmp, mean, std):
            (t.mul_(s).add_(m))
        if clamp:
            tmp = torch.clamp(tmp, min=0.0, max=1.0)
        return tmp.double()

    @staticmethod
    def get_bounds(mean, std):
        bounds = [-1, 1]
        if type(mean) == type(list([])):
            c = len(mean)
            _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
            _max = (np.ones([c]) - np.array(mean)) / np.array([std])
            bounds = [np.min(_min).item(), np.max(_max).item()]
        elif type(mean) == float:
            bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
        return bounds

    @staticmethod
    def get_size(dataset_id):
        INPUT_SIZE = {
            "CIFAR10": 32,
            "CINIC10": 32,
            "CelebA": 32,
            "LFW": 32,
            "VGGFace2": 32,
            "BCN20000": 32,
            "HAM10000": 32,
            "GTSRB": 32,
            "ImageNet": 224,
        }
        for i in range(40):
            INPUT_SIZE[f"CelebA+{i}"] = 32
        return INPUT_SIZE[dataset_id]

    def get_seed_samples(self, dataset_id, batch_size, rand=False, shuffle=True, with_label=False, unormalize=False):
        """
        Return only $batch_size samples from train set
        :param dataset_id:
        :param batch_size:
        :param rand:
        :param shuffle:
        :param with_label:
        :param unormalize:
        :return:
        """
        datapath = os.path.join(DATA_ROOT, dataset_id)
        assert os.path.exists(datapath)

        cfg = self.get_config(dataset_id=dataset_id)
        if rand:
            batch_input_size = (batch_size, *cfg.input_shape)
            images = np.random.normal(size=batch_input_size).astype(np.float32)
        else:
            train_loader = self(dataset_id=dataset_id, split='train', batch_size=batch_size, shuffle=shuffle)
            images, labels = next(iter(train_loader))
            unnormalize_images = train_loader.unnormalize(images).to('cpu').numpy()
            images = images.to('cpu').numpy()
            labels = labels.to('cpu').numpy()
            bounds = train_loader.bounds
            if not with_label:
                if unormalize:
                    return images, unnormalize_images
                return images
            else:
                if unormalize:
                    return images, unnormalize_images, bounds, labels
                return images, labels
        logger.info(f"-> get_seed_samples from:{dataset_id} batch_size:{batch_size}")
        return images


def get_dataloader(dataset_id, batch_size):
    train_loader = DLoader(dataset_id, batch_size)(split="train", shuffle=True)
    val_loader = DLoader(dataset_id, batch_size)(split="val", shuffle=True)
    test_loader = DLoader(dataset_id, batch_size)(split="test", shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    a, b, c = get_dataloader("AIDS", batch_size=10)
    print(len(a), len(b), len(c))











