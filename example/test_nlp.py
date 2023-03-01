#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2023/02/28, homeway'

import torch
from torch.nn import functional as F
from datasets.nlp import SST2
from models.nlp import gpt2
from utils import helper
args = helper.get_args()


def main():
    # step1: Load dataset
    sst2_dst = SST2(split="train")
    data_loader = torch.utils.data.DataLoader(sst2_dst, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    # step2: Load pretrained model
    model, tokenizer  = gpt2.gpt2_sst2()

    model.train()
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for idx, (x, y) in enumerate(data_loader):
        inputs = tokenizer(list(x), return_tensors="pt", padding=True)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)

        optimizer.zero_grad()
        y = y.to(args.device)
        pred = model(**inputs).logits
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        accuracy = 100.0 * pred.argmax(dim=1).view_as(y).eq(y).sum() / len(y)
        print(f"-> step:{idx} loss:{loss.detach().cpu().item()} acc:{accuracy}%")


if __name__ == "__main__":
    main()













