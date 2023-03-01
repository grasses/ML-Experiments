#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/07/29, homeway'


import types
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification


def feature_list(self, **kwargs):
    """
    Return feature map of each layer
    Args:
        self: Densenet
        x: Tensor
    Returns: Tensor, list
    """
    out_list = []
    def get_attn(name):
        def hook(model, input, output):
            out_list.append(output[name])
        return hook
    h = self.transformer.register_forward_hook(get_attn(name="last_hidden_state"))
    y = self(**kwargs).logits
    h.remove()
    return y, out_list


def gpt2_sst2():
    tokenizer = GPT2Tokenizer.from_pretrained("michelecafagna26/gpt2-medium-finetuned-sst2-sentiment")
    model = GPT2ForSequenceClassification.from_pretrained("michelecafagna26/gpt2-medium-finetuned-sst2-sentiment")
    model.feature_list = types.MethodType(feature_list, model)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = gpt2_sst2()
    inputs = tokenizer("I love it", return_tensors="pt")
    y, out_list = model.feature_list(**inputs)











