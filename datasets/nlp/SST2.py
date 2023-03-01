import os.path
import pickle
import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import DataLoader
from torchtext.datasets import SST2 as SSTDataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

batch_size = 16


"""tokenizer for GPT2-medium"""
tokenizer = GPT2Tokenizer.from_pretrained("michelecafagna26/gpt2-medium-finetuned-sst2-sentiment")
def gpt2_medium_transform(batch):
    x = [d[0] for d in batch]
    y = [d[1] for d in batch]
    return tokenizer(x, return_tensors="pt", padding=True), y


class SST2:
    def __init__(self,
        root: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        shots: int = -1,
        seed: int = 0,
        preload: bool = False):

        if split == "train":
            datapipe = SSTDataset(split="train")
        elif split == "test":
            datapipe = SSTDataset(split="test")
        else:
            datapipe = SSTDataset(split="dev")
        self.num_classes = 2
        self.transform = gpt2_medium_transform if transform is None else transform
        self.datapipe = list(datapipe)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.datapipe[index][0]
        y = self.datapipe[index][1]
        return x, y

    def __len__(self) -> int:
        return len(list(self.datapipe))



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    sst2_dst = SST2(root="", split="train")
    data_loader = DataLoader(sst2_dst, batch_size=3)

    for idx, (x, y) in enumerate(data_loader):
        print(idx, x, y)
        print()
        if idx == 1:
            break