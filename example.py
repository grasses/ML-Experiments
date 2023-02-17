import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import dloader
from models import mloader
from utils import helper, ops, metric


def main():
    args = helper.get_args()
    ops.set_default_seed(args.seed)

    # support model: resnet34, resnet18, resnet50, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, densenet121, densenet161, densenet169, mobilenet_v2, googlenet
    # support 32x32 dataset: CIFAR10, CNICN10, GTSRB, HAM10000, BCN20000, VGGFace, CelebA+20, CelebA+31
    # support 224x224 dataset: ImageNet

    dataset_id = "CIFAR10"
    arch_id = "vgg19_bn"

    train_loader, val_loader, test_loader, = dloader.get_dataloader(dataset_id=dataset_id, batch_size=128)
    model = mloader.get_model(dataset_id=dataset_id, arch_id=arch_id)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    model.to(args.device)
    for epoch in range(100):
        model.train()
        phar = tqdm(enumerate(train_loader))
        for step, (x, y) in phar:
            x = x.to(args.device)
            y = y.to(args.device)

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            phar.set_description(f"Run epoch: [{epoch}/100] step:[{step}/{len(train_loader)}]")
        metric.topk_test(model=model, test_loader=test_loader, device=args.device, debug=True, epoch=epoch)


if __name__ == "__main__":
    main()