import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from resnet34 import ResNet34
from dataset import MyDataset, OrgDataset


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Tester(object):
    def __init__(self, args):
        """
        Initialize the Tester with arguments from the command line or defaults.
        """
        self.args = args

        self.deep_model = ResNet34(num_classes=50)
        self.deep_model = self.deep_model.to("cuda")

        if not os.path.isfile(args.ckpt):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        model_dict = {}
        state_dict = self.deep_model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.deep_model.load_state_dict(state_dict)
        if args.q1 == "baseline":
            self.dataset = OrgDataset
        elif args.q1 == "anyInput":
            self.dataset = MyDataset

    def test(self):
        self.deep_model.eval()
        device = next(self.deep_model.parameters()).device

        val_loader = DataLoader(
            self.dataset("images", mode="test", size=(256, 256)),
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
        torch.cuda.empty_cache()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            loop = tqdm(val_loader, desc="[Testing]", leave=False)
            for x, labels in loop:
                x = x.to(device)
                labels = labels.to(device)

                outputs = self.deep_model(x)

                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        acc = total_correct / total_samples

        print(f"[Testing] Acc: {acc:.4f}")
        return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default="output/org_resnet34/20250419_154159/best_model.pth",
    )
    parser.add_argument(
        "--q1",
        type=str,
        default="baseline",
        choices=["anyInput", "baseline"],
        help="anyInput or baseline",
    )

    args = parser.parse_args()

    tester = Tester(args)
    tester.test()


if __name__ == "__main__":
    start = time.time()
    seed_everything(777)

    main()

    end = time.time()
    print("The total testing time is:", end - start)
