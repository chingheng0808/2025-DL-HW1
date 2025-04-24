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


class Trainer(object):
    def __init__(self, args):
        """
        Initialize the Trainer with arguments from the command line or defaults.
        """
        self.args = args

        self.deep_model = ResNet34(num_classes=50)
        self.deep_model = self.deep_model.to("cuda")

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.save_path, args.model_name, now_str)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.Adam(
            self.deep_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        if args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optim, args.epoch, eta_min=args.lr * 1e-4
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optim, step_size=args.decay_epoch, gamma=args.decay_rate
            )
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if args.q1 == "baseline":
            self.dataset = OrgDataset
        elif args.q1 == "anyInput":
            self.dataset = MyDataset

    def training(self):
        best_acc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_data_loader = DataLoader(
            self.dataset("images", mode="train", size=(256, 256)),
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self.deep_model.train()
        for epoch in range(1, self.args.epoch + 1):
            loop = tqdm(
                enumerate(train_data_loader), total=len(train_data_loader), leave=False
            )
            # self.deep_model.eval()
            # self.validate()
            loss_mean = 0.0
            for _, (x, label) in loop:
                x = x.to("cuda")
                label = label.to("cuda")

                pred = self.deep_model(x)
                self.optim.zero_grad()

                final_loss = self.cross_entropy_loss(pred, label)
                loss_mean += final_loss.item()

                final_loss.backward()

                self.optim.step()
                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                loop.set_postfix(loss=final_loss.item())

            print(
                f"[{epoch}/{self.args.epoch}], avg. loss is {loss_mean / len(train_data_loader)}, learning rate is {self.optim.param_groups[0]['lr']}"
            )
            if epoch % self.args.epoch_val == 0:
                self.deep_model.eval()
                acc, avg_loss = self.validate()

                if acc > best_acc:
                    torch.save(
                        self.deep_model.state_dict(),
                        os.path.join(self.model_save_path, f"best_model.pth"),
                    )
                    best_acc = acc
                    best_round = {
                        "best epoch": epoch,
                        "best Accuracy": best_acc,
                        "CE Loss": avg_loss,
                    }
                    with open(
                        os.path.join(self.model_save_path, "records.txt"), "a"
                    ) as f:
                        str_ = "## best round ##\n"
                        for k, v in best_round.items():
                            str_ += f"{k}: {v}. "
                        str_ += "\n####################################"
                        f.write(str_ + "\n")
                with open(os.path.join(self.model_save_path, "records.txt"), "a") as f:
                    str_ = f"[epoch: {epoch}], Acc: {acc}, Loss: {avg_loss}"
                    f.write(str_ + "\n")
                self.deep_model.train()
            self.scheduler.step()

        print("The accuracy of the best round is ", best_round)

    def validate(self):
        self.deep_model.eval()
        device = next(self.deep_model.parameters()).device

        val_loader = DataLoader(
            self.dataset("images", mode="val", size=(256, 256)),
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
        )
        torch.cuda.empty_cache()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            loop = tqdm(val_loader, desc="[Validation]", leave=False)
            for x, labels in loop:
                x = x.to(device)
                labels = labels.to(device)

                outputs = self.deep_model(x)
                loss = self.cross_entropy_loss(outputs, labels)

                total_loss += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples

        print(f"[Validation] Acc: {acc:.4f}, Loss: {avg_loss:.4f}")
        return acc, avg_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=20, help="epoch number")
    parser.add_argument("--epoch_val", type=int, default=1, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )  ##
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )  ##
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")

    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--model_name", type=str, default="org_resnet34")
    parser.add_argument("--save_path", type=str, default="./output/")

    parser.add_argument("--resume", type=str)

    parser.add_argument(
        "--q1",
        type=str,
        default="baseline",
        choices=["anyInput", "baseline"],
        help="anyInput or baseline",
    )

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    start = time.time()
    seed_everything(777)

    main()

    end = time.time()
    print("The total training time is:", end - start)
