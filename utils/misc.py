import json
import os
import shutil
import random
import matplotlib.pyplot as plt

import torch
import numpy as np


class Params:
    """Class store params for hyper-parameters"""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class AverageMeter:
    """
    Return total and average of some number
    """

    def __init__(self):
        self.steps = 0
        self.value = 0

    def update(self, val):
        self.value += val
        self.steps += 1

    def __call__(self):
        return self.value / float(self.steps)


def save_dict_to_json(d, json_path):
    """
    save dictionary to `json_path`
    """
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Save model, optimizer, and other information to a checkpoint
    Args:
        - state: dictionary, may contain model, optimizer, epochs, ...
        - is_best: bool, whether current model is the best
        - checkpoint: string, where to save file
    """
    filepath = os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint Directory does not exist! Making directory {checkpoint}")
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Load model, optimizer from checkpoint
    Args:
        - checkpoint: the state file
        - model: model to load weight with
        - optimizer: whether to load parameter for optimizer
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    # return


def seed_everything(seed):
    np.random.seed(42)
    random.seed(42)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


def save_learning_curve(train_summ, model_dir):
    """
    train_summ is like:
    {
        "train": {
            1: {"accuracy": , "some metric":  ,"loss":  }
        }
    }
    """
    epochs = list(train_summ["train"].keys())
    metrics = train_summ["train"][1].keys()

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        train_metric = [train_summ["train"][epoch][metric] for epoch in epochs]
        val_metric = [train_summ["valid"][epoch][metric] for epoch in epochs]
        plt.plot(epochs, train_metric, label=f"Train {metric.capitalize()}")
        plt.plot(epochs, val_metric, label=f"Validation {metric.capitalize()}")
        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(f"{metric.capitalize()}")
        plt.legend(loc="best")
        plt.savefig(os.path.join(model_dir, f"{metric}_learning_curve.png"))
