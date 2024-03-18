import argparse
import logging
import os
import pickle

import torch
import torch.optim as optim

from dataloader.data_loader import fetch_dataloader
from model.net import Net
from model.metrics import metrics
from utils.engine import train_and_evaluate
from utils.misc import Params, seed_everything
from utils.log import set_logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="where has data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--model_dir",
        default="experiments/base_model",
        help="Directory containing params.json, and save model and files",
    )

    parser.add_argument(
        "--restore_file",
        default=None,
        help="The file name of state that store weight of model",
    )

    parser.add_argument(
        "-l", "--learning_rate", default=0.001, help="Learning rate of optimizer"
    )
    
    parser.add_argument(
        "--data_name",
        default=None,
        help='The data name of '
    )

    parser.add_argument("-b", "--batch_size", default=32, help="Batch size")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    json_path = os.path.join(args.model_dir, "params.json")

    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"

    params = Params(json_path)
    params.cuda = torch.cuda.is_available()
    params.learning_rate = args.learning_rate
    params.batch_size = args.batch_size

    seed_everything(42)
    set_logger(os.path.join(args.model_dir, "train.log"))

    logging.info("Loading the datasets...")
    dataloaders = fetch_dataloader(["train", "val"], args.data_dir, args.data_name, params)
    train_dl = dataloaders["train"]
    val_dl = dataloaders["val"]

    logging.info("- done. ")

    model = Net(params).cuda() if params.cuda else Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    criterion = # some loss function
    metrics = metrics
    
    logging.info(f"Starting training for {params.num_epochs} epochs(s)")
    train_and_evaluate(
        model = model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        params=params,
        model_dir=args.model_dir,
        restore_file=args.restore_file,
    )