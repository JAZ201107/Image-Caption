import os
from tqdm import tqdm
import logging
from pickle import dump
import time

import torch
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from utils.misc import (
    AverageMeter,
    load_checkpoint,
    save_checkpoint,
    save_dict_to_json,
    save_learning_curve,
)
from model.metrics import accuracy
import config


def clip_gradient():
    raise NotImplementedError


def train_one_epoch(
    encoder,
    decoder,
    dataloader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    metrics,
    params,
    grad_clip=None,
    lr_scheduler=None,
):
    encoder.train()
    decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_avg = AverageMeter()
    top5acc = AverageMeter()
    start = time.time()

    summ = []

    with tqdm(total=len(dataloader)) as t:
        for i, (imgs, caps, caplens) in enumerate(dataloader):
            if params.cuda:
                train_batch, train_labels, caplens = (
                    train_batch.cuda(non_blocking=True),
                    train_labels.cuda(non_blocking=True),
                    caplens.cuda(non_nblocking=True),
                )

            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )

            # The target is all word after <start>
            targets = caps_sorted[:, 1:]

            # remove timesteps
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            # TODO: what is doubly stochastic attention regularization
            loss += params.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Track of metrics
            top5 = accuracy(scores, targets, 5)
            loss_avg.update(loss.item())
            top5acc.update(top5)
            batch_time.update(time.time() - start)

            start = time.time()

            summary_batch = {
                metric: metrics[metric](scores, targets) for metric in metrics
            }

            summary_batch["loss"] = loss.item()
            summary_batch[""]
            summ.append(summary_batch)

            t.set_postfix(
                loss="{:05.3f}".format(loss_avg()),
                top5_acc="{:05.3f}".format(top5acc()),
            )
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " : ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


@torch.no_grad()
def evaluate(encoder, decoder, dataloader, criterion, metrics, params):
    encoder.eval()
    decoder.eval()

    summ = []

    loss_avg = AverageMeter()
    batch_time = AverageMeter()
    top5accs = AverageMeter()

    references = list()  # references (true captions)
    hypotheses = list()

    with tqdm(total=len(dataloader)) as t:
        for i, (imgs, caps, caplens, allcaps) in enumerate(dataloader):
            if params.cuda:
                imgs, caps, caplens = (
                    imgs.cuda(non_blocking=True),
                    caps.cuda(non_blocking=True),
                    caplens.cuda(non_blocking=True),
                )

            # Forward
            if encoder is not None:
                imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            loss += params.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            summary_batch = {
                metric: metrics[metric](output_batch, labels_batch)
                for metric in metrics
            }

            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

            loss_avg.update(loss.item())

            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )

    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    metrics,
    params,
    model_dir=None,
    restore_file=None,
    lr_scheduler=None,
):
    if restore_file is not None:
        logging.info(f"Restoring parameters from {restore_file}")
        load_checkpoint(restore_file, model, optimizer)

    best_val_acc = None  # some metric to get the best model
    summ = {"train": {}, "valid": {}}
    start_time = time()

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1} / {params.num_epochs}")
        train_summ = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_dataloader,
            metrics=metrics,
            params=params,
        )

        val_summ = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            metrics=metrics,
            params=params,
        )

        val_acc = val_summ["some_metrics"]
        is_best = val_acc >= best_val_acc

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # save best val metrics in a json file
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")

            save_dict_to_json(val_summ, best_json_path)

        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "model_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=model_dir,
        )
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_summ, last_json_path)

        summ["train"][epoch + 1] = train_summ
        summ["valid"][epoch + 1] = val_summ

    with open(os.path.join(model_dir, config.TRAIN_VAL_METRICS_SUMM)) as f:
        dump(summ, f)
    save_learning_curve(summ, model_dir)

    logging.info("- total time taken: %.2fs" % (time() - start_time))
