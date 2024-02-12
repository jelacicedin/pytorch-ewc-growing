#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch
from data import get_apple_datasets
from train import train
from model import MLP
import utils


parser = ArgumentParser("EWC PyTorch Implementation")
parser.add_argument("--hidden-size", type=int, default=400)
parser.add_argument("--hidden-layer-num", type=int, default=2)
parser.add_argument("--hidden-dropout-prob", type=float, default=0.5)
parser.add_argument("--input-dropout-prob", type=float, default=0.2)

parser.add_argument("--task-number", type=int, default=8)
parser.add_argument("--epochs-per-task", type=int, default=20)
parser.add_argument("--lamda", type=float, default=40)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--weight-decay", type=float, default=0)
parser.add_argument("--batch-size", type=int, default=10)
parser.add_argument("--test-size", type=int, default=10)
parser.add_argument("--fisher-estimation-sample-size", type=int, default=100)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--no-gpus", action="store_false", dest="cuda")
parser.add_argument("--eval-log-interval", type=int, default=250)
parser.add_argument("--loss-log-interval", type=int, default=250)
parser.add_argument("--consolidate", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda

    # prepare apple datasets.
    train_datasets, test_datasets, train_dataloaders, test_dataloaders, scalers = (
        get_apple_datasets(download=False, batch_size=args.batch_size)
    )

    # prepare the model.
    mlp = MLP(
        7,
        1,
        hidden_size=args.hidden_size,
        hidden_layer_num=args.hidden_layer_num,
        hidden_dropout_prob=args.hidden_dropout_prob,
        input_dropout_prob=args.input_dropout_prob,
        lamda=args.lamda,
    )

    # initialize the parameters.
    utils.xavier_initialize(mlp)

    # prepare the cuda if needed.
    if cuda:
        mlp.cuda()

    # run the experiment.
    train(
        mlp,
        train_datasets,
        test_datasets,
        train_dataloaders,
        test_dataloaders,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        test_size=args.test_size,
        consolidate=args.consolidate,
        fisher_estimation_sample_size=args.fisher_estimation_sample_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_log_interval=args.eval_log_interval,
        loss_log_interval=args.loss_log_interval,
        cuda=cuda,
    )
