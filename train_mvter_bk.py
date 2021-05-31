import os
import shutil
import json
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.mvDataset import mvterDataset
from tools.mvTrainer import mvterTrainer
from model.MVTER import MVTER

parser = argparse.ArgumentParser(description='Multi-View Transformation Equivariant Representations')
parser.add_argument("-name", "--name", type=str, help="name of the experiment", default="train_mvter_on_modelnet40")
parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=24)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("-momentum", type=float, help="momentum", default=0.9)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=1e-4)
parser.add_argument("-step_size", type=float, help="decay lr every step epochs", default=10)
parser.add_argument("-gamma", type=float, help="lr decay factor", default=0.5)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="backbone cnn model name", default="googlenet")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-origin_path", type=str, default="rawdata/origin_12x")
parser.add_argument("-rotate_path", type=str, default="rawdata/rotate_12x")
parser.add_argument("-results_dir", type=str, help="path to cache (default: none)", default='')
parser.set_defaults(train=False)

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.results_dir == '':
        args.results_dir = '/cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-mvter")

    log_dir = args.name + args.results_dir
    create_folder(log_dir)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    mvter = MVTER(m=args.num_views, nclasses=33, cnn_name=args.cnn_name)

    optimizer = torch.optim.SGD(mvter.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    rotate_patch = np.load('rotate_gd/rotate_patched.npy', allow_pickle=True).item()
    assert(len(rotate_patch) == 9449)
    train_dataset = mvterDataset(args.origin_path, args.rotate_path, rotate_patch, train=True)
    test_dataset = mvterDataset(args.origin_path, args.rotate_path, rotate_patch, train=False)
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=True)

    # trainer
    trainer = mvterTrainer(log_dir, mvter, train_iter, test_iter, optimizer, scheduler, num_views=12, w=1.0)
    trainer.train(1, 200)
