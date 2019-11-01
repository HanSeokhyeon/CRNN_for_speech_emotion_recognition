"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import time
import argparse
import queue
import random
import torch
import torch.nn as nn
import torch.optim as optim
from loader import *
from models import CRNN, CNN, RNN
from data_downloader import data_download

DATASET_PATH = './dataset'


def train():
    pass


train.cumulative_batch_count = 0


def evaluate():
    pass


def bind_model():
    pass


def split_dataset(config, wav_paths, dataset_ratio=[0.7, 0.1, 0.2]):
    if sum(dataset_ratio) != 1.0:
        raise ValueError("Wrong ratio: {0}, {1}, {2}".format(*dataset_ratio))

    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    test_batch_num = math.ceil(batch_num * dataset_ratio[2])
    valid_batch_num = math.ceil(batch_num * dataset_ratio[1])
    train_batch_num = batch_num - test_batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(wav_paths[train_begin_raw_id:train_end_raw_id]))

        train_begin = train_end

    train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

    train_begin_raw_id = train_begin * config.batch_size
    train_end_raw_id = train_end * config.batch_size

    valid_dataset = BaseDataset(wav_paths[train_begin_raw_id:train_end_raw_id])
    test_dataset = BaseDataset(wav_paths[train_end_raw_id:])

    return train_batch_num, train_dataset_list, valid_dataset, test_dataset



def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNN (default: False')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training (default: 32')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model')
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # N_FFT: defined in loader.py
    feature_size = N_FFT / 2 + 1

    cnn = CNN.CNN(feature_size)
    rnn = RNN.RNN(cnn.feature_size, args.hidden_size,
              input_dropout_p=args.dropout, dropout_p=args.dropout,
              n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='gru', variable_lengths=False)

    model = CRNN.CRNN(cnn, rnn)
    model.flatten_parameters()

    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    # bind_model(model, optimizer)

    if args.mode != 'train':
        return

    data_download()

    wav_paths = [os.path.join('./dataset/wav', fname) for fname in os.listdir('./dataset/wav')]

    best_loss = 1e10
    begin_epoch = 0

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, dataset_ratio=[0.7, 0.1, 0.2])

    logger.info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):

        train_queue = queue.Queue(args.workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        train_loader.start()

    pass


if __name__ == '__main__':
    main()