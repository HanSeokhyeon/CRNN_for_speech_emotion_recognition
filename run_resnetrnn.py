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
from models import CRNN, resnet, RNN
from data_downloader import data_download
from result import *

DATASET_PATH = './dataset'


def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5):
    total_loss = 0
    total_num = 0
    total_correct = 0
    total_sent_num = 0
    batch = 0

    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, label, feat_lengths = queue.get()

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        label = label.to(device)

        model.module.flatten_parameters()
        logit = model(feats, feat_lengths).to(device)

        y_hat = logit.max(-1)[1]

        correct = torch.eq(y_hat, label)
        batch_correct = torch.nonzero(correct).size(0)
        total_correct += batch_correct

        loss = criterion(logit.contiguous(), label)
        total_loss += loss.item()
        total_num += logit.size(0)

        total_sent_num += label.size(0)

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                        .format(batch,
                                total_batch_size,
                                total_loss / total_num,
                                elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_correct / total_sent_num


train.cumulative_batch_count = 0


def evaluate(model, dataloader, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0
    total_num = 0
    total_correct = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, label, feat_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            label = label.to(device)

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths).to(device)

            y_hat = logit.max(-1)[1]

            correct = torch.eq(y_hat, label)
            batch_correct = torch.nonzero(correct).size(0)
            total_correct += batch_correct

            loss = criterion(logit.contiguous(), label)
            total_loss += loss.item()
            total_num += logit.size(0)

            total_sent_num += label.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_correct / total_sent_num


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

    valid_end = train_end + valid_batch_num

    valid_begin_raw_id = train_begin * config.batch_size
    valid_end_raw_id = valid_end * config.batch_size

    valid_dataset = BaseDataset(wav_paths[valid_begin_raw_id:valid_end_raw_id])
    test_dataset = BaseDataset(wav_paths[valid_end_raw_id:])

    return train_batch_num, train_dataset_list, valid_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of model (default: 256)')
    parser.add_argument('--layer_size', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--n_class', type=int, default=7, help='number of classes of data (default: 7)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in training (default: 0.2')
    parser.add_argument('--bidirectional', default=True, action='store_true', help='use bidirectional RNN (default: False')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training (default: 32')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=30, help='number of max epochs in training (default: 10)')
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

    cnn = resnet.ResNet(feature_size, resnet.BasicBlock, [3, 3, 3])
    rnn = RNN.RNN(cnn.feature_size, args.hidden_size, args.n_class,
              input_dropout_p=args.dropout, dropout_p=args.dropout,
              n_layers=args.layer_size, bidirectional=args.bidirectional, rnn_cell='gru', variable_lengths=False)

    model = CRNN.CRNN(cnn, rnn)
    model.flatten_parameters()

    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    if args.mode != 'train':
        return

    data_download()

    wav_paths = [os.path.join('./dataset/wav', fname) for fname in os.listdir('./dataset/wav')]

    best_acc = 0
    begin_epoch = 0

    loss_acc = [[], [], [], []]

    train_batch_num, train_dataset_list, valid_dataset, test_dataset = split_dataset(args, wav_paths, dataset_ratio=[0.7, 0.1, 0.2])

    logger.info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):

        train_queue = queue.Queue(args.workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        train_loader.start()

        train_loss, train_acc = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, args.workers, 10)
        logger.info('Epoch %d (Training) Loss %0.4f Acc %0.4f' % (epoch, train_loss,  train_acc))

        train_loader.join()

        loss_acc[0].append(train_loss)
        loss_acc[1].append(train_acc)

        valid_queue = queue.Queue(args.workers * 2)

        valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
        valid_loader.start()

        eval_loss, eval_acc = evaluate(model, valid_loader, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f Acc %0.4f' % (epoch, eval_loss, eval_acc))

        valid_loader.join()

        loss_acc[2].append(eval_loss)
        loss_acc[3].append(eval_acc)

        best_model = (eval_acc > best_acc)

        if best_model:
            best_acc = eval_acc
            torch.save(model.state_dict(), './save_model/best_model.pt')
            save_epoch = epoch

    model.load_state_dict(torch.load('./save_model/best_model.pt'))

    test_queue = queue.Queue(args.workers * 2)

    test_loader = BaseDataLoader(test_dataset, test_queue, args.batch_size, 0)
    test_loader.start()

    test_loss, test_acc = evaluate(model, test_loader, test_queue, criterion, device)
    logger.info('Epoch %d (Test) Loss %0.4f Acc %0.4f' % (save_epoch, test_loss, test_acc))

    test_loader.join()

    save_data(loss_acc, test_loss, test_acc)
    plot_data(loss_acc, test_loss, test_acc)

    return 0


if __name__ == '__main__':
    main()