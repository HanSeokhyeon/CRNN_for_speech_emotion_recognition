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

import sys
import math
import wavio
import torch
import threading
import logging
from torch.utils.data import Dataset

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

N_FFT = 512
SAMPLE_RATE = 16000

def get_spectrogram_feature(filepath):
    (fate, width, sig) = wavio.readwav(filepath)
    sig = sig.ravel()

    stft = torch.stft(torch.FloatTensor(sig),
                      N_FFT,
                      hop_length=int(0.01*SAMPLE_RATE),
                      win_length=int(0.03*SAMPLE_RATE),
                      window=torch.hamming_window(int(0.03*SAMPLE_RATE)),
                      center=False,
                      normalized=False,
                      onesided=True)

    stft = (stft[:, :, 0].pow(2) + stft[:, :, 1].pow(2)).pow(0.5)
    amag = stft.numpy()
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)

    return feat


def get_label(filepath):
    label = filepath[-6]

    label2num = {"W": 0, "L": 1, "E": 2, "A": 3, "F": 4, "T": 5, "N": 6}

    return label2num[label]


class BaseDataset(Dataset):
    def __init__(self, wav_paths):
        self.wav_paths = wav_paths

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        label = get_label(self.wav_paths[idx])
        return feat, label


class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset_count()
        self.thread_id = thread_id


class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataest_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataest_list[i], self.queue, self.batch_size, i))
