"""
Copyright 2019-present Han Seokhyeon.

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

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def save_data(loss_acc, test_loss, test_acc):
    data = np.array(loss_acc)
    test = np.zeros((2, data.shape[1]))
    data = np.concatenate((data, test))
    data[4, 0] = test_loss
    data[5, 0] = test_acc

    now = datetime.now()
    filename = "./result/{}.csv".format(str(now)[:-7])

    np.savetxt(filename, data, delimiter=',', newline='\n')

    return


def plot_data(loss_acc, test_loss, test_acc):
    data = np.array(loss_acc)

    fig, ax1 = plt.subplots()
    plt.title("Train data")

    ax1.set_xlabel("Epoch")
    # ax1.set_xticks(range(data.shape[1]))

    ax1.set_ylabel("Loss")

    ax1.plot(data[0], label='train')
    ax1.plot(data[2], label='valid')

    ax2 = ax1.twinx()

    ax2.set_ylabel("Accuracy")

    ax2.plot(data[1], label='train')
    ax2.plot(data[3], label='valid')

    plt.text(0.85, 0.5, "Test acc:{:.4f}".format(test_acc), ha='center', va='center', transform=ax2.transAxes)

    plt.grid()
    plt.legend()

    plt.show()

    now = datetime.now()
    filename = "./result/{}.png".format(str(now)[:-7])

    fig.savefig(filename)

    return


