"""

Copyright 2017- IBM Corporation

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN


class RNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

        >>> model = RNN(hidden_size)
        >>> output, hidden = model(input)

    """

    def __init__(self, feature_size, hidden_size, output_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=3, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(RNN, self).__init__(hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = self.rnn_cell(feature_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.hidden_size = hidden_size * 2
        self.hidden_neuron = 300
        self.out1 = nn.Linear(self.hidden_size, self.hidden_neuron)
        self.out2 = nn.Linear(self.hidden_neuron, self.output_size)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
                in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """



        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(input_var)
        if self.bidirectional:
            output_forward = output.contiguous()[:, -1, :self.hidden_size//2]
            output_reverse = output.contiguous()[:, 0, self.hidden_size//2:]
            output = torch.cat((output_forward, output_reverse), dim=1)
        else:
            output = output.contiguous()[:, -1, :]
        output = self.out1(output)
        output = self.out2(output)
        predicted_softmax = F.log_softmax(output, dim=1)

        return predicted_softmax
