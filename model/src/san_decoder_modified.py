#
# created by "Arij Al Adel" (Arij.Adel@gmail.com) at 12/23/20
#

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils import weight_norm
from torch.nn import AlphaDropout
import numpy as np
from functools import wraps
from src.common import activation
from src.similarity import FlatSimilarityWrapper
from src.recurrent import RNN_MAP
from src.dropout_wrapper import DropoutWrapper

SMALL_POS_NUM = 1.0e-30
RNN_MAP = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell} # 'gru': nn.GRUCell is used


def generate_mask(new_data, dropout_p=0.0):
    new_data = (1 - dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1) - 1)
        new_data[i][one] = 1
    mask = Variable(1.0 / (1 - dropout_p) * torch.bernoulli(new_data), requires_grad=False)
    return mask


class SANDecoder(nn.Module):
    def __init__(self, x_size, h_size, opt={}, prefix='answer', dropout=None):
        super(SANDecoder, self).__init__()
        self.prefix = prefix  # decoder
        self.attn_e = FlatSimilarityWrapper(x_size, h_size, prefix, opt, dropout)
        self.rnn_type = opt.get('{}_rnn_type'.format(prefix), 'gru')  # gru TODO: try lstm
        self.rnn = RNN_MAP.get(self.rnn_type, nn.GRUCell)(x_size, h_size)
        self.opt = opt
        # h_size != x_size  ==True
        self.proj = nn.Linear(h_size, x_size) if h_size != x_size else None
        if dropout is None:  # False
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.h2h = nn.Linear(h_size, h_size)
        self.a2h = nn.Linear(x_size, h_size, bias=False)
        self.luong_output_layer = nn.Linear(h_size + x_size, h_size)

    def forward(self, input, hidden, context, context_mask):  # word, hidden, doc_mem, doc_mask
        hidden = self.dropout(hidden)  # size = batch_size * 512 (512=  )
        hidden = self.rnn(input, hidden)  # gru
        if self.opt['model_type'] in {'san', 'BERT'}:  # TODO: BERT ADDED in{'san', 'BERT'} or == 'san'
            # context.shape = batch_size, max_doc, 128
            # hidden.shape = batch_size , 512
            # context_mask.shape =  batch_size, max_doc,
            attn = self.attention(context, hidden, context_mask)  # attn.shape = batch_size * 128
            attn_h = torch.cat([hidden, attn], dim=1)  # (batch_size, 512+128)=(batch_size, h_size + x_size)
            new_hidden = F.tanh(self.luong_output_layer(attn_h))  # batch_size, h_size

        elif self.opt['model_type'] in {'seq2seq', 'memnet'}:  # TODO: BERT ADDED {'seq2seq', 'memnet', 'BERT'}
            new_hidden = hidden
        else:
            raise ValueError('Unknown model type: {}'.format(self.opt['model_type']))

        return new_hidden

    def attention(self, x, h0, x_mask):
        end_scores = self.attn_e(x, h0, x_mask)  # batch_size * max_doc
        ptr_net_e = torch.bmm(F.softmax(end_scores, 1).unsqueeze(1), x).squeeze(1)
        ptr_net_in = ptr_net_e  # batch_size * 128
        return ptr_net_in
