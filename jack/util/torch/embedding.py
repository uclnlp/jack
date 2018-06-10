# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
from torch.nn import functional

from jack.util.torch import misc


class ConvCharEmbeddingModule(nn.Module):
    def __init__(self, num_chars, size, conv_width=5):
        super(ConvCharEmbeddingModule, self).__init__()
        self._size = size
        self._conv_width = conv_width
        self._embeddings = torch.nn.Embedding(num_chars, size)
        self._embeddings.weight.data.mul_(0.1)
        self._conv = torch.nn.Conv1d(size, size, conv_width, padding=math.floor(conv_width / 2))

    def forward(self, unique_word_chars, unique_word_lengths, sequences_as_uniqs=None):
        long_tensor = torch.cuda.LongTensor if torch.cuda.device_count() > 0 else torch.LongTensor
        embedded_chars = self._embeddings(unique_word_chars.type(long_tensor))
        # [N, S, L]
        conv_out = self._conv(embedded_chars.transpose(1, 2))
        # [N, L]
        conv_mask = misc.mask_for_lengths(unique_word_lengths)
        conv_out = conv_out + conv_mask.unsqueeze(1)
        embedded_words = conv_out.max(2)[0]

        if sequences_as_uniqs is None:
            return embedded_words
        else:
            if not isinstance(sequences_as_uniqs, list):
                sequences_as_uniqs = [sequences_as_uniqs]

            all_embedded = []
            for word_idxs in sequences_as_uniqs:
                all_embedded.append(functional.embedding(
                    word_idxs.type(long_tensor), embedded_words))
            return all_embedded
