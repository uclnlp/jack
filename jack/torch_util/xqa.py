# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional


class XQAMinCrossentropyLossModule(nn.Module):
    def forward(self, start_scores, end_scores, answer_span, answer_to_question):
        """very common XQA loss function."""
        long_tensor = torch.cuda.LongTensor if torch.cuda.device_count() > 0 else torch.LongTensor
        answer_span = answer_span.type(long_tensor)
        start, end = answer_span[:, 0], answer_span[:, 1]

        batch_size1 = start.data.shape[0]
        batch_size2 = start_scores.data.shape[0]
        is_aligned = batch_size1 == batch_size2

        start_scores = start_scores if is_aligned else torch.index_select(start_scores, 0, answer_to_question)
        end_scores = end_scores if is_aligned else torch.index_select(end_scores, 0, answer_to_question)

        partitioned_loss = []
        for i, j in enumerate(answer_to_question):
            j = j.data[0]
            while j >= len(partitioned_loss):
                partitioned_loss.append([])
            loss = -torch.index_select(functional.log_softmax(start_scores[i]), 0, start[i])
            loss -= torch.index_select(functional.log_softmax(end_scores[i]), 0, end[i])
            partitioned_loss[j].append(loss)

        for j in range(len(partitioned_loss)):
            partitioned_loss[j] = torch.stack(partitioned_loss[j]).min()

        loss = torch.stack(partitioned_loss).mean()
        return loss
