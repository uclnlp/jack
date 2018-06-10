# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn


class XQAMinCrossentropyLossModule(nn.Module):
    def forward(self, start_scores, end_scores, answer_span, answer_to_question):
        """very common XQA loss function."""
        long_tensor = torch.cuda.LongTensor if torch.cuda.device_count() > 0 else torch.LongTensor
        answer_span = answer_span.type(long_tensor)
        start, end = answer_span[:, 0], answer_span[:, 1]

        batch_size1 = start.data.shape[0]
        batch_size2 = start_scores.data.shape[0]
        is_aligned = batch_size1 == batch_size2

        start_scores = start_scores if is_aligned else torch.index_select(start_scores, dim=0, index=answer_to_question)
        end_scores = end_scores if is_aligned else torch.index_select(end_scores, dim=0, index=answer_to_question)

        partitioned_loss = []
        for i, j in enumerate(answer_to_question):
            j = j.data[0]
            while j >= len(partitioned_loss):
                partitioned_loss.append([])
            loss = -torch.index_select(F.log_softmax(start_scores[i], dim=0), dim=0, index=start[i])
            loss -= torch.index_select(F.log_softmax(end_scores[i], dim=0), dim=0, index=end[i])
            partitioned_loss[j].append(loss)

        for j, l in enumerate(partitioned_loss):
            partitioned_loss[j] = torch.stack(l).min()

        loss = torch.stack(partitioned_loss).mean()
        return loss
