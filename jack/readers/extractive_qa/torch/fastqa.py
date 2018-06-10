import torch
import torch.nn as nn
import torch.nn.functional as F

from jack.core import SharedResources
from jack.core.tensorport import Ports
from jack.core.torch import PyTorchModelModule
from jack.readers.extractive_qa.shared import XQAPorts
from jack.util.torch import Highway
from jack.util.torch import embedding, misc, xqa
from jack.util.torch.rnn import BiLSTM


class FastQAPyTorchModelModule(PyTorchModelModule):
    """PyTorch implementation of FastQA."""

    # TODO: does not support multiparagraph yet
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length,
                    # char embedding inputs
                    XQAPorts.word_chars, XQAPorts.word_char_length,
                    XQAPorts.question_batch_words, XQAPorts.support_batch_words,
                    # feature input
                    XQAPorts.word_in_question,
                    # optional input, provided only during training
                    XQAPorts.correct_start, XQAPorts.answer2support_training,
                    XQAPorts.is_eval]

    _output_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                     XQAPorts.answer_span]
    _training_input_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                             XQAPorts.answer_span_target, XQAPorts.answer2support_training]
    _training_output_ports = [Ports.loss]

    @property
    def output_ports(self):
        return self._output_ports

    @property
    def input_ports(self):
        return self._input_ports

    @property
    def training_input_ports(self):
        return self._training_input_ports

    @property
    def training_output_ports(self):
        return self._training_output_ports

    def create_loss_module(self, shared_resources: SharedResources):
        return xqa.XQAMinCrossentropyLossModule()

    def create_prediction_module(self, shared_resources: SharedResources):
        return FastQAPyTorchModule(shared_resources)


class FastQAPyTorchModule(nn.Module):
    def __init__(self, shared_resources: SharedResources):
        super(FastQAPyTorchModule, self).__init__()
        self._shared_resources = shared_resources
        input_size = shared_resources.embeddings.shape[-1]
        size = shared_resources.config["repr_dim"]
        self._size = size
        self._with_char_embeddings = self._shared_resources.config.get("with_char_embeddings", False)

        # modules & parameters
        if self._with_char_embeddings:
            self._conv_char_embedding = embedding.ConvCharEmbeddingModule(
                len(shared_resources.char_vocab), size)
            self._embedding_projection = nn.Linear(size + input_size, size)
            self._embedding_highway = Highway(size, 1)
            self._v_wiq_w = nn.Parameter(torch.ones(1, 1, input_size + size))
            input_size = size
        else:
            self._v_wiq_w = nn.Parameter(torch.ones(1, 1, input_size))

        self._bilstm = BiLSTM(input_size + 2, size)
        self._answer_layer = FastQAAnswerModule(shared_resources)

        # [size, 2 * size]
        self._question_projection = nn.Parameter(torch.cat([torch.eye(size), torch.eye(size)], dim=1))
        self._support_projection = nn.Parameter(torch.cat([torch.eye(size), torch.eye(size)], dim=1))

    def forward(self, emb_question, question_length,
                emb_support, support_length,
                unique_word_chars, unique_word_char_length,
                question_words2unique, support_words2unique,
                word_in_question, correct_start, answer2support,
                is_eval):
        """fast_qa model

        Args:
            emb_question: [Q, L_q, N]
            question_length: [Q]
            emb_support: [Q, L_s, N]
            support_length: [Q]
            unique_word_chars
            unique_word_char_length
            question_words2unique
            support_words2unique
            word_in_question: [Q, L_s]
            correct_start: [A], only during training, i.e., is_eval=False
            answer2question: [A], only during training, i.e., is_eval=False
            is_eval: []

        Returns:
            start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
        """
        # Some helpers
        float_tensor = torch.cuda.FloatTensor if emb_question.is_cuda else torch.FloatTensor
        long_tensor = torch.cuda.LongTensor if emb_question.is_cuda else torch.LongTensor
        batch_size = question_length.data.shape[0]
        max_question_length = question_length.max().data[0]
        support_mask = misc.mask_for_lengths(support_length)
        question_binary_mask = misc.mask_for_lengths(question_length, mask_right=False, value=1.0)

        if self._with_char_embeddings:
            # compute combined embeddings
            [char_emb_question, char_emb_support] = self._conv_char_embedding(
                unique_word_chars, unique_word_char_length, [question_words2unique, support_words2unique])

            emb_question = torch.cat([emb_question, char_emb_question], 2)
            emb_support = torch.cat([emb_support, char_emb_support], 2)

        # compute encoder features
        question_features = torch.autograd.Variable(torch.ones(batch_size, max_question_length, 2, out=float_tensor()))
        question_features = question_features.type_as(emb_question)

        v_wiqw = self._v_wiq_w
        # [B, L_q, L_s]
        wiq_w = torch.matmul(emb_question * v_wiqw, emb_support.transpose(1, 2))
        # [B, L_q, L_s]
        wiq_w = wiq_w + support_mask.unsqueeze(1)
        wiq_w = F.softmax(
            wiq_w.view(batch_size * max_question_length, -1), dim=1).view(batch_size, max_question_length, -1)
        # [B, L_s]
        wiq_w = torch.matmul(question_binary_mask.unsqueeze(1), wiq_w).squeeze(1)

        # [B, L , 2]
        support_features = torch.stack([word_in_question, wiq_w], dim=2)

        if self._with_char_embeddings:
            # highway layer to allow for interaction between concatenated embeddings
            emb_question = self._embedding_projection(emb_question)
            emb_support = self._embedding_projection(emb_support)
            emb_question = self._embedding_highway(emb_question)
            emb_support = self._embedding_highway(emb_support)

        # dropout
        dropout = self._shared_resources.config.get("dropout", 0.0)
        emb_question = F.dropout(emb_question, dropout, training=not is_eval)
        emb_support = F.dropout(emb_support, dropout, training=not is_eval)

        # extend embeddings with features
        emb_question_ext = torch.cat([emb_question, question_features], 2)
        emb_support_ext = torch.cat([emb_support, support_features], 2)

        # encode question and support
        # [B, L, 2 * size]
        encoded_question = self._bilstm(emb_question_ext)[0]
        encoded_support = self._bilstm(emb_support_ext)[0]

        # [B, L, size]
        encoded_support = F.tanh(F.linear(encoded_support, self._support_projection))
        encoded_question = F.tanh(F.linear(encoded_question, self._question_projection))

        start_scores, end_scores, predicted_start_pointer, predicted_end_pointer = \
            self._answer_layer(encoded_question, question_length, encoded_support, support_length,
                               correct_start, answer2support, is_eval)

        # no multi paragraph support yet
        doc_idx = torch.autograd.Variable(torch.zeros(predicted_start_pointer.data.shape[0], out=long_tensor()))
        span = torch.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return start_scores, end_scores, span


class FastQAAnswerModule(nn.Module):
    def __init__(self, shared_resources: SharedResources):
        super(FastQAAnswerModule, self).__init__()
        self._size = shared_resources.config["repr_dim"]

        # modules
        self._linear_question_attention = nn.Linear(self._size, 1, bias=False)

        self._linear_q_start_q = nn.Linear(self._size, self._size)
        self._linear_q_start = nn.Linear(2 * self._size, self._size, bias=False)
        self._linear_start_scores = nn.Linear(self._size, 1, bias=False)

        self._linear_q_end_q = nn.Linear(self._size, self._size)
        self._linear_q_end = nn.Linear(3 * self._size, self._size, bias=False)
        self._linear_end_scores = nn.Linear(self._size, 1, bias=False)

    def forward(self, encoded_question, question_length, encoded_support, support_length,
                correct_start, answer2question, is_eval):
        # casting
        long_tensor = torch.cuda.LongTensor if encoded_question.is_cuda else torch.LongTensor
        answer2question = answer2question.type(long_tensor)

        # computing single time attention over question
        attention_scores = self._linear_question_attention(encoded_question)
        q_mask = misc.mask_for_lengths(question_length)
        attention_scores = attention_scores.squeeze(2) + q_mask
        question_attention_weights = F.softmax(attention_scores, dim=1)
        question_state = torch.matmul(question_attention_weights.unsqueeze(1),
                                      encoded_question).squeeze(1)

        # Prediction
        # start
        start_input = torch.cat([question_state.unsqueeze(1) * encoded_support, encoded_support], 2)

        q_start_state = self._linear_q_start(start_input) + self._linear_q_start_q(question_state).unsqueeze(1)
        start_scores = self._linear_start_scores(F.relu(q_start_state)).squeeze(2)

        support_mask = misc.mask_for_lengths(support_length)
        start_scores = start_scores + support_mask
        _, predicted_start_pointer = start_scores.max(1)

        def align(t):
            return torch.index_select(t, 0, answer2question)

        if is_eval:
            start_pointer = predicted_start_pointer
        else:
            # use correct start during training, because p(end|start) should be optimized
            start_pointer = correct_start.type(long_tensor)
            predicted_start_pointer = align(predicted_start_pointer)
            start_scores = align(start_scores)
            start_input = align(start_input)
            encoded_support = align(encoded_support)
            question_state = align(question_state)
            support_mask = align(support_mask)

        # end
        u_s = []
        for b, p in enumerate(start_pointer):
            u_s.append(encoded_support[b, p.data[0]])
        u_s = torch.stack(u_s)

        end_input = torch.cat([encoded_support * u_s.unsqueeze(1), start_input], 2)

        q_end_state = self._linear_q_end(end_input) + self._linear_q_end_q(question_state).unsqueeze(1)
        end_scores = self._linear_end_scores(F.relu(q_end_state)).squeeze(2)

        end_scores = end_scores + support_mask

        max_support = support_length.max().data[0]

        if is_eval:
            end_scores += misc.mask_for_lengths(start_pointer, max_support, mask_right=False)

        _, predicted_end_pointer = end_scores.max(1)

        return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer
