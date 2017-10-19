import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, size):
        super(BiLSTM, self).__init__()
        self._size = size
        self._bilstm = nn.LSTM(input_size, size, 1, bidirectional=True, batch_first=True)
        self._bilstm.bias_ih_l0.data[size:2 * size].fill_(1.0)
        self._bilstm.bias_ih_l0_reverse.data[size:2 * size].fill_(1.0)
        self._lstm_start_hidden = nn.Parameter(torch.zeros(2, size))
        self._lstm_start_state = nn.Parameter(torch.zeros(2, size))

    def forward(self, inputs, lengths=None, start_state=None):
        if start_state is None:
            batch_size = inputs.size(0)
            start_hidden = self._lstm_start_hidden.unsqueeze(1).expand(2, batch_size, self._size).contiguous()
            start_state = self._lstm_start_state.unsqueeze(1).expand(2, batch_size, self._size).contiguous()
            start_state = (start_hidden, start_state)

        if lengths is not None:
            new_lengths, indices = torch.sort(lengths, dim=0, descending=True)
            inputs = torch.index_select(inputs, 0, indices)
            new_lengths = [l.data[0] for l in new_lengths]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, new_lengths, batch_first=True)

        output, (h_n, c_n) = self._bilstm(inputs, start_state)

        if lengths is not None:
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
            _, back_indices = torch.sort(indices, dim=0)
            output = torch.index_select(output, 0, back_indices)
            h_n = torch.index_select(h_n, 1, back_indices)
            c_n = torch.index_select(c_n, 1, back_indices)

        return output, (h_n, c_n)
