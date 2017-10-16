import torch


def mask_for_lengths(length, max_length=None, mask_right=True, value=-1000.0):
    max_length = max_length or length.max().data[0]
    mask = torch.zeros(length.data.shape[0], max_length)
    for i in range(length.data.shape[0]):
        for j in range(max_length):
            if mask_right and length.data[i] <= j:
                mask[i, j] = value
            elif not mask_right and length.data[i] > j:
                mask[i, j] = value
    return torch.autograd.Variable(mask)
