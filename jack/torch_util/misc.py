import torch


def mask_for_lengths(length, max_length=None, mask_right=True, value=-1e6):
    max_length = max_length or length.max().data[0]
    mask = torch.cuda.IntTensor() if length.is_cuda else torch.IntTensor()
    mask = torch.arange(0, max_length, 1, out=mask)
    mask = torch.autograd.Variable(mask).type_as(length)
    mask /= length.unsqueeze(1)
    mask = mask.clamp(0, 1)
    mask = mask.float()
    if not mask_right:
        mask = 1.0 - mask
    mask *= value
    return mask
