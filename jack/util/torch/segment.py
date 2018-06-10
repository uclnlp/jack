import torch
from torch.autograd import Function


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


def segment_max(inputs, segment_ids, num_segments=None, default=0.0):
    # highly optimized to decrease the amount of actual invocation of pytorch calls
    # assumes that most segments have 1 or 0 elements
    segment_ids, indices = torch.sort(segment_ids)
    inputs = torch.index_select(inputs, 0, indices)
    output = SegmentMax.apply(inputs, segment_ids, num_segments, default)
    return output


class SegmentMax(Function):
    @staticmethod
    def forward(ctx, inputs, segment_ids, num_segments=None, default=0.0):
        # segments must be sorted by ids
        # highly optimized code, do not change if you don't know what you are doing.
        num_segments = num_segments or segment_ids.max() + 1

        zero_t = inputs[0].clone().view(1, -1)
        zero_t.fill_(default)

        if torch.is_tensor(segment_ids):
            segment_ids = segment_ids.cpu()

        lengths = []
        segm_lengths = []
        num_lengths = [0] * 10
        num_zeros = [0]

        lengths_extend = lengths.extend
        num_lengths_extend = num_lengths.extend
        segm_lengths_append = segm_lengths.append
        segm_lengths_extend = segm_lengths.extend

        _lengths = [[l] * l for l in range(len(num_lengths))]
        _zeros = [0] * num_segments

        def add_length(l):
            if l >= len(num_lengths):
                diff = l + 5 - len(num_lengths)
                num_lengths_extend([0] * diff)
                _lengths.extend([new_l] * new_l for new_l in range(len(_lengths), l + 5))
            num_lengths[l] += l
            segm_lengths_append(l)
            lengths_extend(_lengths[l])

        def add_zeros(n):
            segm_lengths_extend(_zeros[:n])
            num_zeros[0] += n

        offset = 0
        prev_s = segment_ids[0]
        if prev_s:
            add_zeros(prev_s)
        for i, s in enumerate(segment_ids):
            if prev_s != s:
                n_z = s - prev_s - 1
                if n_z > 0:
                    add_zeros(n_z)
                add_length(i - offset)
                offset = i
            prev_s = s
        add_length(segment_ids.shape[0] - offset)
        add_zeros(num_segments - prev_s - 1)

        lengths = torch.cuda.LongTensor(lengths) if inputs.is_cuda else torch.LongTensor(lengths)
        segm_lengths = torch.cuda.LongTensor(segm_lengths) if inputs.is_cuda else torch.LongTensor(segm_lengths)
        _, lengths_sorted = torch.sort(lengths, 0)
        _, segm_lengths_sorted = torch.sort(segm_lengths, 0)

        inputs_sorted = torch.index_select(inputs, 0, lengths_sorted)

        offset = [0]
        ctx.maxes = dict()

        def compute_segment(l, n):
            segment = inputs_sorted.narrow(0, offset[0], n)
            if l > 1:
                segment = segment.view(n // l, l, -1)
                ctx.maxes[l] = _MyMax()
                segment = _MyMax.forward(ctx.maxes[l], segment, dim=1)[0]
            offset[0] += n
            return segment

        segments = [compute_segment(l, n) for l, n in enumerate(num_lengths) if n > 0]
        segments = [zero_t.expand(num_zeros[0], zero_t.shape[1])] + segments

        segments = torch.cat(segments, 0)
        _, rev_segm_sorted = torch.sort(segm_lengths_sorted)

        ctx.rev_segm_sorted = rev_segm_sorted
        ctx.num_lengths = num_lengths
        ctx.lengths_sorted = lengths_sorted
        ctx.num_zeros = num_zeros[0]

        output = torch.index_select(segments, 0, rev_segm_sorted)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        size = grad_outputs.size(1)
        segm_sorted = torch.sort(ctx.rev_segm_sorted)[1]
        grad_outputs = torch.index_select(grad_outputs, 0, segm_sorted)

        offset = [ctx.num_zeros]

        def backward_segment(l, n):
            segment_grad = grad_outputs.narrow(0, offset[0], n // l)
            if l > 1:
                segment_grad = _MyMax.backward(ctx.maxes[l], segment_grad)[0].view(n, size)
            offset[0] += n // l
            return segment_grad

        segment_grads = [backward_segment(l, n) for l, n in enumerate(ctx.num_lengths) if n > 0]
        grads = torch.cat(segment_grads, 0)
        rev_length_sorted = torch.sort(ctx.lengths_sorted)[1]
        grads = torch.index_select(grads, 0, rev_length_sorted)

        return grads, None, None, None


class _MyMax:
    @classmethod
    def forward(cls, ctx, input, dim, keepdim=None):
        ctx.dim = dim
        ctx.keepdim = False if keepdim is None else keepdim
        ctx.input_size = input.size()

        output, indices = input.max(dim=dim, keepdim=ctx.keepdim)
        ctx.indices = indices

        return output, indices

    @classmethod
    def backward(cls, ctx, grad_output, grad_indices=None):
        grad_input = torch.autograd.Variable(grad_output.data.new(*ctx.input_size).zero_())
        dim = ctx.dim
        indices = ctx.indices
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(dim)
            indices = indices.unsqueeze(dim)

        grad_input.scatter_(dim, indices, grad_output)

        return grad_input, None, None
