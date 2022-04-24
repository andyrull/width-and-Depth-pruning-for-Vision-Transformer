import torch
from torch import autograd
import math


class TopKBinarizer(autograd.Function):


    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, head_split: int):

        # Get the subnetwork by sorting the inputs and using the top threshold
        # %
        threshold = torch.sigmoid(threshold).item()

        mask = inputs.clone()
        if head_split <= 1:
            _, idx = inputs.flatten().sort(descending=True)
            j = math.ceil(threshold * inputs.numel())
            v = 0
            n = 1
            base_number = 16
            if j % base_number != 0 and inputs.size()[0] %base_number == 0:
                if j > base_number:
                    while v < j:
                        n += 1
                        v = base_number * n
                    if j - (v - base_number) < v - j:
                        j = v - base_number
                    else:
                        j = v
                    if j > inputs.size()[0]:
                        j = inputs.size()[0]
                else:
                    j = base_number

            # flat_out and mask access the same memory.
            flat_out = mask.flatten()
            flat_out[idx[j:]] = 0.
            flat_out[idx[:j]] = 1.
        else:
            # make it as a 12 x 64 matrix! Then do the sorting!
            inputs = inputs.reshape(head_split, -1)
            # the default is column-wise
            _, idx = inputs.sort(-1, descending=True)
            j = math.ceil(threshold * inputs.size(1))

            #
            flat_out = mask.reshape(head_split, -1)
            for i in range(head_split):
                flat_out[i, idx[i, j:]] = 0.
                flat_out[i, idx[i, :j]] = 1.
        ctx.save_for_backward(mask)  # we should try two things

        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        mask, = ctx.saved_tensors
        return gradOutput, ((gradOutput * mask).sum()).view(-1), None
