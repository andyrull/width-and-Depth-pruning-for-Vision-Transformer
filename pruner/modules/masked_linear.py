# This code is modified from https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning
# Licensed under the Apache License, Version 2.0 (the "License");
# We add more functionalities as well as remove unnecessary functionalities
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .binarizer import TopKBinarizer



class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask during training,
    and does real pruning during inference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        head_split: int = -1,
        bias_mask: bool = False,
        head_pruning: bool = False,
        fc_pruning: bool = False,
        threshold_init = 10.0
    ):

        super(
            MaskedLinear,
            self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)

        self.head_split = head_split
        self.bias_mask = bias_mask
        self.head_pruning = head_pruning
        self.fc_pruning = fc_pruning

        self.inference_mode = False

        self.mask_scale = mask_scale
        self.mask_init = mask_init
        if self.fc_pruning:
            self.saliency_scores = nn.Parameter(
                torch.Tensor(
                    self.weight.size(0),
                    1))  # number of output * 1
            self.init_mask(self.saliency_scores)
            self.threshold_fc = nn.Parameter(torch.zeros(1) + threshold_init)
        if self.head_pruning:
            self.head_saliency_scores = nn.Parameter(
                torch.Tensor(self.head_split, 1))  # number of heads * 1
            self.init_mask(self.head_saliency_scores)
            self.threshold_head = nn.Parameter(torch.zeros(1) + threshold_init)

    def init_mask(self, mask):
        if self.mask_init == "constant":
            init.constant_(mask, val=self.mask_scale)
        elif self.mask_init == "uniform":
            init.uniform_(mask, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(mask, a=math.sqrt(5))

    def get_mask(self):
        # get head mask
        if self.head_pruning:
            mask_head = TopKBinarizer.apply(
                self.head_saliency_scores, self.threshold_head, -1)  # for now, only support this
        else:
            mask_head = None

        if self.fc_pruning:
            mask = TopKBinarizer.apply(
                self.saliency_scores, self.threshold_fc, -1)
        else:
            mask = None
        return mask_head, mask

    def make_inference_pruning(self):
        self.inference_mode = True

        mask_head, mask = self.get_mask()
        if not self.head_pruning:
            mask_head = torch.ones_like(self.weight[:, 0]).type(
                'torch.BoolTensor').view(-1)
        else:
            mask_head = mask_head.type('torch.BoolTensor').view(-1)
            retmask_tmp = mask_head
            mask_head = mask_head.repeat_interleave(64).unsqueeze(1).repeat(3, 1).view(-1)

            retmask_tmp = retmask_tmp.repeat_interleave(64).unsqueeze(1).view(-1)
            retmask = torch.ones_like(retmask_tmp).type('torch.BoolTensor').view(-1)
            retmask = torch.logical_and(retmask_tmp, retmask)

        if not self.fc_pruning:
            mask = torch.ones_like(self.weight[:, 0])

        mask = mask.type('torch.BoolTensor').view(-1)
        mask = torch.logical_and(mask_head, mask)


        self.weight = nn.Parameter(self.weight[mask, :])
        if self.bias_mask:
            self.bias = nn.Parameter(self.bias[mask])

        # we do not need those parameters!
        self.saliency_scores = None
        self.head_saliency_scores = None
        self.threshold_head = None
        self.threshold_fc = None
        # we need this mask for some Layer O and FC2 pruning
        if self.head_pruning:
            return retmask
        else:
            return mask

    def make_column_purning(self, mask):
        # make column pruning for Layer O and FC2
        self.weight = nn.Parameter(self.weight[:, mask])

    def forward(self, input: torch.tensor):
        if not self.inference_mode:
            output = self.training_forward(input)
        else:
            output = self.inference_forward(input)
        return output


    def inference_forward(self, input: torch.tensor):
        return F.linear(input, self.weight, self.bias)

    def training_forward(self, input: torch.tensor):
        mask_head, mask = self.get_mask()

        weight_shape = self.weight.size()
        bias_shape = self.bias.size()
        if self.head_pruning:
            weight_thresholded = self.weight * mask_head.repeat_interleave(64).unsqueeze(1).repeat(3, 1)

            if self.bias_mask:
                bias_thresholded = self.bias * mask_head.repeat_interleave(64).unsqueeze(1).repeat(3, 1).view(bias_shape)
        else:
            weight_thresholded = self.weight
            bias_thresholded = self.bias
        # Mask weights with computed mask
        if self.fc_pruning:
            weight_thresholded = mask * weight_thresholded
            if self.bias_mask:
                bias_thresholded = mask.view(
                    self.bias.size()) * bias_thresholded
            else:
                bias_thresholded = bias_thresholded

        return F.linear(input, weight_thresholded, bias_thresholded)
