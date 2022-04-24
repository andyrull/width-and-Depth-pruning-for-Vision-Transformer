"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class LossWithClassifierAndPruning(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, model: torch.nn.Module, R_threshold=0.5):
        super().__init__()
        self.base_criterion = base_criterion
        self.R_threshold = R_threshold
        self.model = model

    def forward(self,  outputs, labels, lambda_1, lambda_2):
        """
        Args:
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_classifier]
            outputs, outputs_classifier = outputs
        base_loss = self.base_criterion(outputs, labels)
        for i in range(len(outputs_classifier)):
            base_loss += self.base_criterion(outputs_classifier[i], labels)

        if self.R_threshold < 1:
            # for pruning
            loss_p = lagrangian_regularization(
                model=self.model, threshold=self.R_threshold,lambda_1=lambda_1,lambda_2=lambda_2)
        else:
            # For baseline training
            loss_p = 0

        base_loss = base_loss + loss_p

        return base_loss

def lagrangian_regularization(model: torch.nn.Module, threshold: float,lambda_1, lambda_2):
    threshold_list = []
    for name, param in model.named_parameters():
        if 'threshold' in name:
            threshold_list.append(torch.sigmoid(param))
    # DeiT-base has 12 blocks
    param_remain = 0
    layer_num = 12
    block_num = len(threshold_list) // layer_num
    for i in range(12):
        param_remain += remain_param_compute(
            threshold_list[i * block_num: (i + 1) * block_num])

    expected_sparsity = param_remain / 144. # 144 comes from count
    target_sparsity = threshold
    if expected_sparsity - target_sparsity <= 0:
        lagrangian_loss = param_remain * 0.
    else:
        lagrangian_loss = (
                lambda_1 * (expected_sparsity - target_sparsity)
                + lambda_2 * (expected_sparsity - target_sparsity) ** 2
        )

    return lagrangian_loss

def remain_param_compute(threshold_list):
    output = 0.

    attn, o_matrix, fc1, fc2 = threshold_list
    output += torch.max(attn, torch.tensor(1 / 12.)).type(fc1.type()) * 3
    output += torch.max(attn, torch.tensor(1 / 12.)).type(fc1.type()) * \
        torch.max(o_matrix, torch.tensor(1 / 768.)).type(fc1.type())
    output += torch.max(fc1, torch.tensor(1 / 3072.)).type(fc1.type()) * 4
    output += torch.max(fc1,
                        torch.tensor(1 / 3072.)).type(fc1.type()) * torch.max(fc2,
                                                                              torch.tensor(1 / 768.)).type(fc1.type()) * 4
    return output