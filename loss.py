import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

class AttentionTransfer(nn.Module):
    def __init__(self):
        super(AttentionTransfer, self).__init__()

    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class GaussianLoss(nn.Module):
    """
    Gaussian loss for transfer learning with variational information distillation.
    """

    def forward(self, y_pred, y):
        """
        Output the Gaussian loss given the prediction and the target.
        :param tuple(torch.Tensor, torch.Tensor) y_pred: predicted mean and variance for the Gaussian
        distribution.
        :param torch.Tensor y: target for the Gaussian distribution.
        """
        y_pred_mean, y_pred_var = y_pred
        loss = torch.mean(0.5 * ((y_pred_mean - y) ** 2 / y_pred_var + torch.log(y_pred_var)))
        return loss


class EnsembleKnowledgeTransferLoss(nn.Module):
    """
    Knowledge transfer loss as an ensemble of individual knowledge transfer losses defined on predicting the label,
    logits of the teacher model, and features of the teacher model.

    :param torch.nn.Module label_criterion: criterion for predicting the labels.
    :param torch.nn.Module teacher_logit_criterion: criterion for predicting the logit of the teacher model.
    :param torch.nn.Module teacher_feature_criterion: criterion for predicting the feature of the teacher model.
    :param float teacher_logit_factor: scaling factor for predicting the logit of the teacher model.
    :param float teacher_feature_factor: scaling factor for predicting the feature of the teacher model.
    """

    def __init__(
            self,
            label_criterion,
            teacher_logit_criterion,
            teacher_feature_criterion,
            teacher_logit_factor,
            teacher_feature_factor,
    ):
        super(EnsembleKnowledgeTransferLoss, self).__init__()
        self.label_criterion = label_criterion
        self.teacher_logit_criterion = teacher_logit_criterion
        self.teacher_feature_criterion = teacher_feature_criterion

        self.teacher_logit_factor = teacher_logit_factor
        self.teacher_feature_factor = teacher_feature_factor

    def forward(self, logit, label, teacher_feature_preds, teacher_logit, teacher_features):
        """
        Output the ensemble of knowledge transfer losses given the predictions and the targets.
        :param torch.Tensor logit: logit of the student model for predicting the label and logit of the teacher model.
        :param torch.Tensor label: target label of the image.
        :param tuple(tuple(torch.Tensor)) teacher_feature_preds: predictions of the student model made on features of
        the teacher model.
        :param torch.Tensor teacher_logit: logit of the teacher model to predict from the the student model.
        :param tuple(torch.Tensor) teacher_features: features of the teacher model to predict from the student model.
        """
        label_loss = self.label_criterion(logit, label)
        teacher_logit_loss = self.teacher_logit_criterion(logit, teacher_logit)
        teacher_feature_losses = [
            self.teacher_feature_criterion(pred, feature) for pred, feature in
            zip(teacher_feature_preds, teacher_features)
        ]
        loss = (
                label_loss
                + self.teacher_logit_factor * teacher_logit_loss
                + self.teacher_feature_factor * sum(teacher_feature_losses)
        )

        return loss
