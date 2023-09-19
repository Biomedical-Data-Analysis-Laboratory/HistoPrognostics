import torch

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, epsilon=1e-3):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + epsilon) / (TP + self.alpha * FP + self.beta * FN + epsilon)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky
