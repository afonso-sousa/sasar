import torch
from torch import nn


class Sinkhorn(nn.Module):
    def __init__(self, num_iterations=10, temperature=1.0):
        super().__init__()
        self.num_iterations = num_iterations
        self.temperature = temperature

    def forward(self, log_alpha):
        """Log-space Sinkhorn normalization"""
        log_alpha = log_alpha / self.temperature

        for _ in range(self.num_iterations):
            # Row normalization in log space
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)

            # Column normalization in log space
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

        return torch.exp(log_alpha)
