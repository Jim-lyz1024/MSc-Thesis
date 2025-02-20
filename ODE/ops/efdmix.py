import random

import torch
import torch.nn as nn


class EFDMixOP(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.mix = mix

    def __repr__(self):
        return f"EFDMix(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"

    def forward(self, x):
        if random.random() > self.p:
            return x

        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        x_value, x_index = torch.sort(x_view)
        lmda = self.beta.sample((B, 1, 1)).to(x.device)

        if self.mix == "random":
            # Random Shuffle
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            # Split into Two Halves and Swap the Order
            perm = torch.arange(B - 1, -1, -1)  # Inverse Index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        inverse_index = x_index.argsort(-1)
        x_view_copy = x_value[perm].gather(-1, inverse_index)
        x_new = x_view + (x_view_copy - x_view) * (1 - lmda)

        return x_new.view(B, C, W, H)
