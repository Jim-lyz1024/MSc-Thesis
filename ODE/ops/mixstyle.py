import torch
import random
import torch.nn as nn


class MixStyleOP(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.mix = mix

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def forward(self, x):
        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True).detach()
        sigma = ((x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()).detach()
        x_normed = (x - mu) / sigma
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)

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

        mu_2, sigma_2 = mu[perm], sigma[perm]
        mu_mix = lmda * mu + (1 - lmda) * mu_2
        sigma_mix = lmda * sigma + (1 - lmda) * sigma_2

        return x_normed * sigma_mix + mu_mix
