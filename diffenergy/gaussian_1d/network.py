# --------------------------------------------------------------------------
# author Sudeep Sarma
# MLP network for training 1D distribution
# --------------------------------------------------------------------------
import torch
from torch import nn
import math

# --------------------------------------------------------------------------

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale)
        self.W.requires_grad_(False)    # Explicitly set time-embeddings as non trainable
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ScoreNetMLP(nn.Module):
    """A time-dependent score-based model built with MLP architecture."""
    def __init__(self, input_dim, marginal_prob_std, embed_dim=512, layers=None):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.marginal_prob_std = marginal_prob_std
        self.mlp = self.build_mlp(input_dim + embed_dim, layers)

    def build_mlp(self, input_dim, layers):
        modules = []
        for dim in layers:
            modules.append(nn.Linear(input_dim, dim))
            #modules.append(nn.BatchNorm1d(dim))
            modules.append(nn.GELU())
            input_dim = dim
        modules.append(nn.Linear(input_dim, 1))  # Output layer
        return nn.Sequential(*modules)

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.embed(t)
        # Concatenate the input with the time embedding
        x_embed = torch.cat((x, embed), dim = 1)
        # Apply the MLP
        h = self.mlp(x_embed)
        # Normalize the output
        h = h / (self.marginal_prob_std(t)[:, None] + 1e-5)
        return h

class NegativeGradientMLP(nn.Module):
    """A time-dependent model with MLP architecture that predicts negative gradient of a function."""
    def __init__(self, input_dim, marginal_prob_std, embed_dim=512, layers=None):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.marginal_prob_std = marginal_prob_std
        self.mlp = self.build_mlp(input_dim + embed_dim, layers)

    def build_mlp(self, input_dim, layers):
        modules = []
        for dim in layers:
            modules.append(nn.Linear(input_dim, dim))
            modules.append(nn.GELU())
            input_dim = dim
        modules.append(nn.Linear(input_dim, 1))  # Output layer
        return nn.Sequential(*modules)

    def forward(self, x, t):
        x.requires_grad_(True)
        # Obtain the Gaussian random feature embedding for t
        embed = self.embed(t)
        # Concatenate the input with the time embedding
        x_embed = torch.cat((x, embed), dim = 1)
        # Apply the MLP
        f_x = self.mlp(x_embed)
        # Gradient of f_x
        grad = torch.autograd.grad(f_x.sum(), x, create_graph=True)[0]
        
        # Comoute negative gradient of f_x and normalize output
        neg_grad = -grad / (self.marginal_prob_std(t)[:, None] + 1e-5)

        return neg_grad
