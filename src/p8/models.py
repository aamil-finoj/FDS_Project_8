"""PyTorch models.

* MSENet           : x -> single point prediction t_hat (trained with squared error)
* MDN(M)           : x -> (log pi, mu, v) of an M-component Gaussian mixture over t
* single Gaussian  : simply MDN(M=1) -> N(mu(x), v(x)), trained with Gaussian NLL

Both networks standardise x internally with the training-set mean/std (stored as
buffers), so callers always pass raw x and the scaling is applied identically in
training, evaluation and plotting.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_trunk(hidden: int, n_layers: int) -> nn.Sequential:
    """Small MLP: 1 -> hidden -> ... -> hidden, tanh activations."""
    layers, d_in = [], 1
    for _ in range(n_layers):
        layers += [nn.Linear(d_in, hidden), nn.Tanh()]
        d_in = hidden
    return nn.Sequential(*layers)


class _Standardise(nn.Module):
    def __init__(self, x_mean: float, x_std: float):
        super().__init__()
        self.register_buffer("x_mean", torch.tensor(float(x_mean)))
        self.register_buffer("x_std", torch.tensor(float(x_std)))

    def forward(self, x):
        return ((x - self.x_mean) / self.x_std).unsqueeze(-1)


class MSENet(nn.Module):
    """Point predictor f_theta(x) trained with mean squared error."""

    def __init__(self, x_mean, x_std, hidden=64, n_layers=3):
        super().__init__()
        self.scale = _Standardise(x_mean, x_std)
        self.trunk = make_trunk(hidden, n_layers)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(self.trunk(self.scale(x))).squeeze(-1)


class MDN(nn.Module):
    """Mixture density network p(t | x) = sum_j pi_j(x) N(t | mu_j(x), v_j(x)).

    Heads on a shared trunk:
      alpha_j(x) -> pi = softmax(alpha)            (weights >= 0, sum to 1)
      mu_j(x)    -> component means (unconstrained)
      s_j(x)     -> v = softplus(s) + var_floor    (VARIANCES, strictly > var_floor)
    """

    def __init__(self, x_mean, x_std, M=3, hidden=64, n_layers=3, var_floor=1e-4,
                 mu_init="default", a=2.0):
        super().__init__()
        self.M, self.var_floor = M, var_floor
        self.scale = _Standardise(x_mean, x_std)
        self.trunk = make_trunk(hidden, n_layers)
        self.logit_head = nn.Linear(hidden, M)
        self.mu_head = nn.Linear(hidden, M)
        self.var_head = nn.Linear(hidden, M)
        if mu_init == "spread":
            # Start the component means evenly spread over the prior support, with small
            # input weights, so every component begins near a different part of [-a, a].
            with torch.no_grad():
                self.mu_head.weight.mul_(0.1)
                centres = torch.linspace(-a, a, M) if M > 1 else torch.zeros(1)
                self.mu_head.bias.copy_(centres)
        elif mu_init != "default":
            raise ValueError(f"unknown mu_init {mu_init!r}")

    def forward(self, x):
        h = self.trunk(self.scale(x))
        log_pi = F.log_softmax(self.logit_head(h), dim=-1)
        mu = self.mu_head(h)
        var = F.softplus(self.var_head(h)) + self.var_floor
        return log_pi, mu, var

    def log_prob(self, t, x):
        """log p(t_i | x_i) for paired 1-D tensors, computed stably with logsumexp."""
        log_pi, mu, var = self(x)
        return mixture_log_prob(t.unsqueeze(-1), log_pi, mu, var)


def gaussian_log_pdf(t, mu, var):
    """log N(t | mu, var) - note: var is a VARIANCE, not a std."""
    return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (t - mu) ** 2 / var)


def mixture_log_prob(t, log_pi, mu, var):
    """log sum_j pi_j N(t | mu_j, v_j) = logsumexp_j [log pi_j + log N_j]; broadcasts over last dim."""
    return torch.logsumexp(log_pi + gaussian_log_pdf(t, mu, var), dim=-1)


def mdn_nll_loss(model: MDN, x, t):
    """Average negative log-likelihood  -(1/N) sum_i log p_theta(t_i | x_i)."""
    return -model.log_prob(t, x).mean()


def mse_loss(model: MSENet, x, t):
    return F.mse_loss(model(x), t)
