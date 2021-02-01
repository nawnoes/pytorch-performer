import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from util import clones

# Numpy z_sin_cos
# def z_sin_cos(x, omega):
#   sin = lambda x: np.sin(2 * np.pi * x)
#   cos = lambda x: np.cos(2 * np.pi * x)
#
#   coef = np.exp(np.square(x).sum(axis=-1, keepdims=True) / 2)
#   product = np.einsum("...d,rd->...r", x, omega)
#   return coef * np.concatenate([sin(product), cos(product)], axis=-1)

def z_sin_cos(x, omega):
  pi = np.pi
  sin = lambda x: torch.sin(2 * pi * x)
  cos = lambda x: torch.cos(2 * pi * x)

  coef = torch.exp(x.pow(2).sum(dim=-1, keepdims=True) / 2)
  product = torch.einsum("...d,rd->...r", x, omega)
  return coef * torch.cat([sin(product), cos(product)], dim=-1)

def attention_hat(q, k, v, random_dim):
  l, d = q.shape
  normalizer = 1 / (d ** 0.25)  # to normalize before multiplication
  omega = torch.randn(random_dim, d)  # generate i.i.d. gaussian features
  q_prime = z_sin_cos(q * normalizer, omega)  # apply feature map z to Q
  k_prime = z_sin_cos(k * normalizer, omega)  # apply feature map z to K

  # rest of attention (note the order of operations is changed for efficiency)
  d_inv = torch.diag(1 / (q_prime @ (k_prime.T @ torch.ones(l))))
  return d_inv @ (q_prime @ (k_prime.T @ v))

def z_positive(x, omega):
  coef = torch.exp(-x.pow(2).sum(dim=-1, keepdims=True) / 2)
  product = torch.einsum("...d,rd->...r", x, omega)
  return coef * torch.exp(product)


dim = 4
max_seq_len = 10
random_dim = 8
x = torch.randn(max_seq_len, dim)
omega = torch.randn(random_dim,dim)

result = z_sin_cos(x,omega)
print(result)

