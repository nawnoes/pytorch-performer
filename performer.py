import math
import torch
import numpy as np
import torch.nn as nn

class FAVORAttention(nn.Module):
  def __init__(self):
    pass

  def z_sin_cos(x, omega):
    pi = np.pi
    sin = lambda x: torch.sin(2 * pi * x)
    cos = lambda x: torch.cos(2 * pi * x)

    coef = torch.exp(x.pow(2).sum(dim=-1, keepdims=True) / 2)
    product = np.einsum("...d,rd->...r", x, omega)
    return coef * np.concatenate([sin(product), cos(product)], axis=-1)
  def forward(self):
    pass

def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.
  ReLU kernel에 대한 Random Features를 계산 from https://arxiv.org/pdf/2009.14794.pdf.
  Args:
    data: 입력데이터 텐서, the shape [B, L, H, D], where: B - batch
                       dimension, L - attention dimensions, H - heads, D - features.
    is_query: 입력데이터가 쿼리인지 아닌지, 쿼리 또는 키인지를 나타낸다. indicates whether input data is a query oor key tensor.
    projection_matrix: [M,D] 모양을 가진 랜덤 가우시안 매트릭스 random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: 수치 안정성을 위한 작은 값의 양의 상수
  Returns:
    대응되는 kernel feature map 반.
  """
  del is_query
  if projection_matrix is None:
    return nn.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / torch.sqrt(projection_matrix.shape[0].float())
    data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
    return nn.relu(data_dash) + numerical_stabilizer

def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """
  FAVOR+ 메커니즘을 사용하여 softmax kernel에 대한 random feature 계산
  :param data: 입력 텐서. [B,L,H,D] B-batch dimension, L- attention dimensions,
              H- Heads, D- features
  :param is_query: 입력 값이 쿼리 또는 인지 나타내는 값
  :param projection_matrix: [M, D]의 모양을 가진 랜덤 가우시안 매트릭스
               M - M은 Random Feature 수를 의미하며,
               각각의 [D,D] 서브 블록은 pairwise orthogonal rows를 가진다.
  :param numerical_stabilizer: 수치 안정성을 위한 작은 값의 양의 상수
  :return:
  """
  data_normalizer = 1.0 / (torch.sqrt(torch.sqrt(data.shape[-1].float())))
  data = data_normalizer * data
  ratio = 1.0 / torch.sqrt(projection_matrix.shape[0].float())
  data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
  diag_data = torch.square(data)
  diag_data = torch.sum(diag_data, dim=-1) # 확인 필요.
  diag_data = diag_data / 2.0
  diag_data = diag_data.unsqueeze(dim=-1)
  last_dims_t = (len(data_dash.shape) - 1,)
  attention_dims_t = (len(data_dash.shape) - 3,)
  if is_query:
    data_dash = ratio * (
        torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdims=True)) + numerical_stabilizer)
  else:
    # torch.max 부분 수정 필요
    """
    data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
        numerical_stabilizer)
    """
    data_dash = ratio * (
        torch.exp(data_dash - diag_data - torch.max(data_dash, axis=-1, keepdims=True)) + numerical_stabilizer)

  return data_dash

def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
  return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)

def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in noncausal attention.
  """
  all_ones = torch.ones([ks.shape[0]])
  ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
  return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)

def favor_attention(query, key, value,
                    kernel_transformation,
                    causal,
                    projection_matrix = None):
  """
  favor_attention 계산
  :param query: 쿼리
  :param key: 키
  :param value: 밸류
  :param kernel_transformation: kernel feature를 얻기 위한 tranformation.
         relu_kernel_transformation나 softmax_kernel_transformation 사용.
  :param causal: causl or not
  :param projection_matrix: 사용될 projection matrix

  :return: Favor+ normalized attention
  """
  # Kernel Transformation
  query_prime = kernel_transformation(query, True, projection_matrix) # [B, L, H, M]
  key_prime = kernel_transformation(key,False,projection_matrix) # [B,L,H,M]

  # Transpose
  query_prime = torch.transpose(query_prime,0,1) # [L,B,H,M]
  key_prime = torch.transpose(key_prime,0,1)   # [L,B,H,M]
  value = torch.transpose(value,0,1) # [L,B,H,D]

  # Causal or Not
  if causal:
    # 구현 필요
    pass
  else:
    av_attn = noncausal_numerator(query_prime,key_prime, value)
    attn_normalizer = noncausal_denominator(query_prime,key_prime)
  av_attn = torch.transpose(av_attn,0,1)
  attn_normalizer = torch.transpose(attn_normalizer,0,1)
  attn_normalizer = attn_normalizer.unsqueeze(-1)

  return av_attn/attn_normalizer


if __name__=="__main__":
  pass
