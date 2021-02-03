# coding=utf-8
# Performer pytorch
# writen by Seonghwan Kim
# Performer References
# Google Research Github: https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py
# teddykoker's numpy performer https://github.com/teddykoker/performer/blob/main/performer.py
# https://github.com/lucidrains/performer-pytorch


import torch
import torch.nn as nn

# generate IID Gaussian random features
def iid_gaussian(m, d):
    return torch.normal(0.0, 1.0, size=(m,d))

# generate orthogonal Gaussian random features
def orthogonal_gaussian_random_feature(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = torch.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = torch.cat(blocks)
    matrix /= torch.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix

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
  data_normalizer = 1.0 / (torch.sqrt(torch.sqrt(data.shape[-1])))
  data = data_normalizer * data
  ratio = 1.0 / torch.sqrt(projection_matrix.shape[0])
  data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)

  diag_data = torch.square(data)
  diag_data = torch.sum(diag_data, dim=-1) # 확인 필요.
  diag_data = diag_data / 2.0
  diag_data = diag_data.unsqueeze(dim=-1)

  if is_query:
    data_dash = ratio * (
        torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdims=True).values) + numerical_stabilizer)
  else:
    data_dash = ratio * (
        torch.exp(data_dash - diag_data - torch.max(data_dash)) + numerical_stabilizer)

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


def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
    result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

  result = torch.cat(result, dim=0)

  return result

def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = torch.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(torch.sum(qs[index] * sums, dim=2)[None, Ellipsis])

  result = torch.cat(result, dim=0)

  return result

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
    av_attn = causal_numerator(query_prime, key_prime, value)
    attn_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attn = noncausal_numerator(query_prime,key_prime, value)
    attn_normalizer = noncausal_denominator(query_prime,key_prime)

  av_attn = torch.transpose(av_attn,0,1)
  attn_normalizer = torch.transpose(attn_normalizer,0,1)
  attn_normalizer = attn_normalizer.unsqueeze(-1)

  return av_attn/attn_normalizer


if __name__=="__main__":
  pass
