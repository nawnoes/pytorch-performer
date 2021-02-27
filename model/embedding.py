import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class Embeddings(nn.Module):
  def __init__(self, vocab_size, dim):
    super(Embeddings,self).__init__()
    self.emb = nn.Embedding(vocab_size,dim)
    self.dim = dim
  def forward(self, x):
    """
    1) 임베딩 값에 math.sqrt(self.d_model)을 곱해주는 이유는 무엇인지 찾아볼것
    2) nn.Embedding에 다시 한번 찾아볼것
    """
    return self.emb(x) * math.sqrt(self.dim)

class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)

class PositionalEncoding(nn.Module):
  def __init__(self, max_seq_len, d_model,dropout=0.1):
    super(PositionalEncoding,self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_seq_len, d_model)

    position = torch.arange(0,max_seq_len).unsqueeze(1)
    base = torch.ones(d_model//2).fill_(10000)
    pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model,dtype=torch.float32)
    div_term = torch.pow(base,pow_term)

    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)

    pe = pe.unsqueeze(0)

    # pe를 학습되지 않는 변수로 등록
    self.register_buffer('positional_encoding', pe)

  def forward(self, x):
    x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)