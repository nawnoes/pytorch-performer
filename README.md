# performer
My Performer by Pytorch

## Model
### FAVOR+ Attention
: Bidrictional 또는 Unidirectional인 경우 다른 방향성의 어텐션을 가지며
실질적으로는 어텐션을 개선한것과 같다. 이때 어텐션을 개선할때 Kernel Trick을 적용하였으며
이 Kernel Trick을 적용한 어텐션을 FAVOR+ Attention으로 정의할 수 있다.

- $Q'$ 과 $K'$ 은 논문의 2.2와 2.3에서 계산하는 방법을 구했고
- $C := [V 1_L]$과 같다.

```
입력: Q, K, V, isBidirectional  --- Q,K,V는 (max_seq_len, hidden_dim) 모양의 텐서. isBidirectional는 True, False
출력: 1) bidirectional인 경우 bi_attn_hat(Q, K, V) <- [max_seq_len, max_seq_len]
     1) Unidirectional인 경우 uni_attn_hat(Q, K, V) <- [max_seq_len, max_seq_len]값

코드:
  if isBidirectional:
    Buf_1 := (K').Transpose @ C  # 이때 Buf_1은 [M, d+1] 텐서
    Buf_2 := Q' @ Buf_1          # 이때 Buf_2는 [L, d+1] 텐서
  else:
    # 논문의 수식 (11)을 따라서
    # G와 prefix-sum 텐서인 G_PS를 논문의 (11) 식에 따라 계산한다.
    Buf_2 :=[G^PS_{1,:,:}Q'_1 ... G^PS_{L,:,:}Q'_L]  # [B, L, d+1] shape

  [Buf3 buf4] := Buf2    # Buf3은 [L, d], buf4는 [L] 로
  return diag(buf4)^-1Buf3
```

## Test
### Language Model
#### Pertrain
#### Finetuing


 
# References
- [Performer's Fast Attention (FAVOR+) Module](https://github.com/google-research/google-research/tree/master/performer/fast_attention)
- [Google Performer Blog](https://ai.googleblog.com/2020/10/rethinking-attention-with-performers.html)
