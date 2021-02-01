


"""

FAVOR+ Attention
: Bidrictional 또는 Unidirectional인 경우 다른 방향성의 어텐션을 가지며
실질적으로는 어텐션을 개선한것과 같다. 이때 어텐션을 개선할때 Kernel Trick을 적용하였으며
이 Kernel Trickㅇ을 적용한 어텐션을 FAVOR+ Attention으로 정의할 수 있다.

Q' 과 K' 은 논문의 2.2와 2.3에서 계산하는 방법을 구했고
C := [V 1_L]과 같다.

입력: Q, K, V, isBidirectional  --- Q,K,V는 (max_seq_len, hidden_dim) 모양의 텐서. isBidirectional는 True, False
출력: 1) bidirectional인 경우 bi_attn_hat(Q, K, V) <- (max_seq_len, max_seq_len)
     1) Unidirectional인 경우 uni_attn_hat(Q, K, V) <- (max_seq_len, max_seq_len)값

코드:
  if isBidirectional:
    Buf_1 := (K').Transpose @ C  # 이때 Buf_1은 (M, d+1) 텐서
    Buf_2 := Q' @ Buf_1          # 이때 Buf_2는 (L, d+1) 텐서
  else:
    # 논문의 수식 (11)을 따라서
    # G와 prefix-sum 텐서인 G^PS를 계산한다
    Buf_2 :=[]

"""