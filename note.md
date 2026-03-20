The token embedding dimensions and the positional encoding dimensions are the same. After encoding, they shall be added to store tokens' meanings and positional information!

**Reasons**

The self-attention mechanism can not understand the positional order, we must fuse the positional information to the encoded tokens

**Self-Attention**

**input size: [n, dims]**, where n means the textual length and dims means the dimentions of token vector.

**W_Q size: [dims, n_q]**, where n_q means the number of querys.

**W_K size: [dims, n_k]**, where n_k means the number of keys.

**W_V size: [dims, dims]**.

Q = matmul(input, W_Q), **Q.size = [n, n_q]**.Each line means a token's n_q querys.

K = matmul(input, W_K), **K.size = [n, n_k]**.Each line means a token's n_k keys.

V = matmul(input, W_V), **V.size = [n, dims]**.Each line means a token's "meaning"

attention = matmul(Q, transpose(K)), **attention.size = [n, n]**. attention_(i, j) means i_th token's querys multiply with j_th token's keys, referring the attention from i_th token to j_th token. ***i_th line means i_th token's attention to all the tokens***.

output = matmul(attention, V), **output.size = [n, dims]**. Each line means an attentioned token!

