The token embedding dimensions and the positional encoding dimensions are the same. After encoding, they shall be added to store tokens' meanings and positional information!

***Reasons***

The self-attention mechanism can not understand the positional order, we must fuse the positional information to the encoded tokens

***Self-Attention***

**input size: [n, dims]**, where n means the textual length and dims means the dimentions of token vector.

**W_Q size: [dims, n_q]**, where n_q means the number of querys.

**W_K size: [dims, n_k]**, where n_k means the number of keys.

**W_V size: [dims, n_v]**. where n_v means the dims of value.

Q = matmul(input, W_Q), **Q.size = [n, n_q]**.Each line means a token's n_q querys.

K = matmul(input, W_K), **K.size = [n, n_k]**.Each line means a token's n_k keys.

V = matmul(input, W_V), **V.size = [n, n_v]**.Each line means a token's "meaning"

attention = matmul(Q, transpose(K)), **attention.size = [n, n]**. attention_(i, j) means i_th token's querys multiply with j_th token's keys, referring the attention from i_th token to j_th token. **i_th line means i_th token's attention to all the tokens**.

output = matmul(attention, V), **output.size = [n, n_v]**. Each line means an attentioned token!

output_final = c_proj(output), mapping the n_v to dims and interacting multiple attention heads.

***How to segment words***

Using algorithm BPE: Training data is a textual data. The core mechanism is to count the frequency of combinations consist of two conponents in current library. The combination with the highest frequency will be added to current library, and iterate continously until the library is filled.

***How to generate the next token***

input:[n, dims]. Using the last token, [1, dims], mapping it to the probabilities of all tokens in the library. matmul(last token, embedding matrix).

***When to mask the attention and why?***

GPTs' task is predicting the next word, so during the training period, the model can't see the next word and the following words. Otherwise the model will pay much attention to the answer word, causing a failed train. In testing period, the model need to perform like how it had been trained, so the attention is masked.

***Embedding and positional encode***

The embedding matrix is 50257*768, as there are 50257 tokens in the library and each token's vector has a dimension of 768.

The position matrix is 1024*768, as the GPT-2's maximum textual length is 1024.
