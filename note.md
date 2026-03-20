The token embedding dimensions and the positional encoding dimensions are the same. After encoding, they shall be added to store tokens' meanings and positional information!

**Reasons**

The self-attention mechanism can not understand the positional order, we must fuse the positional information to the encoded tokens
