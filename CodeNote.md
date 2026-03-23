```python
@dataclass
class A: #定义一类数据
    x: data type = xxx 
```

```python
class GPT(nn.Module)：
    def __init__(self, config): #config是待传入的数据，调用GPT这个类时要显式传入config   
    super().__init__() 
    self.config = config #将传入的config注册为self.config
    self.layer1 = nn.xxxxx
```

```python
#Function: dict() 构造键值对
d = dict(a=1, b=2.5) #即定义键名a，初始值为1 键名b，初始值为2.5
d["a"] = xxx #访问

self.transformer = nn.ModuleDict(dict(wte = nn.Embedding(n1, n2)))
#则是将键名"wte"对应的Embedding矩阵注册为可学习的Module
self.transfoermer.wte(input) #访问
```

```python
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
#h表示transformers的深度，是建立一个可学习的Module列表List，其中有config.n_layer个Block。而每个Block是多头transformer
```

BatchNorm： 逐通道进行归一化，对同一通道的各个样本归一化。

LayerNorm： 逐样本进行归一化，对同一样本的各个通道归一化。

GPT2 先LayerNorm再attn或mlp，归一化位于残差连接内部。

x = x + attn(ln(x)) √

```python
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) .view(1, 1, config.block_size, config.block_size))
self.register_buffer("bias", tensor)   #bias为键名，tensor为待注册的张量 后续使用：self.bias
```

register_buffer是nn.Module的一个函数，用于将tensor注册为模型的缓冲变量 **被模型管理，但不可训练**

tril是lower triangular，下三角矩阵；triu是upper triangular，上三角矩阵

这里形状固定为[block_size, block_size], 后续使用时会根据输入batch的T(tokens数)来裁剪：self.bias[:, :, :T, :T]

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
```

将QKV矩阵合并成大矩阵一起运算随后拆分，GPT-2实现中，n_query=n_key=n_embd

```python
tensor.masked_fill(mask, value) #函数 根据mask来给tensor对应值填上value
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) #att上三角区（除对角线）被填充为-inf
att = F.softmax(att, dim=-1) # softmax over the LAST DIMENSION, not the last two dims
```

做softmax时注意只对**最后一维**进行，表示一个token对其它token的attention的归一化。若对两维一起softmax，则是整张注意力图一起归一化

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

transpose操作只改引索和访问步长，不改数据的存储结构，进行view重构数据结构前需要用contiguous函数使其连续, 而@、Conv2d、Linear、softmax不要求内存连续
       
***GPT forward***

input: idx, targets

idx.size() = [B, T], targets.size() = [B, T]

idx表示B批次T个token的序号，首先根据embedding矩阵将其映射为向量 -> [B, T, n_embd]. 同时进行位置embedding，根据输入样本的T来产生0~T-1的位置引索,再用positional embedding矩阵将其映射 -> [T, n_embd]

二者相加后进入transformer层得到[B, T, vocab_size]的logits，意为每个位置对下一token的预测概率

再与target（记录每个位置的下一token的引索）求loss

**loss函数是用交叉熵损失，而非MSE。cross_entropy(x, y) x.size = [n, p], y.size = [n] 函数先对x进行softmax，随后对n个样本中的每一个样本，y存储了正确答案的引索，通过y找x中正确答案的概率预测值，再求负对数。**

***Data Processing***

tiktoken库：提供预处理好的BPE分词库，**encode为每个token获取对应的id**

文本读取并token化后，截出训练样本

```python
for i in range(0, len(full_encoded) - self.block_size, self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1] # +1 for the target token
            self.encoded_data.append(chunk)
i from 0 to (text_length-block_size), with a stride of block_size
```

每次截i~i+block_size+1的长度，+1是为了获取targets（targets总是滞后一个token）,将其全部放入self.encoded_data内

self.encoded_data内每个引索存的长为block_size+1的张量，前block_size是x，训练数据；后block_size是y，targets

MyDataset返回一个列表，列表每一项是一个元组(x, y)

***training script***

数据流：

dataset = MyDataset('input.txt') #[n_chunks] -> 求Dataset长度，90%用于训练，10%用于验证。随机分为train_data #[n_chunks - 0.1 * n_chunks] 和 val_data #[0.1 * n_chunks]

使用DataLoader将train_data和val_data分成batch， 每个batch数据[B, （x, y）]

得到train_loader和val_loader

```python
for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
```
每个batch内，取出batch_size个(x, y)，x.size() = [B, T]， y.size() = [B, T], 送入model

算loss时是算一个batch的平均loss，将所有batch的loss求总和再除以len(train/val_loader)得到每个epoch训练/验证的平均loss
