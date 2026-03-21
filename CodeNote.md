@dataclass
class A: #定义一类数据
    x: data type = xxx 

class GPT(nn.Module)：
    def __init__(self, config): #config是待传入的数据，调用GPT这个类时要显式传入config
    super().__init__()
    self.config = config #将传入的config注册为self.config
    self.layer1 = nn.xxxxx

Function: dict() 构造键值对
d = dict(a=1, b=2.5)即定义键名a，初始值为1 键名b，初始值为2.5
访问：d["a"] = xxx

self.transformer = nn.ModuleDict(dict(wte = nn.Embedding(n1, n2)))
则是将键名"wte"对应的Embedding矩阵注册为可学习的Module
访问: self.transfoermer["wte"](input)

h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
h表示transformers的深度，是建立一个可学习的Module列表List，其中有config.n_layer个Block。而每个Block是多头transformer

BatchNorm： 逐通道进行归一化，对同一通道的各个样本归一化。
LayerNorm： 逐样本进行归一化，对同一样本的各个通道归一化。
GPT2 先LayerNorm再attn或mlp，归一化位于残差连接内部。
x = x + attn(ln(x)) √

