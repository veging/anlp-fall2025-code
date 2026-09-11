# 对话知识总结 - 神经网络文本分类与 PyTorch（2026-09-03）

## 目录

- [代码实现细节](#代码实现细节)
- [语法知识点](#语法知识点)
- [常见错误与调试](#常见错误与调试)
- [核心概念与原理](#核心概念与原理)
- [最佳实践与建议](#最佳实践与建议)
- [索引](#索引)

## 代码实现细节

### SentencePiece tokenizer 训练

下面的代码从 JSONL 数据中提取文本，并训练 BPE tokenizer。

```python
import sentencepiece as spm
import json
import os

with open("bow_tokenizer_txt.txt", "w", encoding="utf-8") as output_file:
    with open("train.jsonl", "r", encoding="utf-8") as input_file:
        for line in input_file:
            example = json.loads(line)
            output_file.write(example["text"] + "\n")

options = dict(
    input="bow_tokenizer_txt.txt",
    input_format="text",
    model_prefix="bow_tok",
    model_type="bpe",
    vocab_size=2048,
    byte_fallback=True,
    num_threads=os.cpu_count(),
)

spm.SentencePieceTrainer.train(**options)
```

参数含义：

- `input`：训练 tokenizer 使用的文本文件。
- `input_format`：输入文件是普通文本。
- `model_prefix`：输出文件前缀，通常生成 `bow_tok.model` 和 `bow_tok.vocab`。
- `model_type="bpe"`：使用 BPE 子词算法。
- `vocab_size`：词表大小上限。
- `byte_fallback=True`：遇到未知字符时退化为字节表示。
- `num_threads`：使用的 CPU 线程数。

为什么这样写：文本需要先转换为 tokenizer 可以读取的格式；BPE 能把未知词拆成已知子词，减少未知词问题。

### 加载 tokenizer 和查看词表

下面的代码加载训练好的 SentencePiece 模型，并查看部分词表。

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("bow_tok.model")

vocab = [
    [sp.id_to_piece(index), index]
    for index in range(sp.get_piece_size())
]

vocab[1000:1020]
```

`get_piece_size()` 获取词表大小，`id_to_piece()` 把 token 编号转换成 token。必须先成功训练 tokenizer，才能加载 `bow_tok.model`。

### 数据读取、分词和划分

下面的代码逐行读取 JSONL，提取文本和标签，并将文本转换为 token 编号。

```python
import json
import random

random.seed(123)
label_to_text = {}

def read_dataset(filename):
    with open(filename, "r", encoding="utf-8") as data_file:
        for line in data_file:
            example = json.loads(line)
            text = example["text"]
            label = example["label"]
            label_to_text[label] = example["label_text"]
            tokens = sp.encode(text)
            yield tokens, label

dataset = list(read_dataset("train.jsonl"))
random.shuffle(dataset)

train = dataset[:-1000]
dev = dataset[-1000:]

nwords = len(sp)
ntags = 3
```

为什么这样写：`json.loads()` 将每行 JSON 转为字典，`sp.encode()` 将文本转为 token 编号，`yield` 逐条产生 `(tokens, label)`。`label_to_text` 保存数字标签和文字标签的对应关系。

> 注意：`dev = ds[1000:]` 会与 `train = ds[:-1000]` 大量重叠；正确的最后 1000 条验证集写法是 `dev = ds[-1000:]`。

### One-hot 和 embedding

下面的代码将 token 编号转换为 one-hot 向量。

```python
import torch

vocab_size = 10
token_ids = torch.tensor([2, 5, 1], dtype=torch.long)
one_hot_vectors = torch.nn.functional.one_hot(
    token_ids,
    num_classes=vocab_size,
)
print(one_hot_vectors.shape)
```

如果词表大小为 10，token 编号为 2，则长度为 10 的向量中只有索引 2 的位置为 1。

下面的代码展示 one-hot 向量如何通过权重矩阵变成 embedding。

```python
import torch
import torch.nn as nn

vocab_size = 10
embedding_size = 4
weight = nn.Parameter(torch.randn(vocab_size, embedding_size))

token_ids = torch.tensor([2, 5, 1], dtype=torch.long)
one_hot_vectors = torch.nn.functional.one_hot(
    token_ids,
    num_classes=vocab_size,
).float()

embeddings = torch.matmul(one_hot_vectors, weight)
print(embeddings.shape)
```

形状为：`[3, vocab_size] @ [vocab_size, embedding_size] = [3, embedding_size]`。one-hot 向量与矩阵相乘，本质上就是选取权重矩阵中对应 token 的那一行。

### 自定义 Embedding 层

下面手动实现一个 embedding 层，用于理解 embedding 的工作原理。

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, emb_size))
        self.vocab_size = vocab_size
        nn.init.xavier_uniform_(self.weight)

    def forward(self, token_ids):
        one_hot_vectors = torch.nn.functional.one_hot(
            token_ids,
            num_classes=self.vocab_size,
        ).float()
        return torch.matmul(one_hot_vectors, self.weight)
```

`vocab_size` 是 token 总数，`emb_size` 是每个 token 向量的维度。实际项目中通常直接使用 `nn.Embedding`，效率更高。

### BoW 模型

下面的模型将所有 token 的 embedding 相加，然后输出分类分数。

```python
class BoW(torch.nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(BoW, self).__init__()
        self.embedding = Embedding(vocab_size, num_labels)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        sentence_vector = torch.sum(embeddings, dim=0)
        logits = sentence_vector.view(1, -1)
        return logits
```

形状变化：

```text
tokens:          [sentence_length]
embeddings:      [sentence_length, num_labels]
sentence_vector: [num_labels]
logits:          [1, num_labels]
```

为什么这样写：求和把一句话压缩为一个向量，但也会丢失词序信息；`view(1, -1)` 将向量调整成“一个样本、多个类别”的格式。

### CBoW 模型

下面的模型使用独立的 embedding 维度和输出层。

```python
class CBoW(torch.nn.Module):
    def __init__(self, vocab_size, num_labels, emb_size):
        super(CBoW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.output_layer = nn.Linear(emb_size, num_labels)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        embedding_sum = torch.sum(embeddings, dim=0)
        hidden = embedding_sum.view(1, -1)
        logits = self.output_layer(hidden)
        return logits
```

`emb_size` 表示每个 token 的 embedding 维度，不等于类别数；`num_labels` 表示最终的分类类别数。

### 交叉熵损失

下面是交叉熵损失的手动实现。

```python
def ce_loss(logits, target):
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss = -log_probs[:, target]
    return loss
```

`log_softmax()` 将 logits 转为对数概率，`target` 指定真实类别，负号使模型倾向于提高真实类别的概率。也可以直接使用：

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
```

### 训练循环

下面的代码逐条训练样本并更新参数。

```python
import random
import time

model = CBoW(nwords, ntags, emb_size=32)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for iteration in range(5):
    random.shuffle(train)
    train_loss = 0.0
    start_time = time.time()
    model.train()

    for tokens, label in train:
        tokens = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor([label])

        logits = model(tokens)
        loss = criterion(logits, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        "iteration %r: loss=%.4f, time=%.2fs"
        % (iteration, train_loss / len(train), time.time() - start_time)
    )
```

训练的核心顺序是：前向计算 logits，计算损失，清除旧梯度，反向传播，更新参数。

### 验证准确率

下面的代码使用开发集评估模型。

```python
model.eval()
correct_count = 0

with torch.no_grad():
    for tokens, label in dev:
        tokens = torch.tensor(tokens, dtype=torch.long)
        logits = model(tokens)[0]
        prediction = logits.argmax().item()

        if prediction == label:
            correct_count += 1

dev_accuracy = correct_count / len(dev)
print("dev accuracy=%.4f" % dev_accuracy)
```

`argmax()` 返回最大 logit 所在的类别索引；将该索引与真实标签比较即可计算准确率。

### 深层 CBoW 和 batching

Deep CBoW 在求和后增加隐藏层和 `tanh` 非线性：

```text
embedding -> sum -> Linear -> tanh -> Linear -> logits
```

增加少量层可能增强表达能力，但过深会增加训练时间、过拟合风险和优化难度；它仍然无法恢复 BoW 已经丢失的词序信息。

不同长度的文本进行 batching 时，需要使用 `[PAD]` 补齐并使用 mask：

```python
mask = tokens != pad_id
embeddings = self.embedding(tokens)
embeddings = embeddings * mask.unsqueeze(-1)
sentence_vectors = embeddings.sum(dim=1)
logits = self.output_layer(sentence_vectors)
```

形状通常为：

```text
tokens:           [batch_size, max_length]
embeddings:       [batch_size, max_length, emb_size]
mask:             [batch_size, max_length]
sentence_vectors: [batch_size, emb_size]
logits:           [batch_size, num_labels]
```

为什么需要 mask：`[PAD]` 只是补齐长度，不应影响句子表示。

**关键词**：BPE、BoW、CBoW、Embedding、JSONL、SentencePiece、tokenizer、训练循环、交叉熵、验证集、padding、mask

## 语法知识点

### `**` 字典参数展开

```python
def describe(name, age):
    print(name, age)

options = {"name": "Alice", "age": 20}
describe(**options)
```

`**options` 将字典键值展开为关键字参数。

### `yield` 生成器

```python
def numbers():
    yield 1
    yield 2

for number in numbers():
    print(number)
```

`yield` 使函数成为生成器，每次请求时产生一个结果。

### 列表推导式

```python
squares = [number * number for number in range(5)]
print(squares)
```

列表推导式用简洁语法生成列表。

### 切片

```python
items = [0, 1, 2, 3, 4]
print(items[:-2])
print(items[-2:])
```

`[:-2]` 表示去掉最后两个元素，`[-2:]` 表示取最后两个元素。

### `range()`

```python
for index in range(3):
    print(index)
```

`range(3)` 产生 0、1、2，不包含 3。

### `argmax()` 和 `item()`

```python
import torch

scores = torch.tensor([1.2, -0.4, 0.8])
index = scores.argmax().item()
print(index)
```

`argmax()` 返回最大值的索引，`item()` 将单元素 PyTorch 张量转为 Python 数字。

### `view(1, -1)`

```python
import torch

vector = torch.tensor([1.0, 2.0, 3.0])
matrix = vector.view(1, -1)
print(matrix.shape)
```

`view()` 调整张量形状，`-1` 表示自动推断该维度。

### `dim` 参数

```python
import torch

matrix = torch.tensor([[1, 2], [3, 4]])
print(torch.sum(matrix, dim=0))
print(torch.sum(matrix, dim=1))
```

`dim=0` 按列聚合，`dim=1` 按行聚合。

### 原地操作和下划线

```python
import torch

values = torch.tensor([1.0, 2.0])
values.zero_()
print(values)
```

PyTorch 中以 `_` 结尾的方法通常直接修改原对象，例如 `xavier_uniform_()`。

### `dtype=torch.long`

```python
import torch

token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
print(token_ids.dtype)
```

Embedding 索引通常必须使用整数类型，常用 `torch.long`。

### `%` 字符串格式化

```python
iteration = 2
accuracy = 0.85
print("iteration %r: accuracy=%.4f" % (iteration, accuracy))
```

`%r` 显示对象表示形式，`%.4f` 保留四位小数。

### `with` 上下文管理器

```python
with open("example.txt", "w", encoding="utf-8") as file:
    file.write("hello")
```

`with` 代码块结束后会自动关闭文件。

### `super()`

```python
class Parent:
    def __init__(self):
        self.value = 1

class Child(Parent):
    def __init__(self):
        super().__init__()

print(Child().value)
```

`super()` 调用父类方法，在 PyTorch 模型中用于初始化 `nn.Module`。

### `detach()`

```python
import torch

value = torch.tensor(2.0, requires_grad=True)
detached_value = value.detach()
print(detached_value.requires_grad)
```

`detach()` 将张量从计算图中分离，常用于验证或查看结果。

### `model.train()` 和 `model.eval()`

```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
model.train()
model.eval()
```

这两个方法切换模型模式，不会分别自动开始训练或自动评估。

### 分号

```python
value = 1;
print(value)
```

Python 允许语句末尾使用分号，但通常不需要。

**关键词**：argmax、detach、dtype、dim、item、range、super、view、yield、with、列表推导式、参数展开、切片、原地操作

## 常见错误与调试

### 训练集和开发集重叠

错误写法：

```python
train = ds[:-1000]
dev = ds[1000:]
```

错误原因：两个切片的范围重叠，导致验证集包含训练样本，dev accuracy 可能虚高。

解决方法：

```python
train = ds[:-1000]
dev = ds[-1000:]
```

### `train.jsonl` 找不到

曾出现：

```text
Exit Code: 1
```

错误原因：当前目录是仓库根目录，而文件位于 `02_wordrep_classification/train.jsonl`；同时 `head` 是 Unix 命令，在 PowerShell 中不一定可用。

解决方法：

```powershell
Get-Content .\anlp-fall2025-code\02_wordrep_classification\train.jsonl -TotalCount 4
```

或在 Python 中使用正确路径：

```python
from pathlib import Path

path = Path("02_wordrep_classification") / "train.jsonl"
print(path.exists())
```

### tokenizer 模型不存在

错误现象：

```python
sp.load("bow_tok.model")
```

提示文件不存在。

错误原因：前面的 SentencePiece 训练单元没有成功执行。

解决方法：依次执行提取文本、训练 tokenizer、加载模型，并检查：

```python
import os
print(os.path.exists("bow_tok.model"))
print(os.path.exists("bow_tok.vocab"))
```

### one-hot 或 Embedding 类型错误

错误原因：token 是浮点数，或 token 编号超出 `num_classes` 范围。

正确写法：

```python
import torch

tokens = torch.tensor([1, 3, 5], dtype=torch.long)
vectors = torch.nn.functional.one_hot(tokens, num_classes=10)
```

### logits 和标签形状不匹配

单样本分类通常要求：

```text
logits: [1, num_labels]
target: [1]
```

最小示例：

```python
import torch
import torch.nn as nn

logits = torch.tensor([[1.2, -0.4, 0.8]])
target = torch.tensor([2])
loss = nn.CrossEntropyLoss()(logits, target)
print(loss)
```

标签应是类别索引，而不是普通浮点数。

### 验证阶段没有关闭梯度

问题：验证时仍可能构建计算图，浪费内存和计算。

解决方法：

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
```

### 混淆类别索引和 token 位置

`logits.argmax()` 返回的是类别索引，不是输入文本中 token 的位置：

```text
logits[0] -> 类别 0
logits[1] -> 类别 1
logits[2] -> 类别 2
```

标签对应关系来自数据集中的 `label` 字段；文字名称由 `label_to_text[label]` 保存。

**关键词**：Exit Code 1、JSONL 路径、dev 重叠、dtype 错误、文件不存在、标签形状、类别索引、梯度关闭、PowerShell

## 核心概念与原理

### Token、词表和 embedding

文本需要经过以下转换：

```text
文本 -> token -> token 编号 -> embedding 向量
```

例如：

```text
"I love NLP" -> ["▁I", "▁love", "▁NLP"] -> [12, 45, 817]
```

如果词表大小为 2048、embedding 维度为 32，embedding 矩阵形状就是 `[2048, 32]`。

### `emb_size`

`emb_size` 是每个 token 的 embedding 向量维度，不是 token 的数量，也不是类别数量。

- `vocab_size`：token 总数。
- `emb_size`：每个 token 向量的长度。
- `num_labels`：类别数。

### One-hot 与 embedding

one-hot 只表示 token 的身份，不表示语义。one-hot 与 embedding 矩阵相乘后，结果就是对应 token 的 embedding 行：

```text
one-hot token i × embedding matrix = embedding matrix 的第 i 行
```

### BoW 的优缺点

BoW 将所有 token 的向量相加，因此忽略词序。

优点：结构简单、速度快、适合作为文本分类基线。

缺点：难以理解否定、讽刺和复杂上下文；相同词集合的句子可能得到相近表示。

### logits、概率和预测

logits 是类别的原始分数，不是概率：

```python
logits = [1.2, -0.4, 0.8]
prediction = 0
```

预测类别是最大分数所在的索引。softmax 概率为：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### 标签索引的来源

数据集已经提供类别编号：

```json
{"text": "I love this!", "label": 2, "label_text": "positive"}
```

模型输出向量的位置与标签编号约定一致：`logits[0]` 对应标签 0，`logits[1]` 对应标签 1，`logits[2]` 对应标签 2。

因此可以直接比较：

```python
prediction == label
```

### 反向传播如何使用 `y`

标签不会直接传给模型：

```python
logits = model(x)
```

标签传给损失函数：

```python
loss = criterion(logits, y)
```

随后 `loss.backward()` 根据包含真实标签的损失计算梯度，`optimizer.step()` 更新参数。

> `y` 通过损失函数影响反向传播，但不直接作为 `forward()` 的输入。

### `model.train()` 和 `model.eval()`

`model.train()` 切换训练模式，主要影响 Dropout 和 BatchNorm；它不会自动训练。`model.eval()` 切换评估模式，也不会自动评估。

训练时需要：

```python
model.train()
loss.backward()
optimizer.step()
```

验证时需要：

```python
model.eval()
with torch.no_grad():
    ...
```

当前 CBoW 只有 Embedding 和 Linear，因此模式切换对当前计算几乎没有影响，但仍应规范使用。

### Xavier 初始化

```python
nn.init.xavier_uniform_(layer.weight)
```

Xavier 初始化根据输入和输出维度选择合适的随机范围，有助于稳定信号和梯度传播。初始化方式会影响初始 loss、收敛速度和最终 dev accuracy。

### 超参数实验趋势

- 学习率太小：收敛慢。
- 学习率太大：loss 震荡或发散。
- embedding 或 hidden size 太小：表达能力不足。
- embedding 或 hidden size 太大：训练更慢，可能过拟合。
- epoch 太少：欠拟合。
- epoch 太多：可能过拟合。

实验时应一次只改变一个变量，并同时观察 train loss 和 dev accuracy。

### 深层 CBoW

增加隐藏层和 `tanh` 可以提升非线性表达能力，但过深会导致训练更慢、收益变小、过拟合或优化困难。由于 BoW 已经丢失词序，增加层数无法恢复词序信息。

### Batching、padding 和 mask

不同长度的文本不能直接组成规则矩阵，需要用 `[PAD]` 补齐。mask 用于让 padding 不参与求和：

```python
mask = tokens != pad_id
embeddings = self.embedding(tokens)
embeddings = embeddings * mask.unsqueeze(-1)
sentence_vectors = embeddings.sum(dim=1)
```

**关键词**：BoW、CBoW、Embedding、Xavier、反向传播、类别索引、交叉熵、logits、padding、mask、softmax、超参数

## 最佳实践与建议

1. **正确划分数据集**：训练集和开发集必须互斥，推荐 `train = ds[:-1000]`、`dev = ds[-1000:]`。
2. **固定随机种子**：使用 `random.seed(123)` 便于复现实验。
3. **正确切换模式**：训练使用 `model.train()`，验证使用 `model.eval()` 和 `torch.no_grad()`。
4. **检查张量形状**：单样本分类通常为 `tokens: [length]`、`logits: [1, num_labels]`、`target: [1]`。
5. **保持标签编号一致**：必须满足 `0 <= label < num_labels`，模型输出位置和数据标签含义一致。
6. **优先使用内置 embedding**：自定义 one-hot 实现适合理解原理，实际项目使用 `nn.Embedding` 更高效。
7. **一次只改变一个实验变量**：比较学习率、层数或 embedding 大小时固定其他设置。
8. **同时观察 loss 和 accuracy**：train loss 下降不代表泛化能力一定提升。
9. **按依赖顺序运行 Notebook**：导入库 -> 训练 tokenizer -> 加载 tokenizer -> 读取数据 -> 定义模型 -> 训练 -> 验证。
10. **注意 Windows 路径和命令**：PowerShell 使用 `Get-Content`，Python 使用 `pathlib.Path` 检查路径。
11. **先检查文件和环境**：确认 `train.jsonl`、`bow_tok.model` 存在，确认 PyTorch 和 SentencePiece 已安装。
12. **验证阶段不更新参数**：不要在开发集循环中调用 `backward()` 或 `optimizer.step()`。

**关键词**：batching、dev accuracy、eval、实验设计、mask、Notebook 执行顺序、PowerShell、随机种子、张量形状、训练集划分

## 索引

- Adam：[代码实现细节](#训练循环)
- `argmax()`：[语法知识点](#argmax-和-item)
- BoW：[代码实现细节](#bow-模型)
- BPE：[代码实现细节](#sentencepiece-tokenizer-训练)
- CBoW：[代码实现细节](#cbow-模型)
- `CrossEntropyLoss`：[代码实现细节](#交叉熵损失)
- `detach()`：[语法知识点](#detach)
- `dim`：[语法知识点](#dim-参数)
- `dtype=torch.long`：[语法知识点](#dtypetorchlong)
- Embedding：[代码实现细节](#自定义-embedding-层)
- `eval()`：[核心概念与原理](#modeltrain-和-modeleval)
- JSONL：[代码实现细节](#数据读取分词和划分)
- `item()`：[语法知识点](#argmax-和-item)
- logits：[核心概念与原理](#logits概率和预测)
- mask：[核心概念与原理](#batchingpadding-和-mask)
- `model.train()`：[核心概念与原理](#modeltrain-和-modeleval)
- one-hot：[代码实现细节](#one-hot-和-embedding)
- padding：[核心概念与原理](#batchingpadding-和-mask)
- PowerShell：[常见错误与调试](#trainjsonl-找不到)
- SentencePiece：[代码实现细节](#sentencepiece-tokenizer-训练)
- softmax：[核心概念与原理](#logits概率和预测)
- `super()`：[语法知识点](#super)
- `view()`：[语法知识点](#view1--1)
- `with`：[语法知识点](#with-上下文管理器)
- `yield`：[语法知识点](#yield-生成器)
- Xavier 初始化：[核心概念与原理](#xavier-初始化)
- 反向传播：[核心概念与原理](#反向传播如何使用-y)
- 交叉熵：[核心概念与原理](#反向传播如何使用-y)
- 标签索引：[核心概念与原理](#标签索引的来源)
- 数据集重叠：[常见错误与调试](#训练集和开发集重叠)
- 训练循环：[代码实现细节](#训练循环)
- 训练模式：[核心概念与原理](#modeltrain-和-modeleval)
- 词表：[代码实现细节](#加载-tokenizer-和查看词表)
- 文件路径：[常见错误与调试](#trainjsonl-找不到)
- 语法 `**dict`：[语法知识点](#-字典参数展开)
- 语法切片：[语法知识点](#切片)
