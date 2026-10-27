# Mask-wise Paired Cross-Attention：适用于当前 LDP 的 Q/K/V 融合设计

> 工作名：**Mask-wise Paired Cross-Attention (MPCA)**  
> 目标：在当前两种一一对应的 mask-level feature 之间做更直接的 paired fusion。

当前输入已经确定为：

\[
S,C\in\mathbb{R}^{B\times250\times640}
\]

其中：

- \(S_i\)：第 \(i\) 个 proposal 的 mask/segmentation-side representation；
- \(C_i\)：同一个 proposal 的 CLIP semantic representation。

本文不引入 \(S-C\) 这样的人工差分，而只讨论 **Q/K/V 如何组织两种 feature**。

---

## 1. 当前 LDP

当前：

\[
\boxed{Q=C}\qquad[B,250,640]
\]

\[
\boxed{K=[S,C]_{channel}}\qquad[B,250,1280]
\]

\[
\boxed{V=S}\qquad[B,250,640]
\]

可以写为：

\[
\Delta=MHA(C,[S,C]_{channel},S)
\]

\[
G_{ldp}=LN(C+\Delta)
\]

---

## 2. 当前 LDP 实际在选择什么？

虽然 K 的 channel dimension 为 1280，但它仍然拥有 250 个 proposal tokens。

经过 Q/K projection 后，attention score 为：

\[
QK^T:[B,h,250,250]
\]

因此第 \(i\) 个 CLIP semantic query：

\[
C_i
\]

会与全部：

\[
[S_j,C_j],\quad j=1,\ldots,250
\]

计算 attention。

也就是：

\[
\boxed{C_i\leftrightarrow\{S_j,C_j\}_{j=1}^{250}}
\]

当前 LDP 的 attention 选择对象主要是：

\[
\boxed{proposal}
\]

即“应该从哪个 proposal 的 S 中读取信息”。

---

## 3. 新设计要检验的假设

\(S_i\) 与 \(C_i\) 是同一个 predicted mask 对应的两种 representation。

如果关注的是：

> 当前 mask 的 formation representation 如何校正当前 mask 的 semantic recognition representation，

那么可以检验：

\[
\boxed{S_i\leftrightarrow C_i}
\]

是否比：

\[
C_i\leftrightarrow S_j,\quad i\neq j
\]

更适合作为主要 fusion unit。

所以新的设计将：

\[
\boxed{inter-proposal\ attention}
\]

改成：

\[
\boxed{intra-mask\ paired\ attention}
\]

---

## 4. 设计原则

### 4.1 C 作为 receiver

最终 feature 仍需要与 CLIP text embeddings 分类，因此希望保持：

\[
\boxed{C_i=semantic\ anchor}
\]

所以 Q 仍然由 C 产生。

### 4.2 S 作为 complementary source

\(S_i\) 提供 segmentation-aware / mask-formation information，但不直接替代 CLIP semantic space。

### 4.3 每个 mask 独立融合

只允许：

\[
S_i\leftrightarrow C_i
\]

不允许：

\[
C_i\leftrightarrow S_j,\quad i\neq j
\]

### 4.4 保留 self candidate

借鉴 CrossViT：receiver 自己也放入 K/V，使 attention 可以在“保持 C”与“读取 S”之间选择。

---

## 5. 将两种 feature 作为两个 token

对第 \(i\) 个 mask：

\[
C_i,S_i\in\mathbb{R}^{640}
\]

构造：

\[
\boxed{
X_i=
\begin{bmatrix}
C_i\\
S_i
\end{bmatrix}
}
\]

这是 **token-dimension concat**。

所以：

\[
X_i\in\mathbb{R}^{2\times640}
\]

整个 batch：

\[
X\in\mathbb{R}^{B\times250\times2\times640}
\]

实现时 reshape 为：

\[
[B\cdot250,2,640]
\]

这样每一个 mask 都成为一个独立的两 token fusion problem。

---

## 6. Q 设计

只让 C 做 Query：

\[
\boxed{Q_i=W_QC_i}
\]

shape：

\[
[1,640]
\]

多头后：

\[
[h,1,d_h]
\]

其中：

\[
d_h=640/h
\]

解释：

> 当前 CLIP semantic representation 是最终要被更新的 receiver。

---

## 7. K 设计

Key 来自两个 feature tokens：

\[
\boxed{
K_i=W_K[C_i;S_i]_{token}
}
\]

shape：

\[
[2,640]
\]

多头后：

\[
[h,2,d_h]
\]

两个 Key 分别对应：

- \(K_i^C\)：semantic self candidate；
- \(K_i^S\)：mask-side complementary candidate。

---

## 8. V 设计

主方案使用：

\[
\boxed{
V_i=W_V[C_i;S_i]_{token}
}
\]

shape：

\[
[2,640]
\]

这样最终被读取的内容可以来自：

1. 原 semantic representation；
2. mask-side representation。

与当前 LDP 的关键区别是：

当前固定：

\[
V=S
\]

而这里不提前规定真正的 information source 必须只有 S。

---

## 9. Attention

第 \(i\) 个 mask：

\[
A_i=Softmax\left(\frac{Q_iK_i^T}{\sqrt{d_h}}\right)
\]

shape：

\[
\boxed{A_i:[h,1,2]}
\]

每个 head 只需要学习两个 source weight：

\[
a_{i,h}^{C},\quad a_{i,h}^{S}
\]

并满足：

\[
a_{i,h}^{C}+a_{i,h}^{S}=1
\]

所以 attention 回答的问题从当前 LDP 的：

> “应该读哪个 proposal？”

变成：

> **“对于这个 mask，应该更多依赖 semantic feature 还是 mask-side feature？”**

---

## 10. 输出与 residual

\[
R_i=A_iV_i
\]

多头 merge + output projection：

\[
\hat R_i=W_OR_i
\]

最终：

\[
\boxed{G_i=LN(C_i+\hat R_i)}
\]

输出：

\[
G\in\mathbb{R}^{B\times250\times640}
\]

所以后续：

- CLIP text similarity；
- classification loss；
- ensemble；
- objectness / void；

均可以保持现有逻辑不变。

---

## 11. 当前 LDP 与 MPCA 的 shape 对比

### 当前 LDP

\[
Q:[B,250,640]
\]

\[
K:[B,250,1280]
\]

\[
V:[B,250,640]
\]

attention：

\[
[B,h,250,250]
\]

### MPCA

reshape 后：

\[
Q:[B\cdot250,1,640]
\]

\[
K:[B\cdot250,2,640]
\]

\[
V:[B\cdot250,2,640]
\]

attention：

\[
\boxed{[B\cdot250,h,1,2]}
\]

---

## 12. 两者真正的结构差别

当前 LDP：

\[
\boxed{attention\ over\ proposals}
\]

MPCA：

\[
\boxed{attention\ over\ feature\ sources}
\]

当前 LDP 问：

> 第 \(i\) 个 C 应该从第几个 proposal 的 S 中读取信息？

MPCA 问：

> 对同一个 mask \(i\)，C 和 S 两种表示应该如何组合？

---

## 13. 为什么不是 Q=C，K=S，V=S？

如果每个 mask 单独处理，并写成：

\[
Q_i=C_i,\quad K_i=S_i,\quad V_i=S_i
\]

那么只有一个 Key。

softmax 中只有一个元素：

\[
Softmax(q_ik_i^T)=1
\]

attention 不再具有选择能力，基本退化为：

\[
G_i=C_i+W_VS_i
\]

因此需要至少两个候选 token。

CrossViT 中的 self candidate 思想恰好提供：

\[
[C_i;S_i]
\]

这两个候选。

---

## 14. 为什么不是 channel concat？

如果：

\[
[S_i,C_i]_{channel}\in\mathbb{R}^{1280}
\]

再用 Linear 压回 640，两个 feature 会立即被混合。

模型不会显式产生：

\[
a_i^C,\quad a_i^S
\]

而 token concat：

\[
[C_i;S_i]_{token}\in\mathbb{R}^{2\times640}
\]

保留两个 feature source 的身份，attention 可以显式学习它们的相对贡献。

---

## 15. PyTorch 最小实现

```python
import torch
import torch.nn as nn


class MaskWisePairedCrossAttention(nn.Module):
    def __init__(self, dim=640, num_heads=8, dropout=0.0):
        super().__init__()

        self.pre_norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, s, c):
        # s, c: [B, N, D]
        B, N, D = c.shape

        # token 0 = C, token 1 = S
        x = torch.stack([c, s], dim=2)
        # [B, N, 2, D]

        x = x.reshape(B * N, 2, D)
        # [B*N, 2, D]

        x_norm = self.pre_norm(x)

        # only C is Query
        q = x_norm[:, 0:1, :]
        # [B*N, 1, D]

        kv = x_norm
        # [B*N, 2, D]

        delta, attn = self.attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=True,
        )

        delta = delta.squeeze(1).reshape(B, N, D)

        out = self.out_norm(c + delta)

        attn = attn.reshape(B, N, 1, 2)

        return out, attn
```

---

## 16. 如果希望观察每个 attention head

```python
delta, attn = self.attn(
    query=q,
    key=kv,
    value=kv,
    need_weights=True,
    average_attn_weights=False,
)
```

此时：

\[
attn:[B\cdot250,h,1,2]
\]

可以直接统计每个 head：

```text
head 1: C 0.80 / S 0.20
head 2: C 0.35 / S 0.65
...
```

这可以分析不同 mask、不同类别、seen/unseen 时模型更依赖哪种 representation。

---

## 17. 可选变体：Correction-only MPCA

一个后续可做的消融是：

\[
Q_i=C_i
\]

\[
K_i=[C_i;S_i]
\]

但：

\[
\boxed{V_i=[0;S_i]}
\]

于是：

\[
R_i=a_i^C\cdot0+a_i^S\cdot V(S_i)
\]

最终：

\[
G_i=C_i+a_i^S\Delta S_i
\]

含义是：

- 选择 C key：不修改 semantic anchor；
- 选择 S key：注入 mask-side correction。

这相当于一个 learnable keep-or-correct mechanism。

但它不是标准 CrossViT 形式，建议作为第二阶段 ablation，而不是首个主版本。

---

## 18. 与 MLP fusion 的公平对比

还应该加入一个简单 baseline：

\[
G_i=C_i+MLP([C_i,S_i]_{channel})
\]

用于回答：

> paired attention 本身是否真的比普通 channel fusion 有价值？

建议至少比较：

| Variant | Fusion |
|---|---|
| Current LDP | \(Q=C,K=[S,C]_{channel},V=S\) |
| MLP Fusion | \(C+MLP([C,S])\) |
| **MPCA** | \(Q_i=C_i,K_i=[C_i;S_i],V_i=[C_i;S_i]\) |
| Correction-only MPCA | \(Q_i=C_i,K_i=[C_i;S_i],V_i=[0;S_i]\) |

---

## 19. 推荐监控指标

除了最终 AP/PQ/mIoU，建议记录：

### Mask 正确条件下分类准确率

\[
IoU\ge0.5
\]

和：

\[
IoU\ge0.75
\]

分别统计 top-1。

### Seen / unseen

分别看两类的分类和最终 segmentation 性能。

### Source attention weight

统计：

\[
a_i^C,\quad a_i^S
\]

观察模型什么时候依赖 semantic branch，什么时候依赖 mask-side branch。

### Feature drift

\[
\cos(G_i,C_i)
\]

用于判断是否过度破坏原 CLIP semantic space。

---

## 20. 计算量

当前 attention map：

\[
250\times250=62500
\]

MPCA：

\[
250\times(1\times2)=500
\]

attention score 数量约下降为：

\[
1/125
\]

这只是 attention-map 部分，不能直接等同整个模块 FLOPs 降低 125 倍，因为 Q/K/V projection 仍然存在。

---

## 21. 推荐的第一版

第一轮不要加入：

- \(S-C\)；
- gate；
- local pixel tokens；
- underwater degradation cue；
- 双向更新；
- 额外复杂 MLP。

直接测试：

\[
\boxed{Q_i=C_i}
\]

\[
\boxed{K_i=[C_i;S_i]_{token}}
\]

\[
\boxed{V_i=[C_i;S_i]_{token}}
\]

\[
\boxed{G_i=LN(C_i+MHA(Q_i,K_i,V_i))}
\]

这样能最干净地回答：

> **当两个 feature 是同一个 mask 的配对表示时，把 attention 从 proposal selection 改成 feature-source selection 是否更有效？**

---

## 22. 一句话总结

当前 LDP：

\[
\boxed{Q=C,\quad K=[S,C]_{channel},\quad V=S}
\]

主要是在 250 个 proposals 之间做信息读取。

建议的 MPCA：

\[
\boxed{Q_i=C_i,\quad K_i=[C_i;S_i]_{token},\quad V_i=[C_i;S_i]_{token}}
\]

把 attention 的选择对象从：

\[
\boxed{哪个\ proposal}
\]

改成：

\[
\boxed{同一个\ mask\ 应该使用哪个\ feature\ source}
\]

并始终用：

\[
\boxed{C_i}
\]

作为 semantic anchor 和 residual 主分支。

---

## References

1. Chen, C.-F., Fan, Q., Panda, R. *CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification*. ICCV 2021.  
   https://arxiv.org/abs/2103.14899

2. Official CrossViT implementation:  
   https://github.com/IBM/CrossViT
