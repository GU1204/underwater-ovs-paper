# CrossViT：双分支 Cross-Attention 融合设计详解

> 论文：**CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification**  
> Chun-Fu Chen, Quanfu Fan, Rameswar Panda, ICCV 2021  
> 论文：https://arxiv.org/abs/2103.14899  
> 官方代码：https://github.com/IBM/CrossViT

本文只关注 CrossViT 最值得借鉴的部分：**当存在两种不同来源的特征时，如何确定信息接收方、信息提供方，以及 Q/K/V。**

---

## 1. CrossViT 的基本结构

CrossViT 维护两个 ViT 分支：

- Large-patch branch（L-branch）：较大 patch，作为 primary branch；
- Small-patch branch（S-branch）：较小 patch，作为 complementary branch。

两路 token 分别为：

\[
X^l=[x^l_{cls},X^l_{patch}],\qquad
X^s=[x^s_{cls},X^s_{patch}]
\]

其中：

\[
x^l_{cls}\in\mathbb{R}^{D_l},\quad
X^l_{patch}\in\mathbb{R}^{N_l\times D_l}
\]

\[
x^s_{cls}\in\mathbb{R}^{D_s},\quad
X^s_{patch}\in\mathbb{R}^{N_s\times D_s}
\]

CrossViT 不把两路所有 token 全部丢进一个大 Self-Attention，而是让一个分支的 CLS token 作为 **receiver**，从另一分支 patch tokens 中读取 complementary information。

---

## 2. 为什么不是简单 concat + Self-Attention？

论文比较了四种融合：

| Fusion | Top-1 | FLOPs | Params |
|---|---:|---:|---:|
| None | 80.2 | 5.3G | 23.7M |
| All-Attention | 80.0 | 7.6G | 27.7M |
| Class Token | 80.3 | 5.4G | 24.2M |
| Pairwise | 80.3 | 5.5G | 24.2M |
| **Cross-Attention** | **81.0** | **5.6G** | **26.7M** |

All-Attention 把所有 token 拼接后做 Self-Attention，复杂度为：

\[
O((N_l+N_s)^2)
\]

CrossViT 则只用 **一个 CLS Query** 去读取另一分支的 patch tokens，因此 attention map 对 token 数量近似线性。

---

## 3. 以更新 Large Branch 为例

CrossViT 要把 S-branch 的细粒度信息注入 L-branch。

第一步只取：

\[
x^l_{cls}
\]

而不使用 L-branch 的全部 patch tokens 作为 Query。

因为两个 branch 的 embedding dimension 可能不同，先投影：

\[
\tilde{x}^l_{cls}=f^l(x^l_{cls})
\]

其中：

\[
f^l:\mathbb{R}^{D_l}\rightarrow\mathbb{R}^{D_s}
\]

官方代码中的 projection 由 LayerNorm、激活函数和 Linear 组成。

---

## 4. Token concat

将投影后的 L-branch CLS 与 S-branch patch tokens 沿 **token dimension** 拼接：

\[
\boxed{
X'^l=[\tilde{x}^l_{cls};X^s_{patch}]
}
\]

因此：

\[
X'^l\in\mathbb{R}^{B\times(1+N_s)\times D_s}
\]

注意：这里不是 channel concat。

---

## 5. CrossViT 的真实 Q/K/V

论文 Eq. (6)：

\[
q=x'^l_{cls}W_q
\]

\[
k=X'^lW_k
\]

\[
v=X'^lW_v
\]

因此可以写成：

\[
\boxed{Q=\tilde{x}^l_{cls}}
\]

\[
\boxed{K=[\tilde{x}^l_{cls};X^s_{patch}]}
\]

\[
\boxed{V=[\tilde{x}^l_{cls};X^s_{patch}]}
\]

也就是说：

- Q：当前分支的 summary token；
- K/V：自己的 summary token + 另一分支的 complementary tokens。

---

## 6. Shape

设：

- heads：\(h\)
- dimension：\(D_s\)
- head dimension：\(d_h=D_s/h\)

则：

\[
Q:[B,h,1,d_h]
\]

\[
K:[B,h,N_s+1,d_h]
\]

\[
V:[B,h,N_s+1,d_h]
\]

attention：

\[
A=Softmax\left(\frac{QK^T}{\sqrt{d_h}}\right)
\]

shape：

\[
\boxed{A:[B,h,1,N_s+1]}
\]

所以只有一个 receiver 在从多个候选信息源中读取信息。

---

## 7. 为什么 K/V 中还保留自己的 CLS？

CrossViT 不是纯：

\[
Q=A,\quad K=B,\quad V=B
\]

而更接近：

\[
Q=A,\quad K=[A;B],\quad V=[A;B]
\]

这样 Query 可以在两种来源间选择：

1. 保留自身信息；
2. 从另一分支读取补充信息。

可以理解为：

\[
\boxed{self\ candidate + cross\ candidates}
\]

这是 CrossViT 对双特征融合最值得借鉴的地方之一。

---

## 8. 只更新 Receiver

Cross-Attention 输出：

\[
CA(X'^l)=AV
\]

加入 residual：

\[
y^l_{cls}=\tilde{x}^l_{cls}+MCA(LN(X'^l))
\]

随后 back projection：

\[
\hat{x}^l_{cls}=g^l(y^l_{cls})
\]

再与 L-branch 原来的 patch tokens 拼回：

\[
\boxed{Z^l=[\hat{x}^l_{cls};X^l_{patch}]}
\]

在这个方向里，S-branch patch tokens 只是 information source，并不会被更新。

---

## 9. 为什么自己的 patch 不立即更新？

更新后的 L-branch CLS 已吸收 S-branch 信息。下一层 L-branch Transformer 中，新的 CLS 会再次与自己的 patch tokens 做 Self-Attention，因此跨尺度信息通过：

```text
S-branch patches
      ↓
L-branch CLS
      ↓
next L Transformer
      ↓
L-branch patches
```

传播回本分支。

---

## 10. Small Branch 同样反向融合

另一个方向完全对称：

\[
Q=\tilde{x}^s_{cls}
\]

\[
K=V=[\tilde{x}^s_{cls};X^l_{patch}]
\]

所以 CrossViT 整体是双向信息交换，但每个方向内部都是：

\[
\boxed{one\ receiver\ token\leftarrow complementary\ tokens}
\]

---

## 11. 官方代码中的关键实现

官方 `CrossAttention` 的核心逻辑：

```python
B, N, C = x.shape

# only the first token is Query
q = self.wq(x[:, 0:1, ...])

# all tokens are K/V
k = self.wk(x)
v = self.wv(x)

attn = (q @ k.transpose(-2, -1)) * self.scale
attn = attn.softmax(dim=-1)

out = attn @ v
```

`MultiScaleBlock` 中则先构造：

```python
tmp = torch.cat(
    (
        projected_own_cls,
        other_branch_patch_tokens,
    ),
    dim=1,
)
```

再进入 cross-attention。

官方 fusion block 设置 `has_mlp=False`，因此 cross-attention 主要承担轻量的信息交换，而不是完整 Transformer 编码。

---

## 12. CrossViT 的设计哲学

如果抽象为两种 feature：

\[
A,\ B
\]

而最终希望更新 A，那么 CrossViT 的思路是：

\[
\boxed{Q=A_{receiver}}
\]

\[
\boxed{K=[A_{receiver};B_{source}]}
\]

\[
\boxed{V=[A_{receiver};B_{source}]}
\]

最终：

\[
\boxed{A'=A+\Delta(A,B)}
\]

而不是让 A/B 所有 tokens 无差别互相交换。

三个最值得迁移的原则：

1. **明确 receiver**：最终希望保持在哪个 representation space；
2. **明确 complementary source**：另一路 feature 提供补充，而不一定改变自身；
3. **保留 self candidate**：K/V 同时包含 receiver 自己和 complementary source，让 attention 学习“保持自己还是读取对方”。

---

## 13. 对当前 mask/semantic 双特征融合的启发

如果：

\[
C_i=\text{semantic representation}
\]

\[
S_i=\text{mask/segmentation-side representation}
\]

且最终需要保持 CLIP semantic space，那么最自然的角色分配是：

\[
\boxed{C_i=receiver}
\]

\[
\boxed{S_i=complementary\ source}
\]

根据 CrossViT 的原则，可以构造：

\[
Q_i=C_i
\]

\[
K_i=[C_i;S_i]_{token}
\]

\[
V_i=[C_i;S_i]_{token}
\]

最终只更新：

\[
C_i
\]

这不是 CrossViT 原论文结构，而是对其 **receiver + self/complementary source** 原则的迁移。

详细实现见：`MaskWise_Paired_QKV_Design.md`。

---

## References

1. Chen, C.-F., Fan, Q., Panda, R. *CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification*. ICCV 2021.  
   https://arxiv.org/abs/2103.14899

2. Official implementation:  
   https://github.com/IBM/CrossViT

3. Key source file:  
   https://github.com/IBM/CrossViT/blob/main/models/crossvit.py
