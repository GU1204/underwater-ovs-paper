# DA-VPT（2025）论文写作脉络

> **论文**：*DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers*  
> **作者**：Li Ren, Chen Chen, Liqiang Wang, Kien Hua  
> **会议**：CVPR 2025  
> **方法名**：Distribution Aware Visual Prompt Tuning（DA-VPT）  
> **核心设计**：Prompt–Visual Token Metric Learning + Prompt–[CLS] Metric Learning + Dynamic Class–Prompt Mapping  
> **增强版本**：DA-VPT+（加入 Efficient Bias Tuning）

## Abstract

- **[2]** 从 VPT 只更新少量 learnable prompt 的参数高效优势切入，随后把现有研究缺口落在 **prompt 与 image token 的内在相关性和分布没有被显式研究**；这一写法把问题从“prompt 结构怎么设计”转向“prompt 应该在特征空间中学成什么分布”。
- **[2]** 引入 metric learning 研究 prompt distribution 对微调性能的影响，再提出 DA-VPT，通过类别相关语义学习 prompt 与视觉特征之间的距离关系；方法名称中的 Distribution Aware 由这一核心机制直接得到解释。
- **[1]** 将 prompt 定义为 image patch 与 class token 之间传递语义信息的桥梁，提前给出全文后面理论分析和 attention visualization 要验证的核心机制。
- **[2]** 用 recognition 与 segmentation 多类 benchmark 的实验收束摘要，强调方法不仅提升精度，也保留 PEFT 的参数效率。

## 1. Introduction

- **[3]** 先从 ViT/VFM 规模扩大和下游迁移需求展开，再列出 full fine-tuning 的计算成本、过拟合与 catastrophic forgetting；引用 PEFT 工作说明只更新少量参数已经成为合理替代方案，为 VPT 出场建立背景。
- **[3]** 按 NLP adapter → 视觉 adapter → VPT 的顺序回顾 PEFT 演进，再引用 VPT、GateVPT、E2VPT 等工作说明现有改进主要集中在 cross-layer connection、prompt position 和 prompt quantity；这些引用承担“已有研究主要改结构”的定位作用。
- **[3]** 作者把批评点明确落在现有 VPT 通常随机初始化 prompt，并仅靠 downstream task loss 学习；即使已有 data-driven initialization，也没有利用 discriminative、class-aware information 显式约束 prompt distribution。
- **[1]** 用一个直接问题提出本文主线：能否引导 prompt 在 image token 与 class token 之间促进语义信息流，而不是让 prompt 自由吸收任意视觉特征？
- **[3]** Figure 1 对比 VPT-Deep 与 DA-VPT：前者 prompt 分布无约束，可能从不同类别吸收信息；后者同时受主任务和 semantic metric learning 约束，使 prompt、visual token、[CLS] token 按类别语义聚合。图承担“现有问题—本文修正”的总览作用。
- **[3]** 正式提出在 ViT 深层构造 prompt–visual token semantic metric：拉近 prompt 与同类 patch，推远异类 patch；再构造 prompt–[CLS] metric，使 class token 更倾向从相关 prompt 获取信息。
- **[2]** 将方法意图解释为让每个 prompt 有更高概率捕获同类视觉信息、过滤无关信息，prompt 由普通可学习 token 转化为具有类别倾向的信息中继。
- **[2]** 用 FGVC、VTAB-1K、MoCo、MAE 和 segmentation 结果说明方法跨任务、跨预训练范式有效，为后文多 benchmark 实验安排做预告。
- **[3]** 贡献列表依次总结 semantic metric-guided prompt learning、prompt 作为 patch–[CLS] bridge 的机制解释、24 个 recognition + 2 个 segmentation 任务的验证；贡献结构对应“方法—机制—实验”。

## 2. Related Works

### Parameter-Efficient Fine-Tuning

- **[3]** 从只训练 classification head、BitFit bias tuning、scale/shift tuning 到 adapter，梳理 PEFT 如何逐步减少需要更新的 backbone 参数；引用作用是把 DA-VPT 放在“保留预训练主干、只开放少量参数”的大框架内。
- **[1]** 这部分没有声称新的 PEFT 范式，而是为后面的 Efficient Bias Tuning 说明 bias 参数本身已有成熟依据。

### Visual Prompt Tuning

- **[3]** 从 VPT-Shallow/VPT-Deep 回顾 visual prompt 的基本形式，再列举 test-time adaptation、video recognition、动态 prompt 数量/位置、cross-layer connection、spatial selection 等扩展。
- **[2]** 相关工作刻意突出这些方案在**结构与连接机制**上做文章，为本文“不改复杂连接，而研究 prompt–data distribution”建立差异点。

### Integration of PEFT Approaches

- **[2]** 简述多种 PEFT 可以组合使用，但本文只选择 VPT + bias optimization；作者主动限制研究范围，避免 DA-VPT+ 被理解为对各种 PEFT 组合的穷举。

### Metric Learning

- **[3]** 从 contrastive loss、triplet loss 过渡到 NCA 与 Proxy-Anchor，说明 metric learning 的核心目标是利用类别关系塑造 embedding distribution；这些引用为后文直接把 prompt 当 proxy 提供方法来源。
- **[2]** 引用 NCA 在 ViT 与视觉表征学习中的应用，说明“显式管理特征分布”已有有效先例，但 prompt、visual token、class token 三者关系仍缺少系统研究。
- **[1]** Related Work 最后把传统 metric learning 与 visual prompt tuning 连接起来，直接导入方法章节。

## 3. Methodology

### 3.1 Preliminary

- **[3]** 先形式化 ViT：图像被切成 patch token，与 [CLS] token 一起进入多层 MHSA；保留 \(Q,K,V\) 和 attention 计算形式，是因为后文需要把 cosine similarity、prompt distribution 与 attention weight 建立联系。
- **[3]** 介绍 VPT-Shallow 和 VPT-Deep，并明确本文采用 VPT-Deep；公式 (1) 写出每层 \([x_{\mathrm{cls}},P,X]\) 的输入形式，使后面的 metric regularization 可以精确落到每层 prompt \(P^l\) 上。
- **[1]** 强调标准 VPT 微调时只更新 visual prompts 和 classification head，建立 DA-VPT 所基于的参数高效基线。
- **[3]** 接着介绍 NCA：同类样本在 metric space 中应拥有更高邻近概率，并给出公式 (2)；这部分不是泛泛补充背景，而是为 DA-VPT 的 Proxy-Anchor loss 提供数学起点。
- **[2]** 将 metric-learning 中的 learnable proxy 概念映射到 visual prompt，提出**深层 prompt 可以作为若干语义相近类别的 proxy**；这一句完成 VPT 与 metric learning 的真正连接。

### 3.2 Metric Learning on the Learnable Prompts

- **[3]** 先提出核心假设：一个 prompt 不应在所有类别上无差别搜索信息，而应负责一组语义相关类别；这种分工能提高 prompt 的判别性，也让 [CLS] token 更有效聚合 task-specific information。
- **[2]** Figure 2 给出完整方法流程：先把 \(C\) 个类别聚为 \(M\) 个 semantic cluster 并分配给 \(M\) 个 prompts，再分别构造 prompt–visual token 与 prompt–[CLS] metric loss；图承担方法的数据流和语义映射总览。
- **[2]** 公式 (3) 定义最基本的 margin constraint：prompt 与同类 visual token 的 cosine similarity 应高于异类 token；该式把“prompt 应具有类别倾向”从概念描述变成显式几何约束。
- **[2]** 作者特意把 cosine similarity 与 attention 中的 Query–Key matching 联系起来，说明 metric-space 中更近的 prompt–token pair 更可能在 attention map 中形成更高匹配，为后面的 Theorem 1 预埋机制解释。
- **[3]** 公式 (4) 将上述约束实现为 smoothed NCA / Proxy-Anchor loss \(L_{\mathrm{ML}}(X,P)\)：同类 visual token 被拉向对应 prompt，异类 token 被推远；Proxy-Anchor 引用的作用是提供适合“少量 prompt 对大量 token”不对称关系的损失形式。
- **[1]** 实现时使用 prompt 经过当前层 Query projection 后的 \(Q=P^lW_Q^l\) 做比较，使 metric learning 与真实 self-attention 的 Query 空间更一致，而非只约束原始 prompt embedding。
- **[2]** 再定义 \(L_{\mathrm{ML}}(P,x_{\mathrm{cls}})\)，把 [CLS] token 拉向与当前类别相关的 prompt、推离其他 prompt；方法由“prompt 学会找对 patch”扩展到“[CLS] 学会从对的 prompt 取信息”。
- **[2]** 公式 (5)
  \[
  L=L_{\mathrm{CE}}+\beta L_{\mathrm{ML}}(X,P)+\lambda L_{\mathrm{ML}}(P,x_{\mathrm{cls}})
  \]
  将主任务分类监督与两级 semantic metric 合并；公式承担完整训练目标的收束作用，也明确 DA-VPT 没有替换原任务目标，而是在其上增加分布约束。

### 3.3 Projection and Saliency Patch Selection

- **[3]** 作者随后处理一个实现问题：并非所有 visual token 都适合拿来做 prompt metric learning，需要优先选择包含关键图像信息的 token，减少 false positive 对 prompt distribution 的干扰。
- **[2]** 直接从 attention map 找 salient patch 虽然直观，但在 Flash Attention 等实现下计算代价较高；作者用这个工程限制解释为什么不采用最显然的 attention-map sampling。
- **[2]** Figure 2 中改用 MHSA 后的 output representation \(X^l\)，其已经聚合多头信息，可作为 saliency representation；这一设计用现成中间特征近似显著性选择，避免额外 attention extraction。

### 3.4 Dynamically Mapping Classes and Prompts

- **[3]** 提出 \(M\ll C\)，不为每个类别单独分配 prompt；理由是 prompt 数量过多会增加优化困难，并造成不同 prompt 获得不均衡的训练机会。
- **[3]** 训练前额外跑一个 epoch，用预训练 ViT 的 [CLS] token 对每类取均值得到 class representation，再以 k-means 聚成 \(M\) 个 cluster；每个 cluster 对应一个 prompt，使 prompt 的监督类别来自数据语义关系而不是人工指定。
- **[2]** Figure 2 中的 clustering 不只是初始化步骤，还决定 metric loss 的正负样本关系；它把“一个 prompt 对应一组语义相近类别”的假设变成可执行映射。
- **[2]** 每个 epoch 重新统计 class representation，并以上一轮 centroid 初始化 k-means 更新 class–prompt mapping；动态更新用于适应微调过程中 representation space 本身的变化，避免初始聚类长期固定。

### 3.5 Efficient Bias Tuning

- **[3]** 在主方法之外探索少量释放 ViT bias，使 visual-token distribution 也获得一定可调空间；引用 BitFit 类工作说明只调 bias 是已有参数高效策略。
- **[2]** 实验发现 self-attention 的 Key/Value projection bias \(b_K,b_V\) 最有效，据此构成 DA-VPT+；这部分不是 DA-VPT 的核心语义机制，而是给已经受 metric guidance 的 token distribution 增加有限自由度。
- **[1]** “metric-guided prompts + selective bias tuning”的组合用来强化 accuracy–parameter trade-off，后面的 Table 4 与 Figure 4b 专门验证这项增强。

## 4. Technical Discussion

### 4.1 Connection Between Similarity and Attention

- **[2]** 作者单独设立 Technical Discussion，目的在于回答一个关键机制问题：在 embedding space 中把 prompt 拉近某个 visual token，为什么真的会影响 Transformer 的信息流？
- **[3]** Theorem 1 从 softmax attention 推导
  \[
  \Delta a_i \approx a_i(1-a_i)\Delta s_i
  \]
  并把 \(\Delta s_i\) 写成 prompt 变化与 visual token 的内积变化；定理承担 metric similarity 与 attention weight 之间的数学桥接。
- **[2]** 当 prompt 朝 \(x_i\) 靠近时，相应 attention weight 增大；远离时 attention 减小。这个推导为前文“用 metric learning 控制 prompt 接收哪些 token 信息”提供机制层支撑，而不是只依赖性能实验。

### 4.2 Analysis of Guided Attention Maps

- **[3]** Figure 3(a) 比较浅层和深层的 VPT-Deep/DA-VPT attention map：浅层两者都会关注物体局部，但 DA-VPT 的关注更丰富、更精确；深层差异更明显，标准 VPT prompt 选择信息较弱，而正 prompt 能定位随后被 [CLS] 使用的关键区域。
- **[2]** Figure 3(b) 把观察抽象成 information-flow 图：DA-VPT 让特定 prompt 从同类 visual token 聚合细粒度特征，再把这些特征传给 [CLS]；该图直接验证 Introduction 中“prompt 是 semantic bridge”的核心叙事。
- **[2]** 作者也讨论 ViT attention artifact，引用 Darcet 等人的 register 研究说明该现象并非 DA-VPT 独有；DA-VPT 虽未完全消除 artifact，但使其更集中在语义物体区域，用来限制可视化证据的解释边界。

### 4.3 Compatibility with Metric Learning Methods

- **[3]** 解释为什么选择 Proxy-Anchor 而不是普通 Proxy-NCA 或 triplet loss：prompt 数量 \(M\) 远少于 visual token 数量 \(N\)，把二者当成完全对等样本会产生不平衡优化。
- **[2]** Proxy-Anchor 中 proxy 本身就是类别代表，与本文“prompt 作为 semantic cluster representative”的角色更一致；这一段把损失选择从经验超参数提升为与方法假设相匹配的设计。
- **[1]** 补充实验显示常规 metric loss 容易训练不稳定，为后面附录中的 loss variant 对比预先给出解释。

## 5. Experiments

### 5.1 Experimental Setup

- **[3]** 将评测分成三类：FGVC 五个细粒度识别数据集、VTAB-1K 十九个 transfer task、ADE20K/PASCAL Context 两个 semantic segmentation 数据集；这种设置用于证明方法不局限于标准分类。
- **[2]** 分类采用 ViT-B，分割采用 ViT-L，并分别测试 ImageNet-21K supervised pretraining 与 MAE、MoCo-v3 self-supervised pretraining；多种初始化专门验证 semantic metric guidance 是否依赖特定预训练方式。
- **[2]** 明确 DA-VPT = VPT-Deep + 两个 metric losses，DA-VPT+ = DA-VPT + Efficient Bias；后面所有表格都围绕这两个版本展开，避免把 bias tuning 与核心方法混在一起解释。
- **[2]** prompt 数量多数任务约为 20，Proxy-Anchor 采用固定 margin 和 temperature；这些配置为后面的“更少 prompt 也能提高性能”提供统一实验基础。

### 5.2 Result Comparison with the State-of-the-Art

- **[3]** Table 1 在 24 个 recognition task 上比较 Full Tuning、VPT-Shallow/Deep、GateVPT、E2VPT 等方法，并覆盖 supervised、MAE、MoCo-v3 backbone；表格主要证明分布约束带来的收益具有跨预训练方式稳定性。
- **[2]** supervised ViT-B 上，DA-VPT+ 相对 VPT-Deep 在 FGVC 与 VTAB-1K 上分别提升 2.83 和 4.18 个百分点；作者用这些数字直接强调 semantic guidance 的增益不是边际改进。
- **[2]** self-supervised backbone 上提升更明显，并可超过 full fine-tuning；这一结果用来支撑“prompt distribution guidance 能提高 transferability”，而不只是减少参数。
- **[3]** Table 2 转向 ADE20K 与 PASCAL Context segmentation，验证同一 prompt-learning 原理可以用于 dense prediction；作者特别强调 DA-VPT+ 只使用 full tuning 约 4.3% 的可训练参数仍获得接近的分割表现。
- **[2]** Table 3 单独聚焦 FGVC，与 SNF、MoSA 等 PEFT/SOTA 方法逐数据集比较；这一表承担细粒度判别能力验证，与方法强调 class-specific prompt information 相呼应。

### 5.3 Ablation Studies and Discussion

#### Component Ablation

- **[3]** Table 4 依次开关 \(L_{\mathrm{ML}}(X,P)\)、\(L_{\mathrm{ML}}(P,x_{\mathrm{cls}})\) 和 Efficient Bias，在 CUB 与 VTAB-Natural 上拆分贡献；实验结构严格对应方法章节的“两级 metric + 一个增强项”。
- **[2]** 单独使用任一 metric loss 增益有限，两者联合后提升明显，证明只让 prompt 对齐 patch 或只让 [CLS] 对齐 prompt 都不够；完整的 patch → prompt → [CLS] 信息链才是核心。
- **[2]** 加入 Efficient Bias 后性能继续提升，而 latency、memory 和参数增量较小；该结果支撑 DA-VPT+ 是参数高效增强，而非依靠大规模解冻 backbone。

#### Layer-wise Impact

- **[3]** Figure 4(a) 比较 metric loss 施加在不同 Transformer block，以及从某层持续施加到最后一层的结果；实验直接检验作者为什么强调 **deep-layer semantic metric**。
- **[2]** 最深层通常表现最好，作者将其解释为深层 token 含更高级的语义信息；这一结果回扣 Sec. 3.2 中选择深层 prompt 作为 semantic proxy 的设计判断。

#### Efficient Bias Components

- **[2]** Figure 4(b) 分别测试 \(QKV\)、MLP、projection、normalization、[CLS] 等 bias/参数类别，发现不同 bias 的收益并不均匀；图用于证明只选择有效 bias 比统一放开更多参数更合理。

#### Semantic Mapping Updates

- **[2]** Figure 4(c) 记录每个 epoch 的 k-means 开销，前期较高、类别表征稳定后迅速下降；作者用这一实验回应动态 class–prompt mapping 会不会造成明显训练成本的问题。

#### Prompt Initialization

- **[3]** Figure 4(d) 按已有工作尝试用数据均值初始化 prompt，却发现会削弱 DA-VPT；这一负结果用于区分“好的 prompt initialization”和“好的 prompt distribution guidance”并不是同一问题。
- **[2]** 作者解释均值初始化过于 homogeneous，使 prompt 更难在 metric supervision 下分化并学习 discriminative information；这再次强化全文“关键不是起点，而是训练过程中如何塑造 prompt distribution”的主张。

## 6. Conclusion

- **[2]** 结论回到全文最初的问题：Visual Prompt Tuning 不应只考虑 prompt 是否插入、如何连接，还需要考虑 prompt 与数据表征的内在分布关系。
- **[2]** 用 prompt–image semantic metric 概括 DA-VPT，再把 prompt 重新定义为 image patch 与 class token 之间的 semantic-information bridge；结论与 Introduction、Theorem 1、Figure 3 保持同一个核心解释。
- **[2]** 以 24 个 recognition task 和 2 个 segmentation task 的结果收尾，强调少量 prompt/参数即可提升迁移能力，形成“分布机制—信息流解释—参数高效性能”的完整闭环。

## 整体写作主线

- **研究缺口不是 prompt 数量或连接方式，而是 prompt 学成了什么分布**：作者先承认 VPT/GateVPT/E2VPT 等结构设计的进展，再把空缺移到 prompt–data representation 的内在关系。
- **核心假设是 prompt 应具有语义分工**：一个 prompt 只负责一组相近类别，而不是从所有 image token 中无差别吸收信息；k-means class–prompt mapping 把这个假设落实成训练关系。
- **方法构成一条明确的信息链**：\(L_{\mathrm{ML}}(X,P)\) 让 prompt 从正确 visual token 聚合信息，\(L_{\mathrm{ML}}(P,x_{\mathrm{cls}})\) 让 [CLS] 从正确 prompt 获取信息，形成 **visual token → prompt → [CLS]**。
- **公式不是只定义 loss，还服务于机制论证**：margin constraint 定义同/异类几何关系，Proxy-Anchor 实现 distribution guidance，Theorem 1 再证明 embedding similarity 的改变可以传递到 attention weight。
- **Figure 1、2、3 承担不同层级的论证作用**：Figure 1 对比问题与解决思路，Figure 2 展开训练机制，Figure 3 验证 semantic bridge 真的改变了 attention information flow。
- **DA-VPT+ 是辅助增强而非论文主线**：Efficient Bias 给 visual-token distribution 增加少量自由度，但论文最主要的创新仍是 semantic metric-guided prompt distribution。
- **实验与方法假设逐项对应**：Table 4 验证两级 metric 是否互补，Figure 4(a) 验证深层语义假设，Figure 4(c) 回应动态聚类开销，Figure 4(d) 用负结果说明 distribution guidance 比 prompt initialization 更关键。
