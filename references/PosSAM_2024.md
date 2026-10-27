# PosSAM（2024）论文写作脉络

> **论文**：*PosSAM: Panoptic Open-vocabulary Segment Anything*  
> **作者**：Vibashan VS, Shubhankar Borse, Hyojin Park, Debasmit Das, Vishal Patel, Munawar Hayat, Fatih Porikli  
> **发表形式**：arXiv, 2024  
> **方法名**：PosSAM  
> **核心设计**：SAM-based Mask Generator + Local Discriminative Pooling（LDP） + Mask-Aware Selective Ensemble（MASE）

## Abstract

- **[2]** 先把 SAM 的优势与缺陷并置：SAM 具有很强的空间定位和 mask 泛化能力，但缺少类别语义，并容易把同一实例过分割；这一对比直接建立“为什么不能直接拿 SAM 做开放词汇全景分割”的问题。
- **[2]** 引用现有 Grounding/Segmentation 组合方法说明通常需要额外模型生成 box 或 mask prompt，再以多阶段、非端到端和难以统一 panoptic segmentation 作为不足，为 PosSAM 的统一框架提供动机。
- **[2]** 提出 PosSAM，用 SAM 提供空间表征、CLIP 提供类别判别；摘要先用两种 foundation model 的能力互补定义整体方法，而不是直接展开模块。
- **[2]** 再分别引出 LDP 与 MASE：LDP 融合 class-agnostic SAM 和 discriminative CLIP 表征做开放词汇分类，MASE 在推理阶段根据 mask 与分类置信度自适应融合预测。
- **[1]** 用 COCO→ADE20K 和 ADE20K→COCO 的 PQ 提升收束，实验数字承担“跨数据集开放词汇泛化有效”的直接证据。

## 1. Introduction

- **[2]** 从 SAM 在十亿级 mask 预训练后具备强大的 promptable segmentation 和跨分布泛化能力切入，先确立 SAM 值得作为开放词汇分割基础模型。
- **[3]** 随后马上限制 SAM 的能力边界：缺少 instance/class awareness，容易 over-segmentation，文本 prompt 也只能处理较简单的单类目标；Figure 1 左侧用 SAM 与 PosSAM 的结果直接展示“空间能力强但实例/类别能力不足”。
- **[3]** 列举 GroundingDINO、Semantic Segment Anything、Semantic-SAM 等方案，说明现有方法借助额外 detector/segmenter 获取更强 prompt；引用目的在于把问题落到 **额外步骤造成效率低、缺乏端到端协同，box-based 方法也难处理完整 panoptic segmentation**。
- **[2]** 提出统一端到端框架，保留 SAM 的空间表征来生成 class-agnostic panoptic masks；这一段先解决“怎样利用 SAM”，尚未进入开放词汇分类。
- **[3]** 转向 CLIP-based OVS，引用 MaskCLIP、OVSeg、FC-CLIP、OMG-Seg 等说明现有方法主要依赖 CLIP 做 alignment、mask generation 或 classification；作者随后提出 SAM 的空间表征更适合 mask generation，把 SAM 和 CLIP 的职责明确拆开。
- **[2]** Figure 2 对 frozen SAM 与 CLIP backbone feature 做 K-means clustering，SAM 的 cluster boundary 更贴合物体轮廓；图承担**选择 SAM 而不是 CLIP 作为 mask backbone 的经验依据**。
- **[2]** 作者补充 SAM 原生 decoder 的 panoptic mask 效果较差，并据此采用 Mask2Former decoder；这一步避免“使用 SAM”被等同于完整复用 SAM encoder+decoder。
- **[3]** 正式概括 PosSAM 的三个问题：生成 instance-aware masks、给 class-agnostic SAM region 分类、区分 seen 与 unseen；后面 FPN/IoU、LDP、MASE 分别围绕这三项展开。
- **[2]** 对 mask 分支，引入 FPN 将单尺度 SAM ViT feature 转为多尺度表示，并增加 category-agnostic IoU prediction 约束 mask quality；作者用冻结 SAM backbone 解释为什么需要额外 mask-level regularization。
- **[2]** 对分类分支，提出 LDP，让 CLIP 的类别判别信息与 SAM 的 class-agnostic region feature 交互，目标是减少只依赖 discriminative CLIP 时对 seen classes 的偏置。
- **[2]** 对推理阶段，提出 MASE，根据 IoU 和 LDP/CLIP classification confidence 自适应处理 seen 与 unseen；Introduction 在这里提前交代“训练分类器”和“开放词汇推理”采用不同策略。
- **[4]** 贡献列表按 PosSAM 统一框架、LDP、MASE、跨 benchmark 性能依次总结，结构对应“整体框架—训练分类—推理融合—实验验证”。

## 2. Related Work

### 2.1 Vision-Language Models

- **[3]** 先概括 VLM 通过 image encoder、text encoder 和跨模态训练建立共享表征空间，再列举 contrastive learning、prefix prediction、masked language/image-text matching 等训练范式；作用是交代 CLIP 类模型为什么可以承担开放词汇类别预测。
- **[1]** 这一小节只建立 CLIP 的语义能力来源，不讨论空间定位，为下一节 SAM/CLIP 能力分工做准备。

### 2.2 Open-Vocabulary Segmentation

- **[2]** 从开放词汇分割识别训练时未见类别的目标定义切入，再概括多数工作使用 CLIP 等 foundation model 特征并配合 Mask2Former decoder，建立 PosSAM 所采用的基础范式。
- **[3]** 依次引用 MaskCLIP 的两阶段 mask+CLIP alignment、ODISE 的 diffusion representation、OPSNet 的 embedding modulation、FreeSeg 的多粒度文本、FC-CLIP 的 frozen CLIP single-stage framework；这些引用用于覆盖现有“语义 foundation model 如何进入 segmentation”的主要路线。
- **[2]** 相关工作最后把空缺收束到 **SAM 的 spatially rich fine-grained feature 尚未用于 open-vocabulary panoptic segmentation**，直接对应 Introduction 中 Figure 2 的观察。

### 2.3 Segment Anything

- **[3]** 回顾 SAM 在医学、伪装/透明物体、Grounded-SAM、轻量化以及 BLIP/CLIP class-aware extension 中的应用，说明 SAM 的迁移能力已经得到广泛验证。
- **[2]** 用现有扩展仍未解决 open-vocabulary panoptic segmentation 结束小节，把“扩展 SAM”与前一节“开放词汇分割”两条相关工作线汇合到本文问题。

## 3. Method

### 3.1 Problem Setting

- **[2]** 用训练图像、binary masks 和 \(C_{\text{train}}\) 类别集合形式化训练数据，再定义测试图像可包含 seen 与 unseen 的 \(C_{\text{test}}\)；这一段只负责把开放词汇泛化条件写成数学对象，为后面的 \(C_{\text{train}}\cap C_{\text{test}}\) 推理划分做准备。

### 3.2 Panoptic Open-Vocabulary Segment Anything

- **[3]** Figure 3 先给训练 pipeline：冻结 SAM encoder 得到 \(F_{\text{sam}}\)，FPN 生成多尺度 \(F_{\text{ms}}\)，Mask2Former decoder 得到 region feature \(F_{\text{masked}}\) 和 class-agnostic masks；图先把 mask generation 路径固定下来。
- **[2]** 同一批 \(F_{\text{masked}}\) 送入 IoU head 预测每个 mask 的质量，形成与 mask head 并行的辅助任务；Figure 3 用并行支路强调 IoU score 后面不仅训练 mask，还会参与 MASE。
- **[3]** 另一条冻结 CLIP image encoder 得到 \(F_{\text{clip}}\)，再与 \(F_{\text{masked}}\)、预测 mask 一起进入 LDP；这种画法把 CLIP 明确定位为**分类增强分支**，不参与主要 mask generation。
- **[2]** CLIP text encoder 提供类别 embedding，LDP region embedding 与 text embedding 点积得到分类预测；训练框架由 mask loss、IoU loss、class loss三种监督闭合。
- **[1]** Figure 3 的总体论证作用是把 SAM 的 localization、CLIP 的 semantics 和 Mask2Former 的 instance decoding 分工可视化，说明 PosSAM 并非简单串联 SAM 与 CLIP。

### 3.3 Mask Generator

#### FPN

- **[2]** 作者先指出 SAM ViT 只有单尺度 feature，对不同尺寸物体的 panoptic segmentation 不利；这一缺陷直接引出 FPN，而不是把 FPN 当作普通工程组件。
- **[2]** 通过 convolution/deconvolution 产生 \(1/32,1/16,1/8,1/4\) 多尺度 feature，目标是在保留 SAM 鲁棒空间表征的同时补充 hierarchical representation。
- **[1]** 这段写法把 SAM 的“强空间特征”与“单尺度不适合 panoptic decoder”区分开，说明作者保留 encoder、重构下游 feature hierarchy 的原因。

#### Mask Decoder

- **[3]** 采用 Mask2Former pixel decoder 和 object queries，通过 masked cross-attention、self-attention、FFN 得到 \(F_{\text{masked}}\)，再由 mask head 输出 class-agnostic binary masks；引用 Mask2Former 的目的在于直接借用成熟 instance-aware decoding。
- **[2]** 使用 Hungarian matching 对预测 mask 与 GT mask 建立一一对应，公式 (1) 用 BCE 监督：
  \[
  \mathcal L_{\text{mask}}(i,j)=\mathrm{BCE}(Y_{\text{mask}}^i,P_{\text{mask}}^j)
  \]
  公式承担“只学习空间 mask、不在该分支注入类别语义”的训练定义。

#### IoU Head

- **[2]** 从 \(F_{\text{masked}}\) 预测 mask 与 GT 的 IoU，作为额外 category-agnostic mask quality estimation；作者强调 category-agnostic 是为了让 quality estimator 能泛化到 unseen classes。
- **[2]** 公式 (2) 定义 IoU regression loss；该辅助监督的论证作用是让后续推理拥有一个独立于类别的 mask confidence，而不是只依赖 classification score。
- **[2]** 推理时 \(P_{\text{iou}}\) 用于重新加权分类 logits，使高质量 mask 获得更大权重；这里提前把训练时的 IoU head 与 Sec. 3.5 MASE 连接起来。

### 3.4 Mask Classification

- **[2]** 先说明直接从 SAM region feature ROI pooling 得到 mask embedding 仍缺少类别语义，因为 SAM 和 mask generator 都按 class-agnostic 方式训练；问题由 mask generation 自然过渡到 classification。
- **[1]** 引入冻结 CLIP image encoder补充 semantic discrimination，再提出 LDP 同时利用 SAM 与 CLIP 表征，形成该小节的核心方案。

#### Local Discriminative Pooling（LDP）

- **[3]** Figure 5 展示双流结构：用同一预测 mask 分别在 SAM-side mask feature 和 CLIP feature 上做 mask pooling，得到 \(G_{\text{mask}}\) 与 \(G_{\text{clip}}\)；图首先强调 region correspondence 由同一个 mask 建立。
- **[2]** 两类 region feature 分别投影到可交互的 embedding space，再拼接并通过 self-attention 融合；这一过程不是简单 average pooling，而是学习 SAM spatial information 与 CLIP semantic information 的局部交互。
- **[2]** self-attention 后经 residual/normalization 得到每个 mask 的 \(G_{\text{ldp}}\)；Figure 5 将 LDP 的最终目标明确成一个可以直接与 CLIP text embedding 做相似度的 region representation。
- **[2]** 作者把两类输入解释为互补：SAM feature 提供更精确、class-agnostic 的局部结构，CLIP feature提供类别判别；这种解释用来支撑 LDP 可以减少只训练 discriminative classification feature 时的 seen-class overfitting。
- **[2]** 对 \(C_{\text{train}}\) 的类别文本计算 \(G_{\text{text}}\)，再以
  \[
  P_{\text{class}}=\mathrm{Softmax}(G_{\text{ldp}}G_{\text{text}}/\tau)
  \]
  完成开放词汇形式的分类；公式把 LDP 的 region feature 接回 CLIP text space。
- **[2]** 公式 (3) 用 cross-entropy 对 Hungarian matched mask 做类别监督；该 loss 说明 **LDP 实际在训练词表上被监督学习**，也为下一节为什么推理时还需要 frozen CLIP 分支埋下伏笔。
- **[2]** 公式 (4)
  \[
  \mathcal L_{\text{total}}
  =\gamma_a\mathcal L_{\text{class}}
  +\gamma_b\mathcal L_{\text{mask}}
  +\gamma_c\mathcal L_{\text{iou}}
  \]
  收束训练阶段，把 category classification、mask generation、mask quality 三个目标统一到一个 end-to-end framework。

### 3.5 Open-Vocabulary Inference

- **[2]** 开头明确训练与推理策略不同：mask generation 已经足够稳定，所以推理时固定 mask pipeline，只调整 classification branch 来处理 unseen classes；这一转折是方法写作中的关键节点。
- **[3]** 作者主动承认 LDP 在 \(C_{\text{train}}\) 上训练，因此对 seen 类更有优势；对真正 unseen 类重新直接 mask-pool frozen CLIP 得到 \(G_{\text{clip}}\)，利用 CLIP 未经当前闭集训练破坏的开放词汇能力。
- **[2]** 由此形成两种分类表征：训练得到的 \(G_{\text{ldp}}\) 偏向 in-vocabulary discrimination，原始 \(G_{\text{clip}}\) 提供 out-of-vocabulary generalization；MASE 的必要性来自二者不存在单一固定最优权重。

#### Mask-Aware Selective Ensemble（MASE）

- **[3]** Algorithm 1 先用 \(P_{\text{iou}}\) 分别加权 LDP logits 与 CLIP logits；作者解释 decoder 会产生大量 void masks，所以分类置信度必须先受到 mask quality regularization。
- **[2]** 再从每个分类分布的 top-1/top-2 差异构造相对 confidence，用于估计当前 mask 上 LDP 和 CLIP 谁更可信；算法的作用从固定 ensemble 转成 per-mask confidence-aware ensemble。
- **[3]** 参考 ODISE/FC-CLIP 的 geometric mean 保留 \(\alpha,\beta\) 作为 seen/unseen 基础权重，但不再固定使用；根据 LDP 与 CLIP 的相对 confidence 动态得到 \(\hat\alpha,\hat\beta\)。
- **[2]** 用 category-overlap mask \(M\) 分别计算 in-vocabulary 和 out-of-vocabulary prediction，再合并得到最终 \(P_{\text{class}}\)；这一步把训练集与测试集的类别重叠关系显式编码进 inference。
- **[2]** Algorithm 1 的整体论证作用是：**LDP 与 CLIP 不被视为两个等价分类器，而是各自在 seen/unseen 和不同 mask confidence 条件下具有不同可靠度**，所以采用 mask-aware adaptive fusion。

## 4. Experiments

### 4.1 Implementation Details

- **[3]** 使用 SAM-B/L/H 作为冻结 backbone、FPN 构造层级特征、Mask2Former 生成 250 个 class-agnostic masks、CLIP 负责 image-text discrimination；配置直接对应方法章节的三类组件。
- **[2]** 训练主要在 COCO panoptic 上完成，再用单个 checkpoint 测试 ADE20K、PASCAL Context 等数据集；这一设置专门体现 zero-shot/open-vocabulary cross-dataset generalization。
- **[2]** 推理沿用 prompt engineering 与已有 CLIP template，作用是控制文本端变量，让主要比较聚焦 PosSAM 的 mask/classification 设计。

### 4.2 Results

#### Open-Vocabulary Panoptic Segmentation

- **[3]** Table 1 同时设置 COCO→ADE20K 与 ADE20K→COCO，和 MaskCLIP、FreeSeg、ODISE、OPSNet、MaskQCLIP、FC-CLIP 比较；双向迁移避免结果只来自一个 source-target 组合。
- **[2]** PosSAM-H 在两种方向分别达到 29.2 PQ 和 31.6 PQ，较 FC-CLIP 提升 2.4 和 4.6；作者将提升解释为 SAM localization 与 CLIP classification 的协同。
- **[2]** 单独强调 SQ 持续较高，把总体 PQ 增益中的一部分明确归因于 SAM 的空间定位能力，回扣 Introduction 的 Figure 2。

#### Open-Vocabulary Semantic Segmentation

- **[3]** Table 2 用同一 COCO-trained checkpoint 测 A-847、PC-459、A-150，并与 GroupViT、LSeg、OVSeg、SAN、ODISE、FC-CLIP 等比较；这一实验用来说明 PosSAM 不只适用于 panoptic output。
- **[1]** 多种词表规模下 mIoU 均保持竞争力，支撑框架中的 open-vocabulary region classification 能迁移到 semantic segmentation。

#### Seen / Unseen Analysis

- **[3]** Table 3 专门拆 \(PQ_{\text{seen}}\) 与 \(PQ_{\text{unseen}}\)，把总体 PQ 分解成论文真正关心的 bias/generalization 问题；这是验证 LDP 论点比主结果更直接的实验。
- **[2]** PosSAM 的 unseen PQ 从 FC-CLIP 的 17.3 提升到 21.3，而 seen PQ 仍保持 39.5；作者据此主张加入 class-agnostic SAM feature 后没有通过牺牲 seen 类换 unseen 类。
- **[2]** Figure 6 只展示 unseen categories，如 painting、dishwasher、exhaust；图承担 Table 3 的定性证据，强调真正训练未见类别也能被分割和识别。

### 4.3 Ablation / Analysis

#### Component Contributions

- **[3]** Table 4 按 baseline → FPN → LDP → IoU → MASE 逐步加入组件，顺序与方法 pipeline 基本一致；这种累加消融用于说明整体提升并非单一模块造成。
- **[2]** FPN 带来最大一段基础提升，验证 SAM 单尺度 feature 需要重新构造 hierarchy；LDP 提高 classification 相关指标，IoU 稳定 mask，MASE 完成最终 inference 增益。
- **[1]** 最终组合性能最高，用来支撑三个分支不是相互替代，而是在 mask representation、mask quality、class fusion 上承担不同作用。

#### Pooling Strategy

- **[3]** Table 5 直接比较 SAM mask pooling、CLIP mask pooling、SAM→CLIP cross-attention 与 LDP；这一表专门回答 **为什么需要重新设计 mask-level pooling，而不是直接对 CLIP feature 做 mask pooling**。
- **[2]** 单独 SAM/CLIP pooling 分别只有一种表征优势，Cross-Attention 融合后有所提升，LDP 达到最高 29.2 PQ；结果被用来证明 dual-stream pooling + projection + self-attention 更适合结合两种 foundation feature。
- **[1]** 该消融与 Figure 5 形成直接闭环：Figure 5 解释 LDP 如何融合，Table 5 证明这种融合方式优于更简单的 region pooling。

#### SAM Decoder vs. Mask2Former Decoder

- **[3]** Table 6 比较 SAM two-way decoder 单头、增加到 9 heads 与 Mask2Former；实验直接验证 Introduction 中“原生 SAM decoder 不适合 panoptic mask generation”的判断。
- **[2]** 增加 SAM decoder heads 只能有限改善，Mask2Former PQ 显著更高；作者把原因归到 multi-resolution representation 和 deformable/masked attention，更适合同时处理 things 与 stuff。
- **[1]** 该表说明 PosSAM 利用的核心是 **SAM encoder feature，而不是 SAM 原生完整 segmentation pipeline**。

#### Ensembling Strategy

- **[3]** Table 7 先分别只用 \(G_{\text{ldp}}\) 或 \(G_{\text{clip}}\)，再比较固定 geometric ensemble 与 MASE；实验顺序直接模拟 Sec. 3.5 的推理设计选择。
- **[2]** 单独 LDP 为 23.4 PQ、单独 CLIP 为 25.6、固定 geometric ensemble 为 28.4、MASE 为 29.2；这些结果证明两分支互补，也说明固定 seen/unseen 权重仍不是最优。
- **[2]** MASE 相对固定 geometric fusion 的额外提升专门支撑“每个 mask 应根据自身 quality 和 classification confidence 动态决定融合”的方法假设。

## 5. Conclusion

- **[2]** 结论先重新定义 PosSAM 的贡献：不是简单把 SAM 加到 OVS，而是补足 SAM 的 class/instance awareness，使其可以承担 open-vocabulary panoptic segmentation。
- **[2]** 用 LDP 减少 seen-class bias、MASE 根据 IoU 与 classification confidence 区分 seen/unseen 两点回顾全文核心模块，保持与 Introduction 的问题划分一致。
- **[1]** 以跨 benchmark 的 SOTA 结果结束，全文形成“SAM 能力缺口 → SAM/CLIP 分工 → region classification → seen/unseen inference”的闭环。

## 整体写作主线

- **先拆 SAM 的能力**：SAM encoder 的空间表征值得保留，但原生 decoder 缺 instance-aware panoptic decoding，模型本身也缺类别语义；作者不是整体接受或否定 SAM，而是逐组件判断。
- **用 Figure 2 决定 backbone 分工**：SAM 比 CLIP 更适合 localization，所以 SAM 负责 class-agnostic mask generation；CLIP 的优势保留给 region-level semantic classification。
- **mask 分支先独立做稳**：FPN 补多尺度、Mask2Former 补 instance-aware decoding、IoU head 补 category-agnostic mask quality，得到可以跨类别泛化的稳定 proposal。
- **LDP 解决的是 mask 与类别之间的 region representation**：同一 mask 分别 pool SAM-side region feature 与 CLIP feature，经过 projection 和 self-attention 得到 \(G_{\text{ldp}}\)，再与 CLIP text embedding 分类。
- **方法并没有假设训练后的 LDP 足以完成全部开放词汇推理**：作者明确承认 LDP 在 \(C_{\text{train}}\) 上监督，因此推理时保留 frozen CLIP mask pooling 作为 unseen 分支。
- **MASE 是由这种 train–test 不对称自然推出的**：LDP 更利用训练后的 discrimination，CLIP 保留原始 zero-shot generalization；IoU 判断 mask 是否可信，相对分类 confidence 判断哪个分类分支更可信。
- **实验逐项回扣上述判断**：Table 3 看 seen/unseen bias，Table 5 验证 LDP 对直接 mask pooling/cross-attention 的优势，Table 6 验证 Mask2Former 替换 SAM decoder，Table 7 验证 MASE 优于固定 ensemble。
- **全文最核心的写作模式是**：SAM 空间强但语义弱 → CLIP 语义强但空间不如 SAM → 先生成稳定 class-agnostic masks → LDP 学 seen-domain region classification → frozen CLIP 保留 unseen classification → MASE 在推理阶段按 mask 自适应融合。
