# FC-CLIP 水下开放词汇实例分割写作蓝图

## Metadata

- 生成日期：2026-10-15（UTC）
- 默认体裁：计算机视觉会议型原始研究论文；具体会场尚未指定
- 技术事实主来源：[`content_analysis.md`](content_analysis.md)
- 补充新证据：FathomNet prediction-free full test 已完成，40k checkpoint 的 `segm AP/AP50/AP75 = 15.5106/17.8786/16.1916`；原始证据见 `output/eval_full_test_fathomnet_current_retained_visual_common_v3_mapped_complete_id_pgc_ldp_evp/launcher.log`
- 参考语料：FC-CLIP、SAN、PosSAM、CAT-Seg、SED、Explicit Visual Prompting（EVP）
- 推荐中心对象：**PGC-LDP 为主贡献，frequency-conditioned EVP 为次级适配，组合配置为系统结果，FathomNet 为压力测试与局限性证据**
- 当前成熟度：可进入摘要、引言、方法和结果初稿；尚不具备 SOTA、统计显著性、通用泛化或端到端效率主张所需证据

## 0. 首要写作决策

### 0.1 推荐定位

论文应围绕以下问题展开：

> 如何在保留 FC-CLIP 冻结视觉—语言对齐的同时，用与每个 mask query 一一对应的下游结构信息修正区域分类，并避免跨 query 混合和 no-object/void 竞争被破坏？

主线不是“把 PosSAM 和 EVP 拼到 FC-CLIP”，而是：

1. 从 FC-CLIP 的双重区域表示出发，识别出缺少受约束的同 mask token 交互这一问题；
2. 将 PosSAM 的局部判别融合思想重构为适配 FC-CLIP query 语义的 **query-aligned paired gated correction**；
3. 将原始 EVP 改造成冻结 ConvNeXt CLIP 的输入条件化辅助适配；
4. 用受控消融显示主要收益来自 PGC-LDP，并如实呈现 EVP 的有限增量和跨域结果的边界。

### 0.2 为什么 PGC-LDP 必须是唯一一级贡献

最终 checkpoint 的 matched WaterOVS 结果给出清楚的归因：

| 配置 | Final AP | 相对基线 | 相对 PGC-only |
|---|---:|---:|---:|
| FC-CLIP baseline | 47.5934 | — | — |
| EVP-only | 49.7202 | +2.1268 | — |
| PGC-LDP-only | 53.9612 | +6.3678 | — |
| PGC-LDP + EVP | 54.0845 | +6.4911 | +0.1233 |

因此：

- PGC-LDP 是主要且可归因的改进来源；
- EVP 单独有效，但叠加到 PGC 后的最终增量很小；
- 组合模型可以作为最高配置报告，但不能写成“两个模块强协同”；
- 40k 时组合模型为 `54.4971 AP`，比同为 40k 的 PGC-only `53.4909 AP` 高 `1.0062`，但 final 差距缩小至 `0.1233`，应描述为 checkpoint-dependent incremental benefit。

### 0.3 推荐工作标题

主标题暂用：

> **Query-Aligned Paired Correction for Frozen-CLIP Open-Vocabulary Instance Segmentation in Underwater Imagery**

标题中暂不使用 `efficient`、`robust`、`generalizable`、`state-of-the-art` 或 FathomNet benchmark 等尚未充分支持的词。

---

## 1. 参考论文语料与写作惯例

### 1.1 语料概览

| 论文 | 角色 | 与本文的关系 | 主要可借鉴写作动作 | 匹配度 |
|---|---|---|---|---|
| FC-CLIP, NeurIPS 2023 | 直接基线 | 冻结卷积 CLIP、单阶段 mask generation/classification、in/out-vocabulary ensemble | 以“已有范式的结构性矛盾”驱动方法；从问题定义和 baseline 推导新框架；结果同时报告精度与成本 | 高 |
| SAN, CVPR 2023 | 近邻适配路线 | 冻结 CLIP + 轻量 side adapter；mask proposal 与 recognition 解耦 | 从 image-level 到 pixel/region-level 的粒度差距展开；突出冻结模型、轻量适配、单次前向的设计逻辑 | 高 |
| PosSAM, arXiv 2024 | PGC-LDP 的直接思想来源 | SAM 空间特征与 CLIP 语义特征的 LDP；mask-aware ensemble | 先区分空间能力与语义能力，再提出局部区域融合；方法按 mask generation、classification、inference 展开 | 高，但架构对象不同 |
| CAT-Seg, CVPR 2024 | 对照性近邻 | 在 similarity/cost space 聚合，并展示 CLIP fine-tuning 并非必然破坏 unseen 性能 | 用可视化/诊断建立观察，再把观察转化为方法；对“冻结 CLIP 是唯一正确路线”形成必要反例 | 中高 |
| SED, CVPR 2024 | 层级卷积与效率近邻 | hierarchical encoder、cost map、gradual fusion、category rejection | 把局部空间信息、分层表示和准确率—速度权衡串成一条简洁主线 | 中高 |
| Explicit Visual Prompting, CVPR 2023 | EVP 来源 | 冻结主干、patch embedding 与高频输入构造 image-conditioned prompt | 解释为何 prompt 应来自每张输入而非固定 token；按 prompt 来源、adaptor、注入位置展开 | 方法来源匹配，任务匹配较低 |

语料与目标论文的会场风格为中等匹配：五篇为 CVPR/NeurIPS 会议论文，一篇采用 arXiv 版本；主题均围绕视觉基础模型适配和分割，但只有 FC-CLIP 与 PosSAM 直接覆盖开放词汇全景/实例式 mask classification，EVP 的原始任务是低层结构分割。

### 1.2 共同结构惯例

这些论文的主文普遍采用以下顺序：

1. `Introduction`：任务价值 → 现有范式 → 具体结构性缺陷 → 方法观察/设计问题 → 方法概览 → 定量结果 → 贡献；
2. `Related Work`：按研究路线组织，而非逐篇罗列；每个小节最后落回“本文与该路线的区别”；
3. `Method`：先定义任务或 baseline，再给总览，随后按数据流拆模块；
4. `Experiments`：datasets/protocol → implementation → main comparison → ablation/analysis → efficiency/qualitative；
5. `Conclusion`：只复述已经验证的贡献，并用一两句指出未解决问题。

常见的摘要动作是：背景一到两句、缺口一到两句、方法两到三句、关键数字一到两句。方法型论文通常把贡献压缩为三项：核心框架、关键模块或机制、实验验证。

### 1.3 可借鉴但不可照搬的论证策略

- 从 FC-CLIP 借鉴“一个自然问题 + 两个明确障碍 + 对应设计”的引言逻辑，而不是复述其 two-stage 对 single-stage 的原始问题。
- 从 SAN 借鉴“保持冻结预训练能力，同时让轻量可训练路径适配 dense task”的逻辑，但本文不是 side network，也没有其单次前向或速度证据。
- 从 PosSAM 借鉴“空间/结构表示与 CLIP 语义表示互补”的问题分解；必须明确本文的 paired correction 与 PosSAM dual-stream self-attention 的差别。
- 从 CAT-Seg 借鉴“诊断先于方法”的写法。CAT-Seg 也说明不能把“fine-tuning CLIP 一定损害开放词汇能力”写成普遍定律。
- 从 SED 借鉴 hierarchical ConvNet 对局部空间结构的叙事，但本文没有 SED 的 category rejection，也没有对应速度结果。
- 从 EVP 借鉴 input-conditioned explicit prompt 的来源解释；本文只能把原始 EVP 称为先验方法，把 low-frequency-conditioned gate 称为本项目的适配设计，除非后续文献检索确认其新颖性。

---

## 2. 目标读者、论文目的与叙事风险

### 2.1 目标读者

默认读者是熟悉 CLIP、Mask2Former、开放词汇语义/实例/全景分割的计算机视觉审稿人，但不一定熟悉 WaterOVS、FathomNet 或水下类别体系。

因此正文可以默认读者理解：

- mask query、mask pooling、CLIP image/text embedding；
- in-vocabulary 与 out-of-vocabulary classifier；
- AP、AP50、AP75 和 COCO-style instance segmentation evaluation。

正文必须补充解释：

- WaterOVS 的训练/验证类别数、评测口径和为何称为 in-domain；
- FathomNet 的 retained test 为何有 prediction-selection bias；
- prediction-free full test 的 173 类设置及其与 retained test 不可直接比较的原因；
- `void/no-object` 在 FC-CLIP query classification 中不是普通语义类别。

### 2.2 论文目的

主目的：检验 FC-CLIP 的区域分类能否通过 query-aligned、null-preserving 的结构修正获得明确提升。

次目的：评估 input-conditioned frequency prompt 是否能在冻结 ConvNeXt CLIP 条件下提供额外适配。

第三目的：用跨数据集与 checkpoint 曲线暴露当前方法的域专化和评测局限，而不是把 FathomNet 包装成第二条“性能贡献”。

### 2.3 最大叙事风险

1. **来源归属不清**：LDP 来自 PosSAM，EVP 来自 Explicit Visual Prompting；必须明确“采用的思想”和“本项目重构的机制”。
2. **贡献双主线失焦**：若把 PGC 与 EVP 写成同等创新，会与结果归因冲突。
3. **开放词汇过度解释**：现有 diagnostics 没有证明 unseen top-1 改善，不能把整体 AP 增益直接解释为 unseen recognition 增强。
4. **效率主张缺证据**：只有模块级参数/MAC 计算，没有端到端 latency、throughput、memory 或 FLOPs。
5. **FathomNet 指标误读**：训练 self-eval `90.2243 AP` 只代表拟合；retained test 有偏；full test `15.5106 AP` 没有 matched baseline。

---

## 3. Central Contribution Definition

### 3.1 一句话核心命题

> FC-CLIP 的冻结 CLIP 区域 token 可以由同一 mask query 的结构 token 进行选择性残差修正；将修正限制在一一配对的 query 内，并保留原始 no-object logit，可在不改动辅助 decoder 监督的情况下提高 WaterOVS 实例分割 AP。

### 3.2 方法的一句话定义

> PGC-LDP 以 mask-pooled CLIP region token 为 query，以 `[CLIP token, paired structure token]` 为两项 key，并使用 `[zero, paired structure]` 作为 value，使每个 attention head 在“保持不写入”与“写入同 mask 结构修正”之间选择。

这里不能写成“原始 CLIP token 被严格保留”：`clip_projection` 可训练，残差输出还经过 LayerNorm。更准确的说法是“修正从投影后的 CLIP region token 出发，且 self 分支不写入新的 value”。

### 3.3 核心问题—缺口—响应—证据

| 层级 | 内容 |
|---|---|
| Problem | 水下开放词汇实例分割需要同时保留预训练视觉—语言语义和适应下游 mask 结构。 |
| Gap | FC-CLIP 的下游结构表示与 frozen-CLIP 区域表示主要在独立分类器/分数层面结合，缺少受同 mask 对齐约束的 token-level correction；直接让融合模块控制 void 还会破坏 no-object 竞争。 |
| Response | PGC-LDP 只允许 query `i` 的 CLIP token 读取 query `i` 的结构 token，以零 self value 实现 per-head keep-or-correct，并只替换 final non-void logits。 |
| Main evidence | matched ConvNeXt-Large/BS9/FP32/seed-42：PGC-only final `53.9612 AP`，baseline `47.5934 AP`，提升 `6.3678`。 |
| Boundary evidence | 5k control：LDP 同时控制 foreground+void 为 `1.2070 AP`，保留 base void 为 `42.4391 AP`。 |
| Current limit | single seed；缺少 main-protocol global-LDP control、重复种子、端到端成本和完整 novelty audit。 |

---

## 4. Motivation

### 4.1 Motivation 逻辑链

1. **任务价值**：开放词汇实例分割减轻固定类别集合对真实水下场景的限制，但水下标注有限、类别长尾且跨数据来源分布差异明显。
2. **强基线**：FC-CLIP 用单个冻结卷积 CLIP 同时支撑 mask generation 与开放词汇分类，提供了兼顾语义保持和 dense prediction 的自然起点。
3. **核心缺口**：在 FC-CLIP 中，同一个 mask 同时对应下游 decoder structure token 与 frozen-CLIP region token；前者更贴近任务监督，后者保留预训练语义，但二者缺少严格同 query 约束的可学习交互。
4. **为何不能直接复制 PosSAM**：PosSAM 的 LDP 面向 SAM+CLIP 双流；FC-CLIP 具有固定数量 mask queries、未匹配 query 和独立 void 竞争。跨 query 混合与替换 void 都可能改变原有 query 语义，其中后者已有明显崩溃证据。
5. **设计要求**：所需模块应只利用同一 mask 的两个表示，允许模型选择是否写入结构修正，并把 no-object 处理留给原 FC-CLIP。
6. **次级适配问题**：冻结主干保护视觉—语言对齐，但限制了对当前图像局部结构的适配。原始 EVP 提供 input-conditioned high-frequency prompt；本文进一步让 low-frequency context 只控制 high-frequency residual 的空间注入强度。
7. **研究问题**：query-aligned correction 能否成为主要增益来源？void preservation 是否必要？frequency prompt 能否独立有效并稳定补充 PGC？这些问题由四路 matched ablation 和边界消融回答。

### 4.2 推荐的 Introduction 段落推进

**P1 — 场景与任务。** 从“固定标签集合不足以覆盖真实水下类别”进入开放词汇实例分割。避免一开始罗列水下成像物理机制；当前项目没有量化吸收、散射或颜色偏移，因此相关陈述需要后续水下文献支持。

**P2 — 预训练 VLM 与 FC-CLIP。** 说明 CLIP 将视觉和文本映射到共享空间，FC-CLIP 进一步证明单个 frozen convolutional CLIP 可服务 mask generation 与 classification。把 FC-CLIP 写成强起点，而不是需要被否定的旧方法。

**P3 — 表示层缺口。** 明确列出同一预测 mask 的两种表示：decoder structure token 和 frozen-CLIP pooled token。前者经下游监督适配结构与类别，后者保留开放词汇语义。指出 baseline 缺少“同一 mask 内、可选择的 token correction”。

**P4 — 相关路线为何不足。** SAN 表明 frozen CLIP 可通过 side adapter 适配；PosSAM 表明空间和语义特征可以局部融合；CAT-Seg 表明 similarity-space adaptation 是另一条有效路线。但这些方法没有处理 FC-CLIP 中 query-index alignment、unmatched queries 和 base void competition 的组合约束。

**P5 — PGC-LDP 概览。** 用一句结构描述和一句行为描述：same-mask pairing；zero-self value；per-head keep-or-correct；final non-void replacement；base void preservation。此处配一张 baseline/PGC 对比图。

**P6 — EVP 次级模块。** 说明冻结主干下仍需要输入条件化适配，引出 high-pass prompt 和 low-frequency-conditioned spatial gate。只用一个短段，不让它与 PGC 抢主线。

**P7 — 结果和贡献。** 先报 final checkpoint 的 `+6.37 AP` PGC 结果，再报组合 final `54.0845 AP` 与最高 checkpoint `54.4971 AP`。紧接一句限制：四路主实验为 single-seed，组合增量较小。

### 4.3 可直接转写的 Motivation 核心段

开放词汇实例分割需要把密集 mask prediction 与预训练视觉—语言模型的开放类别识别能力结合起来。FC-CLIP 通过共享一个冻结的卷积 CLIP 主干，为这一目标提供了简洁而有竞争力的单阶段基线。然而，在其 mask-wise prediction 中，下游 decoder 产生的结构表示与 frozen CLIP 在同一 mask 内汇聚的区域表示承担不同角色：前者受目标数据监督，后者保留预训练语义。现有 FC-CLIP 主要通过独立分类路径和 score ensemble 利用这些表示，尚未提供一个显式、受 query 对齐约束的 token-level correction 机制。

PosSAM 的 Local Discriminative Pooling 说明空间与语义表征的融合具有价值，但其 SAM–CLIP 双流设计不能直接等价地迁移到 FC-CLIP。FC-CLIP 的固定 mask queries 包含大量未匹配候选，且 no-object/void logit 在 query competition 中具有特殊作用。我们的早期对照也表明，让 LDP 同时替换前景和 void 会导致性能严重下降。因此，本文关注的不是更宽泛的跨 proposal 融合，而是一个更受约束的问题：能否只让一个 CLIP region token 接收其同 mask structure token 的选择性修正，同时保持原始 void 决策路径？

基于这一问题，我们设计 PGC-LDP。每个 mask 的 CLIP region token 只与自身及其 paired structure token 交互，self 分支使用显式零 value，而 structure 分支提供可写入的残差，使各 attention head 可以执行 keep-or-correct。修正仅作用于 final non-void classification，base void logit 与辅助 decoder losses 保持不变。作为次级适配路径，我们还将 Explicit Visual Prompting 引入冻结 ConvNeXt CLIP，并以低频上下文控制高频残差在早期 stages 的空间注入。这样的叙事把主问题落在 query-aligned semantic correction，把频率提示限定为辅助视觉适配。

---

## 5. 贡献设计

### 5.1 推荐写入 Introduction 的三条贡献

1. **Query-aligned paired correction。** 我们面向 FC-CLIP 构建 PGC-LDP，将每个 frozen-CLIP region token 与由同一预测 mask 对齐的 decoder structure token 成对，并以 per-head gated residual 在“保持”与“结构修正”之间选择，避免跨 query 读取。
2. **Null-preserving、final-only 集成与辅助频率适配。** PGC 只替换最终 non-void logits，保留 FC-CLIP 的 base void competition 和所有辅助 decoder 监督；同时，我们将 input-conditioned EVP 适配到冻结 ConvNeXt CLIP，以 low-frequency-conditioned gate 调节 early-stage high-frequency residual。这里应明确 EVP 是已有方法的改造，不把原始高频 prompting 归为本文原创。
3. **受控实证。** 在 matched WaterOVS ConvNeXt-Large/BS9/FP32/seed-42 协议下，PGC-LDP 将 final AP 从 `47.5934` 提高到 `53.9612`（`+6.3678`）；组合配置达到 `54.0845` final AP，最高 checkpoint 为 `54.4971 AP`。void 边界消融进一步显示，保留 base void 对当前实现至关重要。

### 5.2 贡献强度分级

| 候选贡献 | 建议地位 | 当前强度 | 允许的主张 | 不允许的主张 |
|---|---|---|---|---|
| PGC-LDP | 一级、论文中心 | 支持 | 在一组 matched single-seed WaterOVS 实验中提高 AP；实现具备 query pairing、zero-self、final-only、preserve-void | SOTA；统计显著；优于所有 fusion topology；已证明广泛 unseen recognition |
| Preserve-void | PGC 内部关键设计 | 强边界证据 | 对测试配置是必要设计；5k 差距为 41.2321 AP | 一般性理论；所有数据集均必然成立 |
| Frequency-conditioned EVP | 二级组件 | 支持其有效性，novelty 未定 | EVP-only final 比 baseline 高 2.1268 AP；实现为 high-pass residual + low-pass gate condition | 原创 EVP；物理感知恢复；强协同；普遍提升开放词汇泛化 |
| PGC+EVP | 系统配置 | 最高单点受支持 | 最高观察 checkpoint 和 final 数值可报告 | 两模块强互补或超加性协同 |
| FathomNet pipeline/audit | 压力测试和讨论 | 行为证据支持 | 展示 checkpoint/domain specialization 和评测偏差 | 方法优越性；FathomNet SOTA；无偏泛化 |

### 5.3 不建议列为贡献的内容

- FC-CLIP、Mask2Former、geometric ensemble、hard mask pooling 本身；
- PosSAM 原始 LDP 的概念；
- 原始 EVP 的 patch/high-frequency prompting；
- 尚未通过当前实现的 `amplitude_selected_bandpass` 探索分支；
- FathomNet 数据整理本身，除非后续形成可发布的数据集、协议和 matched baselines。

---

## 6. Global Narrative Arc

### 6.1 全文五步叙事

1. **Preserve**：frozen convolutional CLIP 是值得保留的开放词汇语义底座；
2. **Expose the gap**：同一 mask 的结构表示与 CLIP 区域表示缺少受约束的 learned correction；
3. **Correct locally**：PGC-LDP 用 same-query、per-head、zero-self residual 完成选择性修正；
4. **Adapt cautiously**：frequency-conditioned EVP 在不全量 fine-tune backbone 的前提下注入输入相关局部线索；
5. **Validate honestly**：PGC 提供主要 AP 增益，void preservation 是关键边界，EVP 增益有限，跨域结果揭示剩余问题。

### 6.2 开篇与结尾的镜像关系

- 开篇问题：如何同时保护开放词汇语义并适配 mask-specific structure？
- 结尾答案：same-mask paired correction 是当前最有效的适配点；但 single-seed、unseen classification、跨域性能与效率仍未闭环。

---

## 7. 各章节大纲与 rhetorical moves

### Abstract

建议 5 个动作、约 170–220 English words：

1. 一句定义水下开放词汇实例分割及其固定类别限制；
2. 一句提出 frozen FC-CLIP 的优势与 region-representation interaction 缺口；
3. 两句介绍 PGC-LDP 的 paired keep-or-correct、preserve-void，以及次级 EVP；
4. 一句报告 final 的 baseline/PGC/combo 数字，最高 checkpoint 单独标记；
5. 一句收束：结果支持 query-aligned correction，但不声称 SOTA、强协同或广泛泛化。

摘要中不放 FathomNet `90.2243 AP`；在没有 matched baseline 前，也不建议放 full-test `15.5106 AP`。

### 1. Introduction

#### P1. Task and application gap

- 定义 underwater open-vocabulary instance segmentation；
- 指出固定词表和高标注成本限制真实场景覆盖；
- 水下成像退化相关表述等待补充领域文献，避免无引用物理解释。

#### P2. VLM opportunity and FC-CLIP baseline

- CLIP 提供 image–text alignment；
- FC-CLIP 的 shared frozen convolutional backbone 同时支持 dense features 和 open-vocabulary classifier；
- 将其定位为本文承接的 design principle。

#### P3. Representation interaction gap

- 同一 mask 产生 decoder structure token 与 frozen-CLIP region token；
- baseline 在不同分类路径中利用二者；
- 提出对同 mask token-level correction 的需求。

#### P4. Why existing adaptation/fusion is insufficient

- SAN：side adapter/attention bias；
- PosSAM：SAM–CLIP LDP；
- CAT-Seg：cost-space aggregation；
- 指出本文特有约束是 query-index pairing、unmatched slots、void preservation，而不是笼统地说前述方法“不适用”。

#### P5. Core method

- paired keys/values；
- zero-self keep-or-correct；
- no cross-query read；
- final non-void replacement + base void concatenation。

#### P6. Secondary visual adaptation

- 原始 EVP 的 input-conditioned prompting；
- 本文 high-pass 主残差与 low-pass 条件门；
- stage 1/2 gated addition，frozen CLIP weights。

#### P7. Evidence and contributions

- final PGC `+6.3678 AP`；
- final combo `54.0845 AP`，best `54.4971 AP @40k`；
- 三条贡献；
- 可在段尾轻量提示 single-seed 边界，详细限制留给 Discussion。

### 2. Related Work

#### 2.1 Open-Vocabulary Segmentation

组织方式：two-stage region-to-text → single-stage/frozen VLM → dense cost-based methods。

- FC-CLIP：直接基线；
- SAN：frozen CLIP + side adaptation；
- CAT-Seg、SED：pixel/text cost aggregation 与 hierarchical encoding；
- 结尾差异句：本文不重新设计完整 mask generator，而是聚焦 FC-CLIP 内部 same-mask region representations 的受约束修正。

#### 2.2 Region-Level Mask Classification and Feature Fusion

- 讨论 mask pooling、in/out-vocabulary classifiers、score ensemble；
- 重点介绍 PosSAM LDP，但准确说明其 SAM spatial features、CLIP features 和 self-attention fusion；
- 结尾差异句：PGC 的核心不是更大范围 fusion，而是 `i ↔ i` 配对、zero-self value 和 no cross-query mixing。

#### 2.3 Parameter-Efficient Adaptation of Frozen Vision Models

- SAN 的 side adapter；
- prompt/adaptor 类方法；
- Explicit Visual Prompting 的 patch embedding 与 high-frequency prompt；
- CAT-Seg 作为 fine-tuning 也可能成功的反例，避免把冻结策略写成普遍真理；
- 结尾差异句：本文使用 low-frequency content 作为 high-frequency residual 的 gate condition，并限制在 ConvNeXt early stages。

#### 2.4 Underwater Segmentation and Datasets

- 需后续补充专门文献；
- 分开写 closed-vocabulary underwater instance segmentation、open-vocabulary underwater datasets、WaterOVS/FathomNet；
- 不要用项目内部实验记录代替公开文献引用。

### 3. Method

#### 3.1 Problem Formulation and FC-CLIP Baseline

- 定义输入图像、训练词表、测试词表、`N` 个 mask queries；
- 定义 mask、decoder structure token、pixel-decoder pooled representation、frozen-CLIP dense map；
- 简述 in-vocabulary、out-of-vocabulary logits 和 geometric ensemble；
- 明确 void logit 的来源和作用。

推荐符号：

- `M_i`：第 `i` 个预测 mask；
- `S_i`：第 `i` 个 decoder structure token；
- `C_i`：用 `M_i` hard/detached pooling 得到的 CLIP region token；
- `z_i^base`：baseline final logits，最后一维为 void；
- `z_i^pgc`：PGC non-void logits。

#### 3.2 Architecture Overview

- 图示完整数据流：image → frozen ConvNeXt CLIP + optional EVP → pixel/mask decoder → final masks/structure tokens → CLIP pooling → PGC → base void → ensemble；
- 用颜色区分 frozen、trainable、inference-only；
- 明确 PGC/EVP 不增加新监督目标。

#### 3.3 Query-Aligned Paired Gated Correction

##### 3.3.1 Paired token construction

- `S_i` 来自 final decoder query `i`；
- `C_i` 由该 query 自己的 mask `M_i` 汇聚 frozen dense CLIP feature；
- pooling mask 是 hard 且 detached；
- pairing 由 mask/query 语义保证，不是仅凭 tensor shape。

##### 3.3.2 Per-head keep-or-correct

写出：

```text
q_i = W_q C_i
k_i = [W_k C_i, W_k S_i]
v_i = [0, W_v S_i]
a_i = softmax(q_i k_i^T / sqrt(d_h))
R_i = W_o(a_i[structure] * W_v S_i)
O_i = LayerNorm(C_i + R_i)
```

解释三个 inductive biases：

1. paired-only：query `i` 不读取 `j ≠ i`；
2. zero-self：选择 self 表示不写入结构 value；
3. residual correction：结构信息是修正，不是替换 CLIP 起点。

不要称其为“identity-preserving”，因为 projection 与 LayerNorm 可改变表示。

##### 3.3.3 Final-only and preserve-void integration

- 只用 `O_i` 生成 final non-void logits；
- 拼回 baseline 的 final void logit；
- auxiliary classification/mask/dice losses 全部保持 baseline；
- 用 5k collapse ablation 解释这个边界为何必要。

#### 3.4 Frequency-Conditioned Explicit Visual Prompting

##### 3.4.1 Relation to original EVP

- 先引用原始 EVP；
- 明确复用 frozen backbone + explicit image-conditioned prompt 的范式；
- 再说明本文面向 ConvNeXt FC-CLIP 的改造。

##### 3.4.2 High- and low-frequency branches

- FFT centered mask、`freq_rate=0.25`；
- high-pass 为实际注入残差；
- low-pass 只作为 gate condition，不直接加到 backbone feature。

##### 3.4.3 Spatial gate and injection

- gate 输入：`F, Δ_H, C_L, ||Δ_H||_2, ||C_L||_2`；
- `1×1 → GELU → depthwise 3×3 → GELU → 1×1 → sigmoid`；
- `F_next = ConvNeXtBlock(F + G ⊙ Δ_H)`；
- stages 1/2、每个 block 前注入；
- bias `-2` 对应初始 gate 约 `0.1192`，是保守初始化而非精确 identity。

#### 3.5 Training and Inference

- PGC/EVP 不增加专用损失；
- 列明哪些参数 trainable、哪些 frozen；
- 训练时 final-only replacement，推理时保留 FC-CLIP geometric ensemble；
- 如果主文篇幅有限，把全部 YAML 参数放 supplementary。

#### 3.6 Complexity

- 可以报告 ConvNeXt-Base 模块参数：PGC `2,216,960`，global LDP `2,626,560`；
- 可以报告已有 analytic core-MAC：`0.65600G` vs `0.73536G`；
- 标题必须用 `Module-level Analysis`，不能写 `Efficiency Evaluation`；
- 端到端 latency、memory、throughput 尚为 `TBD`。

### 4. Experiments

#### 4.1 Research Questions

- RQ1：PGC-LDP 是否提高 matched WaterOVS AP？
- RQ2：提升主要来自 PGC、EVP，还是二者组合？
- RQ3：preserve-void 是否为必要边界？
- RQ4：现有 gains 更接近 classification/fusion 变化还是 mask localization 改善？
- RQ5：模型在 FathomNet/WaterOVS 跨数据设置和不同 checkpoints 下表现如何？

#### 4.2 Datasets and Evaluation Protocol

##### WaterOVS

- train 84 类，validation 115 类；
- 主表：ConvNeXt-Large、3 GPUs、global BS9、FP32、50k、LR `1e-4`、seed 42；
- metric：COCO-style segmentation AP/AP50/AP75/APs/APm/APl；
- 解释 ID 的具体含义，不把它自动等价为纯 seen-class evaluation。

##### FathomNet

- train：8,661 images、8,675 retained instances、67 classes；
- retained test：1,155 images、1,163 instances、41 classes，Pred-IoU50 filtered，无 COCO-small instances；
- prediction-free full test：8,988 images、13,960 annotations、173 prompt categories；
- 把 retained 与 full test 分成两张表或两个 panel，避免数值被误作同一 protocol。

#### 4.3 Implementation Details

- baseline 和四路消融的 shared settings 放主文；
- exact config、commit、checkpoint 命名放 supplementary；
- 明确 best checkpoint 与 final checkpoint 的 selection rule；
- FathomNet seed 为 `-1`，不得与 WaterOVS fixed seed 42 混写。

#### 4.4 Main Matched Ablation

主表固定顺序：baseline → EVP-only → PGC-only → combined。

建议主文以 final checkpoint 为第一口径：

| Method | AP | ΔAP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| FC-CLIP | 47.5934 | — | 55.0646 | 50.9703 | 14.3429 | 25.5046 | 48.8541 |
| + EVP | 49.7202 | +2.1268 | 57.3223 | 52.9450 | 26.2014 | 30.7396 | 50.6668 |
| + PGC-LDP | 53.9612 | +6.3678 | 62.3810 | 57.8562 | 10.6254 | 32.5768 | 55.0848 |
| + PGC-LDP + EVP | 54.0845 | +6.4911 | 62.7847 | 57.9002 | 9.0898 | 31.4751 | 55.1517 |

表后按以下顺序解释：

1. PGC 是主要 AP 来源；
2. EVP-alone 有独立收益；
3. combined final 仅比 PGC final 高 `0.1233`，不支持强协同；
4. small-object AP 波动明显，不能从总体 AP 推断各尺寸均改善；
5. best-checkpoint 表放在下一小节或 supplementary。

#### 4.5 Checkpoint Sensitivity

- EVP-only best `50.6923 @40k`，final `49.7202`；
- PGC-only `53.4909 @40k`，final `53.9612`；
- combined best `54.4971 @40k`，final `54.0845`；
- 同 checkpoint 40k 时 combined 相对 PGC 为 `+1.0062`，final 为 `+0.1233`。

结论写成“incremental effect varies with checkpoint”，不要写成“EVP consistently enhances PGC”。

#### 4.6 Mechanism and Boundary Ablations

##### 已有证据

- foreground+void replacement：`1.2070 AP @5k`；
- preserve base void：`42.4391 AP @5k`；
- global LDP diagnostic：matched mask IoU 略降、in-vocabulary top-1 上升，支持 classification/fusion 解释；
- seen ensemble top-1 提升而 unseen top-1 未提升，禁止 broad unseen-recognition claim。

##### 投稿前应补齐

- matched Large global-LDP vs PGC-LDP；
- paired-only vs cross-query mixing；
- zero-self value vs learned/nonzero self value；
- final-only vs all decoder layers；
- main-protocol plain high-pass EVP vs low-conditioned gate；
- 至少 3 seeds 的 mean ± std。

#### 4.7 Cross-Dataset and FathomNet Stress Test

按“测量 → 限制 → 解释”的顺序：

- retained train self-eval final：`90.2243 AP`，只代表训练拟合；
- retained test：best `38.9187 AP @40k`，final `32.9940 AP`；
- FathomNet-trained → WaterOVS：best `31.5323 AP @20k`，final `30.3092 AP`；
- prediction-free full FathomNet test：40k model，`15.5106 AP`、`17.8786 AP50`、`16.1916 AP75`、`3.7614 APs`、`11.5631 APm`、`20.3655 APl`。

允许的结论：不同外部 criteria 选择不同 checkpoints，模型存在明显 domain/checkpoint specialization；full test 揭示小目标和长尾类别仍是难点。

禁止的结论：FathomNet 泛化优越、full test 无偏地证明 PGC 有效、退化完全由 conventional overfitting 导致。这里没有 matched FathomNet baseline，也没有因果机制隔离。

#### 4.8 Qualitative and Error Analysis

主文建议一张 4 列图：baseline、EVP、PGC、ground truth；选择：

- 同一 mask 语义被 PGC 修正的成功例；
- void/duplicate proposal 被正确抑制的例子；
- small/low-contrast instance 失败例；
- FathomNet long-tail/zero-AP 类失败例。

必须从实际 predictions 生成，不可根据模块命名臆造“边界更清晰”或“高频恢复纹理”。

### 5. Discussion

#### 5.1 What the evidence supports

- query-aligned final classification correction 是最强支持的机制；
- preserve-void 是 FC-CLIP 集成时的关键语义边界；
- EVP 能独立提升，但与 PGC 的增量有限且 checkpoint-sensitive。

#### 5.2 What remains uncertain

- PGC 与 global/cross-query fusion 的严格优势；
- unseen recognition 是否真正改善；
- gains 是否跨 seeds、代码版本、数据集稳定；
- frequency gate 的新颖性及物理解释；
- 端到端效率；
- FathomNet 上相对 baseline 的改进。

#### 5.3 Broader methodological implication

可以提出但需保持适度：对带固定 query 和特殊 no-object competition 的 open-vocabulary segmentation，融合模块的索引对齐和 null semantics 可能与融合容量同样重要。

使用 `suggests` / “提示”而不是 `demonstrates generally` / “普遍证明”。

### 6. Conclusion

结论只保留三点：

1. 提出适配 FC-CLIP 的 same-mask PGC-LDP；
2. matched WaterOVS single-seed 实验中 PGC 是主要增益来源；
3. frequency prompting、跨域与效率仍需要更严格验证。

不要在结论重新引入摘要和结果中未出现的新主张。

### Supplementary Material

建议包含：

- 完整 architecture/config table；
- best/final 的全 checkpoint curves；
- PGC shape、gradient、hard pooling、preserve-void tests；
- per-category 与 seen/unseen/head-tail 结果；
- FathomNet 类别映射和 retained/full protocol；
- qualitative cases；
- 失败的 exploratory `amplitude_selected_bandpass` 不应作为已支持模块，可放 future work 或完全移除。

---

## 8. Figure/Table Narrative

| 编号 | 内容 | 服务的论点 | 当前状态 |
|---|---|---|---|
| Fig. 1 | FC-CLIP baseline 与 PGC-LDP 对比：separate region paths → same-mask paired correction；显式画出 void bypass | 动机与核心贡献 | 需绘制 |
| Fig. 2 | 完整系统：EVP early-stage + decoder + PGC final-stage | 方法数据流 | 可由代码重建 |
| Table 1 | 与公开方法的 WaterOVS 对比 | 外部竞争力 | 缺少统一协议/公开比较表，TBD |
| Table 2 | 四路 matched final ablation | 归因 PGC/EVP | 已有 |
| Table 3 | PGC mechanism/boundary ablation | paired、zero-self、preserve-void | 部分已有，需补齐 |
| Fig. 3 | WaterOVS 与 FathomNet checkpoint curves | checkpoint sensitivity | 数据已有 |
| Table 4 | FathomNet retained/full/transfer stress test | 局限和域专化 | 已有，但无 matched baseline |
| Table 5 | 参数、MAC、latency、memory | 效率 | 仅 module params/MAC 已有，其余 TBD |
| Fig. 4 | 成功与失败 qualitative cases | 机制解释与限制 | 需从 predictions 生成 |

如果篇幅有限，优先级为 `Fig.1 → Table 2 → Fig.2 → Table 3 → Fig.3`。没有完成统一外部比较前，不要让 Table 1 占据主结果位置。

---

## 9. Style and Terminology Decisions

### 9.1 稳定术语

| 首次定义 | 后续固定用法 | 避免混用 |
|---|---|---|
| Paired Gated Correction for Local Discriminative Pooling | PGC-LDP 或 PGC | paired LDP、local fusion、PGC attention 随意切换 |
| decoder structure token | structure token | spatial token、mask token（除非另行定义） |
| mask-pooled CLIP region token | CLIP region token | semantic anchor（过强） |
| no-object/void logit | void logit | background class |
| frequency-conditioned Explicit Visual Prompting | frequency-conditioned EVP | physical prompt、frequency restoration |
| WaterOVS in-domain validation | WaterOVS ID | seen-only evaluation |
| prediction-free FathomNet full test | full FathomNet test | unbiased benchmark（过强） |

### 9.2 语态与 claim strength

- 代码事实：`implements`, `uses`, “实现/采用”；
- 单次测量：`achieves`, `improves in our matched run`, “在该受控运行中提高”；
- 机制解释：`is consistent with`, `suggests`, “与……一致/提示”；
- 未验证假设：`may`, `we hypothesize`, “可能/我们假设”。

### 9.3 禁用或慎用表达

- `novel`：完成外部 novelty audit 前不用；
- `unbiased classification`：现有 unseen top-1 证据不支持；
- `preserves the original CLIP feature exactly`：projection 和 LayerNorm 使其不成立；
- `restores underwater details`：没有图像恢复或频率因果证据；
- `efficient`：除非限定为 module-level parameter/MAC；
- `robust/generalizable`：缺少多种子、多数据集 matched baselines；
- `synergistic/complementary`：组合 final 增量不足。

### 9.4 数字报告规则

- 默认以 final checkpoint 作主表；
- best checkpoint 必须标 iteration，不能和 final 混为一栏；
- 所有提升注明比较对象；
- single-seed 数字不配显著性措辞；
- `APs = NaN` 或 `0` 要解释数据集中是否存在 small objects，不做模型能力推断。

---

## 10. Claim-Evidence Map

| Claim | 证据 | 可用措辞 | 强度 |
|---|---|---|---|
| PGC improves WaterOVS AP | baseline `47.5934`, PGC final `53.9612` | “improves AP by 6.37 points in the matched single-seed setting” | 强，范围受限 |
| preserve-void is necessary in current integration | `1.2070` vs `42.4391 @5k` | “is essential in the tested configuration” | 强，配置受限 |
| EVP independently helps | EVP final `49.7202` vs `47.5934` | “provides a 2.13-point final AP gain” | 中强 |
| combo is best observed configuration | `54.4971 @40k`, `54.0845 final` | “reaches the highest observed checkpoint” | 强 |
| PGC+EVP have strong synergy | final incremental `+0.1233` | 不主张；写 “limited incremental gain” | 不支持 |
| PGC improves unseen recognition | unseen top-1 `-0.556pp` in related global-LDP diagnostic | 不主张；只能写 overall AP/classification-fusion changes | 不支持 |
| PGC improves mask localization | matched mask IoU 反而略降于 global-LDP diagnostic | 不主张 | 不支持 |
| method generalizes to FathomNet | full test `15.5106 AP`, no baseline | “evaluated as a stress test” | 仅描述性 |
| model is efficient | module params/MAC only | “has lower analytic module-level cost than global LDP” | 有限 |

---

## 11. Cross-Section Consistency Plan

1. Abstract、Introduction、Results、Conclusion 都将 PGC 写为主要贡献；
2. EVP 始终是 secondary adaptation，不在 Discussion 中突然升级为主创新；
3. Methods 中的 paired/no-cross-query/zero-self/final-only/preserve-void，必须分别在 ablation 或 limitations 中有对应；
4. Results 所有 best/final 数字与 `content_analysis.md` 保持一致；
5. FathomNet full-test 新结果引用 raw evaluator log，并注明 `content_analysis.md` 的 pending 状态已过时；
6. Related Work 必须明确 PosSAM LDP 和原始 EVP 的归属；
7. 不把 global-LDP diagnostic 直接写成 PGC 的机制证明，应明确它是 related diagnostic；
8. `in-vocabulary/out-of-vocabulary` 与 `seen/unseen` 只有在类别划分严格定义时才互换。

---

## 12. Downstream Writing Handoff

### 给 `research-abstract-writer`

- 主问题：同 mask structure/CLIP representations 缺少 query-aligned correction；
- 主方法：PGC-LDP + preserve-void；
- 次方法：frequency-conditioned EVP；
- 主数字：`47.5934 → 53.9612 final AP`；combo `54.0845 final`, `54.4971 best @40k`；
- 必须保留：single-seed、无 SOTA、无强协同。

### 给 `research-introduction-writer`

- 按本蓝图第 4.2 节的 7 段结构；
- FC-CLIP 是设计起点；PosSAM/EVP 是思想来源；
- 核心 gap 是 query-aligned correction 与 null semantics，不是笼统的“CLIP 缺少空间能力”。

### 给 `research-methods-writer`

- 以 `content_analysis.md` 的 3.1–3.5 为事实源；
- 必写 hard detached pooling、paired token provenance、zero self value、LayerNorm、final-only、base void、无新增 loss；
- EVP 必写 high/low 两支的不同作用和 gate bias。

### 给 `research-results-writer`

- final-first，best-second；
- 先四路 matched ablation，再 boundary/mechanism，再 FathomNet stress test；
- 把 measured result 与 interpretation 分段；
- full FathomNet test 使用 `15.5106 AP` 新结果，而不是 pending。

### 给 `research-discussion-writer`

- 主讨论：classification/fusion vs localization、void semantics、checkpoint sensitivity；
- 局限：single seed、missing global-LDP control、unseen evidence、runtime、FathomNet baseline、dataset bias。

### 给 `research-conclusion-writer`

- 只总结 PGC 的 matched gain、preserve-void 边界和 EVP 的有限补充；
- 结尾落在未来需要 multi-seed、cross-dataset baselines 和 efficiency measurement。

### 给 `research-title-writer`

- 推荐关键词：query-aligned、paired correction、frozen CLIP、open-vocabulary instance segmentation、underwater imagery；
- 避免：SOTA、robust、efficient、generalizable、FathomNet benchmark。

---

## 13. 投稿前证据门槛与待决事项

### 必须补齐

1. 四路 matched experiment 至少 3 个固定 seeds，报告 mean ± std；
2. main Large protocol 下 global-LDP vs PGC-LDP；
3. paired/cross-query、zero-self、preserve-void、low-condition 的核心消融；
4. 统一的公开方法比较协议；
5. end-to-end trainable/total params、FLOPs、peak memory、throughput、latency；
6. FathomNet baseline/EVP/PGC/combo matched grid；
7. per-class、head/medium/tail、exact seen/unseen 与 size-stratified analysis；
8. 更新 FathomNet full-test `实验记录.md` 与 `content_analysis.md` 中已经过时的 pending 状态。

### 需要作者决定但不阻塞初稿

1. 具体投稿会场及页数；
2. 是否把 FathomNet 放主文的压力测试，还是只放 supplementary/discussion；
3. PGC-LDP 的最终英文全称；
4. frequency-conditioned gate 是否经过完整 novelty audit 后作为独立方法点，或明确保持为工程适配；
5. 论文最终聚焦 instance segmentation，还是在补齐 PQ/mIoU 后扩大到 universal/panoptic segmentation。

---

## 14. 参考论文来源

1. [FC-CLIP: Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP](https://papers.nips.cc/paper_files/paper/2023/hash/661caac7729aa7d8c6b8ac0d39ccbc6a-Abstract-Conference.html)
2. [SAN: Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Side_Adapter_Network_for_Open-Vocabulary_Semantic_Segmentation_CVPR_2023_paper.html)
3. [PosSAM: Panoptic Open-vocabulary Segment Anything](https://arxiv.org/abs/2403.09620)
4. [CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Cho_CAT-Seg_Cost_Aggregation_for_Open-Vocabulary_Semantic_Segmentation_CVPR_2024_paper.html)
5. [SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_SED_A_Simple_Encoder-Decoder_for_Open-Vocabulary_Semantic_Segmentation_CVPR_2024_paper.html)
6. [Explicit Visual Prompting for Low-Level Structure Segmentations](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Explicit_Visual_Prompting_for_Low-Level_Structure_Segmentations_CVPR_2023_paper.html)

