# ICASSP 场景研究档案：水下开放词汇分割 Motivation 适配性

## 结论摘要

当前 motivation **在研究主题上适合 ICASSP 2027，但在论文组织上只能判为“有条件适合”**。

适合之处在于：水下开放词汇分割属于 ICASSP 2027 明确列出的 `Computer Vision`、`Image, Video & Multidimensional Signal Processing`、`Machine Learning & Generative AI` 范围；现有方法也具有可被信号处理审稿人识别的技术对象——频率分解、条件门控、冻结视觉语言表征的受限适配、同 query 区域表征校正，而不只是场景应用。

主要风险在于：当前 motivation 将论文组织为两个连续接口——`underwater image -> visual representation` 与 `predicted mask -> open-vocabulary region representation`——但两个接口没有共同的 failure mechanism。这个表述作为完整论文的逻辑框架是诚实且合理的；作为 ICASSP 四页技术正文的 controlling motivation，则容易形成“两个各自合理但篇幅不足以充分证明的模块”。因此，投稿版应把核心贡献收束为：**在冻结开放词汇模型的数据流中，对两个表征传递接口进行轻量、受限、选择性的适配**。水下频率条件适配是主场景问题，same-query semantic correction 是下游 readout 的配套机制。不要声称两者解决同一种退化，也不要并列成两个同等展开的宏大理论。

当前实验给出了较强的 ID 性能信号，但尚不足以支撑“提升水下开放词汇泛化”或“两个接口分别有效”的完整结论。最稳妥的 ICASSP 定位是**一个紧凑的信号处理/表征适配方法论文**，而不是新的水下退化理论、通用 OVS 框架或已被充分验证的跨域泛化方案。

## Venue Requirements

### 1. 投稿届次与时间

按当前日期与官方公开信息，本档案采用 **ICASSP 2027** 作为目标届次。官方 Call for Papers 给出的 regular full paper 截止日期为 2026-09-16，录用通知为 2027-01-13，会议于 2027-05-16 至 2027-05-21 在 Toronto 举行。

### 2. 篇幅与形式约束

- Regular paper 最多 **4 页技术内容**；可加第 5 页，但第 5 页只能放 references、funding acknowledgements 和 Compliance with Ethical Standards statement。
- 论文必须是原创、未发表且不得一稿多投。
- 投稿会先检查模板、篇幅、结构、研究主题、语言、作者信息，并进行抄袭/自抄袭检查。
- 评审为 single-anonymous；至少三位 reviewer。作者可提交 rebuttal。
- 官方鼓励公开数据、代码和 supporting content，以增强可复现性。

对本文最重要的现实含义是：四页内不能同时充分展开水下成像物理、频率路由、开放词汇 mask readout、PGC 注意力细节和大规模实验。Introduction 中的 motivation 必须在一至两段内完成“场景问题—接口缺口—方法选择—证据”闭环；Method 应优先保留可公式化、可消融的机制，避免长篇铺陈两个互不相同的 failure mechanism。

### 3. 主题范围与建议 track

ICASSP 2027 官方范围明确包含：

- Computer Vision；
- Image, Video & Multidimensional Signal Processing；
- Computational Imaging；
- Machine Learning & Generative AI；
- Signal Processing in Emerging Technologies。

本文首选应落在 **Image, Video & Multidimensional Signal Processing / Computer Vision 相关 EDICS**，次选可考虑 Machine Learning for Signal Processing。不要仅以“水下机器人应用”或“开放词汇视觉”投向过宽泛的 track；官方政策指出错误 EDICS 会被检查和重分配，而 reviewer 分配直接依赖 track/专业匹配。

### 4. 官方来源

- [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/)
- [ICASSP 2027 官方 CFP PDF](https://2027.ieeeicassp.org/wp-content/uploads/sites/13/2019/08/ICASSP2027-CallForPapers-July1.pdf)
- [ICASSP 2027 Editorial Policies](https://2027.ieeeicassp.org/about/editorial-policies/)
- [ICASSP 2027 Publishing and Paper Presentation Options](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/)

## Review Criteria

ICASSP 2027 官方列出六项评审标准。下面将其直接映射到当前 motivation。

| 官方标准 | 对本文的要求 | 当前状态 | 主要风险 |
|---|---|---|---|
| Language and Clarity | 在很短篇幅内清楚区分两个接口、两个模块和各自 claim boundary | motivation 文档的边界意识较强 | “两个连续接口”若写得太抽象，会掩盖模块间不存在共同 failure mechanism |
| Importance/Relevance | 不仅说明水下场景困难，还要说明冻结 VLM 向恶劣观测和 mask-based recognition 迁移时的普遍接口问题 | 有潜在广泛价值；任务也在 scope 内 | 若主要叙述 WaterOVS 单数据集增益，会被视为场景工程优化或 limited interest |
| Novelty/Originality | 说明 low-conditioned selective residual 与 same-query correction 相对 EVP、adapter、FC-CLIP score ensemble、mask adapter 的结构差异 | 两个模块均存在具体 architecture gap | 两种轻量模块组合容易被质疑 incremental；必须突出“受限写入/keep-or-correct”的共同设计原则，而不是组件堆叠 |
| Technical Correctness | 公式与真实实现一致，避免不受支持的频率语义解释和“preserve semantics”绝对化 | 设计文档提供了精确实现边界 | `low frequency = degradation`、`high frequency = object`、严格保持 CLIP 语义等表述均不成立 |
| Experimental Validation | 每个主要 claim 都应有匹配实验；性能论文需要充分评估 | Large BS9 final AP 54.0845，较 Large BS9 baseline 47.3413 高 6.7432；有强候选信号 | 当前是 single-run ID，代码路径/AMP/训练配置未完全匹配；缺 CD 与模块级严格消融，无法归因 |
| Reference to Prior Work | 覆盖 underwater OVS、FC-CLIP、EVP/prompt adaptation、frequency-aware vision、mask/region semantic adaptation | 当前 motivation 已意识到 SAN / Mask-Adapter 等相关讨论 | 尚未由本档案验证完整引用覆盖；缺关键相关工作可直接造成拒稿风险 |

### 评审阈值下的判断

- **Relevance：通过。** 主题与 ICASSP scope 直接对齐，不需要勉强包装。
- **Clarity：有条件通过。** 两接口视角比“同一退化贯穿两层”更可信，但必须压缩成一条中心问题。
- **Novelty：待论证。** 现有结构具有技术具体性，但组合式贡献需用统一约束原则和对应消融来避免 incremental 印象。
- **Technical correctness：当前文字方向正确。** 前提是保留所有谨慎边界，不升级成因果或物理恢复主张。
- **Validation：当前最弱项。** 现有结果支持“值得投稿准备”，不支持 submission-ready 的强结论。

## Accepted Paper Patterns

以下是由 ICASSP 官方篇幅、scope 与评审标准推导出的目标场景模式，不是对若干已录用论文进行统计后得到的经验比例；具体 exemplar 盘点应由 `exemplar_learning_dossier.md` 单独承担。

### 1. 单一中心问题，少量可验证机制

四页技术正文更适合“一项清楚的问题 + 一套紧凑机制 + 关键消融”，而不适合两个并列论文故事。本文可保留两个模块，但应将它们写为同一预测链上的两个受限适配点，并明确主次：

1. 主问题：自然图像预训练的冻结视觉语言表征如何适应水下观测；
2. 主机制：频率条件下的浅层选择性视觉残差注入；
3. 配套机制：在 mask-based readout 后，用 same-query task evidence 对 CLIP region embedding 做 keep-or-correct 式校正。

### 2. 信号处理对象必须显式

ICASSP 的适配版本应把方法写成具体的信号与算子：FFT high/low responses、空间 gate、残差写入位置、same-query paired tokens、attention gate 和 correction-only value。水下应用是问题场景；真正让论文落入 ICASSP 语境的是对多尺度/频率信号和表征传递的明确处理。

### 3. Introduction 迅速抵达 architecture gap

可采用以下压缩结构：

1. 水下观测与自然图像预训练域存在显著外观差异，但全面微调可能损害或扰动已有开放词汇能力；
2. FC-CLIP 类 mask-based OVS 还存在一个可利用的结构机会：同一 query 同时产生 CLIP-derived 与 task-learned region evidence，而原路径主要在 score level 融合；
3. 因此提出受限的两点适配，只在浅层写入 frequency-conditioned residual，并只对同 query CLIP region anchor 写入 gated correction；
4. 用 matched baseline、模块消融、ID/CD 或 seen/unseen 分析验证对应承诺。

### 4. 每项机制对应一个最小证据闭环

- EVP 必须有：无 EVP、high-pass 普通 gate、low-conditioned gate、Stage 1/2 与其他 stage 范围的比较，最好再有 gate/频率响应诊断。
- PGC-LDP 必须有：原 FC-CLIP score ensemble、global LDP 或非配对替代、PGC-LDP；还应证明 same-query pairing 或 correction-only design 的必要性。
- 联合模型必须有：baseline、EVP only、PGC only、EVP + PGC 的匹配四格消融。
- 开放词汇主张至少需要 seen/unseen、ID/CD 或跨数据集证据之一；只有 overall ID AP 不能独立验证开放词汇泛化。

### 5. Claim 小于 evidence

ICASSP 官方把技术正确性与实验验证明确列为评审项。适合的写法是“selectively adapts”“avoids direct modification”“provides task-learned correction”；不适合的写法是“restores underwater details”“preserves high-level semantics”“solves recognition-support mismatch”“improves generalization”，除非有对应诊断与控制实验。

## Constraints for This Paper

### 1. Motivation 的可保留部分

以下判断与当前代码/设计文档一致，可以保留：

- 把完整预测链分成 `image -> visual representation` 与 `mask -> region representation` 两个连续接口；
- 明确两者位于同一数据流，但不具有相同 failure mechanism；
- 将 EVP 定位为 `frequency-conditioned shallow visual adaptation`，其中 high-frequency response 是候选 residual，low-frequency response 仅作为 gate condition；
- 将 PGC-LDP 定位为 `region-level semantic adaptation after mask-based readout`；
- 强调 CLIP-derived 与 task-learned region representations 来自同一 query、同一 predicted mask，因而可以严格配对；
- 用“限制显式适配位置”而非“严格保留深层语义”解释 shallow-only design。

### 2. 必须收束或删除的表述

- 不要把 `low frequency` 等同于水下退化，也不要把 `high frequency` 等同于目标信息。
- 不要声称 low-frequency branch 能判定 high-frequency response 是否可靠；实现上它只是 gate 的一个条件。
- 不要声称 EVP 恢复了丢失频率、增强了真实边缘或保证了高层语义不变。
- 不要声称 PGC-LDP 改进了 mask pooling、学习 recognition mask、使用 mask 外 context，或直接解决 segmentation support 与 recognition support mismatch；当前 hard detached MaskPooling 未改变。
- 不要把 PGC 的 self choice 写成严格恒等映射；零 correction 时输出仍是 `LayerNorm(C_i)`。
- 不要把 Base/Large 或不同 batch/AMP/代码路径的结果写成严格单变量因果证据。

### 3. 当前证据允许的最大 claim

现有实验记录允许写：

> 在 WaterOVS ID、ConvNeXt-Large、BS9 的已完成单次运行中，联合 PGC-LDP + low-conditioned high-pass EVP 在 40k checkpoint 达到 54.4971 AP，在 50k final 达到 54.0845 AP；对应 BS9 baseline final 为 47.3413 AP。该结果表明联合设计具有较强性能潜力。

但必须紧接以下限制：baseline 与方法运行的代码路径、AMP/FP32 和训练配置不完全一致，因此 `+6.7432 AP` 不是严格消融所得的可归因增益。相对 BS15 baseline 的比较也受 batch protocol 混杂。40k best 与 50k final 均应报告。

现阶段不能写：

- “显著提升开放词汇泛化”；
- “对跨域/类别分布变化稳健”；
- “两个接口分别贡献了性能提升”；
- “PGC 优于 global LDP”；
- “Large backbone 证明方法具有尺度一致性”。

### 4. ICASSP 投稿前的最低证据门槛

按当前 motivation 的承诺，至少需要：

1. **严格 matched Large BS9 baseline**：相同代码 checkout、precision、global batch、schedule、seed 和数据协议；
2. **四格模块消融**：baseline / EVP only / PGC only / EVP + PGC；
3. **EVP 关键设计消融**：普通 high-pass gate 对 low-conditioned gate，Stage 1/2 对合理替代；
4. **PGC 关键设计消融**：score-level ensemble 或 global/non-paired correction 对 paired correction-only topology；
5. **开放词汇证据**：至少补 CD full、seen/unseen breakdown 或跨数据集评估中的一种，优先 CD full；
6. **稳定性披露**：至少明确 best/final checkpoint 规则；若资源允许，报告多 seed 或误差范围。

若这些证据无法在截止前完成，建议进一步缩小论文：选择 EVP 或 PGC-LDP 作为唯一主贡献，另一模块作为增强项或留待后续完整论文。对 ICASSP 四页稿而言，一个得到严格验证的紧凑贡献通常比两个证据不完整的模块更安全。

### 5. 推荐的 controlling motivation

推荐将当前主线压缩为：

> 将冻结视觉语言模型迁移到水下开放词汇实例分割，不仅要求适应受水下成像条件影响的浅层视觉信号，还要求在 mask-based readout 后有效利用与同一 proposal 对齐的任务区域证据。我们因此在预测链的两个表征接口采用受限的选择性适配：以低频条件控制高频候选残差在浅层 backbone 中的写入，并以 same-query task representation 对 CLIP region anchor 进行 keep-or-correct 式语义校正。两种机制处理不同问题，但共享“只在有明确配对或条件证据时写入 correction”的设计原则。

这一定义适合 ICASSP 的原因是它：

- 以 signal/representation processing 为中心，而非仅以水下应用为中心；
- 保留真实实现中的 frequency conditioning 和 paired correction；
- 不强行制造统一的物理 failure mechanism；
- 能在四页内通过一个共同的“受限选择性写入”原则连接两个模块；
- 为每个主要 claim 指定了可验证的消融。

### 6. 最终适配判断

| 维度 | 判断 |
|---|---|
| ICASSP scope | 合适 |
| 方法的信号处理属性 | 合适，但需前置呈现频率条件与表征校正机制 |
| 当前两接口 motivation 的逻辑真实性 | 合适 |
| 四页内双模块叙事 | 偏拥挤，必须形成主次或进一步裁剪 |
| 当前新颖性表达 | 有潜力，但存在 incremental/组件组合风险 |
| 当前实验支撑 | 强候选信号，但不满足完整归因与开放词汇泛化 claim |
| 综合结论 | **有条件适合 ICASSP；motivation 可用，但需收束贡献并补严格对照与 OOD/类别泛化证据** |

