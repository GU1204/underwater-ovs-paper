# BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation

## Abstract

- [3] 水下实例分割支撑细粒度场景理解；水下退化造成显著空气域差异，SAM 的水下分割性能随之下降。
- [3] 提出 BiPA 适配 SAM 至水下域；构建双提示 underwater SAM 与 foreground-attentive injection block，并将 dense prompt 学习建模为刻画 prompt—model 相互依赖的双层优化。
- [3] 采用两阶段学习：先以 Bayesian optimization 适配 dense prompt，再冻结优化后的 prompt 微调模型参数，实现跨域适配。
- [2] 实验与分析验证 BiPA 的性能与效率；代码公开。

## 1. Introduction

- [3] 水下实例分割分离实例并预测像素级掩码，可用于生态监测和资源开发；水下介质中的波长相关吸收、散射和浑浊不同于空气场景，使现有实例分割方法性能下降。

### 1.1. Related Works

**Underwater Instance Segmentation Datasets.**

- [2] UIIS 是首个水下实例分割数据集，并配套 WaterMask；UIIS10K 提供更大规模数据集及利用 SAM 先验的 UWSAM。
- [2] 后续工作提出 underwater salient instance segmentation 与 USIS-SAM；USIS16K 进一步发布大规模 underwater salient instance segmentation 数据集。

**Underwater Instance Segmentation Methods.**

- [3] 基于 UIIS 与 USIS10K 出现多种专用方法：WaterBox 处理前景与背景强纠缠，UIS-Mamba 增强水下实例特征建模。
- [3] 近期方法将 SAM 视觉先验迁移到水下域，但缺少显式跨域桥接机制；散射、色偏等退化改变数据分布并削弱空气域学得的 prompt-to-mask 泛化。

### 1.2. Our Main Contributions

- [2] 基于 bilevel optimization 提出 BiPA 学习水下域 prompts；Figure 1 展示其定量与定性结果。

- [2] 设计 Bilevel Prompt Adaptation，学习域特定 prompts 引导 SAM 适配水下场景；将 bilevel optimization 引入水下实例分割。

- [2] 对 dense prompt 建立双层模型，刻画 learnable prompt 与模型参数的训练依赖；以 foreground-attentive injection 融合互补信息，增强双提示 underwater SAM。

- [1] 两阶段策略先以 Bayesian optimization 优化上层 prompt，再固定该 prompt 微调模型参数。

- [2] 标准实验与算法分析评估方法效果；未见类别上的 zero-shot 结果用于考察 SAM 在水下场景中的原有泛化能力。

## 2. The Overall Architecture

### 2.1. Underwater SAM with Dual Prompts

- [2] 以受 USIS-SAM 启发的双提示 underwater SAM 为基础模型；adapter-augmented ViT-Huge 提取特征，SAM mask decoder 在两类 prompt 条件下预测分割结果。
- [2] 不同于冻结 dense prompt 的 USIS-SAM，dense 与 sparse prompt 端到端联合学习：前者提供粗区域先验，后者细化边界；训练时仅微调双提示、adapter 与 FAI block。

### 2.2. Foreground-Attentive Injection (FAI) Block

- [2] 基于全局 self-attention 的 ViT 容易遗漏局部细节并受低质量水下图像影响；FAI block 注入前景线索以增强局部表示。

**Foreground-Attentive Feature Extraction.**

- [3] ResNet 的层级多尺度局部特征补充 ViT 全局表示；输入同时送入冻结的 ResNet-50 与 ViT，在选定层级生成并对齐 foreground-attentive features 与 ViT features。

**Global-Guided Feature Injection.**

- [3] 以 ViT 全局上下文条件化特征融合；分别用 Efficient Multi-scale Attention 与 scale-aware sampling 细化、对齐 ResNet 和 ViT 特征，并依据空间分辨率执行上采样、下采样或 multi-conv。
- [2] 通过权重生成器融合 foreground-attentive 与 ViT 特征；权重由异构 backbone 特征生成，其结构见 Figure 3。

## 3. BiPA: Bilevel Prompt Adaptation

### 3.1. Bilevel Modeling for Dense Prompt

- [3] sparse 与 dense prompts 均输入 mask decoder；sparse prompt 由生成器产生，dense prompt 从头学习；dense prompt 与模型参数紧密耦合，前者影响损失景观与下降轨迹，后者约束 prompt 空间。

- [2] 直接端到端联合训练混合两类变量并忽略潜在对应关系；将 dense prompt 学习定义为双层问题，上层优化 dense prompt、下层更新网络参数。
- [2] ρ 表示 dense prompt、ω 表示模型参数，Dtr 与 Dval 分别为训练集和验证集；将 dense prompt 作为模型参数的超参数，并仅用 Dval 选择 prompt。
- [2] 模型权重仅由 Dtr 学习、不参与上层目标的梯度更新；下层在给定 ρ 时优化 ω，上层在给定 ω 时优化 ρ。

- [1] 形成显式层级关系：下层解 ω(ρ) 引导上层 prompt 学习，上层 prompt ρ 反向条件化下层损失景观与下降轨迹。

### 3.2. Two-Stage Learning Strategy

- [3] 常见双层优化通过展开内层训练或隐式求导计算 hypergradient；BLO-SAM 用于仅含 2–8 张训练图像的 few-shot semantic segmentation；本任务内层在 USIS10K 的 7,442 张图像上训练，规模近三个数量级更大。
- [2] 每次上层更新需反向传播大量内层梯度步骤；其显存与时间成本随下层迭代数、训练集规模及上层循环放大，不适用于当前设置。

- [1] 不显式计算 hypergradient，而用 Bayesian Optimization 将内层求解器视为黑盒，在计算预算内搜索上层超参数空间。

**Adaptation Stage (Prompt Learning).**

- [2] 内层从头学习 256 维 dense prompt；Bayesian optimization 将其划分为 16 组，并为每组分配标量权重。
- [2] 上层搜索空间由 256 维降至 16 维，仅通过验证目标更新组权重；Optuna 根据当前组权重、下层参数、初始搜索空间、试验数与验证目标返回下一候选 prompt。

- [2] 每个外层迭代先在训练目标上执行 Ka 次更新，再固定所得 ω 对 ρ 执行 Kb 次黑盒上层优化。
- [2] 下一轮以上一轮 ω warm start；重复循环得到适配后的 prompt ρ*。

**Fine-tuning Stage (Model Refinement).**

- [2] 固定 ρ* 后，在 Dtr 上继续更新网络参数至 Kc 步或收敛；最终参数取最后一次更新所得 ω*。

**Discussion.**

- [3] Figure 4 以假想优化过程解释机制：两种方法从相同初始化出发并运行相同 T 次迭代，naive prompt learning 最终停在次优区域。
- [2] bilevel prompt adaptation 采用更短、更定向的轨迹到达更接近最优点的解；对应分割可视化及 Sec. 5.2 的实验用于对照两类结果。

### 3.3. Training Loss

- [2] 采用标准 Mask R-CNN 损失 ℓs=ℓc+ℓb+ℓm，其中包括分类、回归与像素级二元交叉熵损失；上层目标设为 ℓm，下层目标设为 ℓs。

## 4. Experiments

### 4.1. Implementation Details

**Benchmarks and Metrics.**

- [3] 在 UIIS 与 USIS10K 两个公开基准上评估；前者包含数千张水下实例分割标注图像，后者含超过 10k 张图像并支持 multi-class 与 category-agnostic salient instance segmentation；所有比较方法遵循 USIS-SAM 的训练协议。
- [3] USIS10K 按 7,442/1,596/1,596 划分训练、验证、测试集，并在 UIIS10K 上评估泛化；指标包括 mAP、AP50、AP75 及 APs/APm/APl。

**Parameter Settings.**

- [3] 冻结参数采用 ViT-Huge 权重与 ImageNet 预训练 ResNet-50；输入缩放至 1024×1024，batch size 为 2，AdamW 初始学习率 1×10^-4、weight decay 1×10^-3；迭代参数设为 Ka=30、Kb=21、Kc=7、T=3。
- [2] 按最佳验证性能选择最终模型；实验基于 PyTorch/MMDetection，并在 NVIDIA RTX 4090 上训练。

**Compared Methods.**

- [3] UIIS 对比 Mask R-CNN、Mask Scoring R-CNN、Cascade Mask R-CNN、PointRend、QueryInst、Mask Transfiner、Mask2Former、ConvNeXt-V2、PolarNeXt；USIS10K 对比 S4Net、RDPNet、QQTR、ConvNeXt-V2、Mask2Former、RSPrompter、SAM+Mask R-CNN 与 SAM+Fast R-CNN；两种设置均加入 WaterMask 与 USIS-SAM。

### 4.2. Performance Evaluation

**Underwater Instance Segmentation.**

- [3] Table 1 给出 UIIS 结果；BiPA 在所报告指标上取得最高结果，mAP 相对 USIS-SAM 提升超过 8%。
- [3] Figure 5 的可视化与数值趋势一致；其他方法在低对比区域易漏检、产生不完整边界或碎片掩码，BiPA 恢复细结构；Figure 6 的补充示例呈现更清晰的实例分离、边缘贴合及更少浑浊水域误检。

**Underwater Salient Instance Segmentation.**

- [3] Table 2 报告 USIS10K 的 multi-class 与 class-agnostic 结果；BiPA 各项指标均提升，两种设置的 mAP 至少高于对应第二名 5%；相关可视化见 Figure 5 与 Figure 6。
- [3] 相比其他方法，BiPA 的边界更清晰、碎片更少并保留细结构；其他方法易遗漏细部或合并相邻实例，而 BiPA 保持实例分离与语义细节；该现象在两种评估设置中均出现。

### 4.3. Generalization Evaluation

- [2] SAM 基于超大规模 mask 语料具有较强泛化并可 zero-shot 迁移到新图像分布；BiPA 的水下跨域适配引出是否仍保留 SAM 原有 zero-shot 泛化的问题。

- [1] 考察 BiPA 是否在分布外水下数据、尤其未见类别上保留 SAM 的 zero-shot 泛化，而非仅拟合目标域。

- [3] 通过定性证据从两方面评估，并将 UIIS 预训练模型直接应用于含 10 类的 UIIS10K；其中 Artiodactyla、Mollusk、Garbage 三类相对 UIIS 未见。
- [3] 仅这三类在两个数据集间严格不重叠，其余同名类别的标签定义存在合并或部分对应；因此无法建立可靠的一一类别映射，强制对齐会混合或拆分语义并影响量化结果解释。
- [1] 采用较充分的定性结果展示泛化表现。

**Cross-Scene Generalization within Seen Categories.**

- [3] Figure 7 比较 UIIS10K 多种场景；左侧示例中 WaterMask 与 USIS-SAM 均有漏检且后者总体更强，BiPA 可恢复这些实例并改善弱边界和低对比区域。
- [3] 右侧示例中 WaterMask 掩码结构不完整，USIS-SAM 仍难捕获细小结构；BiPA 在不同场景中呈现更完整实例、更少碎片与更清晰边界。

**Zero-Shot Generalization to Unseen Categories.**

- [3] Figure 8 比较三类未见类别；其他方法常无法恢复完整结构或识别未见目标，且 USIS-SAM 不再稳定优于 WaterMask。
- [2] 部分垃圾目标中 WaterMask 仅恢复部分结构而 USIS-SAM 未识别；BiPA 生成更接近标签的掩码。

- [2] 其他示例中 USIS-SAM 对未见类别边缘刻画不精确，显示 SAM 的预期泛化在该设置下降；同样使用 SAM 先验的 BiPA 在一定程度上保留未见类别泛化能力。

### 4.4. Efficiency Evaluation

- [3] 与 Tables 1–2 平均性能第二的 USIS-SAM 比较，两者使用相同 ViT-Huge；BiPA 额外加入 multi-scale ResNet50 foreground-attentive feature injection，参数/FLOPs 为 737M/3314G，对比 USIS-SAM 的 698M/3300G。
- [3] 参数与 FLOPs 开销分别增加约 5.59% 与 0.42%，UIIS 和 USIS10K 平均 mAP 提升 6.95%；精度增益对应较小容量增加；Sec. 4.3 中 BiPA 的 zero-shot 泛化也高于 USIS-SAM。

## 5. Algorithmic Analyses

- [1] 前两项分析比较 frozen、naive joint update 与 BiPA 三种学习策略。

### 5.1. Effects of the FAI Block

- [3] Figure 9 给出 BiPA 在 UIIS 各项指标上的结果；Figure 10 在每种学习策略内部比较 w/o FAI 与 w/ FAI；三种策略加入 FAI 后均改善细弱结构、边界及碎片/错误合并。
- [3] 该一致性对应 FAI 的前景关注与水下退化鲁棒设计；另在 USIS10K 上测试 ResNet50 与 ViT 多尺度特征的简单相加；该变体为 43.5 mAP，低于 FAI 的 45.2 mAP。

### 5.2. Effectiveness of BiPA

- [3] Table 3 报告两个水下实例分割数据集结果；BiPA 在各项报告指标上高于 frozen 与 naive，并在不同阈值保持差距。
- [3] Figure 11 比较三种策略的水下场景结果；frozen 保留 SAM 先验但主要覆盖粗形状、局部细节不足，naive 相比 frozen 提升有限且覆盖与边界质量接近。
- [2] naive 的表现与 Figure 4 中未到达近标签最优点的路径对应；BiPA 的边界更贴合目标，小实例保持分离。

### 5.3. Intermediate Results in Stages

- [3] 两阶段学习是 BiPA 的核心设计；Table 4 分别报告两阶段中间结果，第一阶段通过学习水下导向 dense prompt 已获得可用实例分割。
- [2] 第二阶段冻结适配后的 prompt 再微调，性能继续提升；Figure 12 显示边界、漏检与掩码准确性同步改善。

### 5.4. Parameter Analysis

**Optimization Interval.**

- [2] 分析 prompt adaptation 阶段的优化区间，并在 UIIS 上比较 Table 5 的三种区间设置。
- [2] 区间增大时优化更稳定、性能提高；默认采用 B=[1,3]。

**Number of Groups.**

- [2] 将 256 维 dense prompt 分组以提高优化效率，并比较不同 group number。
- [2] Table 5 中多数指标在 Ng=16 时最好；更大分组数不必然带来更高性能，因此默认采用 Ng=16。

## 6. Concluding Remarks

- [2] 提出 BiPA 将 SAM 适配到水下域；将 dense prompt 学习定义为与模型参数相互依赖的双层优化。
- [2] 采用两阶段策略完成跨域适配；多基准实验与算法分析报告 BiPA 的性能与效率结果。
- [1] 当前方法受实时性能限制，后续计划用 knowledge distillation 进行硬件感知轻量化，并扩展双层优化框架以适应更严格的效率约束。

## Acknowledgments

- [1] 工作获得中国国家自然科学基金、辽宁省重点研发计划、中央高校基本科研业务费及宁波数字港技术重点实验室等项目支持。
