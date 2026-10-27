# DiveSeg（2026）论文写作脉络

> **论文**：*Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter*  
> **作者**：Zhiyang Chen, Chen Zhang, Hao Fang, Runmin Cong  
> **会议**：AAAI 2026  
> **方法名**：DiveSeg  
> **核心模块**：AquaStyle Aligner + ObjectPrior Prompter

## Abstract

- **[2]** 先说明水下实例分割同时要求像素级理解和实例级区分，再引入 DINO 这类视觉基础模型的强表征能力；开头先建立“UIS 任务重要，但现有模型尚未充分利用 foundation representation”的背景。
- **[2]** 直接给出本文判断：DINO 可以作为 UIS 的有效特征学习器，并提出 DiveSeg；摘要没有把 DINO 本身包装成创新，而是把重点放在**如何让预训练 DINO 适应水下场景**。
- **[2]** 将适配拆成两个模块：AquaStyle Aligner 将水下颜色风格注入 DINO 微调过程，ObjectPrior Prompter 用二值前景提示提供 object-level prior；两者分别对应域风格适配与目标级适配。
- **[1]** 用 UIIS、USIS10K 上的 SOTA 结果收尾，提前说明实验将同时验证常规多类 UIS 与 class-agnostic salient instance segmentation。

## 1. Introduction

- **[3]** 从海洋探索、生态监测、资源利用等应用说明 UIS 的价值，再与 underwater semantic segmentation 对比，强调 UIS 能区分重叠鱼群、珊瑚等多个实例；这一段先解释为什么不能只做像素语义分割。
- **[3]** 转向水下成像问题：吸收、散射、颜色失真、低对比度和低可见度会造成深度相关且非均匀的外观变化；Figure 1(a) 用成像过程与退化图像把 domain gap 具体化。
- **[2]** 回顾 WaterMask 等 CNN-based UIS，指出受限于 backbone 表征能力，复杂水下环境中仍表现不足；Figure 1(b) 用 WaterMask、USIS-SAM 与本文结果做视觉对照，提前展示基础模型路线的潜力。
- **[3]** 引入 SAM-based USIS-SAM，说明 foundation model 已被用于水下实例分割，但仍需要大量水下数据缓解 domain misalignment，性能增益有限；这段把问题从“模型容量不足”推进到“预训练表征如何有效适配水下域”。
- **[3]** 选择 DINOv2 作为新的 foundation encoder，并引用 detection、tracking、segmentation 中的泛化表现；作者特别强调其 self-supervised、task-agnostic 表征，相比 SAM 的 task-specific mask pretraining 更适合标注稀缺且域差异显著的水下场景。
- **[1]** 紧接着承认直接迁移 DINOv2 并不简单，避免形成“换 backbone 即可解决问题”的叙事，为专门的 adaptation strategy 留出空间。
- **[3]** 将 DINOv2 适配明确拆成 **underwater scene adaptation** 和 **underwater object adaptation**；这一二分是全文方法组织的核心，也是两个模块产生的直接依据。
- **[3]** scene adaptation 对应颜色失真和光散射，提出 AquaStyle Aligner：通过 Fourier decomposition 提取水下 style，再用轻量参数将其注入 DINOv2，处理 pretrained natural domain 与 underwater domain 的风格错位。
- **[3]** object adaptation 对应珊瑚、水母、海龟等类别在 DINOv2 预训练数据中相对稀缺，提出 ObjectPrior Prompter：先学习所有前景共享的 class-/instance-agnostic object prior，再用该 prior 引导实例级学习。
- **[2]** 作者把这一过程解释为把“是否是水下目标”与“具体属于哪个实例/类别”解耦；写作作用是说明二值 mask supervision 并不是最终任务降级，而是用于降低直接学习细粒度实例的难度。
- **[3]** 贡献列表依次强调首次将 DINO 引入 UIS、AquaStyle Aligner、ObjectPrior Prompter；结构保持“基础模型选择—场景级适配—目标级适配”的顺序，不额外扩展新模块。

## 2. Related Work

### Visual Foundation Models

- **[3]** 从 SimCLR、MoCo 的对比学习，写到 DINO 的 self-distillation、MAE 的 masked reconstruction，再到 SAM 的大规模 mask 预训练；引用序列用于交代视觉基础模型如何从通用表征学习发展到可迁移分割。
- **[2]** 单独强调 DINOv2 在大规模无监督数据上获得高维语义特征，并与 SAM 的精确边界能力对比；这段服务于本文为什么选 DINOv2 作为 encoder，而不是继续围绕 SAM 设计。
- **[1]** 相关工作把 DINOv2 定位成高层语义和通用表征来源，为后文“DINO 有能力，但需要水下适配”建立前提。

### Underwater Instance Segmentation

- **[3]** 从 WaterMask 正式建立 UIS 和 USIS 数据集写起，指出其虽针对光衰减、颜色失真做专门设计，但 CNN 表征能力有限；引用对象承担早期任务基线的角色。
- **[3]** 再介绍 USIS10K 与 USIS-SAM，用 adapter 和 prompt 将 SAM 迁移到水下域；作者随后指出其 prompt 设计难以充分适应水下高目标密度和外观多样性，把最相关 foundation-model baseline 的不足明确化。
- **[2]** Related Work 最后提出 DINOv2 + lightweight scenario-specific adaptation，强调只增加少量可训练参数；这一句直接过渡到方法章节，而不是继续扩展更多 UIS 方法。

## 3. Method

### 3.1 From Motivation to Overall Architecture

- **[3]** 方法开头不直接画网络，而是再次讨论 DINOv2 的能力与水下成像差异；作者用“蓝绿色偏移来自长波光吸收与散射”解释 domain gap 的物理来源，让后面的 style modeling 有具体依据。
- **[3]** Figure 2 用 PCA 比较 DINOv2 在自然图像和水下图像上的特征：vanilla DINOv2 能捕获主要目标，但水下特征更易受背景噪声影响，也可能漏掉部分目标；该图承担**证明需要适配，而不是假定需要适配**的作用。
- **[2]** 基于 Figure 2 的观察，作者得出 DINOv2 需要 specialized fine-tuning strategy；写作上由“foundation model 很强”转入“强模型仍存在具体失败模式”。
- **[3]** Figure 3 给出 DiveSeg 总体结构，并重新按两个层级解释：AquaStyle Aligner 从 scene level 解决颜色 domain misalignment，ObjectPrior Prompter 从 object level 解决目标类别与实例学习困难。
- **[1]** Figure 2 中 DiveSeg 的 PCA 表征更聚焦前景，被用于反向佐证两个适配模块确实使 DINO 表征更适合水下对象。

### 3.2 AquaStyle Aligner

- **[2]** 小节开头先把模块拆成 Style Extraction 与 Style Injection：前者负责得到水下颜色风格，后者负责把风格与 frozen DINOv2 representation 交互；写作上先区分“提什么信息”和“怎么注入”。

#### Style Extraction

- **[3]** 作者先提出自然图像与水下图像的主要差异之一是水介质造成的颜色偏移，并指出这种差异主要体现在 low-level feature；随后引用 Fourier spectrum 中 amplitude 能保留低层统计信息，建立使用频域 amplitude 表征 style 的依据。
- **[2]** 公式 (1)–(3) 依次定义 Fourier transform、amplitude 和 phase；公式本身不是创新，作用是明确后面的 style image 如何只保留 amplitude 统计。
- **[3]** 公式 (4) 固定 phase 为平均值，仅用原图 amplitude 做 inverse Fourier transform 得到 style image；这一操作主动去掉 object-related content，只留下颜色/风格特征，避免 style branch 偷带实例语义。
- **[2]** Figure 4 对原图与 style image 做可视化，展示物体内容被弱化而水下综合色调被保留；图承担“amplitude 确实能隔离水下 style”的直观证据。
- **[1]** 最后用多层卷积和 global average pooling 将 style image 编码为紧凑的 style vector \(p_x\)，为后续 cross-attention 注入准备统一表示。

#### Style Injection

- **[3]** 引用 Houlsby Adapter 作为轻量微调基础，但作者不满足于普通 MLP adapter，而改用 cross-attention，使 image feature 能显式查询当前图像的 style vector；引用的作用是说明结构来源，同时突出本文的 style-conditioned 改动。
- **[3]** 公式 (5) 中 ViT feature 作为 query，style vector 作为 key/value；公式把 style injection 定义为原始 frozen attention 的补充分支，而不是替换预训练 attention。
- **[2]** 作者强调 MHA 参数保持冻结，cross-attention 产生 complementary representation；这一解释服务于“适配水下风格，同时避免破坏预训练知识”的 PEFT 逻辑。
- **[2]** 公式 (6) 在 FFN 位置增加 bottleneck MLP residual，对上一阶段融合后的表征做更深层适配；方法由 style-aware attention interaction 延伸到轻量 feature refinement。
- **[2]** 将 DINOv2 层平均划为四个 block，只在每个 block 的第一层加入 AquaStyle Aligner；这一设计明确用稀疏插入控制计算量和参数增长，后面的效率优势由此产生。

### 3.3 ObjectPrior Prompter

- **[3]** 小节开头用明显的逻辑接力：“颜色适配基本解决后，水下特定对象仍然难”；作者把剩余问题归因于这类目标在 DINOv2 预训练数据中稀缺，使直接从通用表征学习 instance-level segmentation 仍然困难。
- **[2]** 提出先利用 binary mask 学 class-agnostic、instance-agnostic object feature，再把这些 feature 作为 prompt 注入 DINOv2；写作意图是把复杂实例分割拆成“先找到所有目标—再区分实例”。
- **[3]** Figure 3 上方展示 multi-scale encoder → pseudo mask → object feature → cross-attention prompt 的完整流程；图承担 ObjectPrior Prompter 的主要数据流解释。
- **[2]** 轻量 multi-scale encoder 用三层卷积、stride-2 downsampling 和 \(1\times1\) projection 生成三尺度特征；多尺度设计服务于不同尺寸水下目标，而没有额外引入复杂 backbone。
- **[3]** 公式 (7) 用 \(1\times1\) convolution + Sigmoid 从每个尺度动态预测 pseudo mask，并用由 GT 合并得到的单通道 binary foreground mask 监督；该式负责把“object prior”落到可训练的前景概率图。
- **[2]** 训练与推理阶段都动态生成 pseudo mask，说明 GT binary mask 只承担训练监督，不是推理时额外输入；这一点避免 prompter 被误解为依赖人工 mask prompt。
- **[3]** 公式 (8) 用 pseudo mask 过滤背景，再通过 residual 保留原始语义；公式承担“增强 object-related information，但不过度破坏原 feature”的作用。
- **[2]** 将三个尺度的 object-aware feature 展平并拼接成 \(O_{\mathrm{prompt}}\)，把二值前景学习结果转换为可以进入 ViT 的 object-level prompt。
- **[3]** 公式 (9) 中 ViT feature 为 query，object prior 为 key/value；这一式完成“外部前景先验 → DINO 表征”的真正 prompt 注入。
- **[2]** ObjectPrior Prompter 被放在每个 DINO block 之后，更新后的 feature 与原 ViT feature 相加再送入 decoder；逐 block 注入使 object prior 持续影响后续表征，而不是只在 segmentation head 前做一次融合。
- **[1]** 整个模块的论证逻辑保持一致：binary object supervision 降低目标搜索难度，再让 Mask2Former 处理具体类别与实例区分。

## 4. Experiments

### 4.1 Experimental Setup

- **[2]** 使用 UIIS 和 USIS10K 两个主流 UIS 数据集；UIIS 包含 7 个水下类别，USIS10K 除多类别标注外还提供 class-agnostic labels，使实验能够验证 ObjectPrior Prompter 的 object-level 思路。
- **[2]** 采用 mAP、AP50、AP75 作为 mask AP 指标，分别覆盖总体实例分割性能与不同 IoU 精度要求；AP75 对边界与实例区分更敏感，后面的较大增益能支撑精细 segmentation 改善。
- **[3]** 采用 DINOv2 encoder + Mask2Former segmentation head，分类/mask loss 监督最终实例分割，BCE、IoU、L1 loss 监督 pseudo mask；损失设置对应“最终实例任务 + 中间 object-prior 学习”两条训练路径。

### 4.2 Comparison with State-of-the-Arts

#### UIIS

- **[3]** Table 1 同时比较 Mask R-CNN、Mask Scoring R-CNN、Cascade Mask R-CNN、Mask2Former、WaterMask 与 USIS-SAM，覆盖传统 CNN、通用 Transformer 和水下 foundation-model 方法；表格用来证明提升不是只针对某一类 baseline。
- **[2]** DiveSeg 达到 35.6 mAP、52.0 AP50、38.5 AP75，明显超过 ViT-H 的 USIS-SAM；作者借此强调 DINOv2 + 专门适配优于直接依赖更大 SAM backbone。
- **[1]** AP75 的明显提升被用于体现实例边界和高质量 mask 的改善，与 ObjectPrior Prompter 的目标级先验相呼应。

#### USIS10K

- **[3]** Table 2 分别报告 class-agnostic 与 multi-class 结果，这种分栏设计与方法中“先 object-level、再 instance/class-level”的两阶段思想高度对应。
- **[2]** class-agnostic 设置达到 64.1 mAP，multi-class 设置达到 48.4 mAP，并超过 USIS-SAM；两种设置同时提升，说明 object prior 不只是帮助找前景，也能传递到具体类别与实例判别。
- **[1]** 作者补充参数量只有 USIS-SAM 的约 55.6%，将 AquaStyle Aligner 的轻量插入和 DINOv2 适配优势转化为工程证据。

### 4.3 Qualitative Results

- **[3]** Figure 5 与 WaterMask、USIS-SAM 对比，选择阴影中的鱼、水下遗迹、漏检珊瑚、鱼和珊瑚重叠、潜水员与设备重叠等困难案例；案例分别覆盖光照/背景干扰、漏检和实例边界混淆。
- **[2]** 作者不仅展示 mask 更完整，也举出其他方法把 fish 误判为 coral 的分类错误；这一部分把 AquaStyle Aligner 的环境干扰适配与 ObjectPrior Prompter 的目标判别同时映射到可视化结果。

### 4.4 Ablation Study

#### Effectiveness of Individual Modules

- **[3]** Table 3 从 frozen DINOv2 + Mask2Former baseline 出发，分别去掉 AquaStyle Aligner、去掉 ObjectPrior Prompter，再与 Full Model 比较；消融结构完全对应全文“scene adaptation + object adaptation”的二分。
- **[2]** AquaStyle Aligner 单独带来约 3.9 mAP 增益，ObjectPrior Prompter 单独带来约 3.2 mAP 增益，组合达到 35.6 mAP；作者用两者均独立有效且组合最好支撑模块互补。
- **[1]** AP50、AP75 同样提升，避免把模块作用解释成只提高低质量检测数量，而是对不同 mask 精度要求都有效。

#### Ablation on AquaStyle Aligner

- **[3]** Table 4 将 Frozen、Full Fine-tuning、LoRA、普通 Adapter 与 AquaStyle Aligner 放到同一 DINOv2 adaptation 对比中；该表不是验证“微调是否有效”，而是验证**水下 style-aware 微调是否优于通用 PEFT**。
- **[2]** Full fine-tuning 相对 frozen 只有很小提升，作者将其解释为大量更新可能导致 pretrained knowledge 的 catastrophic forgetting；这一负结果用于支撑冻结 backbone 的基本策略。
- **[2]** LoRA 和普通 Adapter 均有改善，说明 targeted parameter tuning 本身有效；AquaStyle Aligner 仍取得最高性能，把额外收益归因到 underwater style extraction 与 style injection，而非仅仅增加可训练参数。
- **[1]** 该表形成 AquaStyle Aligner 最关键的实验闭环：Fourier style prior 不是装饰性模块，它相对通用 PEFT 提供了水下域专属价值。

## 5. Conclusion

- **[2]** 结论重新把 UIS 的困难拆成 underwater scene adaptation 和 underwater object adaptation，与 Introduction 和 Method 的组织完全一致。
- **[2]** AquaStyle Aligner 用 Fourier decomposition + cross-attention 注入颜色 style，ObjectPrior Prompter 用 binary segmentation prompt 提供 object prior；结论只重复两条核心机制，不扩展新概念。
- **[1]** 以 UIIS 和 USIS10K 的 SOTA 定量、定性结果收束，完成“DINO 可用，但必须针对水下 scene/object 两层适配”的全文论证。

## 整体写作主线

- **先证明 DINO 值得用，也证明它不能直接用**：先用大规模自监督预训练说明 DINOv2 的通用能力，再用 Figure 2 PCA 展示水下背景噪声和漏检，避免方法建立在纯经验假设上。
- **把 domain adaptation 拆成两个尺度**：scene-level 解决整体水下颜色/风格错位，object-level 解决海洋对象在预训练数据中稀缺和直接实例学习困难。
- **AquaStyle Aligner 的推导链是**：水下颜色偏移 → 差异属于 low-level statistics → Fourier amplitude 表征 style → 平均 phase 去除 object content → cross-attention 将 style 作为补充信息注入 frozen DINO。
- **ObjectPrior Prompter 的推导链是**：水下特定类别稀缺 → 直接实例判别困难 → 先用 binary mask 学所有前景共享的 objectness → pseudo mask 抑制背景 → object feature 作为 prompt 注入 ViT → 再交给实例分割 decoder。
- **两个模块采用相似的注入思想，但输入先验不同**：AquaStyle Aligner 用 scene style vector，ObjectPrior Prompter 用 multi-scale object feature；两者都通过 cross-attention 调制 frozen DINO representation。
- **Figure 2–4 构成方法动机证据链**：Figure 2 证明 vanilla DINO 的水下表征问题，Figure 4 证明 Fourier 重构能隔离 style，Figure 3 再把两个 prior 的注入位置统一展示。
- **Table 3 回答两个模块是否互补，Table 4 回答 AquaStyle 为什么不能用普通 LoRA/Adapter 替代**；实验与方法假设直接对应，没有把消融停留在参数数量比较。
- **全文最核心的写作模式是**：DINO 通用表征强 → 水下域仍失配 → 分成 scene/object 两类失配 → 用 style aligner 修场景，用 object prompt 降低实例学习难度 → 在 class-agnostic 与 multi-class UIS 上同时验证。
