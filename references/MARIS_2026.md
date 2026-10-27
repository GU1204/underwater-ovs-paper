# MARIS（2026）论文写作脉络

> **论文**：*MARIS: Marine Open-Vocabulary Instance Segmentation with Geometric Enhancement and Semantic Alignment*  
> **作者**：Bingyu Li, Feiyu Wang, Da Zhang, Zhiyuan Zhao, Junyu Gao, Xuelong Li  
> **会议**：CVPR 2026  
> **方法/基准名**：MARIS  
> **核心模块**：Geometric Prior Enhancement Module（GPEM） + Semantic Alignment Injection Mechanism（SAIM）

## Abstract

- **[2]** 先从现有水下实例分割受闭集类别限制切入，再提出 MARIS 数据集作为开放词汇水下实例分割的评测基础；开头同时建立“缺少 benchmark”和“无法识别未知海洋类别”两个研究缺口。
- **[2]** 将自然图像 OVS 直接迁移到水下场景时的问题归纳为视觉退化与语义错位，分别对应颜色衰减等成像问题和 VLM 缺少水下类别知识的问题，为后文两个模块建立明确来源。
- **[2]** 用 GPEM 引入稳定的局部结构与几何信息，用 SAIM 向文本表征注入水下领域语义；摘要中的方法介绍维持“视觉问题—几何模块、语义问题—语言模块”的一一对应。
- **[1]** 用 in-domain 和 cross-domain 实验均超过现有 OVS baseline 收尾，提前说明论文既验证水下域内学习，也验证自然域向水下域迁移。

## 1. Introduction

- **[2]** 从海洋生物监测、水下机器人和环境保护等应用说明水下实例分割的重要性，再指出 pixel-level 标注成本高，为开放词汇学习减少类别标注依赖建立现实动机。
- **[1]** 补充传统模型只识别训练类别的问题，将“昂贵标注”与“类别封闭”合并成现有水下实例分割的两个限制。
- **[2]** 引入开放词汇学习能够识别未见类别，但强调自然场景 OVS 向水下场景的直接迁移尚未被系统研究；这句话把已有 OVS 能力与本文研究空白连接起来。
- **[3]** 第一个挑战落在数据层面：现有 UIIS10K、USIS10K 等数据集类别少，且大量物种被统一标为 fish、plants；用 Amphora、Blue Parrotfish 被粗粒度归为 fish 的例子说明这种 taxonomy 无法支持细粒度开放词汇评测。
- **[2]** 由此引出 MARIS benchmark，将类别扩展到 158 个细粒度类别；Figure 1(a) 承担“现有数据粗粒度—MARIS 细粒度”的直观对照作用，先解决论文能否开展该任务的问题。
- **[3]** 第二个挑战转到视觉域差异：水介质造成颜色衰减、低对比度、散射和遮挡，使依赖 appearance 的视觉特征不稳定；Figure 1(b.1) 用水下退化现象将自然场景 OVS 的迁移困难具体化。
- **[3]** 紧接着寻找退化条件下仍稳定的信息，强调鱼体轮廓、鳍结构、珊瑚几何生长模式仍可辨认；Figure 1(b.2) 的作用是把“几何信息更稳定”从直觉变成方法设计依据。
- **[1]** 据此提出 GPEM，以深度/结构先验补偿不可靠的颜色和纹理表征；模块不是独立提出，而是由上一段的稳定几何观察直接推导。
- **[2]** 第三个挑战落到语义侧：现有 VLM 主要由陆地图文数据训练，缺少细粒度海洋类别定义；Figure 1(c) 用语义失败示例支撑“视觉识别之外还有 language prior 不足”。
- **[2]** 引出 SAIM，通过 underwater-aware prompt 和 embedding enrichment 注入领域知识，目标是减少相似海洋类别之间的语义歧义并提升 unseen recognition。
- **[1]** 将 GPEM 与 SAIM 总结为互补设计：前者处理视觉退化，后者处理语义歧义；这一句完成 Introduction 的“两个问题—两个模块”闭环。
- **[3]** 贡献列表依次总结 MARIS benchmark、GPEM+SAIM 框架和全面实验；贡献结构对应“任务基础—方法设计—实验验证”，没有引入新的支线。

## 2. Related Work

### Underwater Segmentation

- **[3]** 先按时间回顾 SUIM、MAS3K、DUT-USEG、UIIS、UIIS10K、USIS10K、Seaclear、USIS16K 等数据集，说明水下分割数据规模持续扩大；随后指出这些数据仍存在 taxonomy 粗和类别覆盖有限的问题，为 MARIS benchmark 保留必要性。
- **[2]** 回顾 UWSegFormer、UISS-Net、CaveSeg 等利用水下先验或增强策略的方法，用这些引用说明视觉退化一直是水下分割的核心问题，也为 GPEM 使用几何先验提供领域背景。
- **[2]** 引用 WaterSAM、UWSAM 等 SAM/VFM 水下适配工作，说明 foundation model 已开始进入该领域，但“大规模 open-vocabulary object segmentation”仍缺失；相关工作最后一句直接落到本文任务空白。

### Open-Vocabulary Segmentation

- **[3]** 从 CLIP 驱动的 LSeg、MaskFormer/ZSSeg 开始，按 pixel-level alignment 与 proposal-based classification 两条路线回顾 OVS 基本范式；引用目的在于说明 MARIS 并非重新定义 OVS，而是把成熟范式迁移到水下场景。
- **[3]** 再按 MaskCLIP、SAN、ODISE、FC-CLIP/MAFT+ 等方法的发展介绍 dense feature、side adapter、diffusion feature 和 one-stage alignment，建立本文视觉编码与 mask-query 框架的技术来源。
- **[2]** 引用 SCAN、CAT-Seg、ERR-Seg 等 structural prior 与 cost aggregation 方法，说明开放词汇研究已经开始重视局部结构和跨模态对齐，但这些设计仍面向自然场景。
- **[1]** 用 frequency-domain、SAM-CLIP fusion 等工作补充其他泛化路线，最后明确本文是首次系统探索 underwater OVS，并把相关工作自然过渡到 MARIS benchmark 与水下特定适配。

## 3. MARIS Benchmark

### 3.1 Data Collection and Annotation

- **[2]** 开头重复现有水下 benchmark 类别少、标签粗的问题，目的不是再次做背景综述，而是解释 MARIS 数据构建时为什么需要重新标注和细分 taxonomy。
- **[3]** 说明 MARIS 汇总多个公开水下数据源并重新标注，最终包含 16K+ 图像、9 个 superclass、158 个细粒度 subclass；这一段用规模和类别层级定义 benchmark 的基本边界。
- **[2]** 用 fish superclass 被细化为 76 个物种作为具体例子，证明 MARIS 的主要变化不是简单增加图像，而是把原来的粗粒度类别转换为适合开放词汇评测的细粒度语义空间。
- **[1]** 强调所有实例均具有 pixel-accurate mask，将细粒度 taxonomy 与 instance segmentation 所需的空间标注同时闭合。
- **[1]** 用“first benchmark”总结该数据集能够支持水下开放词汇实例分割的系统评估，为后续实验设置提供依据。

### 3.2 Dataset Split and Experimental Settings

- **[3]** 给出 5,712 张训练图像、10,439 张验证图像及训练/验证类别关系，并区分 intersection、train-exclusive 和 test-exclusive 类别；这些数字用于形式化 seen/unseen 评估，而不是单纯描述数据规模。
- **[2]** Figure 2(b) 可视化类别划分，Figure 2(c) 展示任务设置；图承担“数据组成如何映射为开放词汇实验 protocol”的解释作用。
- **[2]** 定义 in-domain：在 MARIS 训练集训练、验证集测试，用 intersection 与 OV 类别同时评估模型在已见和未见水下类别上的表现。
- **[2]** 定义 cross-domain：只在 COCO 训练、直接在 MARIS 验证集测试；由于两者没有类别重叠，这一设置同时放大 terrestrial-to-underwater 的视觉域差异与语义差异，成为检验 GPEM/SAIM 泛化作用的更强场景。

## 4. Method

### 4.1 Problem Definition

- **[2]** 用输入图像 \(I\)、文本类别集合 \(\mathcal C\)、实例 mask 集合 \(\mathbf M\) 和类别标签 \(\mathbf Y\) 形式化 OVIS；特别注明 \(y_i\) 可以属于训练阶段未见类别，把 benchmark 中的 OV setting转写为方法目标。

### 4.2 Overall Architecture

- **[3]** Figure 3 和公式 (1) 先定义两条冻结特征分支：Geo-Generator 提取几何特征 \(\mathbf F_G\)，CLIP visual encoder 提取视觉特征 \(\mathbf F_V\)；总体结构一开始就把“水下稳定几何”与“开放词汇视觉语义”并列为两种基础信息源。
- **[2]** 公式 (2) 将多尺度视觉 decoder 输出与几何先验通过 \(\mathcal F_{VG}\) 融合，得到 \(\mathbf F_{VG}\)；该式定义 GPEM 在整体 pipeline 中的位置，使后文所有几何增强操作都服务于统一的 visual-geometric representation。
- **[2]** 公式 (3) 将 \(\mathbf F_{VG}\) 与 CLIP text embedding \(\mathbf E_T\) 输入 SAIM，输出类别预测和 mask；写法上完成“先修复视觉表征，再做语义对齐”的整体处理顺序。
- **[1]** 用 classification loss 与 mask loss 同时监督最终输出，提前说明两个模块最终仍落到标准 instance segmentation 的分类与定位目标。

### 4.3 Geometric Prior Enhancement Module（GPEM）

#### Multi-scale Visual & Geometric Generator

- **[3]** 先从 frozen CLIP 提取多层视觉特征，再用 multi-scale deformable attention 加强局部细节和长程依赖；公式 (4) 的作用是把原始 CLIP 表征整理成适合实例级结构建模的多尺度视觉特征，而不是直接进行几何融合。
- **[2]** 引入冻结 depth encoder 提取多尺度几何特征和 global depth token；公式 (5) 明确几何先验来自独立预训练深度模型，使“geometry”有可执行的特征来源。
- **[1]** 两条 frozen encoder 的并行设计保留 CLIP 的开放词汇语义能力，同时避免直接在有限水下数据上重训练几何/视觉基础模型，为后面的轻量融合提供基础。

#### Visual–Geometric Feature Fusion

- **[2]** 公式 (6) 先把视觉与几何特征分别线性投影到共享 latent space；这一步承担模态维度对齐作用，使后续融合不是直接相加异构特征。
- **[2]** 公式 (7) 根据拼接后的视觉/几何特征计算逐尺度自适应权重 \(\alpha^{(l)}\)；该权重让几何信息的注入强度由当前场景决定，避免所有水下区域固定依赖 depth prior。
- **[2]** 公式 (8) 采用“视觉主干 + 加权几何残差 + MLP”的形式得到 \(\mathbf F_{VG}^{(l)}\)；公式体现作者并未用几何替代 CLIP，而是把结构信息作为补充注入视觉语义。
- **[1]** 这一组公式共同支撑 Introduction 中的核心判断：颜色和纹理可能退化，但结构信息可以作为稳定补充保持物体一致性。

#### Geometry-based Visual & Semantic Bridge

- **[3]** 引入轻量 Q-Former，把各尺度 visual-geometric feature 用于更新 query embeddings；引用 BLIP/InstructBLIP 类 Q-Former 的目的在于说明这种 query bridge 已被验证适合连接视觉与语义空间。
- **[2]** 将多尺度更新后的 query 聚合成 geometry-informed queries，使 GPEM 的作用从 feature enhancement 延伸到后续 mask query；这一段为 SAIM 使用增强 query 和全局几何信息建立接口。

### 4.4 Semantic Alignment Injection Mechanism（SAIM）

- **[2]** 开头把 SAIM 拆成两部分：underwater-aware textual prompt + adaptive template selection，以及 geometry-based global prior；写作上把“补语言知识”和“用当前图像筛选语言知识”分开说明。

#### Adaptation to Underwater Scenes

- **[2]** 先说明 CLIP 的 generic prompt 无法描述 scattering、low contrast、color attenuation 下的水下对象，引用 CLIP 和相关水下工作支撑语言模板存在 domain mismatch。
- **[2]** 构造覆盖环境语境、水介质与可见度、光照与感知、深度信息、场景交互五类 underwater prompt；这些模板的作用是把 category name 扩展为与水下视觉条件更接近的文本描述。
- **[3]** 作者随后主动指出 prompt 越多并不一定越好：不同图像只与部分模板匹配，简单平均会把有效描述稀释；用低光图像更匹配 “\<class\> in low visibility conditions” 的例子具体说明 prompt noise。
- **[2]** 基于视觉特征与各文本模板的相似度对模板排序，选择 top-\(K\) 模板；这一机制把静态 prompt ensemble 改成 image-conditioned selection，使 SAIM 的“alignment”落到具体选择过程。

#### Category Discrimination

- **[2]** 将 global depth token 与聚合 mask feature 融合得到增强表示，再池化形成 compact feature；这里把 GPEM 产生的全局几何信息带入语义分类，形成两个模块之间的实际连接。
- **[2]** 公式 (9) 用 pooled feature 与适配后的文本 embedding 计算类别预测 \(\mathbf Y_{\text{cls}}\)；该式是 SAIM 从 prompt enrichment 落到最终 open-vocabulary classifier 的关键接口。
- **[2]** 同一个 global depth token 也用于引导 query 生成 mask，使几何先验同时进入“分类表征”和“空间预测”两条路径；这一设计强化 GPEM 不只是 mask enhancement，而是参与视觉—语义桥接。

### 4.5 Training

- **[1]** 公式 (10) 使用分类损失监督 \(\mathbf Y_{\text{cls}}\)，让经过 SAIM 对齐后的区域特征直接学习类别判别。
- **[2]** 公式 (11) 采用 Dice Loss + BCE 监督 mask，并引用 MaskFormer/FC-CLIP 类标准 mask loss；引用作用是说明空间监督沿用成熟范式，论文创新主要集中在 feature/prior alignment 而非重新设计 loss。
- **[1]** 用分类与 mask 两类损失共同训练收束方法章节，对应 OVIS 的“类别识别 + 实例定位”两个输出目标。

## 5. Experiments and Results

### 5.1 Experimental Details

- **[2]** 简要交代 4×RTX 4090、batch size 16，并再次说明采用 in-domain 与 cross-domain 两种设置；实验配置的重点不是训练细节，而是确保后续结果分别回答域内开放类别识别与跨域迁移两个问题。

### 5.2 Main Experiments

#### In-Domain

- **[3]** Table 1 将 OVSeg、ODISE、SAN、FC-CLIP、MAFT+、EOVSeg 等方法放到相同 MARIS in-domain 设置下，并分别报告 intersection、OV 和 overall AP；表格的核心作用是同时验证 seen 与 unseen 类别，而非只看总体分数。
- **[2]** ConvNeXt-B 下 intersection mAP 为 52.68、OV mAP 为 39.77；ConvNeXt-L 下达到 61.55 和 54.02，作者用两个 backbone 的一致增益说明方法收益并非来自单一模型容量。
- **[1]** 特别强调 AP\(_{75}\) 的提升，将总体性能优势解释为更精确、更稳定的实例 mask，而不只归因于类别分类改善。

#### Cross-Domain

- **[2]** Table 2 将所有模型限制为 COCO 训练、MARIS 测试，用性能整体下降先证明确实存在明显 terrestrial-underwater domain gap。
- **[2]** 用 EOVSeg 的明显退化作为反例，说明依赖源域特定信息的 OVS 设计在跨域水下场景中可能失效。
- **[2]** MARIS 在 ConvNeXt-B 上将最佳 baseline 的 30.05 mAP 提升到 32.62，在 ConvNeXt-L 上从 40.27 提升到 46.18；这些结果用于支撑 GPEM 与 SAIM 针对视觉退化和语义差异的设计确实具有跨域价值。

### 5.3 Ablation Experiments

#### GPEM and SAIM

- **[3]** Table 3 采用 baseline、+GPEM、+SAIM、+GPEM+SAIM 四组配置，最直接地对应论文的两个核心模块；实验结构与 Introduction 的“两个问题—两个模块”完全一致。
- **[2]** GPEM 与 SAIM 单独加入均改善性能，组合后达到 intersection 61.55 mAP、OV 54.02 mAP；该表承担证明两者不是重复功能，而是具有互补作用的核心消融。

#### Underwater Prompts and Template Selection

- **[3]** Table 4 从普通 Template 到 UWTemplate，再到 Selection，按 SAIM 的设计顺序逐步验证；实验不是只证明 underwater prompt 有效，还检验“为什么不能直接平均所有水下 prompt”。
- **[2]** UWTemplate 已能改善指标，adaptive selection 对 OV 类别的收益更明显；这一结果回扣 Sec. 4.4 中“部分模板会引入噪声”的判断。

#### Geo-Generator Scale

- **[3]** Table 5 比较不同尺寸的 depth/geometry encoder，并同时报告 in-domain 与 cross-domain；较大 vitl 在域内更强，而 vitb 的跨域表现更稳定，用容量—泛化权衡解释 Geo-Generator 并非越大越好。
- **[1]** 这一实验让几何先验的收益不只停留在“加 depth 有效”，还讨论了几何 foundation model 容量如何影响 domain generalization。

#### Feature Fusion

- **[2]** Table 6 比较 MLP、直接相加与 adaptive alpha fusion，并同时报告精度、GFLOPS 和参数量；表格专门验证公式 (7)–(8) 的自适应融合是否值得额外复杂度。
- **[1]** alpha fusion 获得最佳性能，支撑“根据视觉与几何内容动态控制几何注入强度”优于固定融合。

### 5.4 Per-Class Performance Analysis

- **[3]** Figure 4 展示 AP 最好和最差的类别，将总体指标拆成类别层面；Shark、Turtle、Dolphin 等高频且视觉特征明显的类别保持较高 AP，说明模型对典型海洋对象具有较强泛化。
- **[2]** Sponges、Anemonefish 变体和小型无脊椎动物等稀有或视觉相似类别表现较差；作者用这些结果承认细粒度类别判别仍是当前方法的主要瓶颈，为后续 failure analysis 做铺垫。

### 5.5 Cross-Domain and In-Domain Analysis

- **[2]** 先总结 cross-domain 普遍低于 in-domain，将差距解释为领域特定知识仍然重要；这一分析避免把主实验的提升包装成“已经解决跨域问题”，而是明确剩余 domain gap。
- **[2]** 对 Lunar Fusilier、Pomacentrus Leucogrammicus 等 AP 接近 0 的类别做 failure analysis，归因于 VLM 对专业物种语义编码不足；该段直接回扣 Introduction 中提出的 semantic misalignment。
- **[2]** 用 Plastic Bag 在 cross-domain 反而更好的例子说明：当类别在 COCO 等自然场景中已有充分语义/视觉经验时，跨域并非必然失败；这个反例加强“缺少 marine-specific knowledge 才是关键”的解释。

### 5.6 Inference Efficiency and Model Complexity

- **[3]** Table 7 将 MARIS 与 MAFT+、OVSeg 同时比较 mAP、FLOPs、trainable parameters 和 FPS；作者用约 22–23M trainable parameters 对比 MAFT+ 的 108–223M，证明方法不是依靠大规模微调获得增益。
- **[1]** 该表把“冻结 CLIP/Geo-Generator + 轻量桥接”的架构选择转化为工程证据，补充主实验只关注精度的不足。

### 5.7 Robustness Analysis of SAIM

- **[2]** Table 8 扫描不同 TopN 模板数量，观察 mAP 在较宽范围内保持稳定；实验直接检验 SAIM 中 adaptive template selection 对超参数 \(K\) 的敏感性。
- **[1]** 用稳定结果说明方法不依赖精确 TopN 调参，使 prompt selection 更像可靠机制而非特定超参数带来的偶然提升。

### 5.8 Qualitative Results

- **[2]** Figure 5 单独对比视觉特征、几何特征和融合特征，直观展示 visual-geometric fusion 后结构更清晰；该图承担 GPEM 的 feature-level 证据，而不是最终 segmentation 可视化。
- **[2]** Figure 6(a) 与 FC-CLIP、EOVSeg 比较 Abalone、Atlantic Spadefish、Blacktail Butterflyfish 等实例结果，将定量提升落实到具体海洋类别和 mask。
- **[2]** Figure 6(b) 展示 Anyperodon Leucogrammicus 被误判为 Peacock Grouper 或 Coral Hind 的失败案例，并用相似 grouper morphology 解释原因；案例将全文最后重新落回“视觉相似 ≠ 语义一致”的细粒度语义问题。

## 6. Conclusion

- **[2]** 先总结 MARIS benchmark 解决水下开放词汇实例分割缺少大规模细粒度评测的问题，再用 GPEM 和 SAIM 分别概括几何增强与语言先验注入。
- **[1]** 将 benchmark 与 framework 一起定义为水下 OVS 的基础工作，结论重点仍保持“数据 + 方法”双贡献结构。
- **[2]** limitation 明确极端环境和稀有物种仍覆盖不足，并把未来方向落到数据扩充和模型鲁棒性；这一结尾与 per-class failure analysis 中稀有类别性能差形成呼应。

## 整体写作主线

- **先证明任务本身缺少基础条件**：现有水下数据只有粗粒度类别，无法严谨评估 unseen marine species，所以先构建 MARIS benchmark。
- **再把自然场景 OVS 的水下迁移失败拆成两个问题**：颜色衰减、低对比和散射造成 **visual degradation**；CLIP 缺少海洋物种定义造成 **semantic misalignment**。
- **两个方法模块与两个问题严格对应**：GPEM 用 depth/geometry 补充退化的 appearance feature，SAIM 用 underwater prompt、template selection 和 global geometry prior 修正 region–text/category alignment。
- **GPEM 的写法是“稳定先验 → 自适应注入”**：先说明水下几何比颜色稳定，再由 depth encoder 提取几何特征，最后用 alpha fusion 和 Q-Former 将其注入多尺度视觉特征与 mask query。
- **SAIM 的写法是“缺少语义 → 扩充模板 → 发现模板噪声 → 自适应筛选”**：不是停在 prompt engineering，而是先提出 underwater prompt，再用 image–text similarity 解决不同模板对不同图像适用性不同的问题。
- **实验形成逐层闭环**：Table 1–2 验证 in-domain/cross-domain 总体效果，Table 3 验证两个模块互补，Table 4 验证 prompt 与 selection，Table 5–6 验证几何编码器和融合方式，Table 8 验证 SAIM 稳定性。
- **failure analysis 回到论文核心限制**：专业稀有物种仍因 VLM 语义不足而失败，相似鱼类仍存在细粒度混淆，说明 SAIM 缓解了 semantic misalignment，但没有完全解决 marine fine-grained recognition。
