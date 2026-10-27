# NightAdapter（2025）论文写作脉络

> **论文**：*NightAdapter: Learning a Frequency Adapter for Generalizable Night-time Scene Segmentation*  
> **作者**：Qi Bi, Jingjun Yi, Huimin Huang, Hao Zheng, Haolan Zhan, Yawen Huang, Yuexiang Li, Xian Wu, Yefeng Zheng  
> **会议**：CVPR 2025  
> **方法名**：NightAdapter

## Abstract

- **[2]** 从夜间场景分割受复杂光照影响切入，把现有方法的核心不足收束为对**未见夜间光照条件泛化不足**，由常规夜间分割问题转向本文关注的 generalizable night-time segmentation。
- **[2]** 提出 NightAdapter 作为参数高效的 VFM 微调方案，并用 DST 频域分析给出方法依据：日间与夜间特征可分成 illumination-sensitive 和 illumination-insensitive bands。
- **[3]** 用两个对应模块概括方法：Illumination-Insensitive Band Adaptation 保留跨光照稳定信息，Illumination-Sensitive Band Adaptation 对随机化后的敏感频带进行适配；两者分别服务于 domain invariance 和 domain-gap mitigation。
- **[1]** 以 day-to-night、night-to-night、in-domain 三类实验的领先结果收尾，提前覆盖论文后续三条实验验证线。

## 1. Introduction

- **[2]** 先回顾 CNN、ViT、VFM 在日间场景分割上的进展，再用夜间光照差异导致性能下降建立任务缺口；Figure 1 把本文目标画成“单一日间或夜间源域训练 → 任意未见夜间域测试”，先固定 generalization setting。
- **[3]** 将已有夜间分割归为 fully-supervised 与 domain adaptation 两类，并引用 NightCity、ACDC 及多种 DA 方法说明两条主流路线；两者都依赖训练/测试分布或可见目标域，无法覆盖未知光照条件，用来引出本文的 domain generalization 设定。
- **[3]** 从夜间高质量标注难获取转入 VFM：大规模预训练可缓解数据不足，但 vanilla VFM 在不同夜间外观下仍存在明显 distribution shift；这段把“为什么用 VFM”和“为什么还需要 adapter”连续交代清楚。
- **[3]** 引用频域 domain generalization 工作 [5, 26, 32, 42, 46, 77] 引出 DST，并用 Figure 2 展示核心观察：低频在日/夜间模式相近，中高频变化明显；图不是方法示意，而是为后续频带拆分提供经验依据。
- **[3]** 将 Figure 2 的观察直接映射为方法设计：低频视为 illumination-insensitive content，用 side adapter 强化；中高频视为 illumination-sensitive style，通过 band randomization 减弱域特异性但保留边缘等细节。
- **[4]** 贡献列表依次强调新任务范式、illumination-insensitive adaptation、illumination-sensitive adaptation 和实验优势；四点与“问题设定—两个模块—验证结果”保持一一对应，不再引入新的技术分支。

## 2. Related Work

- **[3]** **Night-time Semantic Segmentation** 先回顾 day-to-night domain adaptation，再强调其默认目标夜间分布在训练中可见；随后引入 DGSS，并说明已有 DG/VFM 方法并非专门针对剧烈 illumination shift，为 NightAdapter 的任务定位留出空缺。
- **[2]** 同一小节补充 fully-supervised 夜间分割与 NightCity、ACDC-Night 等数据集，作用是说明“增加夜间标注”是另一条路线，但并没有解决从一个源域泛化到任意夜间域的问题。
- **[2]** **Domain Generalization with Frequency Prior** 用已有频域工作建立“低频偏 domain-invariant content、高频偏 domain-dependent style”的共同认识，为本文用频带解释光照差异提供理论和经验来源。
- **[3]** **Parameter-Efficient Fine-Tuning** 将现有 PEFT 分成 side adapter 与 LoRA 类方法，再补充频域 PEFT；末句强调这些方法没有针对夜间低对比和强光照变化设计，把通用 PEFT 与本文 illumination-aware frequency adapter 区分开。

## 3. Preliminary

### 3.1 Problem Definition

- **[3]** 明确定义单源泛化问题：训练时只给一个日间或夜间 source domain，测试面向任意未见 night-time target domain；随后定义 VFM 层数及中间特征 \(f_i\in\mathbb{R}^{C\times N}\)，为后续逐层频域操作统一符号。

### 3.2 Discrete Sine Decomposition

- **[3]** 介绍 DST 的实值频域表示和 energy compaction 特性，再给出将冻结 VFM 特征 \(f_i\) 映射为频域特征 \(V_i\) 的公式；公式的作用是建立 NightAdapter 后续所有 band 划分与适配的基础表示，而不是单独贡献一个变换。

### 3.3 Spectrum Analysis

- **[3]** 将 DINOv2 最后一层的 \(1024\times1024\) 特征经 DST 后按 sine coefficient 划成 8 个、每个 128 channel 的频带；band-pass 公式负责形式化“如何切频带”，为后面的 band rejection 实验提供可控变量。
- **[3]** 设计 band-rejection classification：每次去掉一个频带，用两层 MLP 区分 day/night，并分别采用 Cityscapes→ACDC-Night、NightCity→ACDC-Night 两组实验；这里不是正式分割实验，而是验证“哪些频带决定跨光照域差异”的诊断实验。
- **[3]** Figure 3 显示去掉高频可明显改善目标域泛化，去掉中频有一定收益，去掉低频会损害源域与目标域表现；图承担本文最关键的**方法因果依据**，把经验上的频率先验具体化到 DINOv2 夜间特征。
- **[1]** 用上述结果定义两类频带：高频及部分中频为 illumination-sensitive bands，低频为 illumination-insensitive bands；这一句完成从分析章节到方法章节的逻辑桥接。

## 4. Methodology

### 4.1 Overview

- **[3]** Figure 4 先给出 NightAdapter 总体结构：冻结 VFM 主干，在每层旁路中对 DST 后的敏感/不敏感频带分治适配，再通过 IDST 返回空间特征；图负责说明 adapter 插入位置和两条频域支路的数据流。
- **[2]** 低频分支承担 scene-content preservation，中高频分支承担 illumination-gap mitigation；末尾用“IDST 后得到更 illumination-robust 的 VFM feature”把两个局部操作统一到跨夜间域泛化目标。

### 4.2 Illumination-Insensitive Band Adaptation

- **[3]** 先回扣 Sec. 3.3：最低两个频带 \(V_{[768,896)}^i\)、\(V_{[896,1024)}^i\) 对日间和夜间场景都重要且对光照较稳定；这一段先说明**为什么低频不能随机化或丢弃**，再引出专门的 IBA。
- **[3]** 为每层引入 illumination-insensitive token \(T_L^i\)，通过低频特征与 token 的相似度图 \(S_L^i\) 聚合场景信息；相似度公式承担“从固定低频 prior 中检索与当前 patch 对应的稳定语义”的作用。
- **[2]** token 经线性映射投影回 VFM feature space，得到 \(	ilde f_L^i\)；该公式把抽象 token 转成可与原频带融合的增量特征。
- **[2]** 另一线性层将 \(	ilde f_L^i\) 融入低频 \(V_L^i\)，得到适配后的 \(	ilde V_L^i\)；这一式完成 IBA 的目标：不是重新学习低频，而是在已有稳定 content 上做轻量增强。

### 4.3 Illumination-Sensitive Band Adaptation

- **[2]** 开头明确中高频不能沿用低频策略：它们包含较强 illumination/domain variation，需要先做 Band Randomization，再做 Band Adaptation；这一顺序体现“先破坏域特异模式，再学习可泛化表示”。

#### Band Randomization

- **[3]** 回扣 Figure 3 说明高频和六个中频带会损害未见夜间域泛化，但直接删除又会损失边缘等像素级预测所需结构；这段构造“不能保留原值，也不能直接丢弃”的设计约束。
- **[3]** 用 \(V_{j,k}^i\sim U[0,T]\) 对 illumination-sensitive response 做有限范围随机化，并设 \(T=0.3\)；公式的作用是把 domain-specific frequency pattern 转为训练时的扰动源，在保留频率位置和结构信息的前提下扩大光照变化。
- **[2]** 将六个敏感频带两两拼接成 \(V_H^i,V_{M1}^i,V_{M2}^i\)，与低频分支保持相同处理粒度；这一组织方式为后续共享同构的 token adaptation 结构做准备。

#### Band Adaptation

- **[3]** 分别为高频和两个中频组引入 \(T_H^i,T_{M1}^i,T_{M2}^i\)，并以高频为例计算 token 与随机化频带的相似度 \(S_H^i\)；写法先完整展开一个分支，避免对三个同构模块重复推导。
- **[2]** token 经相似度图和线性层投影得到 \(	ilde f_H^i\)，再与随机化后的高频特征融合为 \(	ilde V_H^i\)；两组公式分别承担“提取随机化频带中的可用场景信息”和“完成敏感频带适配”的作用。
- **[2]** 中频 \(V_{M1}^i,V_{M2}^i\) 复用相同流程，最终与 \(	ilde V_L^i\) 一起经 IDST 得到下一层输入 \(f_{i+1}\)；最后一个公式把四组独立适配重新合成为完整空间表征，闭合单层 NightAdapter。
- **[3]** Discussion 用 DST 的 energy compaction 重新解释方法：不敏感频带负责增强稳定模式，敏感频带通过随机化削弱域特异性；频域单点变化会在空间域产生全局影响，用来说明这种 divide-and-conquer 适配为何能以较少参数影响全局 scene representation。

## 5. Experiments

### 5.1 Datasets & Implementation Details

- **[3]** 依次定义 Cityscapes、NightCity、NightCity-fine、BDD100K-Night、ACDC-Night、Dark Zurich 等数据源，并明确哪些作为 source、哪些只作为 unseen target；数据集介绍服务于后面 day-to-night、night-to-night、in-domain 三种 protocol 的区分，而不是简单罗列 benchmark。

### 5.2 Day-to-Night Generalization

- **[3]** 先按 DA/UDA、PEFT、style hallucination、VFM baseline 列出比较对象，以 Cityscapes 为唯一源域，在多个未见夜间数据集上测试；这种对比设置用于证明 NightAdapter 不依赖目标夜间域参与训练。
- **[2]** Table 1 与传统 DA/UDA 比较，突出在 ACDC-Night、Dark Zurich 等未见域上的大幅优势；表格承担“generalization paradigm 优于 target-aware adaptation”的核心证据。
- **[2]** Table 2 将比较范围收紧到 LoRA、AdaptFormer、VPT、EVP、WHT、FourierFT、SET、FADA 等 PEFT，且统一 DINOv2 backbone；这一表专门证明增益来自**夜间定制的频域设计**，而非单纯使用 VFM/PEFT。

### 5.3 In-Domain Night-time Segmentation

- **[3]** 将 NightAdapter 与 NightCity、NightLab、FDLNet、DTP、NightFormer 等 fully-supervised 夜间方法比较，在 NC、NC-f、BN 内部训练测试；这个实验回答“为了泛化而设计的方法会不会损失常规夜间分割能力”。
- **[2]** Table 3 在给出 mIoU 优势的同时报告 trainable parameters，11.5M 参数明显低于常见 CNN/ViT 全量训练规模；表格把性能优势与 parameter efficiency 两项贡献放在同一证据中。

### 5.4 Night-to-Night Generalization

- **[3]** 以 NightCity 或 NightCity-fine 为唯一夜间源域，对 RobustNet、SHADE、SAW、HGFormer、CMFormer、SET、DGInStyle、FADA、REIN 等 DGSS 方法进行跨夜间数据集测试；这一 protocol 排除“日→夜差异”因素，专门检验不同夜间 illumination/style 之间的迁移。
- **[2]** Table 4 在多个 unseen night domains 上保持稳定提升，作用是证明 NightAdapter 学到的不是 Cityscapes→night 的特定转换，而是可跨夜间场景迁移的 illumination robustness。

### 5.5 Ablation Studies

- **[3]** Table 5 从 REIN baseline 依次加入 IBA、SBR、SBA，拆开验证 illumination-insensitive enhancement、敏感频带随机化和敏感频带适配；表格直接对应 Sec. 4 的三个实际操作，建立模块级证据链。
- **[2]** SBR 本身即可带来明显提升，配合 SBA 后增益扩大；作者借这一结果强调 randomization 是削弱 domain gap 的关键，而 SBA 负责在扰动后恢复可用于分割的稳定表示。
- **[2]** Table 6 分别关闭/启用 \(V_L,V_{M1},V_{M2},V_H\) 的 adaptation，证明不同频带均有贡献；该表验证 Sec. 3 的分带并非只需要保留低频，而是要对不同频段采用不同策略。
- **[2]** Figure 5 用 t-SNE 比较 REIN 与 NightAdapter 的 unseen-night feature distribution；不同数据集颜色在 NightAdapter 中混合得更均匀，用特征空间证据支撑“domain gap 被压缩”的解释，而不是只报告最终 mIoU。

### 5.6 Visualization

- **[2]** Figure 6 在 day-to-night 设置下与 DAFormer、HRDA、MIC、CoDA、REIN 做可视化比较，展示更完整、准确的目标区域；定性结果对应 Table 1–2 的跨域性能提升。
- **[2]** Figure 7 改用 NightCity 源域，与 RobustNet、SAW、HGFormer、CMFormer、REIN 比较；与 Figure 6 成对出现，用相同视觉证据分别覆盖 day-to-night 与 night-to-night 两条实验主线。

## 6. Conclusion

- **[2]** 结论先回到本文最初的频域观察：DST 后不同 VFM 频带对日夜光照变化具有不同敏感性，再用这一观察概括 NightAdapter 的设计来源。
- **[2]** 用 illumination-insensitive adaptation 与 illumination-sensitive adaptation 对应 robustness 和 domain-gap mitigation，并以三类实验设置的领先表现收束全文；结论保持“频域观察 → 两类设计 → 三类验证”的主线，不扩展新的论点。

## 整体写作主线

- **先定义新的问题边界**：不是只做夜间分割，也不是已知目标域的 day-to-night adaptation，而是“一个 source domain → 任意未见 night-time domain”的泛化问题。
- **先做诊断，再给方法**：Sec. 3 用 DST + band-rejection experiment 找到 illumination-sensitive / insensitive bands，Figure 3 是 NightAdapter 最重要的方法动机证据。
- **方法严格由诊断结果推导**：低频不能丢弃，所以用 IBA 增强稳定 content；中高频直接保留会携带域偏差、直接删除会损失结构，所以采用“randomization + adaptation”。
- **实验按三层问题验证**：day-to-night 验证未知夜间泛化，night-to-night 排除昼夜转换因素，in-domain 验证泛化设计没有牺牲常规夜间分割能力。
- **消融回扣每个设计假设**：Table 5 验证 IBA/SBR/SBA，Table 6 验证各频段，Figure 5 验证特征域混合；每项实验都能对应 Sec. 3–4 中的具体判断。
