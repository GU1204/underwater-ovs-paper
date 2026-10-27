# PPW Repo-to-Paper Outline — H2 Draft

Status: H1 confirmed; H2 structure awaiting confirmation.

This draft was independently derived from the current source code, inherited configurations, the clean FC-CLIP comparison tree, curated experiment records, and completed evaluation logs. The existing `writing_blueprint.md` and `content_analysis.md` were excluded.

> **[Chinese]** 状态：H1 已确认，H2 等待确认。本稿依据当前源码、继承配置、纯净 FC-CLIP 对照目录、实验整理记录与已完成评估日志独立生成；未使用 `writing_blueprint.md` 和 `content_analysis.md`。

# 1. Introduction

Establish the domain-shift and recognition--localization challenges of open-vocabulary underwater segmentation, motivate adapting an FC-CLIP baseline, and state evidence-bounded contributions.

> **[Chinese]** 说明开放词汇水下分割中的域偏移及识别—定位矛盾，阐明适配 FC-CLIP 的必要性，并给出受证据约束的贡献陈述。

## 1.1 Open-Vocabulary Underwater Instance Segmentation

Introduce underwater instance segmentation as an open-vocabulary recognition and localization problem in which training and test vocabularies, visual conditions, and category distributions differ substantially.  
*Sources: `README.md`; `/data/gjf/output/fc-clip/docs/WaterOVS注册与训练配置说明.md`; `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`.*

> **[Chinese]** 将水下实例分割表述为开放词汇识别与定位问题，其中训练和测试词表、成像条件及类别分布均存在显著差异。

## 1.2 Recognition--Localization and Domain-Shift Gap

Motivate the study from two observed limitations: region classification and mask localization do not improve in lockstep, while in-domain gains and late-training behavior do not reliably transfer across underwater datasets.  
*Sources: `/data/gjf/files/exp/FC-CLIP整体实验文档.md`; `/data/gjf/files/exp/FathomNet_v3_PGC_LDP_EVP_checkpoint_experiment_2026-10-14.md`; `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`.*

> **[Chinese]** 从两类实证局限引出研究动机：区域分类与掩码定位并非同步改善，且域内增益和训练后期行为不能稳定迁移到不同水下数据集。

## 1.3 Research Objective and Evidence-Bounded Contributions

State the objective as adapting a frozen convolutional FC-CLIP through paired structure--semantic correction and frequency-conditioned visual prompting, while restricting contribution claims to matched ablations and completed evaluations.  
*Sources: `fcclip/modeling/local_discriminative_pooling.py`; `fcclip/modeling/backbone/evp_prompter.py`; `/data/gjf/files/exp/FC-CLIP_ConvNeXt-Large与BS9_PGC-LDP_EVP_2026-08-13_2026-08-19.md`.*

> **[Chinese]** 将研究目标界定为通过成对结构—语义修正和频率条件视觉提示适配冻结卷积式 FC-CLIP，并仅依据严格配对消融与已完成评测陈述贡献。

# 2. Related Work

Position the study relative to open-vocabulary segmentation, frozen vision--language adaptation, mask/prompt-based segmentation, and underwater visual recognition.

> **[Chinese]** 从开放词汇分割、冻结视觉—语言模型适配、掩码/提示驱动分割以及水下视觉识别四条线索梳理相关工作。

## 2.1 Frozen Vision--Language Models for Open-Vocabulary Segmentation

Review FC-CLIP and related approaches that reuse frozen image--text representations for class-agnostic masks and vocabulary-flexible region classification.  
*Sources: `README.md`; `/data/gjf/output/fc-clip/README.md`; `fcclip/fcclip.py`.*

> **[Chinese]** 回顾 FC-CLIP 及相关方法如何复用冻结图文表征，同时实现类别无关掩码生成与词表可扩展的区域分类。

## 2.2 Side-Adapter and Prompt-Guided Segmentation: SAN and PosSAM Context

Contrast side-adapter and prompt-guided segmentation with the present design, emphasizing that PGC-LDP is PosSAM-inspired but does not introduce a SAM encoder, IoU head, or heterogeneous SAM--CLIP fusion.  
*Sources: `/data/gjf/output/fc-clip/docs/FC-CLIP_LDP_PGC当前实现设计.md`; `fcclip/modeling/local_discriminative_pooling.py`.*

> **[Chinese]** 对比 SAN、PosSAM 所代表的侧适配与提示引导分割路线，并强调当前 PGC-LDP 仅受相关思想启发，没有引入 SAM 编码器、IoU head 或异构 SAM--CLIP 融合。

## 2.3 Frequency-Aware Visual Prompt Adaptation

Discuss efficient visual prompting and frequency-aware adaptation as mechanisms for injecting task-specific cues into a frozen visual backbone without full fine-tuning.  
*Sources: `/data/gjf/output/fc-clip/docs/FC-CLIP_EVP_low_conditioned_gate当前实现.md`; `fcclip/modeling/backbone/evp_prompter.py`; `fcclip/modeling/backbone/clip.py`.*

> **[Chinese]** 讨论高效视觉提示与频率感知适配如何在不完整微调视觉骨干的前提下写入任务相关线索。

## 2.4 Underwater Open-Vocabulary Data and Evaluation

Summarize vocabulary mismatch, long-tail distributions, incomplete annotation, and evaluation bias that distinguish WaterOVS and FathomNet from standard open-vocabulary benchmarks.  
*Sources: `/data/gjf/output/fc-clip/docs/WaterOVS注册与训练配置说明.md`; `docs/FathomNetCurrentRetainedTaxonomyCleanedEval.md`; `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`.*

> **[Chinese]** 总结 WaterOVS 与 FathomNet 相较常规开放词汇基准所特有的词表错位、长尾分布、标注不完整与评测偏差问题。

# 3. Proposed Approach

Describe the FC-CLIP starting point and organize PGC-LDP and EVP into a coherent recognition-and-localization pipeline.

> **[Chinese]** 说明 FC-CLIP 起点，并将 PGC-LDP 与 EVP 组织为统一的识别—定位流程。

## 3.1 FC-CLIP Baseline and Dual Classification Paths

Describe the shared frozen ConvNeXt CLIP backbone, class-agnostic mask decoder, in-vocabulary classifier, mask-pooled out-of-vocabulary classifier, and their inference-time ensemble.  
*Sources: `README.md`; `/data/gjf/output/fc-clip/fcclip/fcclip.py`; `fcclip/fcclip.py`.*

> **[Chinese]** 说明共享冻结 ConvNeXt CLIP 骨干、类别无关掩码解码器、域内分类器、基于掩码池化的域外分类器及其推理期融合。

## 3.2 Paired Gated Correction for Local Discriminative Pooling

Present PGC-LDP as a per-query, per-head gate that retains a CLIP region token as the semantic anchor and writes only its paired decoder structure token as a residual correction.  
*Sources: `fcclip/modeling/local_discriminative_pooling.py`; `fcclip/fcclip.py`; `/data/gjf/output/fc-clip/docs/FC-CLIP_LDP_PGC当前实现设计.md`.*

> **[Chinese]** 将 PGC-LDP 描述为逐查询、逐注意力头的门控修正：以 CLIP 区域 token 为语义锚点，仅将同索引解码器结构 token 作为残差写入。

## 3.3 Preserve-Void Open-Vocabulary Logit Integration

Explain why PGC-LDP replaces only non-void class logits while retaining FC-CLIP's learned void/objectness logit and leaving auxiliary decoder supervision unchanged.  
*Sources: `fcclip/modeling/local_discriminative_pooling.py`; `fcclip/fcclip.py`; `/data/gjf/files/exp/FC-CLIP整体实验文档.md`.*

> **[Chinese]** 解释 PGC-LDP 为何只替换非 void 类别 logit，同时保留 FC-CLIP 已学习的 void/objectness 路径，并维持辅助解码层监督不变。

## 3.4 Low-Frequency-Conditioned High-Pass Visual Prompting

Describe EVP as high-pass and embedding prompt residuals injected into ConvNeXt stages 1--2, with a low-pass branch conditioning conservative spatial gated-add fusion.  
*Sources: `fcclip/modeling/backbone/evp_prompter.py`; `fcclip/modeling/backbone/clip.py`; `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_waterovs_id_pgc_ldp_evp_low_conditioned_highpass_stage12_gated_add_gb2_full.yaml`.*

> **[Chinese]** 将 EVP 描述为向 ConvNeXt 第 1--2 阶段注入高通与嵌入提示残差，并由低通分支对保守的空间 gated-add 融合进行条件控制。

# 4. Experimental Protocol

Define datasets, implementation settings, metrics, baselines, ablations, and checkpoint-selection rules from reproducible artifacts.

> **[Chinese]** 基于可复现实物证据说明数据集、实现设置、评价指标、基线、消融方案与检查点选择规则。

## 4.1 Datasets, Vocabularies, and Evaluation Splits

Define WaterOVS train/validation vocabularies and FathomNet v3 training, retained-test, and full-test protocols, explicitly separating diagnostic from generalization claims.  
*Sources: `/data/gjf/output/fc-clip/docs/WaterOVS注册与训练配置说明.md`; `docs/FathomNetCurrentRetainedTaxonomyCleanedEval.md`; `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`.*

> **[Chinese]** 定义 WaterOVS 训练/验证词表以及 FathomNet v3 训练集、retained test 和 full test 协议，并明确区分诊断性结论与泛化结论。

## 4.2 Training Configuration and Implementation Details

Report backbone scale, enabled modules, batch size, training length, precision setting, stage selection, gate initialization, and checkpoint interval from inherited configurations.  
*Sources: `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_waterovs_id_pgc_ldp_evp_low_conditioned_highpass_stage12_gated_add_gb2_full.yaml`; `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_fathomnet_current_retained_visual_common_v3_mapped_complete_id_pgc_ldp_evp_low_conditioned_highpass_stage12_gated_add_gb2_full_bs15.yaml`; `fcclip/config.py`.*

> **[Chinese]** 根据继承配置报告骨干规模、启用模块、批量大小、训练长度、精度设置、注入阶段、门控初始化与检查点间隔。

## 4.3 Metrics and Checkpoint-Selection Protocol

Use COCO mask AP, AP50, AP75, and scale-stratified AP, while reporting both best and final checkpoints and avoiding model selection on the full external test set.  
*Sources: `/data/gjf/files/exp/FathomNet_v3_PGC_LDP_EVP_checkpoint_experiment_2026-10-14.md`; `/data/gjf/files/exp/FC-CLIP_ConvNeXt-Large与BS9_PGC-LDP_EVP_2026-08-13_2026-08-19.md`; `output/eval_full_test_fathomnet_current_retained_visual_common_v3_mapped_complete_id_pgc_ldp_evp/log.txt`.*

> **[Chinese]** 使用 COCO mask AP、AP50、AP75 及尺度分层 AP，同时报告最佳与最终检查点，并避免在完整外部测试集上选择模型。

## 4.4 Baselines, Ablations, and Comparison Boundaries

Organize comparisons by matched backbone, batch size, precision, code path, and training schedule so PGC-LDP, EVP, and their combination are not credited using unmatched runs.  
*Sources: `/data/gjf/files/exp/FC-CLIP_BS15_LDP_EVP消融实验_2026-08-19.md`; `/data/gjf/files/exp/FC-CLIP_BS9_noAMP核心模块对比简表_2026-08-21.md`; `/data/gjf/files/exp/FC-CLIP_ConvNeXt-Large与BS9_PGC-LDP_EVP_2026-08-13_2026-08-19.md`.*

> **[Chinese]** 按骨干、批量大小、数值精度、代码路径和训练日程对齐对比，避免使用非配对实验将收益错误归因于 PGC-LDP、EVP 或其组合。

# 5. Results and Discussion

Organize verified findings into in-domain comparisons, component attribution, checkpoint dynamics, cross-dataset transfer, and bounded interpretation.

> **[Chinese]** 将经核验的发现组织为域内对比、组件归因、检查点动态、跨数据集迁移及边界明确的讨论。

## 5.1 In-Domain WaterOVS Performance

Compare the strongest aligned FC-CLIP baselines and PGC-LDP/EVP variants on WaterOVS, separating ConvNeXt-Base from ConvNeXt-Large and BS9 from BS15 protocols.  
*Sources: `/data/gjf/files/exp/FC-CLIP整体实验文档.md`; `/data/gjf/files/exp/FC-CLIP_ConvNeXt-Large与BS9_PGC-LDP_EVP_2026-08-13_2026-08-19.md`; `/data/gjf/files/exp/FC-CLIP_BS15_LDP_EVP消融实验_2026-08-19.md`.*

> **[Chinese]** 在 WaterOVS 上比较经过协议对齐的 FC-CLIP 基线与 PGC-LDP/EVP 变体，并分别讨论 ConvNeXt-Base/Large 与 BS9/BS15。

## 5.2 Component Attribution and Interaction

Use PGC-LDP-only, EVP-only, global-LDP, preserve-void, and combined controls to identify which improvements are supported and where complementarity remains unproven.  
*Sources: `/data/gjf/files/exp/FC-CLIP_BS9_noAMP核心模块对比简表_2026-08-21.md`; `/data/gjf/files/exp/FC-CLIP_BS15_LDP_EVP消融实验_2026-08-19.md`; `/data/gjf/files/exp/FC-CLIP整体实验文档.md`.*

> **[Chinese]** 利用 PGC-LDP-only、EVP-only、global-LDP、preserve-void 与组合对照，区分已获证据支持的改进和仍未被证明的互补性。

## 5.3 FathomNet Adaptation and Checkpoint Dynamics

Analyze training fit, retained-test trajectories, early cross-dataset peaks, and late degradation to show why checkpoint choice is part of the empirical finding rather than a reporting detail.  
*Sources: `/data/gjf/files/exp/FathomNet_v3_PGC_LDP_EVP_checkpoint_experiment_2026-10-14.md`; `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`; `/data/gjf/files/exp/FC-CLIP整体实验文档.md`.*

> **[Chinese]** 分析训练拟合、retained-test 轨迹、跨数据集早期峰值与后期退化，说明检查点选择本身是实验发现的一部分，而非单纯报告细节。

## 5.4 Full-Test Transfer, Failure Modes, and Claim Boundaries

Report the completed mapped-full-test result alongside category and scale failures, then discuss the effects of vocabulary coverage, long-tail supervision, missing labels, and test construction on generalization.  
*Sources: `output/eval_full_test_fathomnet_current_retained_visual_common_v3_mapped_complete_id_pgc_ldp_evp/log.txt`; `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`; `configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_fathomnet_current_retained_visual_common_v3_mapped_complete_id_eval_full_test.yaml`.*

> **[Chinese]** 报告已完成的 mapped full-test 结果及类别/尺度失败模式，并讨论词表覆盖、长尾监督、漏标和测试集构建对泛化结论的影响。

# 6. Conclusion

Summarize validated methodological and empirical contributions, state limitations, and identify realistic next steps.

> **[Chinese]** 总结已被证据验证的方法与实验贡献，说明局限性，并提出切实的后续方向。

## 6.1 Validated Findings and Contribution Boundaries

Recap the supported role of paired structure--semantic correction, the conditional value of frequency prompting, and the observed gap between in-domain improvement and cross-domain transfer.  
*Sources: `/data/gjf/files/exp/FC-CLIP整体实验文档.md`; `/data/gjf/files/exp/FC-CLIP_ConvNeXt-Large与BS9_PGC-LDP_EVP_2026-08-13_2026-08-19.md`; `output/eval_full_test_fathomnet_current_retained_visual_common_v3_mapped_complete_id_pgc_ldp_evp/log.txt`.*

> **[Chinese]** 概括成对结构—语义修正的已验证作用、频率提示的条件性价值，以及域内提升与跨域迁移之间的实证差距。

## 6.2 Limitations and Future Work

Prioritize cleaner supervision, fixed unbiased evaluation splits, explicit seen/unseen and long-tail reporting, repeated matched controls, and small-object analysis before stronger generalization claims.  
*Sources: `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md`; `/data/gjf/files/exp/FathomNet_v3_PGC_LDP_EVP_checkpoint_experiment_2026-10-14.md`; `/data/gjf/output/fc-clip/docs/FC-CLIP_LDP_PGC当前实现设计.md`.*

> **[Chinese]** 在提出更强泛化主张前，优先完善监督完整性、固定无偏评测划分、显式 seen/unseen 与长尾报告、重复配对对照及小目标分析。

