# PPW Repo-to-Paper Outline

Status: H1 structure awaiting confirmation

This outline was generated independently from a filename-level scan of the current repository, the clean FC-CLIP baseline, and the curated experiment directory. The existing `writing_blueprint.md` and `content_analysis.md` were explicitly excluded from the evidence set.

> **[Chinese]** 状态：等待确认一级章节结构。本大纲仅依据当前仓库、纯净 FC-CLIP 基线和实验整理目录的文件名级扫描独立生成；既有 `writing_blueprint.md` 与 `content_analysis.md` 已明确排除在证据来源之外。

## Scan Summary

- **Documentation:** project README files and method/dataset design documents; likely evidence for problem framing and implementation intent.
- **Configuration:** FC-CLIP training and evaluation configurations; likely evidence for model variants, datasets, and experimental settings.
- **Code:** baseline and modified model, backbone, prompting, pooling, dataset registration, training, and evaluation modules; primary evidence for the method section.
- **Results:** curated FC-CLIP ablation, checkpoint, dataset-diagnosis, and aggregate result files; primary evidence for the results section.
- **Figures:** the project teaser and experiment visualizations discovered during the later evidence pass.
- **Dependencies:** environment and installation files; reproducibility evidence.

> **[Chinese]** 扫描材料被归入六类：项目与方法说明、训练和评估配置、核心模型代码、整理后的实验结果、图像资源以及环境依赖。后续细化将以代码和配置支撑方法描述，以真实结果文件支撑定量结论。

# 1. Introduction

Establish the domain-shift and recognition--localization challenges of open-vocabulary underwater segmentation, motivate adapting an FC-CLIP baseline, and state the paper's evidence-bounded contributions.

> **[Chinese]** 说明开放词汇水下分割中的域偏移以及识别—定位矛盾，阐明在 FC-CLIP 基线上开展适配的必要性，并给出受实验证据约束的贡献陈述。

# 2. Related Work

Position the study relative to open-vocabulary segmentation, frozen vision--language adaptation, mask/prompt-based segmentation, and underwater visual recognition without conflating prior methods with the proposed components.

> **[Chinese]** 从开放词汇分割、冻结视觉—语言模型适配、掩码/提示驱动分割以及水下视觉识别四条线索梳理相关工作，并严格区分已有方法与本文组件。

# 3. Proposed Approach

Describe the FC-CLIP starting point and the repository-indicated PGC, LDP, and EVP components as one coherent recognition-and-localization pipeline, with exact operations deferred to source-code verification.

> **[Chinese]** 说明 FC-CLIP 起点，并将仓库文件名所指向的 PGC、LDP 与 EVP 组织为统一的识别—定位流程；具体运算将在源代码核验后写入二级和三级大纲。

# 4. Experimental Protocol

Define datasets and label spaces, train/validation/test construction, implementation settings, metrics, comparison methods, ablation design, and checkpoint-selection rules from reproducible artifacts.

> **[Chinese]** 基于可复现实物证据说明数据集及标签空间、训练/验证/测试划分、实现设置、评价指标、对比方法、消融方案与检查点选择规则。

# 5. Results and Discussion

Organize verified findings into main comparisons, component ablations, scale and checkpoint analyses, cross-dataset evaluation, qualitative behavior, failure cases, and bounded interpretation.

> **[Chinese]** 将经核验的发现组织为主要对比、组件消融、规模与检查点分析、跨数据集评估、定性表现、失败案例及边界明确的讨论。

# 6. Conclusion

Summarize the validated methodological and empirical contribution, state limitations, and identify realistic next steps without introducing unsupported claims.

> **[Chinese]** 总结已被证据验证的方法与实验贡献，说明局限性，并提出切实的后续方向，不引入缺乏支撑的新结论。
