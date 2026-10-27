# 水下开放词汇实例分割论文：修订与验证执行计划（含 Agent 提示词）

> **用途**：将 `content_analysis.md`（技术事实与证据底稿）和 `writing_blueprint.md`（暂定写作蓝图）的审查意见转成可交付、可验收、可逐项执行的 Agent 任务。
>
> **项目**：FC-CLIP + PGC-LDP + frequency-conditioned EVP，目标为 underwater open-vocabulary instance segmentation。
>
> **原则**：先核对事实，再验证机制，最后确定论文主线；不要为了吻合预设 Motivation 而选择性报告实验。
>
> **证据状态说明**：本计划依据两份已有报告和前期审查意见制定，**不是**新完成的代码复验或实验结果。执行时以实际代码、有效配置、checkpoint 与 evaluator 原始输出为准。

---

## 0. 如何使用这份计划

### 0.1 给 Agent 的项目路径

以下路径来自已有技术分析，Agent 必须先检查是否存在，不得假定路径或版本仍然有效。

| 用途 | 候选路径 |
|---|---|
| 当前开发代码 | `/data/gjf/code/fc-clip-amp` |
| 已整理的 clean 实现 | `/data/gjf/output/fc-clip` |
| 实验资料与汇总 | `/data/gjf/files/exp` |
| 方法分析 | `content_analysis.md`（通过当前 Agent 工作区/附件定位） |
| 写作蓝图 | `writing_blueprint.md`（通过当前 Agent 工作区/附件定位） |
| 建议输出根目录 | `/data/gjf/files/exp/paper_revision_agent`（若无权限，可改为项目工作区下的 `paper_revision_agent/`） |

**给 Agent 的通用约束**（每个独立任务也会强调关键点）：

1. 先检查 `git status`、当前 branch/commit、未提交修改；**禁止**对现有用户工作执行 `git reset`、`git clean`、强制覆盖、删除或自动提交。
2. 以 `log.txt` / `eval.log` / `metrics.json`、run-local `config.yaml`、checkpoint 和真实源码为高可信证据。目录名、旧汇总表和报告文字仅作检索线索。
3. 每一个技术结论标记为 `CODE-FACT`（源码可证）、`MEASURED`（原始运行支持）、`INFERENCE`（解释/假设）或 `OPEN`（尚缺证据）。
4. 现有全套 matched WaterOVS 主消融为**单种子**；`best` 与 `final`、40k 与 50k、AMP 与 FP32、不同 batch/代码版本不能混成一组比较。
5. **global-LDP 的分类诊断不是 PGC 的直接证据**。不要用前者的 seen/unseen 或 mask-IoU 数字证明后者的机制。
6. 不得虚构文献、引用、消融、运行日志、GPU 资源或实验指标。缺失时写 `NOT VERIFIED` / `PENDING`。
7. 默认先产出检查结果、对照表、脚本与 dry-run。耗时的大规模训练必须在确认环境、资源、数据协议和配置后启动；如果当前环境不允许执行，就交付可运行命令与配置，不得声称实验完成。
8. 写作修改尽量生成**独立修订稿和 diff**，不要破坏现有 `content_analysis.md` / `writing_blueprint.md`；代码必要改动也应最小化，并保留测试结果。
9. 每项任务均输出：`发现 → 证据路径 → 判断及局限 → 已执行动作 → 剩余工作`。

### 0.2 现有主实验数字（只作为审查锚点，执行时重新核实）

报告中的 matched WaterOVS ConvNeXt-Large / BS9 / FP32 / seed 42 / final checkpoint：

| 方法 | Final AP | APs | 相对 baseline AP |
|---|---:|---:|---:|
| FC-CLIP | 47.5934 | 14.3429 | — |
| EVP-only | 49.7202 | 26.2014 | +2.1268 |
| PGC-only | 53.9612 | 10.6254 | +6.3678 |
| PGC + EVP | 54.0845 | 9.0898 | +6.4911 |

- 组合模型 final 比 PGC-only 高 **0.1233 AP**，不支持“强协同”的直接结论。
- 组合模型 40k 达到 **54.4971 AP**；应始终标出 checkpoint，不与其他方法的 final 无条件比较。
- `preserve-void` 的强烈效果来自特定 **5k 早期对照**（1.2070 vs 42.4391 AP），尚不等于所有场景的普遍定理。
- FathomNet **prediction-free full test 已完成**：报告记载 40k checkpoint、**15.5106 AP**；不要沿用旧文档中的 `pending`，但需再核对原始 log。

### 0.3 任务总览与依赖

| ID | 优先级 | 计划 | 主要属性 | 依赖 | 交付物 |
|---|---|---|---|---|---|
| **P00** | P0 | 证据与运行协议审计 | 无需 GPU | 无 | `P00_evidence_registry.md/csv` |
| **P01** | P0 | PGC / EVP 表征与语义定义核对 | 代码审查+轻量测试 | P00 | `P01_method_semantics.md` |
| **P02** | P1 | 单元测试与实现一致性修复 | 代码/CPU | P00、P01 | `P02_test_audit.md` |
| **P03** | P0 | PGC 必要性与融合拓扑消融 | GPU 训练 | P00、P01、P02 | `P03_pgc_ablation.md` |
| **P04** | P0 | PGC 分类 vs 定位与 seen/unseen 诊断 | 评估/可能需 GPU 推理 | P00、P01 | `P04_pgc_diagnostics.md` |
| **P05** | P1 | EVP 高频、低频门控与浅层注入消融 | GPU 训练 | P00、P01、P02 | `P05_evp_ablation.md` |
| **P06** | P1 | 四路多种子复现与交互分析 | GPU 训练 | P00、P02；可并行 P03/P05 | `P06_reproducibility.md` |
| **P07** | P1 | 水下特性分层分析与真实案例 | 评估/可视化 | P00、P04 | `P07_underwater_evidence.md` |
| **P08** | P1/P2 | FathomNet 协议审计与 matched controls | 先数据审计，后 GPU | P00 | `P08_fathomnet_protocol.md` |
| **P09** | P2 | 参数量、延迟与显存核算 | GPU/性能测试 | P00、P01 | `P09_complexity.md` |
| **P10** | P0 | 相关工作 / novelty 审查 | 文献研究 | P01 | `P10_novelty_matrix.md` |
| **P11** | P0 | Motivation 与写作蓝图重构 | 写作/逻辑审查 | P01、P04、P10；可先做候选稿 | `P11_revised_blueprint.md` |
| **P12** | P0 | 论文主张一致性与最终交接 | 文稿/证据审计 | 以上已完成部分 | `P12_claim_evidence_audit.md` |

**建议执行顺序**：`P00 → P01 → (P02、P04、P10) → P03/P05/P06/P07/P08/P09 → P11 → P12`。

> 不必等所有训练做完再开始写作。P11 可以先给出两个候选 Motivation，但必须把依赖 P03–P08 的断言标为“待验证”；获得新结果后更新，而不是倒推结论。

---

## P00｜建立单一可信的代码—配置—结果证据台账

**优先级**：P0。**目标**：确定现有所有关键数字对应的具体源码、数据、训练及评估协议，避免“表面 matched，实际不一致”。

**执行任务**：定位主消融四个 run、5k preserve-void 对照、global-LDP 诊断、FathomNet 全量评估；对齐 checkpoint、配置、原始日志；标出历史汇总中已过时的状态。特别检查四路 run 是否具有同一代码快照，报告目前尚未确定此点。

**可直接复制给 Agent 的提示词**：

```text
你现在是本项目的 reproducibility / experiment-audit agent。请基于 @content_analysis.md 和 @writing_blueprint.md，直接审查我实际的 FC-CLIP 项目与历史运行结果。

项目候选位置：
- 开发仓库 /data/gjf/code/fc-clip-amp
- clean 实现 /data/gjf/output/fc-clip
- 实验汇总 /data/gjf/files/exp
- 输出目录 /data/gjf/files/exp/paper_revision_agent

要求：
1. 先核对路径、git branch/commit/status/dirty state，不得修改或清理已有用户代码。
2. 定位 WaterOVS matched Large/BS9/FP32/seed-42 的 baseline、EVP-only、PGC-only、PGC+EVP 四个原始运行；读取 run-local config、metrics/eval logs、checkpoint 名称。逐字段核实 backbone、数据 split、类别映射、训练 iterations、batch、AMP、seed、优化器、学习率、评测器、checkpoint selection、代码版本。不能从 run 名称推断一致性。
3. 为 5k preserve-void、global-LDP 分类诊断、FathomNet retained/full/WaterOVS transfer 建立独立记录。特别查验 FathomNet prediction-free full-test 原始日志，纠正过时的 pending 状态。
4. 如无法确定某个 run 的代码快照，标记 NOT VERIFIED，说明需要什么证明；不得默认 matched。
5. 建立 evidence registry，至少包含 run_id、module flags、code_version、dirty_status、train_data、eval_data、class_vocabulary、seed、batch、AMP、checkpoint、best/final、AP/AP50/AP75/APs/APm/APl、config/log/checkpoint paths、可信度、是否可直接比较。
6. 检查原始日志与两份文档数字是否一致，提供差异报告；不要覆盖原文档。

输出：P00_evidence_registry.csv、P00_evidence_registry.md、P00_discrepancies.md。每条差异给出来源路径及修订建议；最后给出哪些对照可以进入主论文表，哪些不能。
```

**验收标准**：主四路实验每行都能追溯到原始日志+有效配置；显式说明代码快照匹配是否已证实；FathomNet full-test 状态统一。

---

## P01｜纠正 PGC 与 EVP 的技术表述和方法边界

**优先级**：P0。**目标**：解决“structure token 是否纯结构”“冻结 CLIP 是否等于保持对齐”“EVP 高频/低频究竟做了什么”等表述问题。

**执行任务**：以源码逐步追踪 `S_i`、`C_i` 来源、训练梯度、最终 logits、void 路径及 `clip_vis_dense` 的生成位置；明确 EVP 注入能改变什么；给出与 FC-CLIP baseline 的功能差异，不凭模块命名猜测机制。

**可直接复制给 Agent 的提示词**：

```text
请担任模型实现审计与论文 Methods 审稿人，读取 FC-CLIP baseline、PGC-LDP、EVP 的实际源码，并检查 @content_analysis.md 与 @writing_blueprint.md 的技术表述。

重点文件在 /data/gjf/output/fc-clip/fcclip/ 下的 fcclip.py、modeling/local_discriminative_pooling.py、modeling/transformer_decoder/fcclip_transformer_decoder.py、modeling/backbone/evp_prompter.py、modeling/backbone/clip.py；如果开发仓库实现不同，要记录两者的差异与对应实验版本。

请完成：
1. 追踪每个 mask/query 对应的 decoder 表征 S_i 与 mask-pooled CLIP token C_i 的来源、维度、归一化、池化是否 hard/detached、投影是否 trainable、反向传播路径。
2. 判断 decoder structure token 这一术语在实现上成立到什么程度；如果不能证明其为纯类无关空间信息，建议更精确名称（例如 task-adapted mask-aligned decoder representation），同时尽量保留作者已有符号。
3. 精确复核 PGC 的 paired-only、zero-self value、每头门控、LayerNorm、final-only non-void replacement、base void preservation、辅助损失、几何 ensemble；检查哪些部件已经存在于 baseline，不能将 baseline 已有操作包装成新贡献。
4. 明确区分 frozen CLIP weights、独立 CLIP 分类路径、实际 feature alignment preservation；指出 EVP 注入后是否仍可把 clip_vis_dense 直接称为原始未改变的 frozen-CLIP representation。
5. 解释 EVP high-pass residual 和 low-pass gate condition 的真实信息流，stage 1/2 的代码含义，以及 gate bias=-2 不是 identity 的原因。
6. 给每项论文级断言打 CODE-FACT、MEASURED、INFERENCE、OPEN 标签；纠正公式或方法图中任何不精确表达。

输出 P01_method_semantics.md：包含真实数据流、关键公式、冻结/可训练参数表、术语替换表、错误表述清单和可直接用于 Methods 的中英文准确表述。涉及源码事实须带文件路径、函数名或行号。不要修改模型算法。
```

**验收标准**：任何“纯结构信息”“完全保留 CLIP”“恢复纹理”“保持原样”等表述均有源码依据，或被明确降级为假设；Methods 可据此准确重建模型。

---

## P02｜修复测试覆盖和探索分支的不一致

**优先级**：P1。**目标**：消除测试失败引起的实现可信度问题，区分 main path 与 exploratory path。

**已知背景**：报告记录 `test/test_local_discriminative_pooling.py` 与 `test/test_evp_frequency_router.py` 共 **33 passed / 3 failed**；失败均与探索性的 `amplitude_selected_bandpass` 接口不匹配。通过排除该分支可得到 **33 passed / 3 deselected**。该记录应在当前代码重新验证。

**可直接复制给 Agent 的提示词**：

```text
请在不破坏现有工作的前提下审核 PGC-LDP 和 EVP 测试。先检查 git status，并使用当前项目所需的 Python 环境，复现：

/data/gjf/env/miniconda3/envs/MARIS/bin/python -m pytest -q test/test_local_discriminative_pooling.py test/test_evp_frequency_router.py

再完成：
1. 记录当前真实通过/失败测试，而不是照搬旧报告；对比两个测试文件与实现接口。
2. 确认 selected low_conditioned_gate 路径、paired query、zero-self、preserve-void、shape/gradient 等不变量是否有测试覆盖。
3. 对 amplitude_selected_bandpass 的失败提出最小修复方案：如果该路径并非当前受支持功能，可以通过明确标记/拆分测试使默认测试集与实际支持范围一致；不要伪造通过，也不要直接删掉失败记录。若确需支持，先提出实现与测试同步的改动，再在最小范围内修复。
4. 运行相关测试及必要 smoke checks，并记录代码 diff、运行命令、退出码与结果。

输出 P02_test_audit.md、必要的最小代码 diff 和可复现测试命令。禁止重置用户未提交修改；不对未经验证的探索分支声明支持。
```

**验收标准**：主要 PGC/EVP 路径的测试可复现通过；探索分支要么修复且测试通过，要么明确隔离并作为未支持功能记录。

---

## P03｜验证 PGC-LDP 的设计必要性，而不仅是有效性

**优先级**：P0。**目标**：回答“为什么必须 same-mask pairing、zero-self、保留 void，而不是简单融合或全局 LDP”。

**实验设计建议**（均须先审查配置的可比性）：

| 对照 | 核心差别 | 对应问题 |
|---|---|---|
| FC-CLIP baseline | 不加入 PGC | PGC 是否整体有效 |
| Simple Add / Concat (+投影) | 相同 token、简单融合 | 复杂门控是否必要 |
| Global / Cross-query LDP | 允许读取其他 query | Paired-only 是否有优势 |
| PGC (full) | same-mask + zero-self | 主方法 |
| PGC without zero-self | self value 改为可写入/常规值 | keep-or-correct 机制是否必要 |
| PGC replacing void（审慎小规模） | 同时改写 no-object logits | preserve-void 的边界 |
| 可选：final-only vs additional layers | 修正层数改变 | final-only 是否必要 |

**注意**：不同结构可能有不同参数量/训练稳定性。除性能外须报告参数量，并尽可能提供容量匹配或公平性讨论。对会造成崩溃的 void 控制，优先使用已存在的有效记录或短程实验，不需要盲目耗费完整训练预算。

**可直接复制给 Agent 的提示词**：

```text
请作为 PGC-LDP 机制验证 Agent，在 WaterOVS 主实验协议下设计并实施严格可归因的融合消融。

输入：@content_analysis.md、@writing_blueprint.md、P00_evidence_registry、P01_method_semantics；代码以实际运行快照为准。主协议目标为 ConvNeXt-Large、WaterOVS ID、global BS9、FP32、50k、固定 seed，与已验证的 baseline/PGC run 一致。

要求：
1. 首先查找是否已存在真正 matched 的 baseline、global-LDP、simple fusion、PGC/zero-self 等运行；禁止重复计算已有有效数据。
2. 根据现有代码给出最小实验矩阵：baseline、simple add/concat、global/cross-query LDP、PGC full、no-zero-self；另单列 preserve-void 和 final-only 的边界实验。
3. 除被消融机制外，保持数据/词表、配置、初始化规则、训练轮次、optimizer、batch、AMP、evaluator、checkpoint 策略一致；记录参数量、额外损失与训练成本。先做配置对比和 CPU/小样本 dry-run。
4. 检查 no-cross-query 是否真正成立；确认 paired-only 没有因为 mask 排序或 query 索引错位而产生伪配对。为每种实现写至少一项自动测试。
5. 可以在可用资源上启动获准的完整实验；资源不足时只生成可执行 YAML/命令/预计运行表，不得声称得到结果。
6. 主表以相同 50k final 为主；若要比较 best checkpoint，单列并明确相同的选择规则。将失败或无增益情况同样记录。
7. 分别判断：PGC 相对 baseline 有效性、paired-only 的必要性、zero-self 的贡献、preserve-void 的特定实验边界。不要把“相对 baseline 有效”直接等同于“每项设计都必要”。

输出：P03_pgc_ablation_plan.md、可运行配置/脚本、P03_pgc_ablation.md（若已运行，含原始日志路径）、汇总 CSV 和公平性/缺失证据清单。不要覆盖原始运行。
```

**验收标准**：至少能够在同一协议下比较 paired PGC 与一个有效的非 paired 融合对照；结果对应真实配置、日志和 checkpoint。无法完成的实验明确列为 pending。

---

## P04｜真正针对 PGC 的分类、定位与开放词汇诊断

**优先级**：P0。**目标**：定位 AP 增益的来源，避免把 global-LDP 的诊断误归因到 PGC。

**分析维度**：使用相同 evaluation protocol 和冻结的类别划分，分别评估 seen/unseen、in-/out-vocabulary、per-class AP、分类 top-1、分数排序与 geometric ensemble、mask recall/IoU、objectness/void 行为。在可能情况下，进行**固定同一组 mask、仅替换分类分数**的对照，以区分分类变化与 mask generation 变化。

**可直接复制给 Agent 的提示词**：

```text
请对已训练的 PGC-only（必要时包括 baseline、EVP-only、combined）执行机制诊断；必须明确与旧的 global-LDP 诊断不同，不能直接继承其结论。

请先从有效 WaterOVS 数据注册、训练词表、评测词表、aliases 和类别 ID 映射中严格定义 seen/unseen、in/out-vocabulary，并锁定映射文件。若这些划分不是同一概念，绝对不要混用。

任务：
1. 对相同图片和 checkpoint 导出预测 mask、query、base/PGC non-void logits、base void、pooled CLIP out-of-vocab scores、ensemble scores、最终类别预测；注意内存，提供分批保存和复现实验脚本。
2. 在同一 evaluator 下对 baseline 与 PGC-only 统计 overall/per-class/seen/unseen 的 AP，以及 top-1、置信度或排名指标；明确 GT matching 规则，避免只选匹配成功的样本造成选择偏差。
3. 诊断 mask generation 与 classification：比较 mask recall、GT-matched IoU / mask quality；在技术允许时固定同一预测 mask 只替换语义分数，检测 AP 变化。区分 GT-pooled CLIP oracle 与真实预测 pipeline，不得以 oracle 代表实际结果。
4. 分析 void/重复 query 的错误：false positives、未匹配预测、objectness/void 竞争变化。
5. 重点解释 PGC-only APs=10.6254、combined APs=9.0898 低于 baseline APs=14.3429 的现象：核对小目标 GT 数量、按类别/面积分布、checkpoint 波动和误差样例；不先假定原因。
6. 每个诊断结果必须指出它能支持什么、不能支持什么。若无法判断 PGC 的机制，明确列出缺失的中间输出或实验。

输出 P04_pgc_diagnostics.md、category_split.json、可复现实验脚本、per_class_metrics.csv、必要的诊断图。未经验证不得声称 PGC 广泛改善 unseen top-1 或 mask localization。
```

**验收标准**：形成 PGC-specific 而非仅 global-LDP 的结论；看到 overall AP 提升背后的类别与错误类型；对 seen/unseen 与 APs 的负面证据诚实解释。

---

## P05｜验证 EVP 的频域设计选择与作用范围

**优先级**：P1。**目标**：回答为什么用频域、为什么高频注入而低频作条件、为什么只在浅层，而不是仅写“参考 EVP”。

**最低限度消融**：baseline；原有 EVP-only；high-pass only/no low gate；high + low-conditioned gate；early stage 1/2 vs 更深 stage 或不同配置（保持其他参数一致）。建议另外做输入频率与 gate 激活的描述性可视化，避免将频域分量绝对解释为“纹理/布局”。

**可直接复制给 Agent 的提示词**：

```text
请审查并验证 frequency-conditioned EVP，而不是从模块名称倒推‘恢复水下细节’的结论。

阅读 evp_prompter.py、clip.py、@content_analysis.md、原始 Explicit Visual Prompting 方法说明以及已有 EVP 实验记录。请完成：
1. 先精确复核高通/低通 FFT mask、freq_rate、prompt pyramid、stage1/2 每 block 注入、high-pass 主残差、low-pass 仅作为 gate conditioning、gate bias=-2 及实际特征更新公式。
2. 设计 matched EVP-only 对照：无提示 baseline、high-pass-only（不引入 low condition）、high-pass + low-conditioned gate；再针对注入 stage 做最少但有解释力的对照。确保不是同时改变门控容量、训练 schedule 和频率窗口却把收益归因于 low frequency。
3. 检索既有运行，能复用则复用；新增实验先实现并运行单元测试与 dry-run；完整训练仅在资源具备时执行。
4. 评估总体 AP、APs/APm/APl、seen/unseen 及稳定性。如果需要可视化频谱/门控热图，必须基于真实输入和实际模型激活，不能使用假造示意作为实验结果。
5. 用‘实现事实—观察结果—潜在解释’三层分别回答：为什么频域、high/low 分工、为什么 early stages、与已有 EVP 的区别。不要直接将高频=全部细节、低频=全部空间布局当成已验证定律。

输出 P05_evp_ablation.md、实验矩阵、可运行命令/配置、频率和 gate 可视化（若有），以及可写入论文的保守机制陈述。明确指出何时只能称为已有 EVP 的适配，而非全新 prompting 范式。
```

**验收标准**：至少区分 low-conditioned gate 与 high-pass only 的独立效果；浅层设计有数据或可靠的消融依据；不能验证的部分标为解释假设。

---

## P06｜重复种子、checkpoint 敏感性与模块交互

**优先级**：P1。**目标**：确认目前单次实验是否稳定，并量化组合收益是否有可重复性。

**设计**：固定至少三个预注册 seeds，四路 `baseline / EVP / PGC / combined`，统一协议，报告每个 seed 的 final、predefined checkpoint selection 的 best（如有），以及均值、标准差与配对差异。可附描述性交互项：

\[
I=\mathrm{AP}_{\mathrm{PGC+EVP}}-\mathrm{AP}_{\mathrm{PGC}}-\mathrm{AP}_{\mathrm{EVP}}+\mathrm{AP}_{\mathrm{Base}}.
\]

该指标只是 AP 上的 difference-in-differences **描述量**，不代表因果协同已被证明，且必须在同 seed、同 checkpoint 规则下计算。现有 final 单种子数字给出 `I = -2.0035 AP`，因此尤其不能预设超加性互补。

**可直接复制给 Agent 的提示词**：

```text
请设计并在资源允许时运行 WaterOVS 主协议的多种子复现实验，重点检验 PGC 主收益和 EVP 叠加增益的稳定性。

请以 P00 台账核实后的 ConvNeXt-Large、BS9、FP32、50k、数据/类别、训练配置和代码快照为基准。预注册至少三个固定 seeds，四路 baseline、EVP-only、PGC-only、PGC+EVP 使用相同 seed 列表。

具体要求：
1. 优先查找可合法复用的同协议历史运行；如果 seed/代码/数据不完全匹配，不能当作新增 seed。
2. 生成配置差异检查、完整运行矩阵、启动与复评脚本、实验注册表；先跑 smoke test。GPU 资源不足时停止在可执行计划，不要模拟结果。
3. 以 final checkpoint 为主口径，另外在固定且提前定义的规则下统计 best；禁止用不同组的 best 与 final 直接比较。
4. 计算每个 seed 的 AP、APs/APm/APl、均值±标准差、PGC-baseline 差异、combined-PGC 差异和描述性交互项 I；强调 I 不是已证明的因果协同。
5. 分析 combined 相对 PGC 的增量是否稳定、是否受 checkpoint 选择驱动，并记录失败、异常和中断 run。

输出 P06_reproducibility.md、P06_run_matrix.csv、可运行脚本、统计汇总 CSV 及原始日志索引；根据实际结果给出可以/不可以使用 stable、consistent、complementary 等措辞的判断。
```

**验收标准**：不再仅依赖一次 seed 判断效果；可追溯 12 个 seed×方法格子的真实执行状态；明确哪些还没完成。

---

## P07｜补齐“为什么是水下”所需的证据

**优先级**：P1。**目标**：把 Motivation 从通用的 FC-CLIP 融合问题提升为有证据的水下识别问题，同时避免伪造水下退化标签或视觉恢复主张。

**关键限制**：缺乏退化强度标注时，不能声称模型在“高散射/强吸收”真实分组中表现更好。可做可复现的图像统计代理指标（如对比度/清晰度/颜色偏移），但需明确代理不等于真实成像物理量。合成退化属于鲁棒性压力测试，不可冒充真实领域分组。

**可直接复制给 Agent 的提示词**：

```text
请调查并验证本文在水下场景中的任务针对性，目标是给 Introduction 提供真实失败现象，而不是写泛泛的水下退化常识。

材料：WaterOVS 评测 split、已训练 baseline/EVP/PGC/combined 的同图预测、P04 的分类和 mask 诊断、已有水下相关文献（文献部分必须提供可核验来源）。

任务：
1. 清点数据集是否具有真实成像条件/水质/浑浊度/颜色退化标注；没有就明确说明。
2. 在同一 GT 与评测设置下分析类别相似性、尺寸、实例密度、低对比度、简单可计算的颜色/清晰度代理特征，说明代理指标的局限。避免把多因素相关性写成因果。
3. 按预定义采样规则选择：baseline 失败而 PGC 修正类别的案例、EVP 与 baseline 预测不同的案例、small-object 退化案例、模型仍失败的案例。提供图像 ID、GT、预测、分数、checkpoint，避免只展示成功样本。
4. 如果设计合成水下退化 stress test，独立标注为‘synthetic perturbation’，保证输入/输出和强度可复现，不能把结果推广为真实水下泛化。
5. 归纳哪些水下特殊困难能被实际证据支持，哪些只能由文献支持，哪些目前无法证明；分别匹配 EVP 与 PGC 的作用层级。

输出 P07_underwater_evidence.md、per_group_metrics.csv、真实案例清单与可复现可视化脚本；额外给出一份可用于 Motivation 的 6–10 句事实链，逐句标明证据类型。
```

**验收标准**：Introduction 所使用的“水下问题”至少具有文献或真实结果依据；失败案例和负面结果同样保留。

---

## P08｜修订 FathomNet 的评测协议与跨域证据

**优先级**：协议审计 P1；完整 matched baselines P2（资源允许时提前）。**目标**：防止用 retained test 的预测筛选偏差或无 baseline 的 full-test 指标宣称泛化优势。

**已知区别**：retained test 为 1,155 图、1,163 实例、41 类并经过 prediction filtering；prediction-free full test 为 8,988 图、13,960 annotations、173 prompt categories；两者**不是可直接比较的等价测试集**。已有 full-test 40k PGC+EVP 指标需复核：15.5106 AP。

**可直接复制给 Agent 的提示词**：

```text
请作为 FathomNet dataset/protocol audit agent，根据真实数据注册脚本、split 生成脚本、类别映射与 evaluator 日志检查现有 FathomNet v3 研究的有效性。

任务：
1. 审计 retained train/test、prediction-filtered retained test、prediction-free full test 及 WaterOVS transfer 的数据来源、图像重复、类目 ID/aliases、prompt vocabulary、缺失标注风险、small-object 分布和是否出现训练/测试泄漏。
2. 核对 40k full-test evaluator 原始 launcher.log，确认 15.5106 AP 对应的 checkpoint、类别数、图像数和是否为 prediction-free；更新技术报告中落后的 pending 状态，但不要覆盖原始日志。
3. 优先形成基于 GT-only 规则的训练/验证 split 与 checkpoint 选择协议；不得使用测试预测或测试 AP 来决定数据筛选。若需要新 split，保存清晰的生成脚本和 seed。
4. 制定同数据、同训练预算的 FathomNet baseline/EVP-only/PGC-only/combined 对照表，明确需要哪些 GPU 运行；若资源未准备，交付配置与运行命令即可。
5. retained、full、transfer 的数字分表报告。解释不同 checkpoint 最优的观察，但不要先断定退化全部由过拟合造成，也不要说 FathomNet 已证明方法优于 baseline。

输出 P08_fathomnet_protocol.md、类别/split 统计 CSV、检查脚本、P08_matched_run_plan.md、更新建议。强调缺失或无法直接比较的结果。
```

**验收标准**：full/retained/transfer 三类协议无混淆；未进行 baseline 对照前不主张泛化提升；存在代码可复现的 split 与 checkpoint 规则。

---

## P09｜实测效率，不再以模块 MAC 推断端到端速度

**优先级**：P2。**目标**：把“轻量/高效”的说法限定在已测指标，而非设计直觉。

**可直接复制给 Agent 的提示词**：

```text
请建立 FC-CLIP baseline、global LDP、PGC-only、EVP-only、PGC+EVP 的公平复杂度与端到端效率测量。

要求：
1. 先确认每个方法的源码、输入尺寸、backbone、device、precision、batch、评估/训练状态和预热规则完全记录。
2. 区分 total/trainable params、模块级理论 MAC/FLOPs 与模型端到端 MAC/FLOPs、推理 latency、throughput、peak GPU memory；说明计量工具对自定义 FFT/attention 算子的支持边界。
3. 用多次重复的相同输入集测量（固定 warm-up、同步 CUDA、统计 median/分布），不要混用不同设备或 AMP 设置；分别说明是否包含预/后处理及文本编码。
4. 比较 PGC/global-LDP 的理论模块参数量与实测整体变化。如资源不足，只保留已有模块级结果，删除未经测量的 fast/efficient/real-time 主张。

输出 P09_complexity.md、测量脚本、环境配置和原始计时/显存日志。每个数字有单位、硬件与统计口径。
```

**验收标准**：若论文使用 `efficient`，必须给出其确切含义和可复现数据；否则只讨论 module-level cost。

---

## P10｜相关工作与真正的 novelty 边界审查

**优先级**：P0。**目标**：避免将 PosSAM 的 LDP、原始 EVP 或 FC-CLIP 已有操作当作本文创新，也防止遗漏近年的水下 OVS 方法。

**可直接复制给 Agent 的提示词**：

```text
请担任 open-vocabulary segmentation / underwater computer vision 的严谨文献审查 Agent，为本论文开展‘可核验的 novelty audit’，而不是直接写一篇宣传式 Related Work。

必须覆盖原始 FC-CLIP、SAN、PosSAM、Explicit Visual Prompting，并检索 CAT-Seg、SED 及 2025–2026 年真实存在且相关的水下开放词汇/实例分割与区域特征融合、频域适配方法（包括 MARIS、已有水下相关路线；实际名称、年份和作者以论文来源核对为准）。

逐篇完成：
1. 核验标题、作者、年份、正式发表渠道/arXiv 版本、可靠 URL/DOI；读原始 method 而不是只看摘要。
2. 抽取其研究问题、输入表示、特征融合拓扑、是否 cross-query、CLIP 是否冻结、in/out classifier、void/no-object 处理、frequency prompting 位置、主要任务和测试协议。
3. 与本文 PGC 的 same-mask pairing、zero-self、final-only、preserve-void，以及 EVP 的 high-pass injection / low-frequency gate / stage 1–2 逐项比较，标注 SAME、RELATED、DIFFERENT、UNKNOWN。
4. 为每个候选创新写出：已有相似思想、真正实现区别、目前必要性证据、能否声称原创、审稿人可能提出的最近邻反驳。
5. 不得虚构论文或引用；未找到 full text 的明确写无法核查。论文间跨数据集 AP 不做直接性能排名。

输出 P10_novelty_matrix.md、verified_references.bib（只包含核实的条目）、P10_reviewer_objections.md。最后建议 2–3 个可防御的贡献表述，禁止直接推导为 SOTA 或首次提出。
```

**验收标准**：PGC/EVP 每个宣称的“新”都有最近邻对照及准确引用；保留不确定性。

---

## P11｜重构 Motivation 与 `writing_blueprint.md`（核心写作任务）

**优先级**：P0。**目标**：修复“从实现倒推研究缺口”和“水下任务不突出”的问题；**不能**只把当前七段式 Introduction 润色得更流畅。

建议先生成两条候选路线：

- **A｜区域语义修正主线**：FC-CLIP mask-level classification 中如何选择性利用同 mask 的 decoder 表征，PGC 为核心；EVP 为辅助。**优点**：最贴近当前 AP 归因。**风险**：领域针对性与机制创新论证可能较弱。
- **B｜视觉—区域两层表征增强主线**：水下成像下的视觉特征适配（EVP）与 mask-level 分类证据利用（PGC）分别处理两个层次的挑战。**优点**：更能解释双模块。**风险**：不能在缺少证据时声称恢复细节或强协同。

**当前倾向**：用 B 组织研究问题，用 A 控制实验证据和贡献权重；但最终选择应由 P03–P07、P10 的结果共同决定。

**可直接复制给 Agent 的提示词**：

```text
请以‘批判性学术写作规划 Agent’的角色，重构 @writing_blueprint.md 的 Motivation，而不是简单润色原文。

输入：@content_analysis.md、@writing_blueprint.md、已完成的 P00/P01/P04/P10 结果，以及可获得的 P03/P05/P06/P07 结果。未完成的任务视为 OPEN，不得预设实验成功。

请提供两套完整、互相竞争的论文论证方案：
A. PGC-centered query-aligned region semantic correction；
B. underwater visual-level adaptation + region-level semantic evidence enhancement。

对每套方案都输出：
1. 一句话研究问题，以及该问题为什么不是‘baseline 没有我们的模块’；
2. 任务动机 → 具体失败现象/观察 → 既有方法局限 → 设计原则 → PGC/EVP 方法响应 → 可验证预测；
3. 原始 FC-CLIP、PosSAM、EVP、近期 underwater OVS 文献在其中的准确位置；
4. 哪些句子由源码支持、由实验支持、由公开文献支持、仍仅是假设；
5. PGC 与 EVP 各自的重要性、未证明的协同、小目标 AP 下降和冻结 CLIP 并不等于完全保持对齐等潜在审稿问题；
6. Introduction 段落级写作大纲：每段逐句用 [1]、[2]、[3] 等表示句子承担的论证功能，不要直接写成冗长英文正文。

随后以证据充足度、创新性可防御性、水下针对性、双模块连贯性、实验成本五个维度比较 A/B，并给出带条件的推荐，不要求一开始就确定 PGC 为‘唯一一级贡献’。

最后生成 writing_blueprint 的独立修订版：重点修改原第 0、3、4、5、7 节，并同步更新 10–13 节 claim–evidence 与待补实验。保留准确的数据、技术术语和方法实现，避免推翻所有原文。不要把未验证机制写成既定科学事实。

输出：P11_motivation_route_A.md、P11_motivation_route_B.md、P11_route_comparison.md、P11_revised_blueprint.md、P11_diff_summary.md。
```

**验收标准**：新 Motivation 先解释可观察问题，再引出设计；区别“论文总问题”和“增益最大的模块”；所有关键句可追溯证据或明确为待验证假设。

---

## P12｜论文写作交接：统一主张、结果与章节间逻辑

**优先级**：P0（建议在 P11 后执行，后续新增实验后重复一次）。**目标**：形成能交给 abstract/introduction/method/results writer 的稳定输入，避免各 Skill 分别夸大贡献。

**可直接复制给 Agent 的提示词**：

```text
请作为论文内部审稿与一致性检查 Agent，检查修订后的写作蓝图、技术分析、实验 registry、Mechanism/FathomNet/EVP 报告及当前稿件（如有）。

要求：
1. 建立 claim-evidence matrix：每个主要陈述对应 source、experiment/run、checkpoint、protocol、support level、建议措辞、适合出现的章节。
2. 核查 Methods 的所有关键技术事实：S_i/C_i provenance、hard detached pooling、zero-self、LayerNorm、final-only non-void、void、geometric ensemble、high-pass/low-pass 的真实作用、stage1/2、frozen/trainable 区别。
3. 核查所有数值：baseline/EVP/PGC/combined，best 与 final，单种子与多种子，FathomNet retained/full/transfer；不能混用 global-LDP 的诊断结论和 PGC 的结论。
4. 审查 Abstract、Introduction、Method、Experiments、Discussion、Conclusion 的主张强度是否一致。特别删除或降级未经证明的 unseen improvement、mask localization improvement、underwater detail restoration、strong synergy、SOTA、efficient、robust generalization。
5. 每一个 OPEN 的核心主张要有对应实验任务或被从正文移除。若尚未完成训练，不要把计划中的结果写成已验证结果。
6. 生成最终写作交接包：推荐路线与替代路线的取舍、稳定术语表、最终章节大纲、可直接引用的证据表、待完成风险清单和 reviewer objections。

输出：P12_claim_evidence_audit.md、P12_writing_handoff.md、P12_open_issues.md、P12_paper_consistency_checklist.md。只对已证实内容给出确定措辞，不改写或伪造原始测量。
```

**验收标准**：同一论文主张在不同章节不会升级或矛盾；任何数值均有完整来源；所有待完成实验在写作交接包里保留待验证标记。

---

## 13. 推荐交付目录

```text
paper_revision_agent/
├── P00_evidence_registry.md
├── P00_evidence_registry.csv
├── P00_discrepancies.md
├── P01_method_semantics.md
├── P02_test_audit.md
├── P03_pgc_ablation_plan.md
├── P03_pgc_ablation.md
├── P04_pgc_diagnostics.md
├── P05_evp_ablation.md
├── P06_reproducibility.md
├── P07_underwater_evidence.md
├── P08_fathomnet_protocol.md
├── P09_complexity.md
├── P10_novelty_matrix.md
├── verified_references.bib
├── P11_motivation_route_A.md
├── P11_motivation_route_B.md
├── P11_route_comparison.md
├── P11_revised_blueprint.md
├── P11_diff_summary.md
├── P12_claim_evidence_audit.md
├── P12_writing_handoff.md
├── P12_open_issues.md
├── P12_paper_consistency_checklist.md
├── configs/                # 新实验有效配置；不覆盖历史文件
├── scripts/                # 统计、评估、可视化、启动脚本
├── figures/                # 仅真实实验生成的图
└── tables/                 # CSV/Markdown 表
```

以上为建议命名；实际执行时可以按 Agent 工具链调整，**关键是交付可追溯、版本明确**。

## 14. 分阶段验收关口（避免无效训练）

### Gate A：允许开展机制实验之前

- [ ] P00 已确认四路主消融的有效协议与未核实项。
- [ ] P01 已厘清 PGC/EVP 的真实实现，避免为错误的架构理解设计消融。
- [ ] P02 核心测试通过，拟新增对照也有 shape/gradient/smoke test。
- [ ] 新实验配置差异表已检查，估算资源并明确 checkpoint/reporting 规则。

### Gate B：允许锁定论文主线之前

- [ ] P04 的 PGC-specific 诊断能解释至少一部分分类/定位收益。
- [ ] P10 明确 PosSAM/EVP/FC-CLIP 的已知方法来源和 novelty 边界。
- [ ] P03 与 P05 的关键消融，已完成或被清楚列为待验证。
- [ ] P07 至少获得一组真实水下预测证据，或删去无法实证的任务独特性断言。
- [ ] 不将强协同、unseen improvement、细节恢复、全面泛化当作既定结果。

### Gate C：允许形成较稳定的论文初稿之前

- [ ] P11 有两条经过比较的论证方案，推荐主线与证据相符。
- [ ] P12 对照原始 evaluator 验证所有关键数字。
- [ ] best/final、single/multi-seed、WaterOVS/FathomNet 协议在所有章节口径一致。
- [ ] 每个 Contributions 关键条目都有方法区别、支持证据及限制。

## 15. 两种实际执行策略

### 快速打牢论文起点（尽量不等待大规模 GPU）

执行：**P00 → P01 → P04（已有 checkpoint 可完成部分）→ P10 → P11 → P12**。

这个顺序优先解决：**研究问题定义是否准确、PGC 解释是否过度、水下故事是否有证据、与现有工作的区别是否成立**。新增训练结论保留 `PENDING`，但可以开始更可靠的 Introduction/Method 初稿。

### 完整投稿前证据建设

在上述基础上完成：**P02 → P03 → P05 → P06 → P07 → P08 → P09**，并回写 P11/P12 的观点和证据表。

新增结果无论正负，都应决定论文保留什么主张；不要因为原先想写“双模块协同”而忽略组合增量有限或 APs 下降的结果。

---

## 16. 一键任务入口：让 Agent 先拆解而不直接盲目训练

如果不想一次性粘贴 13 个提示词，可以**先发送下面这段总控提示词**，让 Agent 规划、建立台账，再逐项执行本文件中的专用提示词。

```text
你是 FC-CLIP 水下开放词汇实例分割论文项目的研究执行 Agent。请读取 @content_analysis.md、@writing_blueprint.md 和《水下开放词汇实例分割论文：修订与验证执行计划（含 Agent 提示词）》。

目标不是立即写完一篇论文，而是逐步完成技术事实审计、设计必要性验证、开放词汇分类诊断、频域设计消融、水下任务证据补全、novelty audit 和 Motivation 重构。

请先执行 P00，输出真实的证据/协议 registry。根据 P00 结果核实 P01–P12 的输入是否具备，按 P0→P1→P2 给出依赖 DAG、可立即完成的任务、需要 GPU 的任务、需作者介入的事项和各项最小交付物。每项任务使用该计划附带的专用提示词和验收标准。

通用规则：严禁虚构结果；不破坏用户未提交代码；未完成 GPU 训练不得写为完成；所有论文主张严格标记 CODE-FACT / MEASURED / INFERENCE / OPEN；对失败或负结果照实报告。P11 可先生成候选 Motivation，但没有被实验验证的机制必须标注为待验证，不应倒推结论。

先交付 P00 和任务执行台账，再按依赖开始无需 GPU 的核查。默认输出目录 /data/gjf/files/exp/paper_revision_agent，路径不存在时先报告并使用可写的工作区目录。
```

---

## 17. 最终研究判断的决策规则

| 如果后续验证发现…… | 论文应如何调整 |
|---|---|
| PGC 相对 baseline 稳定获益，但与 global/simple fusion 接近 | 降低“paired topology 必要性”主张，改写为有效的 FC-CLIP region fusion 实现；重新评估 novelty |
| PGC 的优势主要集中在 seen/in-vocabulary 分类 | 聚焦 mask-wise classification/score calibration，而非宣称普遍的 unseen semantic improvement |
| PGC improves AP 但 small-object AP 持续下降 | 把局限明确写入 Results/Discussion，避免“增强细节/轮廓”作为 PGC 主张 |
| EVP-only 改善显著，但 combined−PGC 不稳定 | EVP 是独立可用适配机制，但双模块强互补不成立；论文主结果以 PGC 为锚点 |
| Low-frequency gate 未优于 plain high-pass | 将门控定位为探索设计或简化方法，不再作为核心技术贡献 |
| 水下退化分组无证据或指标不稳定 | 保留水下任务背景，不把特定物理退化写成已证实的核心失败原因 |
| FathomNet matched baseline 显示无增益 | 如实呈现负面迁移结果或将其保留为局限性实验，不宣称跨域性能优势 |
| PGC 与已有文献非常相似 | 必须重新界定精确方法区别或改变论文贡献重心，不能依靠换名制造 novelty |

**最终目标**不是完成尽可能多的消融，而是让 **Problem → Observation → Design → Evidence → Claim** 每个环节能够追溯并相互一致。

---

## 来源与现有材料（供 Agent 定位）

1. `content_analysis.md`：3.1–3.5 节方法还原；4 节主实验、global-LDP 诊断与 FathomNet；5–6 节主张边界；7 节写作交接。
2. `writing_blueprint.md`：0 节贡献定位；3–5 节 Central Contribution/Motivation；7 节各章节；10–13 节 claim–evidence、统一性与证据门槛。
3. 本计划的路线 A/B 和任务优先级属于**研究组织建议**；只有经过原始实现、真实实验与可核验文献支持的部分，才能成为论文中的确定结论。
