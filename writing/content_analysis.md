# FC-CLIP 水下开放词汇实例分割内容分析

## Metadata

- Project name: FC-CLIP + PGC-LDP + frequency-conditioned EVP for underwater open-vocabulary instance segmentation
- Project root: `/data/gjf/code/fc-clip-amp`
- Clean implementation reference: `/data/gjf/output/fc-clip` (`v1`, commit `c5ff07d9beab32c627eb8ded7c4f312b7c1e4517`)
- Development state: `feature/fcclip-global-amp-text`, commit `ce5f8656e55664481aff6d4c40b3d9dafba6aa7a`, with substantial uncommitted user work
- Analysis date: 2026-10-15
- User guidance: current code and raw runs are under the project root; prioritize the clean implementation and `/data/gjf/files/exp`
- Intended paper topic: inferred as adapting FC-CLIP to underwater open-vocabulary instance segmentation through query-aligned region fusion and frequency-conditioned visual prompting; the user has not yet fixed the paper's central claim
- Target task/datasets: WaterOVS ID/CD instance segmentation; FathomNet taxonomy-cleaned visual-common v3 training, retained-test adaptation, and transfer to WaterOVS val
- Primary framework: PyTorch, Detectron2/Mask2Former, OpenCLIP ConvNeXt
- Trust notes:
  1. Highest trust: evaluator-backed raw `log.txt`/`eval.log`, saved `config.yaml`, and machine-readable `metrics.json` tied to a checkpoint.
  2. High trust: clean implementation files and run-local configuration snapshots.
  3. Moderate trust: curated CSV/Markdown summaries when their numbers agree with raw artifacts.
  4. Low trust: directory names, stale `实验记录.md` status fields, or configuration names without evaluator output.
  5. The FathomNet summary dated 2026-10-14 is stale for WaterOVS transfer: raw logs now contain all checkpoints and differ from the earlier partial table. Section 4 records the current raw values.

## 1. Executive Technical Summary

- The project extends frozen-convolutional-CLIP FC-CLIP with two trainable paths: final-query Paired Gated Correction LDP (PGC-LDP) and stage-1/2 frequency-conditioned EVP prompting.
- PGC-LDP is the best-supported technical center. It pairs each decoder structure token only with the CLIP token pooled from the same predicted mask, applies a per-head keep-or-correct gate, replaces only final non-void logits, and preserves FC-CLIP's original void/objectness logit.
- EVP injects a gated high-pass prompt before each enabled ConvNeXt block. A low-pass prompt conditions the spatial gate but is not itself written into the backbone. The frozen CLIP weights remain frozen; prompt, adaptor, and gate parameters are trainable.
- The strongest controlled WaterOVS evidence is the ConvNeXt-Large, BS9, FP32, seed-42 group: baseline final `47.5934 AP`; EVP-only best/final `50.6923/49.7202`; PGC-LDP-only final `53.9612`; PGC-LDP+EVP best/final `54.4971/54.0845`.
- In that matched group, PGC-LDP explains most of the gain: `+6.3678 AP` over baseline at final. EVP adds only `+0.5359 AP` when comparing the combination's best checkpoint with PGC-only final, and `+0.1233 AP` at final. Strong PGC–EVP synergy is therefore not demonstrated.
- The global-LDP diagnostic indicates that improvement is mainly associated with the in-vocabulary/seen classification and score-fusion path, not better mask matching or a stronger frozen-CLIP region oracle. Claims of broadly improved unseen-region recognition should be avoided.
- FathomNet v3 provides convergence and failure-mode evidence, not method-superiority evidence: there is no matched FathomNet baseline or module ablation. Retained-test AP peaks at `38.9187` at 40k and falls to `32.9940` at 50k; WaterOVS transfer peaks at `31.5323` at 20k and ends at `30.3092`.
- The largest uncertainties are single-seed WaterOVS results, incomplete efficiency measurement, a biased retained test, a large full-versus-retained protocol gap, and absence of a literature audit establishing novelty.

## 2. File And Artifact Map

| Role | Path | Why it matters | Confidence |
|---|---|---|---|
| Baseline/model integration | `/data/gjf/output/fc-clip/fcclip/fcclip.py` | Builds LDP/PGC, pools final predicted masks over dense CLIP features, replaces final non-void logits, and performs open-vocabulary geometric ensembling | High |
| PGC/global LDP core | `/data/gjf/output/fc-clip/fcclip/modeling/local_discriminative_pooling.py` | Authoritative paired/global topology, explicit zero self value, preserve-void combiner, validation | High |
| Decoder structure token | `/data/gjf/output/fc-clip/fcclip/modeling/transformer_decoder/fcclip_transformer_decoder.py` | Produces mask-aligned `pred_region_features`; auxiliary layers remain on the base classifier | High |
| EVP core | `/data/gjf/output/fc-clip/fcclip/modeling/backbone/evp_prompter.py` | FFT high/low prompts, prompt pyramid, block adaptors, low-frequency-conditioned spatial gate | High |
| EVP backbone integration | `/data/gjf/output/fc-clip/fcclip/modeling/backbone/clip.py` | Freezes CLIP, constructs EVP after freezing, and injects residuals before enabled ConvNeXt blocks | High |
| Model defaults | `/data/gjf/output/fc-clip/fcclip/config.py` | Defines LDP/PGC and EVP configuration surface | High |
| WaterOVS registration | `/data/gjf/output/fc-clip/fcclip/data/datasets/register_waterovs_instance.py` | Registers 84-class train and 115-class val vocabularies from MARIS definitions | High |
| Clean method config | `/data/gjf/output/fc-clip/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_base_waterovs_id_ldp_pgc_evp_high_low_router_stage12_gated_add_gb2_full.yaml` | Minimal clean PGC-LDP+EVP activation; useful for implementation reconstruction, not the authoritative config for every reported run | High |
| FathomNet v3 registration | `/data/gjf/code/fc-clip-amp/fcclip/data/datasets/register_fathomnet_train_ovs_v42_current_retained_taxonomy_cleaned_visual_common_v3_instance.py` | Enforces 67 annotation/prompt categories and exact ID agreement | High |
| FathomNet mask pipeline | `/data/gjf/code/fc-clip-amp/fcclip/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py` | Supports RLE/BitMasks and converts masks to dense tensors after augmentation | High |
| Empty-target handling | `/data/gjf/code/fc-clip-amp/fcclip/modeling/matcher.py` | Returns empty assignments after LSJ removes all instances from an image | High |
| FathomNet training config | `/data/gjf/code/fc-clip-amp/configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_fathomnet_current_retained_visual_common_v3_id_pgc_ldp_evp_low_conditioned_highpass_stage12_gated_add_gb2_full_bs15.yaml` | Connects the method to the 67-class FathomNet v3 dataset and bitmask path | High |
| Consolidated WaterOVS table | `/data/gjf/files/exp/FC-CLIP_all_experiment_results.csv` | Index of full/checkpoint/smoke runs and artifact locations | Moderate–high |
| Matched Large/BS9 ablation | `/data/gjf/files/exp/FC-CLIP_BS9_noAMP核心模块对比简表_2026-08-21.md` | Best concise attribution table for baseline, EVP-only, PGC-only, and combination | High after raw-log verification |
| Broad experiment history | `/data/gjf/files/exp/FC-CLIP整体实验文档.md` | Negative results, protocol differences, invalid-run exclusions, and checkpoint trajectories | Moderate–high |
| Classification diagnostic | `/data/gjf/files/exp/LDP_preserve_void_noamp_3gpu_bs15_分类定位分析.md` | Separates classification/fusion effects from mask localization and frozen-CLIP oracle behavior | Moderate; analyzes global LDP rather than PGC directly |
| WaterOVS matched raw runs | `/data/gjf/code/fc-clip-amp/output/new_fcclip_convnext_large_waterovs_id_*` and `/data/gjf/code/fc-clip-amp/output/fcclip_convnext_large_waterovs_id_pgc_ldp_evp_*_bs9` | Saved effective configs, logs, metrics, and final evaluator outputs for the central ablation | High |
| FathomNet retained curve | `/data/gjf/code/fc-clip-amp/output/eval_retained_test_fathomnet_current_retained_visual_common_v3_id_pgc_ldp_evp/checkpoints` | Complete 5k-step checkpoint curve on the biased retained test | High for within-run comparison |
| FathomNet-to-WaterOVS curve | `/data/gjf/code/fc-clip-amp/output/eval_waterovs_val_fathomnet_current_retained_visual_common_v3_id_pgc_ldp_evp_all_checkpoints/checkpoints` | Complete cross-dataset checkpoint curve; newer than the curated partial summary | High |
| FathomNet data audit | `/data/gjf/files/exp/FathomNet_v3_dataset_problems_and_improvements_2026-10-14.md` | Class imbalance, missing-annotation risk, test-selection bias, vocabulary coverage, and recommended protocol | Moderate–high; some conclusions are evidence-backed in the document but source dataset-generation scripts were not fully re-audited here |
| Targeted tests | `/data/gjf/code/fc-clip-amp/test/test_local_discriminative_pooling.py`, `/data/gjf/code/fc-clip-amp/test/test_evp_frequency_router.py` | Validate paired values, gradients, parameter count, preserve-void behavior, and selected router semantics | High for tested invariants |

## 3. Core Method Reconstruction

### 3.1 Baseline

FC-CLIP uses a single frozen convolutional CLIP backbone to provide multi-scale visual features and a dense CLIP map. A Mask2Former-style pixel decoder and transformer decoder produce 250 masks and query features. The baseline final region feature is the sum of a mask-pooled pixel-decoder feature and the normalized decoder query, projected to CLIP space and classified against prompt-derived text embeddings.

At inference, FC-CLIP also hard-pools the dense CLIP feature under each predicted mask to obtain an out-of-vocabulary classifier. In-vocabulary and out-of-vocabulary probabilities are geometrically combined with separate weights for train-vocabulary-overlapping and non-overlapping categories. The void probability remains a separate final factor.

### 3.2 PGC-LDP

Inputs for query `i` are:

- `S_i`: the final decoder's 256-dimensional mask-aligned structure feature;
- `C_i`: the CLIP region token obtained by hard, detached pooling of `clip_vis_dense` using the same final predicted mask and applying the frozen CLIP visual projection.

The clean implementation maps both to common dimension `D` and computes, per query and per head:

```text
q_i = W_q C_i
k_i = [W_k C_i, W_k S_i]
v_i = [0, W_v S_i]
a_i = softmax(q_i k_i^T / sqrt(d_head))
correction_i = a_i[structure] * W_v S_i
output_i = LayerNorm(C_i + W_o correction_i)
```

The self candidate has an explicit zero value created after projection, so selecting it means no structure correction is written into the residual. The output is not a strict identity copy of raw CLIP: `clip_projection` is trainable and the residual passes through LayerNorm.

Pairing is semantically grounded rather than based only on equal tensor lengths: `S_i` comes from decoder query `i`, and `C_i` is pooled with that query's own predicted mask. Unlike global LDP, query `i` cannot read structure token `j`.

Only the final decoder layer uses PGC-LDP. Its non-void logits replace the base final non-void logits, while the original base void logit is concatenated back. Auxiliary decoder classifiers, auxiliary CE losses, mask losses, and dice losses remain on the FC-CLIP path.

For the ConvNeXt-Base setting (`D=640`), the tested implementation has 2,216,960 parameters versus 2,626,560 for global LDP. The design document's analytic core-MAC estimates are 0.65600G versus 0.73536G. These are module-level calculations, not measured end-to-end latency or memory.

### 3.3 Frequency-conditioned EVP

The selected main path uses:

```yaml
EVP_INPUT_TYPE: fft
EVP_PROMPT_TYPE: highpass
EVP_FREQ_RATE: 0.25
EVP_TUNING_STAGE: 12
EVP_EMBEDDING_TUNE: true
EVP_FUSION_TYPE: gated_add
EVP_GATE_BIAS: -2.0
EVP_FREQUENCY_ROUTER: low_conditioned_gate
```

For each input image, the code creates high-pass and low-pass FFT responses with a centered frequency mask. The high-pass response is the main handcrafted prompt; the low-pass response is a conditioning signal. Both pass through the same hierarchical prompt pyramid. The high branch is added to a stage-derived embedding prompt and adapted to the ConvNeXt stage channel dimension; the low branch is adapted separately with the same block adaptor.

For current feature `F`, high residual `Delta_H`, and low condition `C_L`, the spatial gate concatenates `F`, `Delta_H`, `C_L`, and their channelwise L2 norms. A `1x1 -> GELU -> depthwise 3x3 -> GELU -> 1x1 -> sigmoid` network produces one gate value per location:

```text
G = sigmoid(g(F, Delta_H, C_L, ||Delta_H||_2, ||C_L||_2))
F_next = ConvNeXtBlock(F + G * Delta_H)
```

The low-frequency branch affects the gate only and is not directly added to the feature. With output weight zero and bias `-2`, the initial gate is `sigmoid(-2) ~= 0.1192`, which is conservative but not an exact identity initialization. Stage `12` means ConvNeXt stages 1 and 2 (`res2/res3`), with prompt injection before every block in those stages.

The implementation describes itself as “SegFormer-EVP style.” Its literature novelty relative to EVP and other frequency prompting methods has not been established by this repository analysis.

### 3.4 Combined data flow

```text
image
  -> frozen ConvNeXt CLIP + trainable stage-1/2 EVP residuals
  -> res2/res3/res4/res5 + dense CLIP feature
  -> pixel decoder + transformer decoder
  -> final masks + final structure tokens
  -> hard detached mask pooling on dense CLIP feature
  -> paired PGC correction of each CLIP region token
  -> PGC non-void logits + base void logit
  -> geometric ensemble with original pooled-CLIP out-of-vocabulary scores
  -> instance predictions
```

EVP changes upstream visual features and therefore can affect masks, decoder representations, and dense features. PGC-LDP acts downstream on final mask-wise classification. The main configuration introduces no additional PGC/EVP-specific loss; it uses the existing classification, mask, and dice objectives.

### 3.5 Training and data procedures

- WaterOVS registers 84 training categories and 115 validation categories. The central matched ablation uses ConvNeXt-Large, three GPUs, global batch 9, FP32 (`AMP=false`), 50k iterations, base LR `1e-4`, seed 42, and COCO-style segmentation AP.
- FathomNet v3 uses 8,661 images, 8,675 retained instances, and 67 classes; ConvNeXt-Large, three GPUs, batch 15, FP32, 50k iterations, LR `1e-4`, checkpoint every 5k, and seed `-1`.
- FathomNet COCO RLE masks are read as `BitMasks`, filtered after LSJ augmentation, and converted to dense tensors. Empty post-augmentation targets are handled explicitly in the matcher.
- The FathomNet run trains and self-evaluates on the same retained training split; its 90.2243 final AP is a fit diagnostic only.

## 4. Experiment And Result Map

### 4.1 Central matched WaterOVS ID ablation

All rows use ConvNeXt-Large, BS9, FP32, seed 42, the same WaterOVS evaluator, and saved effective configs in `fc-clip-amp`. Results are single-seed.

| Experiment | Checkpoint | AP | AP50 | AP75 | APs | APm | APl | Comparison | Reliability |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| FC-CLIP noAMP baseline | 50k | 47.5934 | 55.0646 | 50.9703 | 14.3429 | 25.5046 | 48.8541 | Reference | High for this run |
| EVP-only | 40k best | 50.6923 | 58.0966 | 54.3649 | 16.3899 | 27.7329 | 52.1091 | `+3.0989 AP` vs baseline final | High; checkpoint-selected |
| EVP-only | 50k final | 49.7202 | 57.3223 | 52.9450 | 26.2014 | 30.7396 | 50.6668 | `+2.1268 AP` | High |
| PGC-LDP-only | 40k | 53.4909 | 62.0613 | 57.0710 | 6.6346 | 33.5645 | 54.6437 | `+5.8975 AP` | High |
| PGC-LDP-only | 50k final | 53.9612 | 62.3810 | 57.8562 | 10.6254 | 32.5768 | 55.0848 | `+6.3678 AP` | High |
| PGC-LDP + EVP | 40k best | 54.4971 | 63.0998 | 58.1406 | 7.6982 | 33.5863 | 55.6528 | `+6.9037 AP`; `+0.5359` vs PGC final | High; checkpoint-selected |
| PGC-LDP + EVP | 50k final | 54.0845 | 62.7847 | 57.9002 | 9.0898 | 31.4751 | 55.1517 | `+6.4911 AP`; `+0.1233` vs PGC final | High |

Measured interpretation: PGC-LDP is the dominant source of the Large/BS9 gain. EVP-only is useful, but its incremental contribution on top of PGC is small and checkpoint-dependent.

### 4.2 Additional ablations and diagnostics

| Experiment | Dataset/protocol | Key result | Evidence and interpretation | Reliability |
|---|---|---|---|---|
| LDP controls both foreground and void | WaterOVS ID, ConvNeXt-B, 5k | 1.2070 AP | Replacing void/objectness with LDP collapses performance | Strong within tested 5k pair |
| LDP preserve-void | Same 5k setting | 42.4391 AP | Preserving the FC-CLIP void logit is a necessary implementation boundary | Strong within tested 5k pair |
| Global LDP preserve-void | WaterOVS ID, ConvNeXt-B, BS15, final | 49.0105 AP vs historical base 44.8520 | Positive, but baseline differs in AMP/batch protocol | Moderate |
| Global LDP classification diagnostic | WaterOVS ID, 10,439 images / 11,437 GT | In-vocab top-1 rises from 12.851% to 27.164% at final; mean matched mask IoU falls from 0.919855 to 0.908056 | Supports a classification/fusion explanation rather than better masks | Moderate; global LDP, not PGC |
| Seen/unseen diagnostic | Same | seen ensemble top-1 `+10.164pp`; unseen `-0.556pp`; unseen macro AP still rises 42.829 -> 46.327 | Unseen AP change likely includes ranking/fusion effects, not broad top-1 recognition gains | Moderate |
| ConvNeXt-B PGC + band-amplitude EVP | WaterOVS ID, BS15, FP32 | 50.7442 AP final | Strong single run, but PGC-only was resumed and EVP-only used AMP; not a clean attribution set | Moderate |
| WaterOVS CD high-low EVP | CD, ConvNeXt-B | best 30k 41.1382; final 35.3896; baseline best 20k 38.1967 and final 35.4915 | Mid-training gain does not survive to final; open-vocabulary generalization is checkpoint-sensitive | Moderate |
| Text-adapter branches | WaterOVS ID/CD | Best ID variants remain below the main PGC path; several CD finals degrade | Useful negative evidence; should not be a central paper contribution | Moderate |

### 4.3 FathomNet v3

Retained test contains 41 classes, 1,155 images, and 1,163 instances. It was filtered using same-class Pred-IoU50 and has no COCO-small instances, so it is appropriate for within-run checkpoint comparison but not an unbiased benchmark.

| Evaluation | Checkpoint | AP | AP50 | AP75 | APs | APm | APl | Reliability |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Retained train self-eval | 50k | 90.2243 | 96.6704 | 94.9216 | 78.8459 | 87.4809 | 91.7792 | High as fit evidence only |
| Retained test | 35k | 38.8720 | **51.2392** | 41.1804 | NaN | **27.7919** | **42.3310** | High within biased test |
| Retained test | 40k | **38.9187** | 49.7430 | **41.4181** | NaN | 27.4378 | 42.1677 | High within biased test |
| Retained test | 50k/final | 32.9940 | 41.7445 | 34.6641 | NaN | 19.5228 | 36.6353 | High; 5.9247 AP below 40k |
| WaterOVS transfer | 15k | 31.5102 | **39.9591** | 33.4404 | 0.0000 | 8.9805 | 32.3261 | High, raw log |
| WaterOVS transfer | 20k | **31.5323** | 39.4981 | **33.6860** | 0.0000 | 9.8747 | **32.3871** | High, raw log |
| WaterOVS transfer | 50k/final | 30.3092 | 37.7128 | 32.4291 | 0.0000 | 8.0019 | 31.5192 | High, raw log |
| Prediction-free FathomNet full test | 40k model, 173 classes, 8,988 images | 15.5106 | AP50 17.8786; AP75 16.1916; APs 3.7614; APm 11.5631; APl 20.3655 | High, raw evaluator log |

The complete current WaterOVS-transfer AP curve is `26.2392, 28.6377, 31.5102, 31.5323, 30.0341, 29.0098, 29.6378, 30.2672, 30.2518, 30.3092` at 5k through 50k. This supersedes the partial and slightly mismatched values in `/data/gjf/files/exp/FathomNet_v3_PGC_LDP_EVP_checkpoint_experiment_2026-10-14.md`.

Measured interpretation: retained-test selection favors 35k/40k, while independent WaterOVS transfer favors 15k/20k. Final training fit is very high, but both external criteria are worse at 50k. This supports checkpoint/domain specialization, not a quantitative claim that all degradation is conventional overfitting.

### 4.4 Validation status

Command:

```text
/data/gjf/env/miniconda3/envs/MARIS/bin/python -m pytest -q \
  test/test_local_discriminative_pooling.py test/test_evp_frequency_router.py
```

Result: 33 tests passed and 3 failed. All failures concern the exploratory `amplitude_selected_bandpass` path: the tests expect `AmplitudeSelectedBandPrompt` and constructor arguments absent from the current development implementation. A follow-up run with `-k 'not amplitude_selected'` passed (`33 passed, 3 deselected`). The selected `low_conditioned_gate` path and PGC/LDP invariants are green, but the repository-wide exploratory EVP test/implementation mismatch remains an open issue.

## 5. Contribution Candidates

### 5.1 Query-aligned Paired Gated Correction LDP

- Contribution: A final-query fusion topology that preserves a mask's CLIP semantic anchor while allowing only its paired decoder structure token to write a gated correction.
- Implementation evidence: `local_discriminative_pooling.py`, `fcclip.py`, and final `pred_region_features` from the transformer decoder.
- Experimental evidence: matched Large/BS9 PGC-only final `53.9612 AP` versus `47.5934` baseline (`+6.3678`).
- Baseline or ablation: baseline, EVP-only, and combination are matched; preserve-void has a strong 5k boundary ablation. A matched Large global-LDP control is missing.
- Claim strength: Supported. The single-seed gain is clear; mechanism superiority over every alternative topology and literature novelty are not yet demonstrated.
- Best paper section: central Methods contribution and primary WaterOVS Results ablation.
- Safer wording: “We develop a query-aligned paired correction module that improves WaterOVS ID AP by 6.37 points in a matched single-seed ConvNeXt-Large/BS9 experiment.”
- Missing evidence: multi-seed confidence, matched global-LDP control, end-to-end cost, and external novelty comparison.

### 5.2 Preserve-void semantic division

- Contribution: LDP/PGC handles foreground semantic correction while FC-CLIP retains no-object competition.
- Implementation evidence: `combine_ldp_non_void_logits()` always concatenates the base void logit.
- Experimental evidence: 5k foreground+void replacement `1.2070 AP` versus preserve-void `42.4391 AP`.
- Baseline or ablation: direct early-training pair.
- Claim strength: Demonstrated for the tested configuration as a critical design boundary, not necessarily a general theorem.
- Best paper section: Methods design choice and ablation.
- Safer wording: “Preserving the original no-object logit is essential in our implementation; replacing it with LDP logits caused a 41.23-point AP collapse at 5k.”
- Missing evidence: repeated seeds and a final-training version of the destructive control, if reviewers require it.

### 5.3 Low-frequency-conditioned high-pass visual prompting

- Contribution: A stage-1/2 ConvNeXt prompting path in which high-frequency residuals determine what is injected and low-frequency content conditions where it is injected.
- Implementation evidence: `evp_prompter.py` and `clip.py`; selected router tests pass.
- Experimental evidence: matched Large/BS9 EVP-only best/final improves baseline by `+3.0989/+2.1268 AP`.
- Baseline or ablation: baseline and EVP-only are matched; plain high-pass versus low-conditioned controls exist mainly in other protocol groups.
- Claim strength: Supported as an effective adaptation. Novelty is unverified, and the code explicitly describes the generator as SegFormer-EVP style.
- Best paper section: secondary Methods component and ablation.
- Safer wording: “We adapt frequency-conditioned visual prompting to frozen ConvNeXt CLIP and observe a 2.13-point final AP improvement under the matched protocol.”
- Missing evidence: literature comparison, multi-seed repeat, and a matched plain-EVP/router ablation on the main Large protocol.

### 5.4 Combined PGC-LDP + EVP configuration

- Contribution: An upstream visual adaptation plus downstream mask-wise semantic correction configuration yielding the highest observed WaterOVS checkpoint.
- Implementation evidence: saved configs enable both selected modules.
- Experimental evidence: 54.4971 AP at 40k and 54.0845 at final.
- Baseline or ablation: matched baseline, EVP-only, and PGC-only.
- Claim strength: Supported as the highest observed single run; only Suggested as evidence of module synergy.
- Best paper section: main result table, with PGC-only attribution immediately adjacent.
- Safer wording: “The combination reaches the highest single-checkpoint AP, while most of the improvement is already obtained by PGC-LDP.”
- Missing evidence: repeated seeds and a stable incremental gain over PGC-only.

### 5.5 FathomNet v3 adaptation and evidence audit

- Contribution: A strict 67-class registration/RLE training pipeline plus a checkpoint and dataset audit exposing split bias, class imbalance, missing-annotation risk, and domain-specific checkpoint selection.
- Implementation evidence: FathomNet registration, mapper, matcher, and training/evaluation configs.
- Experimental evidence: train `90.2243 AP`, retained-test peak `38.9187`, WaterOVS-transfer peak `31.5323`, and later degradation on both external criteria.
- Baseline or ablation: no matched FathomNet method baseline; evidence concerns behavior of one combined model.
- Claim strength: Implemented for the pipeline and Supported for checkpoint/domain specialization; Suggested for causal explanations such as missing-label negative supervision.
- Best paper section: Dataset/Experimental Protocol, limitations, and discussion—not the main method-superiority claim.
- Safer wording: “The audit reveals substantial evaluation and data limitations and shows that checkpoint selection differs between the retained FathomNet criterion and WaterOVS transfer.”
- Missing evidence: clean validation split, quantified annotation-completeness review, matched FathomNet baselines, and a prediction-free full-test evaluation of the mapped-complete training run.

## 6. Claim Boundaries

### Strongly supported

- The code implements mask-index-aligned PGC correction, explicit zero self values, final-only replacement, and preserve-void behavior.
- Under the matched Large/BS9/FP32/seed-42 WaterOVS protocol, PGC-LDP is the main measured source of improvement.
- The selected EVP path uses high-pass residuals, low-pass gate conditioning, and stage-1/2 block-wise injection.
- Best and final checkpoints often differ; reporting only a chosen best checkpoint would hide regressions.
- FathomNet retained test is unsuitable as the sole unbiased benchmark because it is prediction-filtered and has no small-object instances.

### Moderately supported

- Query-aligned correction is preferable to unrestricted global proposal mixing. The topology is well motivated and more compact, but the main Large protocol lacks a matched global-LDP comparison.
- FathomNet training becomes domain/checkpoint specialized. Both external curves support the pattern, but causal mechanisms have not been isolated.
- LDP-related gains are primarily classification/fusion effects. The diagnostic is detailed but was run on global LDP rather than PGC.

### Claims that should be softened

- Replace “PGC-LDP improves unseen recognition” with “PGC-LDP improves overall AP and seen/in-vocabulary classification; unseen gains may reflect ranking and fusion.”
- Replace “EVP and PGC are strongly complementary” with “their combination gives the highest checkpoint, but the incremental gain over PGC alone is small.”
- Replace “efficient” with “lower analytic module-level parameter/MAC cost than global LDP”; no end-to-end speed or memory benefit has been measured.
- Use the prediction-free full-test result as the FathomNet generalization headline; retain the prediction-filtered retained set and WaterOVS transfer as complementary diagnostics.

### Not yet supported

- State-of-the-art claims on WaterOVS or FathomNet.
- Statistical significance, robustness across seeds, or robustness across code revisions.
- Novelty of EVP, PGC-LDP, or the combined design relative to published fusion/prompting methods.
- Strong PGC–EVP synergy.
- Better mask localization, better frozen-CLIP oracle classification, or broad unseen top-1 improvement.
- FathomNet small-object performance.
- End-to-end computational efficiency or deployment readiness.

### Experiments needed before submission

1. Repeat the matched Large/BS9 baseline, EVP-only, PGC-only, and combined grid with at least three fixed seeds; report mean, standard deviation, best, and final.
2. Add a matched global-LDP versus PGC-LDP comparison on the same Large protocol.
3. Build a fixed GT-only clean validation split for early stopping, then evaluate the selected mapped-complete checkpoint once on the full test.
4. On FathomNet, run same-data baseline, EVP-only, PGC-only, and combination controls rather than evaluating only the combined model.
5. Report exact-alias seen/unseen, head/medium/tail, supercategory, and size-stratified metrics; freeze these partitions before tuning.
6. Measure end-to-end parameters, trainable parameters, FLOPs/MACs, peak memory, throughput, and latency for baseline/global-LDP/PGC/EVP/combined.
7. Add the main-protocol ablations for preserve-void, paired-only versus cross-query mixing, zero-self value, and low-frequency conditioning.
8. Resolve the three failing amplitude-selected-bandpass tests or remove them from the claimed supported surface.

## 7. Writing Handoff

### `science-writing-foundations`

Use PGC-LDP as the primary contribution candidate and frequency-conditioned EVP as a secondary adaptation. The safest narrative is: FC-CLIP's final mask classification can mix region semantics and structure more selectively; query-aligned correction supplies most of the measured gain, while frequency prompting supplies a smaller visual-adaptation gain. Retain all single-seed, checkpoint-selection, and dataset-bias boundaries.

### `research-abstract-writer`

- Central contribution: query-aligned paired correction of mask-pooled CLIP tokens with same-mask decoder structure.
- Strongest controlled result: `53.9612 AP` PGC-only versus `47.5934` baseline (`+6.3678`) on WaterOVS ID; highest configuration `54.4971 AP` at 40k.
- Do not claim strong synergy or FathomNet superiority.
- Stable terms: FC-CLIP, PGC-LDP, query-aligned correction, preserve-void, frequency-conditioned visual prompting, WaterOVS.
- Use `EVP` only after defining it from the intended cited source; the repository alone does not establish a safe expansion.

### `research-introduction-writer`

Frame the technical problem as preserving CLIP region semantics while injecting mask-specific decoder structure without unrestricted cross-query mixing, plus limited underwater-domain visual adaptation. Prior-work and novelty claims require an external literature review and cannot be derived from this repository.

### `research-methods-writer`

Use the clean LDP/PGC, EVP, CLIP integration, transformer decoder, and FC-CLIP files as authoritative. Explicitly document hard detached pooling, final-only correction, original void preservation, geometric ensembling, stage-1/2 block injection, gate initialization, and absence of new losses. Put exploratory text adapters and unsupported EVP variants outside the main method.

### `research-results-writer`

Lead with the matched Large/BS9 four-way ablation. Report best and final separately. Use raw logs/configs as source of truth. Present the global-LDP localization analysis as a mechanism diagnostic with an explicit non-equivalence warning. Treat FathomNet as checkpoint/dataset analysis until a matched baseline and prediction-free test exist.

### `research-discussion-writer`

Discuss single-seed evidence, checkpoint regression, seen-class concentration, score-fusion effects, absence of runtime measurements, FathomNet imbalance/incomplete annotations, retained-test selection bias, and the different retained-test versus WaterOVS optimal checkpoints.

### `research-conclusion-writer`

Safest conclusion: query-aligned paired correction is the best-supported component and materially improves WaterOVS under one matched protocol; frequency prompting is useful alone but adds little at final when combined with PGC; generalization and efficiency claims remain open.

### `research-title-writer`

Supported keyword scope: open-vocabulary instance segmentation, underwater imagery, FC-CLIP, query-aligned region fusion, paired correction. Avoid “state of the art,” “robust generalization,” “efficient,” or a FathomNet benchmark claim in the title.

### `research-writing-reviser`

Audit every improvement against the matched baseline rather than the historical `44.8520` baseline, preserve best-versus-final distinctions, and remove any statement that attributes the full combined gain to EVP or that treats the retained FathomNet split as unbiased.

## 8. Open Questions For User

1. Should the paper's central object be PGC-LDP, the combined PGC-LDP+EVP model, or the broader FathomNet data/benchmark study?
2. What expansion and prior citation should define `EVP`, and is the low-frequency-conditioned router claimed as new or as an adaptation of an existing EVP method?
3. Were all four Large/BS9 ablation runs launched from exactly the same code snapshot and initialization procedure beyond the saved seed/config values?
4. How should the completed prediction-free full-test result and its gap from retained test be framed in the paper narrative?
5. Is there a fixed target venue and a published WaterOVS comparison table to use for any external-performance claim?
