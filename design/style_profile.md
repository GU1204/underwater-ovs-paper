# ICASSP Style Profile

## Target form

- Four technical pages demand one central problem, one compact method figure and only the most diagnostic ablations.
- Lead with signal and representation processing, not a broad underwater application survey.
- Describe the paper as transferring frozen open-vocabulary semantics through a constrained prediction path.
- Give EVP narrative priority because its FFT decomposition, conditional gate and shallow residual injection are the clearest ICASSP-facing signal-processing elements.
- Present PGC-LDP as a downstream same-query correction mechanism, not as a second independent grand theory.

## Rhetorical discipline

- Use `implementation enforces` for source-verified constraints, `is designed to` for intended behavior, and `the run shows` for measured outcomes.
- Prefer `conditions`, `selectively injects`, `paired correction`, and `avoids direct modification`.
- Avoid `restores`, `preserves semantics`, `recognition-optimal support`, `reliable frequency`, `significant`, and `improves generalization` unless a direct diagnostic supports them.
- Report best and final checkpoints together where they differ.

## Evidence-to-claim profile

- The BS15 ablation gives directional module evidence: EVP-only final 48.4061 AP, PGC-only best/final 49.4622/47.9295 AP, and joint band-amplitude EVP + PGC final 50.7442 AP.
- It is not a strict 2x2 because the baseline is historical, EVP-only uses AMP, the PGC runs use FP32, and the joint EVP router differs from the EVP-only router.
- The Large BS9 joint run gives a strong overall signal (54.4971 best, 54.0845 final versus 47.3413 baseline final), but cannot isolate the two modules.
- Current evidence supports WaterOVS ID performance potential, not cross-domain or unseen-class generalization.

## Recommended four-page emphasis

1. One transfer problem and two constrained insertion points.
2. EVP mechanism and its asymmetric frequency roles.
3. PGC-LDP in one compact formula/diagram as same-query, correction-only readout adaptation.
4. One main table, one near-matched ablation table with explicit confounds, and one diagnostic/complexity table.

