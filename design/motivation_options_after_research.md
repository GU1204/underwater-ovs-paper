# Motivation Options after Research

## Overall judgment

The current two-interface motivation is **directionally correct and suitable for ICASSP, but not suitable unchanged**. It matches the implemented data path and avoids false shared-cause claims. Its main weakness is that two equally weighted stories are too broad for four pages, while current ablations remain directional rather than strictly causal.

## Option A — Controlled representation transfer at two interfaces (recommended)

> Transferring frozen vision-language semantics to underwater open-vocabulary instance segmentation requires controlled adaptation at two representation interfaces. At the visual interface, underwater appearance variation motivates shallow frequency-conditioned residual injection: low-frequency responses condition where a high-frequency candidate residual is written. At the region interface, FC-CLIP already provides paired CLIP-derived and task-learned representations for the same query; we use the latter only as a suppressible correction to the CLIP receiver before text matching. The two modules address different bottlenecks but share one design rule: task-specific information is written only through constrained conditioning or pairing.

Why recommended:

- It matches both source implementations.
- It gives ICASSP a clear signal/representation-processing center.
- It unifies the modules without inventing a common underwater degradation mechanism.
- It permits cautious reporting of the BS15 ablation and strong Large joint result.

Required boundary: call the BS15 evidence a directional ablation, not a strict 2x2; make no CD/generalization claim without additional evidence.

## Option B — EVP-first frequency-conditioned adaptation

> Underwater appearance variation disrupts the direct transfer of natural-image visual-language representations. We therefore restrict explicit adaptation to shallow stages and use low-frequency context to condition the selective injection of a high-frequency candidate residual. A same-query region correction module is included only as a downstream readout enhancement.

Why choose it:

- Safest for ICASSP's four-page limit.
- EVP has the strongest direct signal-processing identity and a clearer ordinary-gate comparison.
- Reduces the risk that reviewers see two incremental modules joined after the fact.

Trade-off: PGC-LDP becomes a secondary component, so the paper claims less architectural breadth.

## Option C — Same-query semantic correction as the main contribution

> Mask-based OVS already produces CLIP-derived and task-learned representations for the same proposal, yet mainly combines their predictions at score level. We adapt the CLIP region receiver before text matching using a paired, correction-only, keep-or-correct topology, while shallow frequency adaptation supplies underwater-aware visual features.

Why choose it:

- The architecture gap relative to score-level fusion is concrete.
- PGC pairing, zero self value and preserve-void boundaries are technically explicit.

Trade-off: current evidence is weakest here. PGC-only regresses at 50k, lacks matched fusion alternatives and gate/readout diagnostics, and therefore cannot yet support a strong superiority claim.

## Recommendation

Choose **Option A** for the paper spine, but write the four-page paper with the emphasis order of **Option B**: EVP is the principal ICASSP-facing mechanism; PGC-LDP completes the prediction path. If the remaining matched controls and CD/open-vocabulary diagnostics cannot be added, use Option B alone.

## Confirmation required

PaperSpine must stop here until the user explicitly confirms Option A, B, C, or supplies a revised motivation. `confirmed_motivation.md` must not be created automatically.
