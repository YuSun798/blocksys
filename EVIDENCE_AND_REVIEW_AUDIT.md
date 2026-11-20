# Reviewer-facing audit — PriGraph × ShareRoute, revision 2026-11-18

## Paper thesis

Do not confuse *task-level elimination of avoidable secret dependence* with *cryptographic protection of residual private operator computation*. The proposed novel link is typed task-to-operator handoff and a security-stage-constrained global planning objective. It remains a framework design, **not a fully tested integration**.

## Source-result provenance matrix

| Research question / measurement | Original PriGraph PDF | Original ShareRoute PDF | Interpretation |
|---|---|---|---|
| Accuracy | 360 synthetic tasks; original 7B local 54.2%, privacy-violating 14B remote 69.2%, rewritten 14B public + private local binding 100%; reported 95% CIs for baselines | 9,800 operator comparisons, 120/120 argmax on d=32; no full-size downstream task accuracy | Different notions of correctness; cannot compare percentages |
| Key confounder | Tool-augmented local 7B also achieves 100% on 400 explicit-formula financial tasks; 38% on 200 implicit-formula tasks (no matched rewritten 14B arm) | Different trust stages: semi-honest 2PC vs honest-majority 3PC; 95x result switches stages | Avoid attribution of deterministic-evaluation accuracy to graph rewriting and 95x to island selection alone |
| Reuse/scheduling | 6 vs 2 generations at 3 variants; 12 vs 2 at 12; joint 0.1551 vs sequential 0.1929 or 0.4110, *calibrated simulation* | Island/protocol/cohort joint plan described but cohort placement/fusion not independently benchmarked | No cross-system real traffic result |
| Cryptographic execution | No MPC within task prototype | Real MP-SPDZ on d=32 mini-GPT; full 2PC 73.2 s / 6313 MB vs selectable 2PC 19.0 s / 1629 MB; selectable 3PC 0.77 s / 144 MB | 3.9x **within 2PC** at 75% public; 95x changes trust assumption |
| Boundary | Private formula: 104/104 fallback; reported 0% rewritten path | 100% private: island selection reverts to Full-MPC | Both boundaries required in any new combined experiment |
| Large-scale evidence | 7B/14B Qwen task-level workloads; heterogeneous 4090/V100 preliminary tests | GPT-2 125M taint-only / plaintext; d=768 MPC estimate analytical, compiler times out at d=128 | No production-scale encrypted model inference |
| Adversarial support | Guarded/unguarded secret egress, prompt injection, cache poisoning, AST execution battery | Information-flow argument and conditional composition; anti-collusion not empirically validated | Not a general malicious-peer proof |

Original source PDFs supplied by the user: `PriGraph_v5_final_2026-10-18(1).pdf` and `ShareRoute_BlockSys2027_final(2).pdf`; the downloadable delivery ZIP retains these unchanged. Values above are reported source results, not results of this revision.

## New experiment: CPU-only causal dependency

Source: `experiments/causal_privacy_cpu.py`; raw data and summary under `experiments/results/`.

- Frozen, random 2-layer d=32, 4-head causal Transformer; PyTorch CPU, sequence length 16.
- 5 private-input fractions x 3 placements x 4 seeds x 12 trials = 720 paired tests, 11,520 position comparisons.
- Exact conservative full-causal dependency rule: output at position t depends on any secret input position i <= t. The test found 0 observed false negatives and 0 observed false positives using a fixed 1e-6 activation-difference threshold.
- A naive label based only on whether *that input token* is private missed 2,133 changed positions across the experiment. At private fraction 25%, permissible public-state fraction was suffix 75%, prefix 0%, interleaved 14.97%.
- Deterministic arithmetic harness: 1,200 tests, with 800 public-formula evaluations and 400 private-formula fallbacks; rejects 800 invalid-version cache lookups.

**Scope exclusion:** no MP-SPDZ executed in the new run, no cryptographic secrecy measurement, no WAN/peer scheduling, no learned model benchmark and no GPU hardware in this environment. These unit-level results support a dependency/architecture point only, not a novel performance headline.

## Top-tier reviewer objections and remedies

1. **No integrated implementation:** implement a genuine task rewrite to operator trace, and a working 2PC backend for the non-decomposable residual graph. Time and bytes must come from one end-to-end run, with no hypothetical cross-layer gain.
2. **No matched counterfactual on implicit reasoning:** compare 7B/14B with *identical* tools, policy, and budgets; measure non-deterministic task quality and unsafe egress explicitly.
3. **Cryptographic assumptions differ:** repeat complete factorial analysis independently at S1 (semi-honest 2PC) and S2 (honest-majority 3PC), without mixing denominators.
4. **Data and model scale:** prefill vs decode, longer contexts, 7B+ protected activations, realistic regions/WAN and heterogeneous peers; separate numerical error from task accuracy.
5. **Cost-model confounding:** source simulated reuse and planner proxy should not be reported as live financial savings; measure planner CPU latency, cache misses, networking, cost gaps and session-unique offline triples.
6. **New CPU test overclaims:** it is random two-layer attention, not proof of soundness. Extend to pretrained decoders and dynamic execution traces, examine public-cache lookups and rejection of illegal declassification.
7. **Bibliography:** corrected the original PPMI entry against arXiv:2506.17336 (Bae et al.; the supplied PDF had an incorrect title/placeholder author), and checked MINIM against the ICML 2026 PMLR entry. Other titles and venue metadata inherited from both PDFs still merit full primary-source verification before submission.
8. **Single-blind submission:** author info must be authentic; the draft intentionally inherits anonymous author placeholders.

## Exact proposed minimum factorial for a genuine future paper

For each fixed stage and matched workload: A local-7B with deterministic tool; B task rewriting + local private binding; C residual full-model MPC; D operator-only island MPC; E rewrite + island MPC (cache off); F rewrite + island MPC + verified reuse; G full joint planner vs sequential and exhaustive oracle on small DAGs. Record task correctness/quality, privacy policy leakage, online/offline bytes, latency breakdown, GPU-hours, WAN effects, cache safety, MPC numeric error, failure recovery, and 95% seed-level uncertainty. Include private formula, private prefix, interleaved secrets, 100% private, replayed cache poison, and unsafe static public masks.

This audit is an explicit boundary on the claims made in the revised paper. It is not a promise of results from experiments not run.


## Matched cross-layer arithmetic ablation

A new paired test evaluates the same 400 synthetic requests under four rewrite × island configurations. The output of every configuration equals the ring reference (1,600/1,600). At width 64, per-request logical peer bytes are 2,048 (full), 2,048 (islands alone), 2,080 (rewrite alone), and 32 (both). All arms batch their openings in one round. The deterministic in-process emulator exercises additive sharing and Beaver openings with independent dealer triples. It excludes client ingress/egress, preprocessing and transport, and supplies no LLM, WAN or cryptographic deployment evidence. The required offload policy isolates the task-to-operator handoff. A local arithmetic evaluator can solve the entire workload.

The latest two attachments match the earlier source files byte-for-byte. SHA-256:

- PriGraph: `b94dc97540927d13bebed179193e19fb33ec85daab07f523d7bf3317925c935d`
- ShareRoute: `ee598947f85de22d288a077e3382621731d027a5d999e05ec62c58fa247fbe25`

## Figure and table references

These papers informed visual conventions. The manuscript figures and values come from the two source studies, the compiler design and the archived CPU experiments.

| Proceedings paper | Layout decision used in this revision |
|---|---|
| [Orca, OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf) | Separate compilation flow and execution into named panels. |
| [FlashAttention, NeurIPS 2022](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html) | Use a spatial diagram to expose structure, with each mark defined in the legend. |
| [PagedAttention, SOSP 2023](https://doi.org/10.1145/3600006.3613165) | Use an aligned cell grid for token position and public-state comparisons. |
| [SecretFlow-SPU, USENIX ATC 2023](https://www.usenix.org/conference/atc23/presentation/ma) | Give compiler stages and runtime endpoints distinct visual roles. |
| [BOLT, IEEE S&P 2024](https://doi.org/10.1109/SP54263.2024.00130) | Identify the protected computation primitive at the point where it executes. |
| [DistServe, OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | Carry comparison conditions and physical units in quantitative headers. |
| [Sarathi-Serve, OSDI 2024](https://www.usenix.org/system/files/osdi24-agrawal.pdf) | Report paired workload conditions with the results they qualify. |
| [PowerInfer, SOSP 2024](https://doi.org/10.1145/3694715.3695964) | Align resource categories so placement can be compared spatially. |
| [Breaking the Layer Barrier, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi) | Give alternative security backends separate grouped columns. |
| [PlanetServe, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/fang) | Route returned artifacts and output shares explicitly to the trusted client. |
