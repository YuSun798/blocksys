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

The color revision uses blue for trusted client computation, teal for public execution, and orange for secure execution. Text labels, marker shapes, line patterns and fill contrast carry the same distinctions in grayscale. The causal sweep uses the archived aggregate CSV and overlays the exact expectation for uniformly placed private inputs. The width sweep uses the archived matched arithmetic results and combines the two configurations with numerically identical secure-product counts. The plot data are regenerated with `python3 experiments/figure_data.py` and checked by CI.

## Submission integrity review

This section records the current disposition of source results discussed in earlier sections.

The latest manuscript distinguishes directly reproduced diagnostics, source-reported results, and analytical counts. The source PDF tables establish transcription provenance. They do not independently verify the original runs.

| Finding | Manuscript disposition |
|---|---|
| The old Marill entry attributed the paper to Cheng et al. and USENIX Security 2024. The identifiable work is by Rathee, Li, Stoica, Zhang and Popa. | Cite the verified [arXiv preprint](https://arxiv.org/abs/2408.03561). |
| Cocoon had an incorrect page range for its ACM article. | Use Article 100 as listed by [ACM](https://doi.org/10.1145/3649817). |
| The Agentic-PPML first-author initial was wrong. The paper is a position paper. | Use the four authors listed by [arXiv](https://arxiv.org/abs/2508.02836). Describe the related design without treating the paper as an empirical baseline. |
| Closest related work includes remote pattern extraction with local answer reconstruction and adaptive plan-template reuse. | Scope PriGraph's distinction to typed interfaces, explicit public subgraph keys and the proposed handoff to operator closure. See [AAAI proceedings PDF](https://ojs.aaai.org/index.php/AAAI/article/download/40779/44740) and [APC proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9549f7d06700f0966d5f938f1d11022a-Abstract-Conference.html). |
| Table 4 in the ShareRoute source compares 0%, 50% and 75% public input across the three configurations. Its 3.9x value mixes workload exposure and configuration. Its 95x wall-clock and 44x per-party communication additionally switch the trust model. The ratio of summed party communication for the 2PC and 3PC endpoints would be approximately 29.2x under consistent per-party counting. | Remove the protocol table, orthogonality language and the isolated island speedup claim. Keep the source 2PC sweep labeled as workload sensitivity. |
| ShareRoute source reports 8,760 comparisons with factors 40 pairs × 79 tensors × 64 positions. The product is 202,240. The unit or filters require the underlying logs. Source wording also mixes single-run measurement with five-seed mean and SD. | The aggregate count is excluded. Keep only the original table values with their stated protocol and summary statistics, qualified as source-reported. |
| PriGraph's accuracy arms change model size and execution path together. The separately reported local-tool control attains the same explicit-formula accuracy. A 100% fallback rate measures route eligibility, whereas fallback accuracy measures task utility. | Retain qualified synthetic source evidence and remove the inference that rewriting caused the accuracy difference. Keep only the fallback rate for the private-formula arm. |
| The 1,200-case local guard script uses an explicitly assigned eligibility flag, constructs fixed public fields, tests absence of keys it never inserts, and increments the egress counter by zero. | Exclude those counts from the paper's evidence. Archived tests may remain available as examples. |
| The causal chart is generated from oracle dependency labels. Output perturbations are checked separately at the specified tolerance. Fixed prefix and suffix masks and 100%/0% endpoints make some comparisons deterministic. | Label the plotted quantity as closure-derived and avoid using agreement as a validation of cryptographic security or of the actual ShareRoute implementation. |
| The arithmetic emulator directly constructs its four schedules. It uses a seeded random generator and counts logical messages, while the local process sees both parties' shares. A direct algebraic baseline attains one protected square under the same forced offload policy, and all-local arithmetic can use zero MPC. | Keep as a scope-limited mechanism example. Disclose the algebraic baseline in the abstract, methods and figure caption. Exclude claims of an optimizing compiler, measured network bytes, round speedup or a production MPC security result. |
| The proposed security argument assumes complete information-flow tracking, cache independence and backend security. Metadata, control-dependent public behavior, malicious peers and cross-session composition require their own models. | Keep only a conditional plaintext egress argument and define its scope. |

For the integrated evaluation, the decisive contrast is a fixed prompt set with identical disclosure policy, 2PC backend and output tolerance. Compare full MPC, same-prompt ShareRoute, task-only rewriting, joint rewriting with residual MPC, and a strong hand-written or compiler optimization baseline. Report end-to-end input/output cost, preprocessing, peer messages, latency, exact or calibrated task quality, and uncertainty by workload and seed.

## Follow-up paired TCP execution

This replaces the former in-process emulator as the manuscript's arithmetic
ablation. The archive retains the emulator as an independent regression check.
Two workers exchange square-correlation openings over TCP loopback, using
OS-random masks and a trusted client as dealer. Each worker sees only its own
operand and preprocessing shares. The client adds the local correction and
reconstructs the weighted sum from the two returned shares.
The host and dealer are trusted for this experiment. Traffic counts are
application payload sizes measured separately for peer openings, client
inputs/outputs, and dealer delivery. The reported cold time starts at plan
construction and includes preprocessing over established sockets. Online time
starts after the dealer acknowledgment and includes input encoding and result
reconstruction. The instrumented latency includes two extra control-report
words omitted from client payload accounting.

Five schedules share each input and backend, including a hand-written
algebraic comparator and a private-template negative control. The 4,000
measured outputs match the plaintext reference in the 64-bit ring. At width 64
on public templates the joint and algebraic schedules each use one square,
16 opening bytes, 48 client payload bytes and 42 dealer/control bytes. Full
MPC uses 64 squares and 1,024 peer opening bytes. At the same width, when all
template operands are private all five schedules use 64 squares. The
experiment measures the restricted arithmetic implementation and does not
establish a gain over algebraic simplification, a general lower bound for
private templates, an MP-SPDZ replication, WAN latency, or secure Transformer
execution.

## Pretrained model follow-up

The additional plaintext check uses official Transformers 4.52.4 eager CPU
FP32 with the pinned Pythia-70M-deduped checkpoint and weight SHA-256.
The 48 pairs use 12 synthetic templates, two value pairs per template,
and two field placements. Moving one secret field preserves the token
inventory. Before the changed token the distributions have zero observed
TV. Beyond that token, all 290 early-placement and all 82 late-placement
unchanged-token positions exhibit TV above the prespecified (10^{-5})
threshold. Prefix cache reuse maintains all 1,928 top-1 predictions and
max TV is 0.001099. This demonstrates a dependency and cache behavior with
pretrained weights. It is a single-checkpoint plaintext diagnostic on
synthetic prompts, with no transport, secure attention, or quality benchmark.

## Two-party pretrained output projection

The same 48 Pythia prompt pairs are evaluated using complete trusted prefill
or one cached public prefix per pair. For each prompt the client splits its
final fixed-point hidden vector into two uniform additive shares. Two TCP
loopback processes apply the public head locally. Row selection is an
output-contract specialization. It discloses four candidate IDs only in
the authorized public-pool arms. In the counterfactual private-pool policy
the same model, tokens and IDs are used but the client uploads no selector
IDs, receives full shares, and performs the selection itself.

All 480 executions match their same-input local fixed-point gather on the
four candidate scores. The reconstructed constrained top-1 matches FP32
in all five arms, 96/96 per arm. Both protocols inherit the underlying
model's 88/96 correct field tokens. Maximum error over candidate logits is
0.0062002, including any cached-prefix kernel difference. Total client
application payload is 813,064 bytes for a full output and 8,296 bytes for
four authorized rows. There are no inter-worker messages.
Pruning alone reaches the same payload as the joint schedule. A local
four-row gather is faster, and prefix reuse increases prefill latency for
these short two-prompt batches. This is a conditional offload comparison
with stable public rows, not evidence for system-wide speedup, open-vocabulary
generation, an offloaded Transformer body, or a private row-selection protocol.

## Interactive attention-score follow-up

The second layer, head 0, uses native pretrained Pythia-70M post-RoPE query
and key vectors. Captured scores reproduce eager attention probabilities
before quantization. Both arms execute the same causal triangle on the same
96 prompts. Under a trusted client/dealer and semi-honest non-collusion, the
all-shared arm uses Beaver multiplication for every coordinate. The
selective arm discloses the approved causal prefix, computes mixed products
linearly on shares, and uses fresh Beaver correlations only for the private
suffix block. Actual independent workers communicate over pre-established
localhost TCP sockets. No future-token score, softmax, autoregressive
decoding, or full-layer secure execution is benchmarked.

The 40,976 fixed-point scores reconstructed across 192 tiles match exactly.
The maximum scaled error against the floating reference is 9.7271e-5.
Median opening traffic is 430,080 bytes in either all-shared cell, 186,368
for early selective, and 20,480 for late selective. Median dealer traffic
is 645,146, 279,578 and 30,746 bytes respectively. Median client upload and
output bytes are 40,960 and 3,360 in every cell. Timings include one extra
eight-byte counter per worker, omitted from the payload columns. The QK
results and output-head results are separately measured operator boundaries.
Do not multiply their ratios or infer a complete full-model speedup.

## Composed Pythia forward

The integrated probe replaces one Pythia second-layer causal score tile
with reconstructed fixed-point values and immediately consumes those values
in the real model's softmax, value aggregation, and residual path. It sends
the resulting final state as shares to the output-head workers. The client
runs the remaining Transformer and reconstructs both protected boundaries.
The benchmark uses the same 96 single-prompt prefills, two separate pairs
of local TCP worker processes, five randomized policies, and two warmups
per policy. Timings include attention preprocessing and both reconstructions
and exclude initial model and connection setup and correctness checks.

All 480 candidate vectors agree with the corresponding integer references.
Across all five policies the archived candidate digests agree exactly per
prompt. Top-1 matches the unmodified FP32 path in 96/96 prompts per policy.
The absolute maximum candidate-logit deviation is 0.011131. Each policy
gets 88/96 target tokens. Median attention peer payload is 430,080 B for
all-shared QK and 77,824 B for selective QK. Dealer payload is 645,146 B
and 116,762 B respectively. Public four-row authorization lowers head
payload from 813,064 B to 8,296 B. The private-pool control returns the full
head while retaining selective QK. Elapsed medians of 14.03 ms and 4.96 ms
compare two offloaded policies on this host. The result is scoped to one
selected attention head and a constrained output head with the rest of the
model local, semi-honest workers, a trusted client/dealer, and no WAN claim.


## Network sensitivity replay

The 18 scenarios overlay analytical serialization and propagation on all
96 composed-request traces. The shared-link model serializes all payloads.
The parallel-link model uses the maximum of two equal-size client-side
messages and serializes both ordered inter-worker openings. Both charge
four RTTs and use the measured loopback elapsed time as the starting point.
The paired ratios are medians of individual full-to-joint ratios.
This model does not measure cross-host performance, change the security
assumptions, or simulate TCP congestion or concurrent traffic.
