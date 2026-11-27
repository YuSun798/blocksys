# Evidence audit for PriGraph and ShareRoute

## Submission revision

The manuscript now distinguishes the proposed compiler from the implemented
two-layout serializer and instrumented Pythia operators. The tested
two-party path computes QK scores for one attention head and projects the
last hidden vector. The client performs the remaining prefill operations.
The QK kernel validates malformed shapes and prefix bounds before sending.
In-range disclosure relies on a trusted caller.

The main paper retains matched traffic, local latency controls, the 160M
accuracy regression on field copying, and the opposite change on the
full SST-2 development split. The paired SST-2 table is generated from the checked TCP
archive. The gate is reported as a repeated check of its enumerated input
domain, with conditional break-even in the table. The synthetic arithmetic
ablation and sender-delay sweep remain in the repository archive. The
closest prompt-separation manuscript is cited as a public OpenReview
manuscript. [OpenReview record](https://openreview.net/forum?id=tFNsosNMHT)

After adding the QK input checks, both model checkpoints were rerun on
the full 128-example SST-2 sample over TCP. The post-change run matched
the prior archive on all 512 prompt fingerprints, native and protected
choices, ring checks and protocol counts. The independent verifier
re-tokenized 256 prompt layouts from the original development split.
The first two runs used the same balanced subset and preceded the full
development experiment below.

## Source-result provenance matrix

| Research question / measurement | Original PriGraph PDF | Original ShareRoute PDF | Interpretation |
|---|---|---|---|
| Accuracy | 360 synthetic tasks; original 7B local 54.2%, privacy-violating 14B remote 69.2%, rewritten 14B public + private local binding 100%; reported 95% CIs for baselines | 9,800 operator comparisons, 120/120 argmax on d=32; no full-size downstream task accuracy | Different notions of correctness; cannot compare percentages |
| Key confounder | Tool-augmented local 7B also achieves 100% on 400 explicit-formula financial tasks; 38% on 200 implicit-formula tasks (no matched rewritten 14B arm) | Different trust stages: semi-honest 2PC vs honest-majority 3PC; 95x result switches stages | Avoid attribution of deterministic-evaluation accuracy to graph rewriting and 95x to island selection alone |
| Reuse/scheduling | 6 vs 2 generations at 3 variants; 12 vs 2 at 12; joint 0.1551 vs sequential 0.1929 or 0.4110, *calibrated simulation* | Island/protocol/cohort joint plan described but cohort placement/fusion not independently benchmarked | No cross-system real traffic result |
| Cryptographic execution | No MPC within task prototype | MP-SPDZ on d=32 mini-GPT; full 2PC 73.2 s / 6313 MB vs selectable 2PC 19.0 s / 1629 MB; selectable 3PC 0.77 s / 144 MB | The 2PC contrast changes public exposure; the 95x comparison also changes the trust assumption |
| Boundary | Private formula: 104/104 fallback; reported 0% rewritten path | 100% private: island selection reverts to Full-MPC | Both boundaries required in any new combined experiment |
| Large-scale evidence | 7B/14B Qwen task-level workloads; heterogeneous 4090/V100 preliminary tests | GPT-2 125M taint-only / plaintext; d=768 MPC estimate analytical, compiler times out at d=128 | No production-scale encrypted model inference |
| Adversarial support | Guarded/unguarded secret egress, prompt injection, cache poisoning, AST execution battery | Information-flow argument and conditional composition; anti-collusion not empirically validated | Not a general malicious-peer proof |

The table records values reported in the two original source PDFs supplied by
the authors. File digests appear below.

## Initial CPU diagnostic of causal dependence

Source: `experiments/causal_privacy_cpu.py`; raw data and summary under `experiments/results/`.

- Frozen, random 2-layer d=32, 4-head causal Transformer; PyTorch CPU, sequence length 16.
- 5 private-input fractions x 3 placements x 4 seeds x 12 trials = 720 paired tests, 11,520 position comparisons.
- Exact conservative full-causal dependency rule: output at position t depends on any secret input position i <= t. The test found 0 observed false negatives and 0 observed false positives using a fixed 1e-6 activation-difference threshold.
- A naive label based only on whether *that input token* is private missed 2,133 changed positions across the experiment. At private fraction 25%, permissible public-state fraction was suffix 75%, prefix 0%, interleaved 14.97%.
- Deterministic arithmetic harness: 1,200 tests, with 800 public-formula evaluations and 400 private-formula fallbacks; rejects 800 invalid-version cache lookups.

**Scope exclusion:** no MP-SPDZ executed in the new run, no cryptographic secrecy measurement, no WAN/peer scheduling, no learned model benchmark and no GPU hardware in this environment. These unit-level results support a dependency/architecture point only, not a novel performance headline.

## Early review before the Pythia experiments

1. **Integration:** the first draft had no timed task-to-operator path. The later Pythia runs integrate one typed rewrite and two operators. General-purpose tracing and full-model MPC remain design considerations.
2. **No matched counterfactual on implicit reasoning:** compare 7B/14B with *identical* tools, policy, and budgets; measure non-deterministic task quality and unsafe egress explicitly.
3. **Cryptographic assumptions differ:** repeat complete factorial analysis independently at S1 (semi-honest 2PC) and S2 (honest-majority 3PC), without mixing denominators.
4. **Data and model scale:** prefill vs decode, longer contexts, 7B+ protected activations, realistic regions/WAN and heterogeneous peers; separate numerical error from task accuracy.
5. **Cost-model confounding:** source simulated reuse and planner proxy should not be reported as live financial savings; measure planner CPU latency, cache misses, networking, cost gaps and session-unique offline triples.
6. **New CPU test overclaims:** it is random two-layer attention, not proof of soundness. Extend to pretrained decoders and dynamic execution traces, examine public-cache lookups and rejection of illegal declassification.
7. **Bibliography:** corrected the original PPMI entry against arXiv:2506.17336 and checked MINIM against the ICML 2026 PMLR entry. The subsequent peer review traced the 30 public references in the merged draft to primary records and identified the additional prompt-separation manuscript.
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
| PriGraph's accuracy arms change model size and execution path together. The local-tool control attains the same explicit-formula accuracy. | Attribute the matched explicit-formula results to the source. Report private-formula fallback (104/104) and rewrite-arm accuracy (0/104) separately. |
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


## Controlled TCP delay

Full and joint policies execute the same 96 prompts at each of three
nominal RTTs, with simulated one-way propagation implemented by sender
waits on real TCP loopback sockets. Each client fan-out incurs one wait.
Each independent worker delays its return, and peer openings are ordered.
The measured ratios decrease with the configured delay. The measured
increments exceed the ideal four-RTT increase on this host, so the
analytical serialized link estimates and the application delay experiment
are reported separately. The archive checker verifies all 576 keys,
unchanged payloads per policy, identical per-prompt score digests and
the timing summary.


## Matched task rewrite in a pretrained forward

The executable record compiler checks all four allowed bindings before
accepting either serialization. It verifies the same typed contents and
token multisets for the two layouts, and per-layout invariant prefix and
shape. It refuses to lift a private note or a note in an order-sensitive
schema. The compiled plan minimizes private QK products under the same
head-row policy. The tested placements use one fixed dealer-assisted
backend with TCP loopback on the M4 host.

The archived experiment covers 12 new schemas × 4 bound values × 5
configurations, or 240 protected executions. The length is equal per
matched request. The two all-shared arms have identical byte counts.
For selective QK the mean prefix increases from 7.92 to 17.08 tokens,
mean QK peer openings move from 206,848 to 25,600 bytes, and mean
total application payload from 573,574 to 120,454 bytes. The median
paired ratios are 4.67 for total bytes and 1.10 for protected request
elapsed time. Median comparison of candidate layouts adds a separate
0.38 ms compilation measurement per schema. Manual public-first
serialization produces the same layout, traffic, and fixed-point outputs.

Both protected boundaries match the integer references in all 240 runs.
Constrained FP32 decisions match within each layout. The original
layout answers 47 of 48 and the lifted layout 48 of 48. All 48
cross-layout fixed-point score digests differ. The single changed
decision is wrong-to-correct on these synthetic field-repetition tasks.
The independent archive checker recomputes payloads and paired outcomes
and checks plan selection, shape invariance and the reported summary.



## Completed TCP context-length and trusted-local controls

Using the same 12 schemas and four bindings, add zero, three or nine
sentences to the public annotation and rerun all five protected policies.
For each layout, also time local eager FP32 with the same prepared token
tensor and author-approved candidate rows, and a deterministic parser with
the same serialized record plus the public candidate map. Each tier has
48 inputs and nine policy arms. The archived benchmark runs 1,296 arms
on the same host, including 720 protected TCP executions. It excludes
compilation, tokenization, connection setup and model loading from measured
latency, and includes dealer preparation and delivery in protected time.
The parser accepts the original or rewritten record and uses the typed
schema to read the private field without access to the experiment target.

The token ranges are 19–25, 46–52 and 105–111. Mean QK peer openings under
the joint policy remain 25,600 bytes across tiers. Operator-only QK peer
openings grow from 206,848 to 10,275,499 bytes. Joint total application
traffic grows from 120,454 to 386,137 bytes, as the trusted client still
sends full-length Q/K and receives the whole causal triangle. At the
longest tier, operator-only, joint and local FP32 medians are 42.12,
10.59 and 8.90 ms respectively. The independent checker recalculates
per-request payload and protocol bounds, validates local zero traffic,
and reconciles row-level accuracy and timing summaries. Local parsing
gets 288/288 fields. The interaction between disclosure and offload is
evaluated with a fixed number of private tokens and one protected QK head.

## Figure 1 illustration provenance

Gemini Apps' Nano Banana rendered a visual reference from a two-panel
specification. The first panel shows typed rewriting, privacy closure,
optional public artifact reuse and policy-eligible routes. The second
shows the measured selective Pythia prefill and separate QK and projection
worker pairs. Nano Banana's second rendering adds a policy checklist,
token glyphs, a causal score grid, layered prefill blocks, candidate
scores and server icons. The vector figure adapts these illustrations
to the implemented share directions and places logit reconstruction
after projection. The grid and bars are schematic.

Visual references included [RLHFless Fig. 5](https://arxiv.org/pdf/2602.22718#page=4),
[FlexPipe Figs. 5–6](https://www.usenix.org/system/files/atc25-zhao-hairui.pdf),
and [DistServe Fig. 6](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf).
The captured Nano Banana reference is saved at
`figures/compiler-nano-banana.jpg` (1401 × 594, SHA-256
`f7cee4864c19daf00ebdc0d01250925b4c4c9f8fc5dffba898ea40ca14a50281`).

## Record-layout illustration provenance

Gemini Apps' Nano Banana image tool rendered a design reference for the
two prompt strips and causal score grids. The captured 989 × 540 JPEG is
`figures/record-lift-nano-banana.jpg` and has SHA-256
`3f98c1ec262c13549ac96a418d7c73da4b82a08a975985a98e2c52641623ae39`.
The paper uses an editable vector adaptation. Its prompt positions
follow the typed record layouts. The score classes follow the causal
mask and the approved-prefix cutoff. With four illustrative positions,
prefix lengths one and two give six and three private-private scores,
respectively, by the triangular count in the method. The caption
identifies the example as a schematic. The acknowledgments disclose
the use of Nano Banana. The numeric tables are generated directly from
archived measurements.



## Two-checkpoint replication and quality-gate replay

The 160M experiment repeats the complete 1,296-run TCP sweep from the
70M checkpoint, including five protected configurations and four local
comparators. Both checkpoints use the same 48 structured records at
each length tier. The pairing checker requires identical candidate IDs,
prompt and multiset token fingerprints, layout, target, disclosed
prefix and QK counters. Only the hidden-vector width changes application
accounting. The four-row output head uses 8,296 bytes for 70M and
12,392 bytes for 160M. Both checkpoints use 64-wide attention heads.
The new official runner records hashes of the pinned weight file,
configuration, tokenizer and companion files. Protected results for
each layout match that layout's native predictions in all 1,440
protected runs across the two checkpoints.

For 160M, the original-layout operator-only arm gets 39/48, 39/48 and
35/48 correct. Joint gets 27/48, 26/48 and 24/48, with 40 regressions
and four gains across the paired tiers. The same regressions appear
in local FP32. The semantic record contract therefore admits changes
that reduce work while changing model predictions. The tier totals
are descriptive because note lengths reuse the same semantic records.
Record order, tokenizer, and communication invariance do not establish
task accuracy or robustness beyond this model family.

An exhaustive replay admits a lift only if all four authorized bindings
of a schema leave its native FP32 decision unchanged. The check uses
all allowed values, without conditioning on the actual private value.
For 160M this selects the lift on 5/12, 5/12 and 2/12 schema instances
and preserves the original decisions. The median paired byte ratio is
1.00 at each tier, while aggregate bytes fall by about 30%, 40% and 16%.
For 70M the gate admits 11/12, 12/12 and 12/12. These are archive replays
under a finite exhaustive domain and pinned checkpoint. The replay
contains no preparation timings. The live study below measures setup
for the finite binding domain.

## Accessible vector figure handoff

The published-facing version of the record-layout diagram keeps the
Nano Banana rendering as its visual reference and typesets the same
two verified field orders as vector graphics. Both the original JPEG
and its editable TikZ adaptation remain in the source bundle.
Descriptions for all three figures are embedded in the official
LNCS source. The class generates a separate description list at build
time. Bibliography items are ordered by first citation, and every grouped
citation lists reference numbers in ascending order.



## Live certification, protected routing and projected reuse

In a new TCP run, both pinned checkpoints build a public-domain
certificate before dispatch. Each certificate compiles both layouts and
evaluates all four public bindings in native FP32. The planner selects
the rewrite only when the four decisions agree and private QK work
decreases. It otherwise uses the original layout with selective QK.
Three timed repetitions for each of 72 model, tier and schema groups
produce 216 certificates and 1,728 native forward calls within the
timed regions. The warm path sends each binding once through each
route in randomized pair order. All 576 protected runs are new, with
fresh correlated randomness. All 288 paired outputs match.

| Checkpoint | Tokens | Admitted | Positive savings among admitted | Median setup (ms) | Warm paired ratio | Conditional break-even median |
|---|---:|---:|---:|---:|---:|---:|
| 70M | 19–25 | 11/12 | 11 | 31.89 | 1.093 | 56 |
| 70M | 46–52 | 12/12 | 12 | 40.52 | 1.788 | 9 |
| 70M | 105–111 | 12/12 | 12 | 69.78 | 4.155 | 3 |
| 160M | 19–25 | 5/12 | 3 | 111.45 | 1.002 | 229 |
| 160M | 46–52 | 5/12 | 5 | 150.95 | 1.057 | 30 |
| 160M | 105–111 | 2/12 | 2 | 253.37 | 1.002 | 9 |

Warm ratios summarize every matched binding, including fallback cases
where the two routes execute the same layout. The projected break-even
divides each eligible schema's median setup by its mean paired warm
time saving and rounds up. Setup excludes checkpoint and tokenizer
loading and includes prompt compilation and eight native forwards.
Warm time includes tensor preparation and protected execution and
excludes connection setup and independent exactness checks.
The projections assume enough repeated requests with a stable mixture
of bindings and costs. Since per-schema warm data has four pairs, the
shortest 160M tier has especially noisy projections. Only three of its
five admitted schemas give positive mean savings. The finite projections
there range from 135 to 1,296 requests. The other two have no observed
break-even. All rejected schemas incurred setup in the experiment.

The archived summary records the full timing decomposition, selected
layout and raw digests. The independent verifier checks against the
canonical matched inputs, confirms all eight model decisions per
certificate and recomputes protocol bytes and projected counts. These
are finite-domain, one-host measurements using two checkpoints in
one model family.



## Initial 128-example SST-2 diagnostic

The run uses 128 rows sampled without replacement from the official
872-row development split, with 64 rows per label and a fixed seed of
20271122. The dataset ZIP and dev.tsv have pinned SHA-256 digests.
SST-2 is public, and the experiment treats each review as the private
input for the protocol. The runner stores sample IDs and hashes.
It does not redistribute the source text.

Two fixed zero-shot layouts move a public note while leaving the final
sentiment query and label set unchanged. On every pair, lengths of
42–89 tokens, token inventories and private tokens match. The
approved prefix grows from 18 to 33 tokens. One attention head's QK scores
and a two-row output projection use the same dealer-assisted 2PC backend as
the other Pythia trials. The result records 512 TCP executions in total
over the two checkpoints and layouts, with exact ring checks and
agreement with each layout's native FP32 prediction throughout.

| Model | Original correct | Rewritten correct | Same decisions | Correct lost | Correct gained |
|---|---:|---:|---:|---:|---:|
| 70M | 65/128 | 68/128 | 113/128 | 6 | 9 |
| 160M | 80/128 | 94/128 | 102/128 | 6 | 20 |

Mean QK peer openings fall from 1,990,368 to 907,968 bytes for either
checkpoint. The independent check verifies all 256 prompt layouts
against the official ZIP, re-tokenizes them and recomputes stage bytes.
The two models use the same tokenized inputs. A pair-level descriptive
bootstrap gives accuracy changes of 2.34 percentage points for 70M
(95% interval [-3.13, 8.59]) and 10.94 points for 160M
(95% interval [3.13, 18.75]). For 70M, correct negative predictions rise
from 55 to 63 of 64, whereas correct positives fall from 10 to 5 of 64.
In the 160M result the direction of the accuracy change
reverses relative to synthetic field copying. The structured-record
gate certifies a specified finite binding domain. This experiment
characterizes prompt sensitivity on a fixed SST-2 validation sample
and does not certify unrestricted review text.

Sources: [SST](https://aclanthology.org/D13-1170/),
[GLUE](https://aclanthology.org/W18-5446/),
[official distribution ZIP](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip).

## Full SST-2 development split

The manuscript uses all 872 official examples in source order, with 428
negative and 444 positive labels. The prompts, model checkpoints and TCP
protocol are identical to the earlier diagnostic. Re-tokenization from
the pinned original split reproduces 1,744 prompt layouts. The paired
prompts have lengths 39–96, equal token multisets and identical within-pair
private token spans. Both checkpoints use an 18-token original approved
prefix and a 33-token rewritten prefix. All 512 rows shared with the
earlier archive match on prompt fingerprints, predictions, ring status
and communication counts.

| Checkpoint | Original / rewrite FP32 correct | Wrong to right | Right to wrong | FP32 agreement of protected runs |
| --- | ---: | ---: | ---: | ---: |
| Pythia-70M | 441 / 452 of 872 | 48 | 37 | 1,742 / 1,744 |
| Pythia-160M | 549 / 649 of 872 | 132 | 32 | 1,744 / 1,744 |

All 3,488 TCP forwards pass the exact fixed-point ring checks. The two
floating-point decision differences are Pythia-70M on row 521 in the
original layout and row 471 in the rewritten layout. Both are negative
examples with absolute native label margins below 0.0005. The original
layout changes from a positive prediction to a negative prediction at
row 521. The rewritten layout changes in the opposite direction at row
471. The resulting protected correct counts are 442 and 451 on 70M.
The 160M predictions agree throughout. These floating-point exceptions
are outside the typed-record gate's input domain.

The 70M FP32 layout change is +1.26 percentage points with a descriptive
paired-bootstrap 95% interval of [−0.80, +3.33]. Correct positive
predictions decline from 58 to 31 of 444. The majority baseline is 444 of
872. The 160M change is +11.47 points with interval [+8.72, +14.33].
Both Pythia models reduce mean measured QK peer payload from about
1.994 MB to 0.909 MB. Results are from one fixed zero-shot prompt pair.

### Plaintext Qwen3 comparison

The archived Qwen3-0.6B checkpoint uses the same literal prompts and
decisions over two fixed single-token label continuations. The test runs
1,744 native CPU FP32 forwards over the same 872 rows. Independent
re-tokenization reproduces all prompt, token multiset and private-span
hashes and both actual label continuations. Paired lengths, token
inventories and private spans match within Qwen3. Its public prefixes
are 17 and 31 tokens. The private-span calculation conservatively
includes a token containing the following line break on 853 examples.

Native correct counts change from 439 to 444, with 15 improvements and
10 regressions. The downstream change of +0.57 points has a descriptive
paired-bootstrap 95% interval of [−0.57, +1.72]. The rewritten layout
reaches the majority baseline. These results quantify the effect of
layout under this fixed prompt in a second architecture. The protected
traffic and fixed-point comparisons apply to the Pythia checkpoints.
The identical 128-row subset is retained in the Qwen3 summary for
comparison with the preliminary Pythia diagnostic.
[Qwen3 technical report](https://arxiv.org/abs/2505.09388)
