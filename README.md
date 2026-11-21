# Cross-Layer Privacy Compilation — BlockSys 2027

**Paper:** *From Task Rewriting to Selective Two-Party LLM Inference*
**Main file:** [`main.tex`](main.tex), official BlockSys 2027 `llncs.cls` v2.25, with figures in `figures/`.
**Status:** research manuscript with causal diagnostics, a trusted-dealer
arithmetic ablation, two protected Pythia operators and an executable typed
rewrite that selects their operating policy for synthetic records.

## Contents

- `main.tex`: submission manuscript. It juxtaposes the results from PriGraph and ShareRoute, with method/assumption boundaries and a new CPU causal-privacy test.
- `experiments/local_lm_quality.py`: pretrained Pythia-70M end-position dependence and public-KV reuse diagnostic, with results in `experiments/results/pythia_probe.json`.
- `experiments/causal_privacy_cpu.py`: pure-CPU reproducible PyTorch diagnostic and deterministic arithmetic contract tests.
- `experiments/results/summary.json`: summary of observed tests and hardware.
- `experiments/results/aggregate.csv`: per-placement/per-private-fraction means and counts.
- `experiments/results/case_results.csv`: 720 per-case position-dependency measurements.
- `experiments/loopback_2pc.py`: paired rewrite × island experiment and algebraic baseline with two worker processes over TCP loopback. Per-case and grouped results are under `experiments/results/loopback_*.csv`.
- `experiments/rewrite_island_cpu.py`: earlier deterministic in-process ring emulator, retained for regression checking.
- `EVIDENCE_AND_REVIEW_AUDIT.md`: source-derived results, confounders, and experiments still needed before claiming top-conference-level systems evidence.

The original PriGraph and ShareRoute PDFs are source manuscripts supplied by the authors and are not included in this repository.

## Reproduce the new CPU diagnostic

Requires Python 3.10+, PyTorch; CPU is sufficient. No GPU, MPC runtime, external model weights or private records are needed.

```bash
python experiments/causal_privacy_cpu.py --seeds 4 --trials 12 \
  --length 16 --tol 1e-6 --output experiments/results
```

The reported run used PyTorch `2.10.0+cpu` and 5 available AMD EPYC vCPUs (the script limits compute threads to 2). There are **720 paired executions** (4 model seeds, 12 trials, 5 secret fractions, 3 position patterns), **11,520 output-position comparisons**, no observed disagreements with conservative full-causal dependency closure, and **2,133 changed activations** misclassified by naive input-token privacy labels. The synthetic arithmetic guard ran 1,200 contract tests (800 accepted public formulas, 400 local fallback) and refused 800 wrong-version cache lookups. These counts are unit/functional checks; **not** proof of information-theoretic secrecy or measured MPC speedups.

**Interpretation caveat:** the `private suffix`, `private prefix` and `interleaved` cases compare *positions* at equal private input fraction. A downstream nominally public token can depend on a prior secret via causal attention. The experiment exercises randomly initialized two-layer d=32 attention; it does not reproduce ShareRoute's MP-SPDZ arithmetic, GPT-2 weight values, or PriGraph's Qwen2.5 task metrics.

## Reproduce the pretrained plaintext probe

Install PyTorch 2.8.0 and Transformers 4.52.4. Download or locate the
`EleutherAI/pythia-70m-deduped` snapshot at commit
`e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`. The script requires the
safetensors version and checks its weight digest before loading.

```bash
python3 experiments/local_lm_quality.py --checkpoint /path/to/snapshot --output /tmp/pythia_probe.json
```

This CPU test compares next-token probability vectors on synthetic records
under two private-field placements. It also checks prefix KV reuse against
complete plaintext prefill.

## Reproduce the Pythia operator experiments

Use the same pinned checkpoint as above and NumPy 2.0.2:

```bash
python3 experiments/pythia_secure_head.py --checkpoint /path/to/snapshot --output ./head-rerun
python3 experiments/pythia_qk_2pc.py --checkpoint /path/to/snapshot --output ./qk-rerun --pairs 24 --warmups 4 --transport tcp
```

To compose both boundaries in one forward:

```bash
TOKENIZERS_PARALLELISM=false python3 experiments/pythia_composed.py --checkpoint /path/to/snapshot --output ./composed-rerun
```

This run injects reconstructed QK scores before the trusted client's native
causal softmax and uses the resulting final state for the two-party head.
It evaluates five policies on each of 96 prompts. All policies use complete
prefill. The two `pythia_composed_*` archives report per-case counters,
FP32 and fixed-point matching, per-prompt score digests and timing.

The client constructs both hidden states and distributes only fresh additive
shares to two TCP loopback processes. A full output mode hides the candidate
IDs from workers. The four-row mode reveals the IDs under an explicit
public-pool policy. Archived per-prompt and grouped evidence is in
`experiments/results/pythia_head_cases.csv` and `pythia_head_summary.json`.
Report application payload and scoped timing separately. Pretrained weights
are obtained from the pinned checkpoint and are not included in the repo.
The attention-score test uses one Pythia head in the second layer. The
trusted client captures post-RoPE inputs and supplies fresh Beaver triples.
Both arms evaluate the same causal triangle. The selective arm exposes
only the approved prefix and computes mixed products on shares.
`pythia_qk_cases.csv` and `pythia_qk_summary.json` record 192 executed tiles.
Projection and QK are separate operator tests. Neither runs softmax or the
Transformer body over the two-party workers.

## Reproduce the matched Pythia rewrite study

Use the pinned Pythia snapshot and environment described above.

```bash
TOKENIZERS_PARALLELISM=false python3 experiments/cross_layer_rewrite.py --checkpoint /path/to/snapshot --output ./rewrite-rerun --transport tcp
python3 experiments/check_cross_layer_evidence.py ./rewrite-rerun/cross_layer_rewrite_cases.csv --summary ./rewrite-rerun/cross_layer_rewrite_summary.json --transport tcp
```

Twelve record schemas and four bound values per schema yield 48 matched
inputs. The rule lifts a public note over a private field when the schema
permits it. The planner minimizes the number of private QK products.
The four factorial arms and independently written public-first control
use the same four public output rows. The archive reports
protected-request elapsed time and per-schema planning time separately.
The checker recomputes traffic and paired outcomes and validates the
summary. Raw fingerprints and target IDs describe synthetic inputs.

## Reproduce the matched rewrite × island ablation

Run from the repository root with Python 3.9+:

```bash
python3 experiments/loopback_2pc.py --trials 100 --warmup 10 --output ./loopback-rerun
python3 experiments/check_manuscript_evidence.py main.pdf
```

Correctness and payload sizes should reproduce. Latencies depend on the host and scheduler. The second command requires Poppler's `pdfinfo` and `pdftotext` and a compiled manuscript. Use `python3 experiments/figure_data.py` to refresh the vector plot data after changing the archived results. CI checks the plotted data and archived causal counts, reruns the arithmetic ablation, and enforces the conference page limit. A separate local rerun reproduced the full causal results on Apple Silicon with PyTorch 2.8.0.

## Source-prototype experimental provenance

The `PriGraph` observations (Qwen2.5-7B/14B; synthetic tasks, calibrated simulations and preliminary multi-peer tests) and the `ShareRoute` observations (mini-GPT d=32 real 2PC/3PC MPC on a single machine/LAN; GPT-2 125M plaintext analysis) **were originally reported in two different PDFs**. The manuscript labels source-prototype results separately from the new arithmetic loopback measurements. In particular, a 95x cross-stage 2PC-to-3PC time ratio is **not** a comparable fixed-security-stage gain; a tool-augmented local PriGraph baseline matches 100% accuracy on explicit-formula tasks.

## Build

With a TeX installation providing `llncs.cls` and `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
python3 experiments/check_manuscript_evidence.py main.pdf
```

**BlockSys 2027** specifies a maximum of 14 Springer LNCS pages, appendix included, single-blind review. The main author block remains anonymous as in the original PDFs; replace it with the appropriate genuine author information before final submission.

## Overleaf synchronization

GitHub pushes do not automatically update an Overleaf project unless a separate synchronization is configured. In an Overleaf project connected to `YuSun798/blocksys`: **Integrations > GitHub > Pull GitHub changes**, then compile `main.tex` with pdfLaTeX. If no connection exists, import the GitHub repo into a new Overleaf project. This README does not claim an Overleaf pull happened.

## Evaluation scope

The executable planner handles an order-insensitive record rewrite, a
fixed trusted-dealer protocol and two protected operator boundaries. A
general compiler, planning with cache and heterogeneous placement costs,
full-model generation and cross-host timing remain separate evaluations.


### Reproduce the network sensitivity model

Run `python3 experiments/network_sensitivity.py --check` to regenerate and
compare the 18 scenario summaries against the archived composed-request
traces. The replay uses a four-RTT dependency term and application-byte
serialization across either a shared bottleneck or equal-capacity parallel
client links. These are modeled scenarios based on measured payloads and
loopback time.


For the controlled TCP delay experiment, run
`python3 experiments/delay_sweep.py --checkpoint /path/to/snapshot --output ./delay-rerun`.
The launcher waits at message send boundaries and measures real local
protocol executions at three nominal delays. Timing depends on the host.
Use `python3 experiments/delay_sweep.py --check --output experiments/results`
to validate coverage, paired output and payload stability, and summaries.
