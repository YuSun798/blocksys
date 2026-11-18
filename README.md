# Privacy-Aware Graph Compilation for Decentralized LLM Inference

BlockSys 2027 submission draft integrating the *ideas and independently reported component results* of PriGraph and ShareRoute.

## Scientific integration
- **PriGraph task compiler:** typed rewrites (Separate-Template, Lift-Public-Subgraph, Late-Private-Fusion) expose policy-safe reusable public programs while retaining private local bindings.
- **ShareRoute operator compiler:** privacy closure and secure-island extraction protect residual private operator computation using S1 semi-honest 2PC or S2 honest-majority 3PC.
- **Cross-layer proposal:** a typed task-to-operator interface and joint objective over task rewrites, cache reuse, secure islands, protocol/cohort, WAN traffic, GPU cost, quality and offline preprocessing.
- **No new integrated benchmark is claimed**; original prototype evidence is discussed and separated.

## Read before submitting
- BlockSys 2027: Springer LNCS, **maximum 14 pages including appendix**, single-blind. Round 1 deadline Nov 30, 2026 and Round 2 Jan 31, 2027 (verify the official CFP).
- Source manuscripts are 14 pages each and have been condensed into a 13-page merged manuscript; they are preserved unchanged in the local downloadable ZIP archive, not committed here as binary PDFs.
- Both source PDFs use anonymous authors; update author metadata before single-blind submission as appropriate.
- The reported 95x MPC speedup compares **different** security models (2PC vs 3PC), not a same-security-stage gain.
- PriGraph's 100% on explicit-formula tasks is matched by the local tool-augmented control; distinguish deterministic execution from reasoning gains.
- Full-scale MPC and unified rewrite-to-MPC benchmarks have **not** been run.

## Build and sync
`main.tex` is the Overleaf root file, built with `llncs.cls` and pdfLaTeX.
`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

GitHub changes do not automatically appear in an existing Overleaf project. If linked to this repo, use **Overleaf → GitHub Sync → Pull GitHub changes**. Otherwise use **New Project → Import from GitHub**.
