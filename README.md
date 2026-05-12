# Automata_SSM_Learning — Artifact

Reproducible pipelines for learning automata from TLSF specifications using:

- Active learning (L*-style queries against the synthesized Mealy machine)
- Passive learning (state-merging from finite trace samples)
- Random-init SSM training (gradient-descent baseline)
- Drift-injection experiments: warm-started Moore SSM (test) vs. from-scratch SSM (control)

All dependencies are pre-installed in a Docker image. Pull, run, go.

---

## 1. Prerequisites

- **Docker** (Docker Desktop on macOS/Windows, Docker Engine on Linux)
- **NVIDIA Container Toolkit** *only if* you want GPU acceleration for SSM training
- **Apple Silicon (M-series Mac)**: works via Rosetta/qemu emulation but training will be slow — use a Linux/x86_64 host with a GPU if available

Disk: ~12 GB for the image.

---

## 2. Pull the image

The image is published on GHCR (linux/amd64 only — CUDA base image, no native ARM build):

```
docker pull --platform linux/amd64 ghcr.io/wfishell/automata_ssm_learning:0.1.3
```

The same image supports both CPU-only and GPU runs — the difference is the `docker run` flags.

---

## 3. Launch the container

Mount this repository into the container at `/workspaces/Automata_SSM_Learning` and start a shell. Run these from the repo root.

### GPU (NVIDIA, Linux x86_64)

```
docker run -it --gpus all --platform linux/amd64 -v "$(pwd)":/workspaces/Automata_SSM_Learning -w /workspaces/Automata_SSM_Learning ghcr.io/wfishell/automata_ssm_learning:0.1.3 bash
```

### CPU-only (any host, including Apple Silicon)

```
docker run -it --platform linux/amd64 -v "$(pwd)":/workspaces/Automata_SSM_Learning -w /workspaces/Automata_SSM_Learning ghcr.io/wfishell/automata_ssm_learning:0.1.3 bash
```

You will land at `root@<id>:/workspaces/Automata_SSM_Learning#`. All commands below are run inside the container.

---

## 4. Benchmark directories

The repo ships with two TLSF benchmark sets:

| Directory                              | Contents                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `Test_TLSF_Warm_Start/`                | Small curated arbiter set — use for quick sanity checks (minutes)        |
| `symbolic_vs_Gradient_learning_TLSF/`  | Full SYNTCOMP TLSF corpus (959 files) — use for paper-style sweeps       |

If `symbolic_vs_Gradient_learning_TLSF/` is empty, repopulate from SYNTCOMP:

```
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/SYNTCOMP/benchmarks.git /tmp/syntcomp_bench
cd /tmp/syntcomp_bench && git sparse-checkout init --cone && git sparse-checkout set tlsf && git checkout && cd -
find /tmp/syntcomp_bench/tlsf -type f -name "*.tlsf" -exec cp {} symbolic_vs_Gradient_learning_TLSF/ \;
rm -rf /tmp/syntcomp_bench
```

---

## 5. Run pipelines

Every pipeline below auto-skips unrealizable TLSFs and synthesis timeouts (>30 s) — sweeps over the full SYNTCOMP set finish without manual intervention. Use `python -u` so progress/epoch lines stream live.

### 5a. Active learning (L*)

```
python -u src/pipelines/active_learning_pipeline.py -d Test_TLSF_Warm_Start -t 1
```

Output: timestamped `active_learning_results_<TS>.csv` with one row per (file, trial): `sample_size` (membership-query count) and `accuracy` (%).

Flags: `-t N` trials per file, `-w N` worker processes (default 16), `--serial` for single-process debugging, `-o file.csv` for explicit output path.

### 5b. Passive learning (state-merging)

```
python -u src/pipelines/passive_learning_pipeline.py -d Test_TLSF_Warm_Start -t 1
```

Iteratively increases trace count (5000 → +5000 per step → max 50000) until target accuracy (100%) is reached or max is hit. Output CSV records `num_traces`, `accuracy`, `converged` per (file, trial).

### 5c. SSM random-init training

```
python -u src/pipelines/ssm_pipeline.py -d Test_TLSF_Warm_Start --samples 10000
```

Trains an SSM from scratch on traces generated from each synthesized Mealy machine. Produces two CSVs:

- `ssm_learned_results.csv` — per-file summary: `converged_epoch`, `sample_complexity`, `test_trace_acc`
- `ssm_epoch_history.csv`   — per-epoch metrics: `loss`, `test_step_acc`, `test_trace_acc`

Flags: `--samples N N N` to sweep multiple training-set sizes in one run.

### 5d. Drift-injection — control (no warm start)

```
python -u src/pipelines/drift_control_pipeline.py Test_TLSF_Warm_Start -o drift_control_results.json
```

Generates traces, injects fairness-bounded drift via `Drift_Test/drunk_arbiter_data_gen.py` (a deterministic transformer that caps any grant signal at `mean + 2` grants), then trains an SSM **from scratch** on the drifted data. Accepts a single `.tlsf` file or a directory of them.

### 5e. Drift-injection — test (warm-started Moore SSM)

```
python -u src/pipelines/drift_test_pipeline.py Test_TLSF_Warm_Start -o drift_test_results.json
```

Same data generation as 5d, but trains the **warm-started Moore SSM** (`Drift_Test/moore_ssm_training.py`) instead of a fresh model. Pair with 5d on the same input set for a direct A/B comparison.

---

## 6. Expected runtimes (rough)

Single file, single trial, on a Linux x86_64 host with a recent NVIDIA GPU:

| Pipeline             | Per file       | Bottleneck                     |
| -------------------- | -------------- | ------------------------------ |
| Active learning      | seconds        | L* queries (CPU)               |
| Passive learning     | tens of seconds| trace generation + AALpy       |
| SSM random-init      | 1–10 minutes   | gradient training              |
| Drift control / test | 5–30 minutes   | full SSM training              |

On Apple Silicon via emulation, multiply SSM-training times by ~10–30×. Use `Test_TLSF_Warm_Start` for sanity checks on Apple Silicon; reserve `symbolic_vs_Gradient_learning_TLSF` sweeps for an x86_64+GPU host.

---

## 7. Repository layout

```
src/
  pipelines/        Entry-point scripts run above
  learning/         active_learning.py, Passive_Mealy_Learning.py, ...
  models/           HOA_SSM.py, HOA_Moore.py, State_Space_Model.py, mealy_to_moore.py
  data/             Dot_Trace_Generator.py, trace_hoa_generation.py, automaton_state_tracker.py
  scripts/          train_fsm_ssm.py, Trace_Checker.py, ...
  analysis/         Plotting and post-hoc stats
Drift_Test/         moore_ssm_training.py, drunk_arbiter_data_gen.py, supporting .pt and .dot files
.devcontainer/      Learning.Dockerfile, devcontainer.json (and cpu/ variant)
```

---

## 8. Troubleshooting

- **`libspot.so.0: cannot open shared object file`** — image was built without `ldconfig` in the final stage. Fixed in `0.1.2`+; if you build locally, ensure your Dockerfile runs `RUN ldconfig` after `COPY --from=spot-builder /usr/local /usr/local`.
- **`no matching manifest for linux/arm64/v8`** on Apple Silicon — add `--platform linux/amd64` to both `docker pull` and `docker run`.
- **Pipeline appears frozen** — most stages now stream progress; if you see no output for >2 minutes, check `docker stats <id>` for CPU activity. The first SSM epoch under emulation can take several minutes before the first `Epoch ...` line appears.
- **Synthesis errors for some TLSFs** — the SYNTCOMP set contains intentionally unrealizable specs and large benchmarks that exceed the 30 s synthesis budget. Pipelines log these as `[SKIP] unrealizable` / `[SKIP] synthesis timeout` and continue.
