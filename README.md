# Predict-Then-Commit

**Stable Prefix Commitment for Streaming Neural Machine Translation**  
Quality–Latency–Stability Optimization under Character-Level Retraction Constraints

> Streaming translation should not merely be fast. It should be stable enough to read.

`Predict-Then-Commit` is an experimental research codebase for **streaming neural machine translation (NMT)** where the system must decide **when to expose translated text to the user**. The core idea is to separate three objects that are usually conflated:

- the model's internal tentative hypothesis, $h^{(t)}$,
- the stable committed prefix, $c^{(t)}$,
- the exact visible output stream, $z^{(t)}$.

This lets us evaluate not only translation quality and latency, but also **visible text retraction**, the flicker users actually see when streaming subtitles or live translation systems revise previously displayed text.

---

## Repository Status

This repository is structured for reproducible end-to-end experimentation. It contains:

- a modular Python package under `src/ptc`,
- runnable scripts under `scripts/`,
- smoke and paper-profile configs under `configs/`,
- a fully commented experimentation notebook under `notebooks/`,
- result table templates under `results/provisional/`,
- report and presentation artifacts under `docs/`,
- basic unit tests for the metrics under `tests/`.

### Important integrity note

The result tables included in `results/provisional/` mirror the cleaned end-term presentation summary. They are included for documentation and reporting structure. Before public claims, papers, or external submissions, regenerate the tables from actual run logs using:

```bash
python scripts/evaluate.py --config configs/paper.yaml --checkpoint outputs/<checkpoint>.pt
```

Do not treat provisional numbers as a substitute for saved experiment logs. Peer reviewers are already dangerous enough without handing them a flamethrower.

---

## Project Motivation

Standard offline NMT receives the full source sentence before generation. Streaming NMT operates under partial context:

$$
x_{\leq g_t} = (x_1, \ldots, x_{g_t}), \qquad g_t \leq g_{t+1}.
$$

A conventional retranslation system may repeatedly decode from each new source prefix. This often gives high final quality, but the visible text can flicker:

```text
Source stream:       The bank will raise rates
Retranslation:       Das Ufer -> Die Bank wird -> Die Zentralbank wird die Zinsen anheben
Stable commitment:   [wait]   -> Die Bank     -> Die Zentralbank wird die Zinsen anheben
```

The model is allowed to change its mind internally. The user-visible display should not behave like a caffeinated typewriter possessed by a committee.

---

## Contributions

1. **Stable-prefix formulation** separating tentative hypotheses, committed prefixes, and visible output streams.
2. **Detokenized Erasure (DE)**, a character-level measure of visible text retraction.
3. **Character-Normalized Average Lagging (CNAL)**, a character-mass latency metric for subword-tokenized cross-lingual streaming.
4. **Policy-gradient stable commit model** that learns when to `READ` more source context and when to `COMMIT` stable translation spans.
5. **Corrected evaluation pipeline** with SacreBLEU, chrF++, optional COMET, deterministic policy rollout, true autoregressive baselines, and prefix-level CSV/JSONL logs.

---

## Formal Setup

Let the source sentence be

$$
x = (x_1, x_2, \ldots, x_n), \qquad x_i \in \mathcal{V}_{src}.
$$

At streaming step $t$, the system has read source prefix

$$
x_{\leq g_t} = (x_1, \ldots, x_{g_t}), \qquad 0 \leq g_t \leq n.
$$

The read schedule is monotone:

$$
g_t \leq g_{t+1}.
$$

The reference target sequence is

$$
y^\star = (y^\star_1, \ldots, y^\star_m), \qquad y_i^\star \in \mathcal{V}_{tgt}.
$$

### Tentative hypothesis

The translation model produces an internal tentative hypothesis:

$$
h^{(t)} = \arg\max_{y \in \mathcal{V}_{tgt}^{\ast}}
P_\phi\!\left(y \mid x_{\leq g_t}, c^{(t-1)}\right).
$$

This sequence is **not required to be monotone**:

$$
h^{(t)} \not\preceq h^{(t+1)} \quad \text{in general.}
$$

### Committed prefix

The committed prefix is the stable target sequence exposed by the system:

$$
c^{(t)} \in \mathcal{V}_{tgt}^{\ast}.
$$

It must satisfy append-only monotonicity:

$$
c^{(t-1)} \preceq c^{(t)} \quad \forall t.
$$

The policy may append a stable extension:

$$
c^{(t)} = c^{(t-1)} \oplus \Delta c^{(t)}.
$$

The new committed prefix must be compatible with the current tentative hypothesis:

$$
c^{(t-1)} \preceq c^{(t)} \preceq h^{(t)}.
$$

### Visible output stream

Let $D(\cdot)$ be detokenization from target BPE tokens to surface characters. The visible output stream is

$$
z^{(t)} = D(c^{(t)}).
$$

All stability metrics are computed over $z^{(t)}$, not over hidden decoder states.

---

## Metrics

### Detokenized Erasure

Let $\mathrm{LCP}(a,b)$ denote the longest common prefix of two character strings. The step-level visible erasure is

$$
e_t = |z^{(t-1)}| - \left|\mathrm{LCP}\left(z^{(t-1)}, z^{(t)}\right)\right|.
$$

Total Detokenized Erasure:

$$
E_{DE} = \sum_{t=2}^{T} e_t.
$$

Normalized Detokenized Erasure:

$$
NDE = \frac{E_{DE}}{\sum_{t=1}^{T}|z^{(t)}| + \epsilon}.
$$

### Character-Normalized Average Lagging

Define token character mass $\chi(v)$ after stripping tokenizer markers such as `▁`, `Ġ`, and `##`.

Source character mass read by time $t$:

$$
C_{src}(t)=\sum_{j=1}^{g_t}\chi_{src}(x_j).
$$

Visible target character mass:

$$
C_{vis}(t)=|z^{(t)}|.
$$

Corpus-level source/target character ratio:

$$
\gamma = \frac{\mathbb{E}_{(x,y)\sim \mathcal{D}}[|D(y)|]}
{\mathbb{E}_{(x,y)\sim \mathcal{D}}[|D(x)|]}.
$$

Instantaneous lag:

$$
L_{CNAL}(t)=\max(0, C_{src}(t)-\gamma C_{vis}(t)).
$$

Sentence-level CNAL:

$$
CNAL = \frac{1}{T}\sum_{t=1}^{T}L_{CNAL}(t).
$$

### Commit Delay

A zero-erasure system can cheat by waiting forever, so commit delay is measured explicitly.

Let $r(i)$ be the source read position when committed token $c_i$ first appears.

$$
D_{commit} = \frac{1}{|c^{(T)}|}\sum_{i=1}^{|c^{(T)}|}r(i).
$$

Character-weighted delay:

$$
D_{char}=\frac{\sum_i \chi(c_i)r(i)}{\sum_i\chi(c_i)}.
$$

---

## System Architecture

```text
Source stream -> BPE prefix -> Masked encoder memory
                                |
                                v
                         Autoregressive decoder
                                |
                                v
                         Tentative hypothesis h(t)
                                |
                                v
Commit policy <--- state features: decoder state, entropy, g_t, |c|, C_src, C_vis
     | READ
     v
Advance source prefix

     | COMMIT
     v
Append stable prefix -> visible output z(t)
```

The policy optimizes:

$$
R = Q(c^{(T)}, y^\star) - \lambda_E E_{DE} - \lambda_L CNAL - \lambda_D D_{commit}.
$$

where $Q$ is a translation-quality reward derived from BLEU/chrF/COMET-style signals.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/<your-username>/predict-then-commit.git
cd predict-then-commit
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Optional COMET metric:

```bash
pip install unbabel-comet
```

---

## Quick Start

### Smoke run

This runs a tiny configuration for verifying that the pipeline works.

```bash
python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoint.pt
```

### Paper-profile run

This is the intended larger experimental setup.

```bash
python scripts/train.py --config configs/paper.yaml
python scripts/evaluate.py --config configs/paper.yaml --checkpoint outputs/paper/checkpoint.pt
```

### Notebook

Open:

```bash
jupyter notebook notebooks/Predict_Then_Commit_End_to_End_Experimentation.ipynb
```

The notebook defaults to a smoke profile. For final experiments, change:

```python
RUN_PROFILE = "paper"
```

---

## Configuration

### Smoke config

```yaml
run_name: smoke
seed: 42
train_samples: 128
valid_samples: 64
test_samples: 64
max_seq_len: 48
batch_size: 8
epochs_stage1: 1
epochs_stage2: 1
wait_k: 3
lambda_erasure: 0.10
lambda_latency: 0.05
lambda_delay: 0.02
```

### Paper config

```yaml
run_name: paper
seed: 42
train_samples: 100000
valid_samples: 3000
test_samples: 3000
max_seq_len: 96
batch_size: 16
grad_accum_steps: 8
epochs_stage1: 5
epochs_stage2: 3
wait_k: 3
lambda_erasure: 0.10
lambda_latency: 0.05
lambda_delay: 0.02
waitk_values: [1, 3, 5, 7]
```

---

## Evaluation Protocol

Every system uses:

- the same tokenizer,
- the same test split,
- true autoregressive decoding,
- deterministic greedy evaluation,
- prefix-level logs,
- corpus-level SacreBLEU and chrF++ format:

```python
BLEU().corpus_score(predictions, [references])
CHRF(word_order=2).corpus_score(predictions, [references])
```

The reference list is passed as `[references]`, not `[[r] for r in references]`. A tiny bug, a huge graveyard of credibility.

---

## Baselines

| System | Description |
|---|---|
| Offline full sentence | Upper bound with full source context. |
| Immediate retranslation | Decode from every source prefix and display the full hypothesis. |
| Wait-k | Fixed lag with $k \in \{1,3,5,7\}$. |
| Local agreement-2 | Commit only after a prefix repeats for two consecutive updates. |
| Confidence commit | Commit when decoder confidence exceeds a threshold. |
| Predict-Then-Commit | Learned policy-gradient stable-prefix commitment. |

---

## Provisional End-Term Result Summary

These are the cleaned end-term presentation numbers included for documentation. Regenerate before public claims.

| System | BLEU ↑ | chrF++ ↑ | COMET ↑ | DE ↓ | CNAL ↓ | Delay ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Offline full sentence | 28.4 | 57.1 | 0.782 | — | — | — |
| Immediate retranslation | 27.1 | 56.2 | 0.758 | 31.8 | 8.1 | 1.2 |
| Wait-k, k=1 | 23.7 | 51.8 | 0.681 | 9.6 | 7.4 | 1.6 |
| Wait-k, k=3 | 25.1 | 53.9 | 0.711 | 5.7 | 10.9 | 3.1 |
| Wait-k, k=5 | 26.0 | 54.8 | 0.728 | 2.9 | 16.8 | 5.0 |
| Local agreement-2 | 26.4 | 55.3 | 0.743 | 1.6 | 12.7 | 3.9 |
| Confidence commit | 26.2 | 55.1 | 0.738 | 1.2 | 13.4 | 4.1 |
| Predict-Then-Commit | **26.9** | **55.8** | **0.751** | **0.0** | **9.3** | **3.3** |

Interpretation: Predict-Then-Commit preserves roughly 94.7% of offline BLEU while eliminating committed visible retraction and maintaining moderate latency.

---

## Ablation Summary

| Policy Variant | BLEU ↑ | DE ↓ | CNAL ↓ | Commit Delay ↓ |
|---|---:|---:|---:|---:|
| No erasure penalty, $\lambda_E=0$ | 27.2 | 24.1 | 7.6 | 1.7 |
| No latency penalty, $\lambda_L=0$ | 27.9 | 0.0 | 34.5 | 8.9 |
| No commit-delay penalty, $\lambda_D=0$ | 27.5 | 0.0 | 21.8 | 6.4 |
| Token-LCP erasure only | 26.4 | 2.8 | 10.1 | 3.5 |
| Balanced full reward | **26.9** | **0.0** | **9.3** | **3.3** |

Key lesson: erasure penalty alone teaches silence. Latency and commit-delay terms force the policy to speak early enough.

---

## Generated Outputs

A typical run writes:

```text
outputs/<run_name>/
├── checkpoint.pt
├── config.json
├── train_stage1.csv
├── train_stage2.csv
├── summary_metrics.csv
├── prefix_logs.csv
├── prefix_logs.jsonl
├── pareto_frontier.png
├── erasure_barplot.png
└── manifest.json
```

The prefix logs contain:

| Column | Purpose |
|---|---|
| `sent_id` | Sentence index. |
| `strategy` | System name. |
| `prefix_step` | Streaming step. |
| `source_prefix` | Observed source text. |
| `h_t` | Tentative hypothesis. |
| `c_t` | Committed prefix. |
| `z_t` | Visible output. |
| `step_erasure` | Character retraction at the step. |
| `step_cnal` | Character lag at the step. |
| `action` | READ / COMMIT / FLUSH. |


