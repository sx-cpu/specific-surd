# Neural Mutual Information Estimation for SURD

## Overview

This project addresses the limitations of SURD (Synergistic–Unique–Redundant Decomposition) in handling high-dimensional continuous variables. We propose a neural network-based mutual information estimation approach that mitigates the curse of dimensionality and avoids information loss from discretization. Additionally, we design and implement a regularized training objective function that significantly improves convergence stability.

The method has been validated on:
- Gaussian distributions with analytical solutions
- Typical causal structures (mediator, confounder, collider)
- Real-world physical scenarios (energy cascade, turbulent boundary layer)

## Key Features

- **Neural Network-Based MI Estimation**: Uses MINE (Mutual Information Neural Estimation) to handle high-dimensional continuous data without discretization
- **Regularized Training**: Implements ReMINE with regularization terms for stable convergence, plus early stopping when the estimate plateaus
- **Seeded Training**: Every MINE training (target x variable subset) gets a deterministic per-task seed, so results are reproducible and independent of the order the trainings run in
- **One Script per Experiment**: Each experiment runs MI estimation *and* SURD decomposition end-to-end
- **YAML Configuration**: Hyperparameters live in `configs/`, no code editing required
- **Multiple Causal Structures**: Supports mediator, confounder, synergistic collider, and redundant collider structures
- **Self-contained Runs**: Every invocation writes one folder, `exp/<experiment>_<timestamp>/`, holding its log, manifest (seeds/timings/values), convergence curves, MI arrays and figures — runs never overwrite each other and can run concurrently
- **Comprehensive Visualization**: Automatic generation of training curves and SURD decomposition plots

## Project Structure

```
.
├── blocks.py                   # Experiment: synthetic block structures (MI + SURD)
├── energy.py                   # Experiment: energy cascade (MI + SURD)
├── inner_outer.py              # Experiment: TBL inner/outer causality (MI + SURD)
├── pipeline.py                 # Shared MI-estimation and SURD-decomposition pipeline
├── pipeline_specific.py        # Specific (per-state) MI estimation + decomposition
├── configs/
│   ├── blocks.yaml             # Config for blocks.py (canonical, commented example)
│   ├── energy.yaml             # Config for energy.py
│   └── inner_outer.yaml        # Config for inner_outer.py
├── model/
│   ├── MLP.py                  # MINE estimator with ReMINE regularization + early stopping
│   └── specific.py             # Per-state specific-MI estimator (K heads, DV value)
├── utils/
│   ├── surd.py                 # Global SURD decomposition + information-flux plotting
│   ├── surd_specific.py        # Per-state allocation loop (port of upstream surd())
│   ├── binning.py              # Target discretization (bit-identical to np.histogramdd)
│   ├── analytic_eqs.py         # Data generators for the block structures
│   ├── datasets.py             # Subset enumeration utilities
│   ├── config.py               # YAML config loading
│   ├── loader.py               # Fast in-memory batching
│   ├── early_stop.py           # Window-averaged patience check shared by both estimators
│   ├── diagnose/
│   │   ├── ground_truth.py     # Gaussian data generator (known true MI)
│   │   └── diagnose_mine.py    # MI convergence-curve plotting
│   └── seed.py                 # Random seed management
├── assets/                     # Figures embedded in this README
├── data/                       # Generated and real datasets (gitignored)
├── exp/                        # Experiment output (gitignored)
│   ├── <experiment>_<timestamp>/  # one folder per run: run.log, config.yaml,
│   │                              #   manifest[_specific].yaml, curves/, MI_results/*.npy, surd_results/*.png
│   └── 000/                    # legacy archive
└── README.md
```

## Installation

### Requirements

```bash
pip install torch numpy matplotlib scipy pyyaml
```

A LaTeX installation (`latex` + `dvipng`) is required by the SURD plots, which
render their labels with `plt.rcParams["text.usetex"] = True`.

### Setup

1. Clone the repository:
```bash
git clone git@github.com:sx-cpu/MINE-for-SURD.git
cd MINE-for-SURD
```

2. Download real-world data from the
   [SURD repository](https://github.com/ALD-Lab/SURD/tree/main/data) into `data/`:
   - Energy cascade: `energy_cascade_signals.mat`
   - Inner-outer experiment: `inner_outer_u_z32_c1.mat`, `inner_outer_u_z32_c2.mat`,
     `inner_outer_u_z32_c3.mat`, `inner_outer_tbl_param.mat`

## Usage

Each experiment is **one script** that runs the full workflow: MI estimation for
every target, then the SURD decomposition and its figure. Run the scripts **from
the repository root** (data, log and result paths are relative to the working
directory).

### Quick start

| Experiment | Command | Output |
|---|---|---|
| Synthetic blocks | `python blocks.py` | `exp/blocks_<block>_<ts>/surd_results/synergistic_collider.png` |
| Energy cascade | `python energy.py` | `exp/energy_<ts>/surd_results/energy_cascade.png` |
| Inner-outer (TBL) | `python inner_outer.py` | `exp/inner_outer_<ts>/surd_results/inner_outer.png` |

Useful flags:

```bash
python blocks.py --block mediator               # study another causal structure
python blocks.py --target 3                     # MI only for target 3, skip SURD (fast iteration)
python energy.py --config configs/energy.yaml --target 4
```

### Configuration

All settings live in YAML files under `configs/`; edit those instead of the
code. `configs/blocks.yaml` is the canonical, fully commented example. The main
keys are:

```yaml
dataset: blocks                  # blocks | energy | diagnose (selects the loader)
block: synergistic_collider      # mediator | confounder | synergistic_collider | redundant_collider
Nt: 2_000_000                    # samples generated
transient: 10_000                # initial samples discarded
lag: 1                           # time steps between inputs and target
targets: [1, 2, 3]               # targets to estimate and decompose
input_vars: [1, 2, 3]            # candidate drivers; every subset is evaluated
seed: 42

mi:                              # MINE estimator
  batch_size: 65536
  epochs: 200                    # maximum; early stopping decides when to stop
  lr: 3.0e-4
  lambda_reg: 0.1                # ReMINE regularizer weight
  C_reg: 0.0                     # ReMINE anchor value
  device: cuda
  early_stop:                    # stop when the window-averaged epoch MI stalls
    patience: 25                 # epochs without (relative) improvement
    min_delta: 0.01              # relative improvement threshold
    window: 5                    # epochs averaged before comparing

surd:                            # SURD decomposition + figure
  nvars: 3
  bins: 100                      # histogram bins for the information-leak estimate
  nlabels: null                  # null -> keep everything above `threshold`
  threshold: -0.01
  label: Q                       # variable symbol in the axis titles
  figure: synergistic_collider    # exp/<run>/surd_results/<figure>.png
```

`configs/energy.yaml` differs in dataset-specific keys (`fname`, `nlags`);
`configs/inner_outer.yaml` adds the cycle files, the probe metadata and the
per-cycle decomposition.

### What the scripts produce

Every invocation creates one folder `exp/<experiment>_<timestamp>/` (printed at
startup, `blocks_<block>_<ts>` for the synthetic structures) holding everything:

- `run.log` — full stdout transcript; every run opens it with the command line,
  the git revision and the complete config it ran with
- `config.yaml` — copy of the source config that produced the run
- `manifest.yaml` — config, seeds, timings, values, output paths (inner-outer;
  `--mode both` adds `manifest_specific.yaml`)
- `curves/*.png` — one convergence curve per sub-training
- `MI_results/global_target_<t>[_cycle<c>].npy` — MI values, and
  `MI_results/specific_target_<t>[_cycle<c>].npy` — per-state blobs
- `surd_results/*.png` — decomposition figures
- The SURD stage reads exactly the arrays written by the MI stage of the same
  run — no "latest file" guessing.

### Inner-outer experiment (turbulent boundary layer)

`inner_outer.py` ports the original SURD application to experimental
turbulent-boundary-layer data (Melbourne HRNBLWT, Re_τ = 14,750): streamwise
velocity at an inner (z⁺ ≈ 4.33) and an outer (z/δ ≈ 0.314) wall-normal probe,
three independent cycles. MINE is trained for every cycle × target × subset (18
trainings), each cycle is decomposed separately and the contributions are
averaged (the cycle-to-cycle std is reported in `run.log` and the manifest).

```bash
python inner_outer.py                       # full run (18 seeded trainings)
python inner_outer.py --target 1            # MI stage only for one target
python inner_outer.py --surd-only           # re-decompose the latest complete run's MI into a new run
python inner_outer.py --surd-only --from inner_outer_20261007_124254  # ... from that run
```

The lag is fixed to 593 samples (ΔT* ≈ 757 plus units), the value selected in
the original study. Every invocation claims its own number, so runs never
overwrite each other:

- `exp/inner_outer_<timestamp>/run.log` — full stdout transcript
- `.../manifest.yaml` (and `manifest_specific.yaml` for `--mode both`) — config
  snapshot, per-task seeds and timings, all MI and decomposition values, output
  paths, `reused_from` when `--surd-only`
- `.../curves/` — MI convergence curves
- `.../surd_results/inner_outer[_specific].png` — decomposition figures
- `.../MI_results/global_target_<t>_cycle<c>.npy` — MI arrays

Trainings are seeded per task (`seed + 1000*(cycle-1) + 100*(target-1) + subset
index`), so a re-run reproduces the same numbers (in a new folder)
regardless of the order trainings finish in.

## Model Architecture

The MINE estimator (`model/MLP.py`) implements:

- **Network**: MLP with two hidden layers of 128 units
- **Objective**: the standard MINE (Donsker–Varadhan) lower bound with a **live**
  negative-sampling denominator, plus the ReMINE regularizer:

  ```
  loss = -(E[T] - log E[e^{T_marg}]) + lambda_reg * (log E[e^{T_marg}] - C_reg)^2
  ```

  Both the joint and the marginal scores receive gradient; the regularizer
  anchors the log-mean-exp term to `C_reg`.
- **Early stopping**: training ends once the window-averaged epoch estimate has
  not improved by more than `min_delta` (relative) for `patience` epochs
- **Evaluation**: full-data estimate averaged over 5 global permutations of `Y`
  (more accurate than batch shuffling)

### Hyperparameters

Set these in the `mi:` block of the relevant config. Values below are those used
by `configs/blocks.yaml`.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | 3e-4 |
| `batch_size` | Batch size | 65536 |
| `epochs` | Maximum training epochs (early stopping decides when to stop) | 200 |
| `lambda_reg` | ReMINE regularization strength | 0.1 |
| `C_reg` | ReMINE anchor value | 0.0 |
| `early_stop.patience` | Epochs without improvement before stopping | 25 |
| `early_stop.min_delta` | Relative improvement threshold | 0.01 |
| `early_stop.window` | Epochs averaged before comparing | 5 |

## Results

### Training Convergence

#### Gaussian Distribution

![Gaussian Convergence](assets/gaussian_convergence.png)

The MINE estimator converges to the analytical mutual information value
(`0.143675` vs `0.143841` for N=2×10⁶ samples, 0.12% error); early stopping
ends training after 47 epochs once the estimate plateaus.

#### Energy Cascade

![Energy Cascade Convergence](assets/energy_cascade_convergence.png)

Training curves show stable convergence for real-world turbulence data; the
curves are from the full four-variable joint MI (Σ₁…Σ₄ → Σ₁⁺).

### SURD Decomposition

#### Energy Cascade Causal Structure

![Energy Cascade SURD](assets/surd_energy_cascade.png)

The SURD decomposition reveals the unique and redundant information flow in the
energy cascade system across four variables (Σ₁, Σ₂, Σ₃, Σ₄): causality is
dominated by the unique contribution of the same scale and by redundancy across
adjacent scales. The information leak (fraction of each target's information
not explained by the observed scales) is 7.6% / 65.3% / 46.5% / 28.2% for
targets Σ₁…Σ₄.

#### Block Structures

![Synergistic Collider SURD](assets/surd_synergistic_collider.png)

Synergistic collider: the collider (Q₁) is driven by strong synergistic
information `S₂₃` from both parents, while Q₂ and Q₃ are each driven by their
own unique past.

![Mediator SURD](assets/surd_mediator.png)

Mediator chain `Q₃ → Q₂ → Q₁`: SURD recovers the unique causalities
`U₃→₃`, `U₃→₂` and `U₂→₁`, i.e. every variable is determined by the unique
causality of the previous one. The leak for `Q₁` is the smallest (37%), as it
is affected by the weakest stochastic forcing.

![Confounder SURD](assets/surd_confounder.png)

Confounder (`Q₁ ← Q₃ → Q₂`): the confounding effect appears as the synergistic
causalities `S₁₃→₁` (0.60) and `S₂₃→₂` (0.75), together with the self-induced
unique causality of each target's own past (`U₁→₁` ≈ 0.34, `U₃→₂` ≈ 0.22). The
largest leak belongs to `Q₃`, which carries the strongest stochastic forcing.

![Redundant Collider SURD](assets/surd_redundant_collider.png)

Redundant collider (`Q₂ ≡ Q₃ → Q₁`): `Q₂` and `Q₃` are identical, and SURD
attributes the influence on `Q₁` to the redundancy of the pair (`R₂₃→₁` ≈ 0.31)
plus the synergy of the target's own past with one parent (`S₁₂→₁` ≈ 0.55,
while `S₁₃→₁` ≈ 0, avoiding double counting of the duplicated pathway). Both
parents share the same redundancy `R₂₃` and causality leak.

#### Turbulent Boundary Layer (Inner-Outer)

![Inner-outer SURD](assets/inner_outer.png)

Streamwise velocity at an inner (z⁺ ≈ 4.33) and an outer (z/δ ≈ 0.314) probe of
a turbulent boundary layer (Re_τ = 14,750, three cycles, ΔT* ≈ 757 plus units).
The outer-layer motion is dominated by its own unique past (`UO`), while the
inner-layer motion is driven by the unique causality of the outer layer
(`UO → I`, `UI ≈ 0`): causality flows top-down, and ≈ 99.6% of each signal's
information is unexplained by the observed past.

## Specific (per-state) MI — the SURD decomposition of `paper/writing-plan.md`

The global path above estimates **one scalar** `I(X; Y)` per subset and
decomposes those scalars (`utils/surd.py:surd_global`).  SURD itself is defined
*per state* of the target (Eq. 3–4): the specific MI

```
I_k(Q_S) = D_KL( p(x_S|k) || p(x_S) ) = E_{p(x_S|k)}[ log p(k|x_S) - log p(k) ]
```

with `sum_k p(k) I_k = I(X; Y)`.  `--mode specific` estimates that length-`K`
vector per subset and runs upstream's per-state allocation loop
(`utils/surd_specific.py`, ported from `SURD/utils/surd.py:91-123`).

```bash
python blocks.py --mode specific                     # per-state decomposition + figure
python blocks.py --mode both                         # both paths + side-by-side table
python blocks.py --mode specific --target 1          # MI stage only, one target
python energy.py --mode both                         # E03, same two paths
python inner_outer.py --mode both                    # E10, both paths
```

The same experiments run under `--mode specific`: the block structures
(`blocks.py`), the energy cascade (`energy.py`) and inner-outer
(`inner_outer.py`).

Estimator (`model/specific.py`): a shared encoder over `x` with one linear head
per state, trained on the Donsker-Varadhan bound with ReMINE.  Because the
optimal critic is `T*_k(x) = log p(k|x) - log p(k)` (a function of `x` alone),
the architecture is exactly a K-way classifier.  The reported value is the DV
objective itself, `E_{p(x|k)}[T_k] - log E_{p(x)}[e^{T_k}]`, read off the
trained critic — the critic is fit and the value read on the same data, so no
held-out split is needed.  And `log E[e^{T_k}]` has the exact target `0` at the
optimum, which is what `C_reg: 0` anchors.

Two things to keep in mind when comparing the modes:

- **They estimate different quantities.** The specific path bins the *target*
  into `K` states (as SURD does), while the global path keeps it continuous, so
  the data-processing inequality gives `sum_k p(k) I_k(S) <= MINE(S)`.  The gap
  closes monotonically as `specific.states` grows — measured on the synergistic
  collider with a single input: 0.22x the continuous MI at `K=5`, 0.75x at
  `K=50`, 0.91x at `K=200`.
- **The aggregate is not the discriminator.** `sum_k p(k) dI(k)` can coincide
  with the global bars even when the per-state structure is entirely different,
  so a matching aggregate shows the implementation is right but not that the
  per-state view added anything.  Use `specific.figure_states` to see it.

### Validation

`mine_benchmark.py` is the Gaussian check: it runs the specific estimator on
the MINE benchmark (componentwise-correlated Gaussians), compares its aggregate
`sum_k p_k I_k` to the analytic `-0.5 log(1 - rho^2)`, and draws the upstream
SURD histogram on the same axes (`fig_a0_rho_sweep`).  The specific path bins
the first target coordinate into `K` states; its aggregate is a lower bound on
`I(X; Y_1)` that approaches the analytic value as the state count grows.

### Where the two modes agree, and where they cannot

Aggregate bars, `--mode both`, after re-running every experiment:

| Experiment | Agreement | Notes |
|---|---|---|
| Gaussian (analytic) | 0.3–1.4% per state | the estimator's own reference |
| E02 mediator, target 1 | total 11% lower; `R23`, `R13`, `R123` within ~1% | the plateau residual lands in `S123` (see below) |
| E02 confounder, target 1 | `U1` −0.5%, `R13` +0.3%, `R123` −1.0%, `S12` −2.1% | `S13` −19% |
| E02 synergistic_collider, target 1 | dominant `S23` in both | `S23` −36% |
| E02 synergistic_collider, targets 2/3 | <1% (0.1425 vs 0.1429, 0.1430 vs 0.1435 nats) | |
| E03 energy cascade | 47–86% of the global value | 21,760 samples — see below |
| E10 inner-outer | every bar within the cross-cycle std | leak 99.81% vs 99.76%; unique outer→inner reproduced |

Three rules explain the table:

1. **Terms agree when the MI is small relative to the target's dynamic range,
   and diverge for near-deterministic relations.**  `K` target states cap the
   measurable information at `I(X; Y_binned) <= I(X; Y_continuous)`, and the
   cost grows with the MI.  Measured on the collider with a single input, as a
   fraction of the continuous MINE value: **0.22 at `K=5`, 0.40 at `K=10`, 0.75
   at `K=50`, 0.91 at `K=200`**.  That is why the confounder's small terms match
   to 0.3% while its 2.2-nat `S13` is 19% low: raise `specific.states` when the
   MI is large, do not read a large gap as an estimator failure.
2. **Flat plateaus make the attribution noise-driven.**  In the mediator `q1`
   depends on `q2` alone, so `I_k(2)`, `I_k(1,2)`, `I_k(2,3)` and `I_k(1,2,3)`
   are all `~1.97` and the truth is a tie; the ±0.2% estimation noise then
   decides how the plateau splits between `U2` and `S123`.  The redundant
   collider is the clearest case, and there the two modes *look* structurally
   different while agreeing on the physics: `q3 ≡ q2` by construction, and the
   specific MI is

   ```
   I(1)=0.232  I(2)=0.925  I(3)=0.924  I(1,2)=1.650  I(1,3)=1.653  I(2,3)=0.923  I(1,2,3)=1.654
   ```

   — both the `2,3` redundancy (`I(2)≈I(3)≈I(2,3)`) and `1` with `2`/`3` being
   synergistic (`I(1,2)≈I(1,3)≈I(1,2,3)`, all far above `I(1)`) are exactly
   right, and the global MINE has the identical shape at 0.66–0.85 of the
   values.  But the top three differ by **0.25%**, so how the 1.65 nats splits
   between `S12`, `S13` and `S123` is decided by noise — the specific run puts
   0.611 in `S13`, the global run 1.384 in `S12`.  Read the violation mass, and
   for near-ties treat the attribution as unresolved rather than as a finding.
3. **E03 is sample-starved, and the two modes fail differently.**  With 21,760
   samples the *global* MINE is not converged in training length (1.89 nats at
   300 epochs, 2.85 at 731, 3.00 in the saved run), while the specific estimate
   moves only ~4% across network capacity.  Both pass the shuffled-target
   control on real data (global 0.0009, specific −0.0007 nats), so neither is
   bias-dominated; but treat E03's specific/global *ratio* as carrying the
   global estimate's spread.

## Citation

This paper is still in preparation. If you use this code in your research, please cite:

```bibtex
@article{SURD,
  title={Neural Mutual Information Estimation for SURD},
  author={Shixing Sun, Zhihao Chen},
  journal={...},
  year={...}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Original SURD methodology and data from [ALD-Lab/SURD](https://github.com/ALD-Lab/SURD)
- MINE estimator based on [Belghazi et al., 2018](https://arxiv.org/abs/1801.04062)
# specific-surd
