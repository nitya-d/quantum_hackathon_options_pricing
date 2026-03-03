# Track B: Quantum Machine Learning for Options Pricing

Feel free to check out the website we made, detailing our process and results:
https://oliviarosel.github.io/fanot_qubits_webpage/

## Final Results


**Level 1 — Swaption surface forecasting (primary metric: QLIKE, lower is better)**

| Model | RMSE | MAE | QLIKE | Notes |
|---|---|---|---|---|
| **Hybrid Temporal QRC (ours)** | **0.003467** | **0.002591** | **0.000636** | Single circuit + LexGrouping + Ridge α=0.1 |
| MLP Baseline | 0.011867 | 0.008495 | 0.012288 | Fairest classical comparison |
| Variational Quantum CircuitBuilder | 0.014969 | 0.009463 | 0.012683 | Best single run, high variance |
| Variational Quantum CircuitBuilder with Barren Plateau fix | 0.014574 | 0.010024 | 0.015849 | Identity-block init, zero variance |
| LSTM Baseline | 0.017637 | 0.013449 | 0.021510 | Insufficient data for recurrent model |

---

## Development Journey

### Step 1 — Classical baselines

Before touching quantum circuits, we established honest classical benchmarks under identical conditions (same data split, same QLIKE loss, same random seed):

- **MLP baseline**: replaces the quantum block with `Linear(16→32)+ReLU`. QLIKE **0.012288**
- **LSTM baseline**: 2 stacked LSTM layers (hidden=64), processes lookback as sequence. QLIKE **0.021510**

The LSTM underperformed the MLP despite being architecturally suited to time series — 489 training samples is too small for recurrent models to generalise. This established the MLP as the classical ceiling to beat.

### Step 2 — First quantum model: QuantumLayer.simple()

Initial model used `QuantumLayer.simple()` with MSE loss:
- 16 PCA components → 16-mode circuit → LexGrouping → MLP readout
- RMSE **0.01959**, QLIKE not yet computed
- Result: competitive with classical but not clearly better


### Step 3 — CircuitBuilder + QLIKE loss

Two simultaneous improvements:

**CircuitBuilder over simple API:** Rewriting the circuit with explicit `add_entangling_layer → add_angle_encoding(scale=np.pi) → add_rotations → add_superpositions` gave direct control over Perceval's phase conventions. The `scale=np.pi` on angle encoding is mandatory — without it, the encoding is incorrect and the model underperforms significantly. Result: **2× improvement** over the simple API.

**QLIKE as training loss:** QLIKE (`mean((σ_true/σ_pred)² − 2·log(σ_true/σ_pred) − 1)`) penalises underprediction more than overprediction — the financially correct behaviour for volatility surfaces. A 1% error on a 1-month vol matters far more than the same error on a 10-year vol. Best QLIKE: **0.012683** (quantum competitive with MLP at 0.012288).

### Step 4 — Barren plateaus: the variance problem

The 0.012683 result was a lucky run. Without a fixed seed, QLIKE ranged from 0.012 to 0.024 across runs — a ±50% spread. This is the **barren plateau** problem: random parameter initialisation scatters the circuit into flat gradient regions where `∂P/∂θ ≈ 0` everywhere.

Two fixes were implemented based on Grant et al. (2019):
- **Identity-block initialisation**: set all rotation parameters to θ=0.01. Circuit starts near-identity, keeping Fock outputs input-dependent with non-zero gradients.
- **Two-stage layerwise training**: freeze quantum params in Stage 1, unfreeze in Stage 2 at lower LR.

Result: QLIKE **0.015849**, perfectly consistent across runs. Trade-off: slightly above the lucky run, but fully deterministic. The variance problem was solved at the cost of some peak performance.

### Step 5 — Literature pivot: Temporal QRC (Li et al. 2025)

The barren plateau analysis revealed the deeper issue: gradient-trained quantum circuits on small financial datasets are fundamentally limited. We read Li et al. (arXiv:2505.13933) — *Quantum Reservoir Computing for Realized Volatility Forecasting* — and identified a better strategy:

**Key insight from the paper:** Use a **fixed** (non-trained) quantum reservoir with:
1. Distinct **input modes** (angle-encoded) and **memory modes** (never re-encoded, carry temporal state)
2. **Sequential timestep encoding** — each day in the lookback window fed through the same circuit in order
3. **Ridge regression** readout — analytical solution, no gradient descent, no barren plateaus

By eliminating gradient training entirely, the barren plateau problem disappears by construction.

### Step 6 — Final architecture and hyperparameter tuning

**Architecture (research_paper V2.ipynb):**
- 8 modes: **5 input + 3 memory** (following Li et al. n₂=3 hidden qubits)
- 3 photons, UNBUNCHED Fock measurement, depth-3 fixed circuit
- Single LexGrouping: Fock space → 8 grouped features per timestep
- 5 days × 8 quantum features = 40 quantum features
- Concatenated with 25 raw PCA values (classical autoregressive signal)
- Total: **65 features** fed to Ridge regression

**Ridge alpha tuning:** The default α=10 from Li et al. was found via ablation sweep to over-regularise the quantum features relative to the 25 classical PCA features. Sweeping α ∈ {0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0} identified **α=0.1** as optimal — low enough for the quantum features to contribute their nonlinear signal alongside the classical autoregressive component.

**Final result: QLIKE 0.000636 — 19× better than MLP baseline, deterministic every run.**

---

## Key Contributions

1. **QLIKE training loss** — industry-standard volatility metric as the optimization objective; documented superiority over MSE for financial forecasting
2. **CircuitBuilder benchmark** — documented 2× gain over `QuantumLayer.simple()` from explicit circuit control and correct Perceval phase conventions
3. **Barren plateau diagnosis and mitigation** — identity-block initialisation (Grant et al. 2019) eliminates run-to-run variance
4. **Temporal QRC adaptation** — photonic realisation of Li et al.'s input/memory qubit separation using MerLin CircuitBuilder; single LexGrouping replaces multi-seed ensemble
5. **Ridge alpha tuning** — ablation sweep identifying the regularisation strength that balances quantum and classical feature contributions
6. **Rigorous baseline comparison** — MLP and LSTM under identical conditions at every stage of development

---

## Team Setup & Environment

We use a Python **venv** (not conda) to avoid corporate network issues with Anaconda channels.

### Prerequisites

- Python 3.11+ installed ([python.org](https://www.python.org/downloads/))
- Git
- VS Code with the Python and Jupyter extensions

### Steps

1. **Clone the repo** (use the fork URL your team lead shared)
   ```bash
   git clone https://github.com/YOUR_USERNAME/quantum_hackathon_options_pricing.git
   cd quantum_hackathon_options_pricing
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate it**

   - **Windows (cmd / Anaconda Prompt):**
     ```bash
     .venv\Scripts\activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If `torch` fails with a DLL error on Windows, install the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), restart, and try again.

5. **Register the Jupyter kernel**
   ```bash
   python -m ipykernel install --user --name quantum --display-name "Python (quantum)"
   ```

6. **Select the kernel in VS Code**

   Open a notebook → click the kernel picker (top-right) → **Select Another Kernel** → **Python Environments** → pick **quantum** or **Python (quantum)**.

### Troubleshooting

| Problem | Fix |
|---|---|
| `numpy.core.multiarray failed to import` | `pip install "numpy<2"` (already pinned in requirements.txt) |
| PyTorch DLL error (`c10.dll`) | Install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and restart |
| Conda channels 403 Forbidden | Don't use conda — follow the venv steps above |

---

## Hackathon Resources

See [HACKATHON_RESOURCES.md](HACKATHON_RESOURCES.md) for the full hackathon overview, all track descriptions, and community links.

## Track B Study Materials

See [preparation_materials/Track_B_Quandela.md](preparation_materials/Track_B_Quandela.md) for the full reading list, SDK docs, and technical constraints.
