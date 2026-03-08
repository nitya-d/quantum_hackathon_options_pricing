# Track B: Quantum Machine Learning for Options Pricing

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

---

## Noise Analysis — Interpreting QRC vs LSTM under Hardware Noise

### Test results (6 days × 224 instruments, QLIKE metric)

| Model | Test QLIKE | vs LSTM |
|:------|:----------:|:-------:|
| MLP | 0.003921 | +263% |
| LSTM | 0.001081 | — |
| **QRC (perfect)** | **0.000879** | **−18.7%** |
| QRC (pessimistic) | 0.001110 | +2.7% |
| QRC (optimistic) | 0.001147 | +6.1% |
| QRC (hardware) | 0.001286 | +19.0% |

### Is QRC (hardware) being 19% worse than LSTM statistically significant?

**No.** With only 6 test days, there is far too little data for statistical significance. A paired test on 6 observations has almost no power. The absolute gap is just 0.000205 (0.001286 vs 0.001081). From the per-day breakdown, LSTM beats QRC_hw on 4/6 days but QRC_hw wins on the later days in the horizon. This looks more like sampling noise than a robust difference. At N=6 we can say "competitive" but not much more.

### Why is optimistic noise worse than pessimistic?

This is counterintuitive — the optimistic profile has objectively better hardware parameters (higher brightness=0.30, higher indistinguishability=0.96, lower phase noise=0.02). But "better hardware simulation" ≠ "better downstream QLIKE" because:

1. **Higher brightness → more detected photons → different output distribution shape.** The Ridge readout was trained on *perfect* features. Noisy features with more photons detected shift the feature space in a different direction than fewer photons. Neither is "closer to perfect" — they're just *different* distortions.

2. **The gap is tiny**: 0.001147 vs 0.001110 = 3.3%. With 6 test days and 3 ensemble seeds, this is well within sampling noise.

3. **Noise model parameters interact nonlinearly** through the Fock space distribution. There is no reason to expect monotonic degradation across all parameters simultaneously.

### Bottom line

The ordering (pessimistic < optimistic < hardware) is **not meaningful** at this sample size. All three noisy profiles cluster at 0.0011–0.0013 — essentially the same performance band, all competitive with LSTM. The real story:

- **Perfect QRC clearly beats LSTM** (−19%)
- **Noise adds a ~26–46% penalty** vs perfect
- **That penalty puts noisy QRC roughly at parity with LSTM**
- **Pessimistic is only 2.7% from LSTM** — noise mitigation (ZNE) or larger ensembles could close the gap
