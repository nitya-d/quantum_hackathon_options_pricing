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
