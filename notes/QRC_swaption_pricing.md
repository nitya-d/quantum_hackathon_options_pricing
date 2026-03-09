# Problem breakdown

## 1. Understanding the Asset: What is a Swaption?
Consider a company that took out a loan with a floating interest rate (one that resets periodically). If rates rise, the loan becomes expensive. The company wants insurance.

**Swap:** A contract to exchange a floating rate for a fixed rate.

**Swaption:** An option to enter that swap at a future date — the right, but not the obligation, to lock in a specific interest rate (the strike).

**The Two Dimensions:** The dataset is not a flat list of prices; it is a *surface*. Every swaption is defined by two time-based variables:
- **Maturity (Expiry):** When must the decision to enter the swap be made? (e.g., in 2 years).
- **Tenor:** If exercised, how long does the swap last? (e.g., for 5 years).

The dataset contains columns for Maturity, Tenor, and Price. Because these are interconnected (a 2-year option on a 5-year swap is mathematically related to a 3-year option on a 4-year swap), the prices form a smooth surface. The task is to predict how this entire surface evolves over time.

## 2. The Task: Prediction vs. Imputation
The data is a time-series of surfaces.

**Task 1 (Future Prediction):** Given swaption prices for days 1â€“5, predict the surface on day 6.

**Task 2 (Imputation):** Given a surface with missing grid points (e.g., the 2Y-into-5Y price), infer the missing values from surrounding data.

## 3. Why Quantum Reservoir Computing (QRC)?
Analogy: a pool of water.

**Input:** A "stone" (the data) is dropped into the water.

**The Reservoir:** The water ripples in complex, nonlinear ways. Here, a fixed quantum circuit plays the role of the water â€” it maps simple input data into a high-dimensional quantum state.

**The Readout:** The ripples are observed and a simple classical model (e.g., Ridge regression) maps them to a price.

**Why QRC?** Quantum systems naturally produce complex, high-dimensional features that are expensive to simulate classically. Because the reservoir (the quantum circuit) is fixed and never trained, the barren plateau problem â€” where variational quantum circuits stop learning â€” is avoided entirely.

# Implementation
## 1. The Core Strategy: Quantum Reservoir Computing (QRC)
A **hybrid model** where a quantum system (the reservoir) handles nonlinear feature extraction from financial data, while a classical readout layer produces the final prediction.
*   **Key Advantage:** Unlike variational Quantum Neural Networks, the quantum circuit is **never trained**. Its internal dynamics are random and frozen. This sidesteps barren plateaus and makes the training loop purely classical (fast).
*   **NISQ Suitability:** The approach is well-suited to noisy, near-term hardware (e.g., Quandela QPU) because it leverages the natural scrambling dynamics of photonic modes to extract features â€” noise is part of the computation, not purely detrimental.

## 2. How the "Memory" Works
Swaption prices depend heavily on historical trends (memory effects). The QRC handles this through a multi-step **iterative loop**:
*   **Input vs. Hidden Modes:** The photonic modes are divided into two groups. **Input modes** receive the encoded financial data for a specific day; **hidden modes** act as memory.
*   **The Iteration:** Data from *tâˆ’3* is fed into the input modes and allowed to scramble across the circuit. The input modes are then discarded while the **hidden modes** are retained. Data from *tâˆ’2* and *tâˆ’1* are fed in sequentially in the same way.
*   **Persistence:** This process carries information from past time steps forward through the hidden-mode quantum state, enabling the reservoir to "remember" previous market conditions when producing features for the current prediction.

## 3. Data Handling: Encoding & Features
*   **Encoding:** Amplitude encoding is not available on photonic hardware, so **angle encoding** is used instead. Swaption data (prices, maturities, tenors) is scaled to [0, Ï€] and applied as rotation angles on the photonic modes.
*   **Key Features:** Literature identifies **lagged volatility RV_{tâˆ’1}** as the most predictive input. Other important features include Market Excess Returns (MKT), Short-Term Reversal factors (STR), and interest rate indicators like the Default Spread (DEF).
*   **Feature Selection:** With a limited mode budget (e.g., 10â€“24 modes), not all features can be encoded simultaneously. A **forward selection** process is appropriate: start with one feature, evaluate performance, and add the next best feature iteratively until performance plateaus.

## 4. Evaluation Metrics
*   **Asymmetric Risk:** In finance, underestimating risk is far more costly than overestimating it.
*   **QLIKE Metric:** The **Quasi-Likelihood (QLIKE)** loss (ratio − ln(ratio) − 1) penalises under-predictions more heavily than MSE, making it the standard evaluation metric for volatility forecasting in quantitative finance.
*   **Imputation (Level 2):** For missing data points, the reservoir's high-dimensional feature space enables the classical readout layer to infer missing values from their correlation with known points on the pricing surface.
