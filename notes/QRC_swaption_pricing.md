# Problem breakdown

## 1. Understanding the Asset: What is a Swaption?
Imagine you are a company that took out a loan with a floating interest rate (it changes every month). If rates go up, your loan gets expensive. You want insurance.

The Swap: A contract where you trade your "Floating" rate for a "Fixed" rate.

The Swaption: An option to enter that swap later. It gives you the right, but not the obligation, to lock in a specific interest rate (the Strike) at a future date.

The Two Dimensions of the data: The dataset isn't just a list of prices; it’s a surface. Every swaption is defined by two time-based variables:
- Maturity (Expiry): When do you have to decide if you want the swap? (e.g., in 2 years).
- Tenor: If you say "yes," how long does the swap last? (e.g., for 5 years).

In the dataset, we have columns for Maturity, Tenor, and Price. Because these are all connected (a 2-year option on a 5-year swap is mathematically related to a 3-year option on a 4-year swap), the prices form a smooth "surface." The job is to predict how this whole surface moves over time.

## 2. The Task: Prediction vs. Imputation
We have a Time-Series of Surfaces.

Task 1 (Future Prediction): You have the "weather" (swaption prices) for Monday through Friday. Predict the weather for next Monday.

Task 2 (Imputation): You have the data for Wednesday, but some specific grid points (e.g., the 2Y into 5Y price) are missing. You need to use the surrounding data to "fill in the blanks."

## 3. Why Quantum Reservoir Computing (QRC)?
The problem statement mentions QRC. To understand this, think of a pool of water:

Input: You throw a "stone" (the data) into the water.

The Reservoir: The water ripples in complex, non-linear ways. In this case, a fixed quantum circuit acts as the water. It turns your simple input data into a very complex quantum state.

The Readout: You observe the ripples and use a simple classical model (like Linear Regression) to map those ripples to a price.

Why use it? Quantum systems are naturally good at creating complex, high-dimensional "ripples" (features) that classical computers struggle to simulate. Since the reservoir (the quantum circuit) is fixed and not trained, you avoid the "Barren Plateau" problem (a common issue where quantum neural networks stop learning).

# The how:
## 1. The Core Strategy: Quantum Reservoir Computing (QRC)
We are using a **hybrid model** where a quantum system (the reservoir) does the heavy lifting of finding complex patterns in financial data, while a classical computer handles the final prediction.
*   **The Big Advantage:** Unlike standard Quantum Neural Networks, we **do not train the quantum circuit**. The internal "quantum sloshing" is random and fixed. This avoids common training hurdles like "barren plateaus" (where the model stops learning) and makes training much faster for a hackathon timeline.
*   **Efficiency:** It is specifically designed for noisy, near-term hardware (like the Quandela QPU) because it uses the natural, "scrambled" dynamics of the qubits to extract features.

## 2. How the "Memory" Works
Swaption prices depend heavily on historical trends ("memory effects"). Our QRC model needs to handle this through a three-step **iterative loop**:
*   **Input vs. Hidden Qubits:** We divide our modes into two groups. **Input qubits** receive the raw financial data for a specific day, while **hidden qubits** act as the "memory".
*   **The Iteration:** For example we feed in data from _t-3_, let it scramble across the qubits, and then discard the input qubits while keeping the **hidden qubits**. We then feed in data from _t-2_ and _t-1_ into new input qubits.
*   **Persistence:** This process allows the quantum state to carry information from the past forward, allowing the model to "remember" previous market conditions when making a prediction for the future.

## 3. Data Handling: Encoding & Features
*   **Encoding (The Constraint):** Since we cannot use amplitude encoding, we must use **Phase/Angle Encoding**. We take our swaption data (prices, maturities, tenors), scale them between -π and +π, and use those numbers as **rotation angles (RY gates)** for the photons.
*   **Key Features:** Research shows that **lagged volatility RV_{t-1}** is the most important factor. Other critical features include Market Excess Returns (MKT), Short-Term Reversal factors (STR), and interest rate indicators like the Default Spread (DEF).
*   **Feature Selection:** Because we are limited to **20–24 modes**, we can't use every piece of data. We should use a **forward selection** process: start with one feature, see if it helps, then add the next best one until the model's performance peaks.

## 4. Evaluation: How to Win
To produce a high-quality "Advanced" solution, we need to look beyond accuracy:
*   **The "Dangerous" Error:** In finance, underestimating risk is much more dangerous than overestimating it. 
*   **QLIKE Metric:** Instead of just using Mean Squared Error (MSE), we should use the **Quasi-Likelihood (QLIKE)** metric. It specifically **penalizes under-predictions** more heavily, which is essential for accurate derivative pricing.
*   **Imputation (Level 2):** Since some data is missing, the reservoir’s ability to map inputs into a high-dimensional space will help the classical layer "infer" the missing values based on how they correlate with known points on the pricing surface.

## Team Action Plan
1.  **Select the Subset:** Identify the 7–10 most important swaption data points (Maturity/Tenor pairs) to fit within our mode limits.
2.  **Map the Loop:** Set up the iterative encoding-evolution-trace loop to capture the "memory" of interest rate shifts.
3.  **Classical Readout:** Train a simple classical model to take the final quantum measurement and output the full 2D pricing surface.