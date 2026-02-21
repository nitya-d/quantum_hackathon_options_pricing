# Track B: Finance & QML - Preparation Guide

**Sponsor:** Quandela

## Required Tools & Technical Constraints

**Primary SDK:**
*   **[MerLin](https://merlinquantum.ai/)** by Quandela
    *   **Installation:** `pip install merlinquantum`

**Crucial Technical Constraints:**
As you design your QML architectures, you must strictly adhere to the following hardware and simulation limits:
*   **In Simulation:** Your models can utilize up to **20 modes** and **10 photons**.
*   **On Quandela’s QPU:** Your models can utilize up to **24 modes** and **12 photons**. 
    *   *🚨 Hardware Limitation Warning:* The QPU currently **does not support** amplitude encoding or state injection. You must plan your data encoding strategy accordingly!

***

## Study Resources (Recommended Reading)

To ensure participants from all backgrounds—whether you are an ML expert or a quantum physicist—can engage effectively, we have structured the provided learning materials from foundational concepts to applied QML models. 

### 1. Foundations: Machine Learning & PyTorch
*   **[Colab Notebook on PyTorch:](https://colab.research.google.com/drive/1PzVEdag-hmRr2kBq2Nu1ckfERw0cNfF5?usp=sharing)** An initiation notebook covering core Machine Learning concepts and basic PyTorch usage. Start here if you are new to classical ML.

### 2. Framework: Mastering MerLin
*   **[Training Center (Quantum Computing Hackathon):](https://training.quandela.com/)** Step-by-step lessons specifically tailored for this challenge. It bridges the gap between ML and quantum concepts to give you the exact knowledge needed to tackle the prompt.
*   **[Ready-to-run MerLin Tutorial:](https://drive.google.com/file/d/1dnD68kCWNFgmMn5xuO1Mhmg1T5Pgb24x/view)** A practical, step-by-step tutorial introducing the MerLin environment so you can start coding immediately.
*   **[MerLin Documentation:](https://merlinquantum.ai/index.html)** The official reference guide for the MerLin framework. Keep this open in a tab while you build!

### 3. Applied QML & Project Inspiration
*   **[Quantum Reservoir Computing for Realized Volatility Forecasting:](https://arxiv.org/pdf/2505.13933)** A concrete example of a QML model applied to a similar financial forecasting subject. This will be highly relevant to your Option Pricing task.
*   **[Paper](https://arxiv.org/abs/2510.25839) & [Code](https://github.com/Quandela/HybridAIQuantum-Challenge) from the Perceval Quest Hackathon:** A research paper detailing the top approaches that made it to the final phases of Quandela's first hackathon. Reviewing this (along with its associated **GitHub repository** containing the code) will give you a major advantage in understanding what makes a winning submission.
