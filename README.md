# TG-SMN: Thermodynamically-governed Sparse Memory Networks

This repository contains the experimental codebase for **TG-SMN**, a project investigating **continual learning** in language models using **sparse memory networks** (Mixture-of-Experts) controlled by a **Thermodynamically-Integrated Update Rule (TIUR)**.

The system features a **Hierarchical Sparse Expert** architecture where a learned or fixed controller dynamically modulates hyperparameters (like sparsity $k$, replay ratio, and routing noise) based on thermodynamic signals (e.g., dissipation/loss gradients) to balance stability and plasticity during continual learning tasks.

## Key Components

-   **Models (`tg_smn/models`):**
    -   **Transformer LM:** Standard dense Transformer baselines.
    -   **Sparse Memory LM:** A hierarchical Mixture-of-Experts (MoE) Transformer. It employs a two-stage routing mechanism (Group $\to$ Expert) to efficiently access a large memory of low-rank experts.
-   **Controllers (`tg_smn/controllers`):**
    -   **Learned Controller:** An RNN-based meta-learner optimized via Reinforcement Learning (PPO-like) to adaptively set system hyperparameters (sparsity, noise, replay) in real-time.
    -   **Fixed Controller:** Static baseline strategies.
-   **Environments (`tg_smn/envs`):**
    -   **Permuted WikiText-2:** A continual learning benchmark where the vocabulary is permuted or drifted over task sequences.
    -   **Multi-Domain:** A sequence of distinct text domains (e.g., WikiText, PTB, AGNews) to test domain adaptation and forgetting.
-   **Training (`tg_smn/train`):**
    -   Supports Experience Replay, Fisher Information Matrix (FIM) regularization, and standard optimization loops.

## Usage

The primary entry point for all experiments is the **Jupyter Notebook**:

`TG_SMN_Run.ipynb`

This notebook orchestrates the entire workflow:
1.  **Setup:** Installs dependencies and configures the environment.
2.  **Configuration:** Defines model, data, and controller hyperparameters.
3.  **Execution:** Runs training loops, baselines, and parameter sweeps.
4.  **Analysis:** Loads checkpoints, visualizes metrics (loss, expert utilization), and compares controller strategies.

## Installation & Quickstart

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```
    *Note: Designed for usage in Google Colab; local setups may require PyTorch installation matching your CUDA version.*
3.  **Run the Notebook:** Open `TG_SMN_Run.ipynb` and execute cells sequentially.

## Directory Structure

| Path | Description |
| :--- | :--- |
| `TG_SMN_Run.ipynb` | **Main Experiment Notebook** (Entry Point) |
| `tg_smn/` | Source code package |
| ├── `models/` | Transformer architectures (Dense & Sparse MoE) |
| ├── `controllers/` | Meta-learning controllers (Learned RNN & Fixed) |
| ├── `envs/` | Data loaders for Continual Learning benchmarks |
| ├── `train/` | Training loops, replay buffers, and logging |
| ├── `sweep.py` | Hyperparameter grid search utilities |
| └── `config.py` | Dataclasses for all configuration schemas |

## Notes

-   **Datasets:** The multi-domain environment leverages Hugging Face Datasets (using Parquet where available) to avoid script-loading security restrictions.
-   **Outputs:** Experiment artifacts (metrics, checkpoints, configs) are saved to a specified output directory (e.g., Google Drive in Colab).