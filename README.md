# RL-HydroFM: Reinforcement-Learned Multi-Source Transductive Foundation Models for Few-Shot Remote Sensing and Water Resources Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE JSTARS](https://img.shields.io/badge/Submitted%20to-IEEE%20JSTARS%20Special%20Issue-blue)](https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/)

Official PyTorch implementation of **RL-HydroFM** (and **RL-LCTIM**), a reinforcement-learned transductive foundation model framework for Earth observation few-shot scene classification, surface water extraction, and flood inundation monitoring.

Submitted to the **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (IEEE JSTARS)** Special Issue on *"Foundation Models and Multi-Source Remote Sensing for Water Resources Monitoring, Assessment, and Management"*.

---

## 📌 Architecture Overview

```
+-----------------------------------------------------------------------------------------------+
|                                      RL-HydroFM PIPELINE                                      |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|  [Query Tile x_i] ──► Frozen Multi-Modal Foundation Backbones (GeoRSCLIP / DINOv3 / SAR)      |
|                                │                                                              |
|                                ▼                                                              |
|                  [14-Dimensional Diagnostic State Representation s_i]                         |
|                  • Normalized Zero-Shot Predictive Entropy H(y_hat_i)                         |
|                  • Top-1 vs. Top-2 Confidence Margin                                          |
|                  • Manifold Density & Candidate Similarity Statistics                         |
|                  • Cross-Modal Optical-SAR Concordance Profile                                |
|                                │                                                              |
|                                ▼                                                              |
|                  [Actor-Critic Graph Policy Network (HydroGraphPolicy)]                       |
|                  ├── Dynamic Neighborhood Size Selector: kappa_i in {1, 3, 5, 8, 12, 16}      |
|                  ├── Adaptive Optical-SAR Router: beta_i in Delta^1                           |
|                  └── Uncertainty-Calibrated Regularizer: lambda_LC,i in [0, 1]                |
|                                │                                                              |
|                                ▼                                                              |
|                  [Vectorized Closed-Form ADM Transductive Solver]                             |
|                  • Adaptive Consensus: p_bar_i = (1/kappa_i) * sum_{j in N_i(beta_i)} p_j     |
|                  • Closed-Form q-Update: q_ik^(t+1) proportional to p_ik^(1+alpha) *          |
|                                           y_hat_ik^gamma * (p_bar_ik)^(lambda_LC,i)          |
|                  • Centroid Updates W^(t+1) via Assignment Matrix Q                           |
+-----------------------------------------------------------------------------------------------+
```

---

## 🚀 Key Features

* **Policy-Driven Dynamic Neighborhoods ($\kappa_i$):** Eliminates the rigid $\kappa=5$ heuristic by dynamically allocating neighborhood sizes based on local topological density.
* **Multi-Modal Optical-SAR Sensor Gating ($\bm{\beta}_i$):** Adaptively balances optical semantics (GeoRSCLIP) and radar backscatter (Sentinel-1 SAR / DINOv3), ensuring cloud resilience.
* **Closed-Form Vectorized ADM Optimization:** Zero gradient backpropagation into the frozen foundation model backbones, yielding sub-second transductive inference.
* **Water Resources & Flood Benchmark Suite:** Standardized few-shot benchmarks on EuroSAT-Water, Kaggle Sentinel-2 Water Bodies, Sen12-Flood, and RESISC45-Water.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/farhansajid/RL-HydroFM.git
cd RL-HydroFM

# Create Python virtual environment
conda create -n hydrofm python=3.10 -y
conda activate hydrofm

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

---

## 📊 Quick Start & Reproducing Paper Results

### 1. Run Complete Benchmark Experiments on GPU
```bash
python experiments/run_water_experiments.py --datasets eurosat_water sentinel2_water sen12_flood resisc45_water --shots 1 2 4 8 16 --seeds 1 2 3 4 5
```

### 2. Run Cloud Degradation & Multi-Sensor Ablation
```bash
python experiments/run_modality_ablation.py
```

### 3. Run Policy Interpretability Analysis
```bash
python experiments/run_policy_analysis.py
```

### 4. Generate Publication Plots & LaTeX Tables
```bash
python experiments/generate_plots_and_tables.py
```

---

## 📁 Repository Structure

```
RL-HydroFM/
├── models/                  # Geospatial foundation backbones & multi-modal encoders
│   ├── __init__.py
│   └── backbones.py
├── rl_core/                 # Actor-Critic policy, transductive environment & PPO agent
│   ├── __init__.py
│   ├── hydro_env.py
│   ├── graph_policy.py
│   └── ppo_agent.py
├── solvers/                 # Vectorized ADM transductive solver & baselines
│   ├── __init__.py
│   ├── rl_transductive_solver.py
│   └── baseline_solvers.py
├── datasets_loader/         # Water resources & Earth observation benchmark suite
│   ├── __init__.py
│   └── water_benchmarks.py
├── experiments/             # Experiment runners, ablation sweeps & plot generators
│   ├── run_water_experiments.py
│   ├── run_modality_ablation.py
│   ├── run_policy_analysis.py
│   └── generate_plots_and_tables.py
├── caches/                  # Cached features, results summaries & checkpoints
├── DATASETS.md              # Detailed dataset descriptions
├── requirements.txt
└── README.md
```

---

## 📖 Citation

```bibtex
@article{zaheer2026hydrofm,
  title   = {Reinforcement-Learned Multi-Source Transductive Foundation Models for Few-Shot Remote Sensing and Water Resources Monitoring},
  author  = {Zaheer, Ahmad Nawaz and Farhan, Muhammad},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  note    = {Special Issue on Foundation Models and Multi-Source Remote Sensing for Water Resources Monitoring, Assessment, and Management}
}
```
