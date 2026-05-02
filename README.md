# Task Arithmetic for Efficient Machine Unlearning: A Study on CNN Dynamics

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

**Authors:** Giulia Pietrangeli, Lorenzo Musso  
**Course:** Deep Learning and Applied AI (2025/2026)  
**Institution:** Sapienza University of Rome  

> **Abstract:** This work evaluates Task Arithmetic for single-class unlearning across VGG-11, ResNet-18, and MobileNetV2. While state-of-the-art metrics (ZRF, KL Divergence, MIA) validate apparent knowledge erasure, we expose critical post-ablation topological vulnerabilities: Semantic Shift and Over-forgetting. Finally, the Anamnesis Index (AIN) quantifies latent residual traces and the resulting “Rebound Effect,” revealing that true permanent amnesia remains an illusion.

## Key Findings

*   **Topology-Bound Efficacy:** Sequential architectures (VGG-11) allow for precise surgical ablation, while residual (ResNet-18) and bottleneck (MobileNetV2) topologies resist erasure, inducing "Fake Unlearning".
*   **Semantic Shift:** Unlearning does not result in random noise, but in deterministic prediction re-routing (e.g., suppressed "Ship" features reroute to "Cat" in ResNet-18 due to skip connections).
*   **The Rebound Effect:** Quantified via the Anamnesis Index (AIN), we prove that vectorial subtraction leaves latent weight-space paths, allowing rapid concept re-acquisition.
*   **Green AI Compliance:** Task Arithmetic achieves an **88.27% computational reduction** compared to full retraining from scratch, offering a scalable GDPR-compliant solution.

---

## Repository Structure

The codebase is designed as a modular, sequential pipeline. Execution order is enforced by script numbering (`01` to `15`).

```text
.
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
├── README.md                          # You are here
│
├── results/                           # Generated outputs (JSON metrics & PNG plots)
│   ├── *.json                         # Experiment results (Privacy, ZRF, Anamnesis, etc.)
│   └── plots/                         # Visual proofs (t-SNE graphs, Confusion Matrices)
│
└── src/                               # Source code
    ├── 01_train_base_models.py        # Trains base models from scratch on CIFAR-10
    ├── 02_train_experts.py            # Fine-tunes 30 Expert models (10 classes x 3 archs)
    ├── 03_test_original_performance.py# Evaluates base & expert accuracies
    ├── 04_study_task_arithmetic.py    # Core Grid Search for optimal (α, ρ) hyperparams
    ├── 05_study_baselines.py          # Baselines: Gradient Ascent & Random Labeling
    ├── 06_comprehensive_ablation.py   # Tests optimal params across all 10 target classes
    ├── 07_train_comparison_models.py  # Trains "Native Ignorance" models (Retrain baseline)
    ├── 08_privacy_evaluation.py       # Computes KL Divergence against retrained models
    ├── 09_tsne_visualization.py       # Generates t-SNE latent space plots
    ├── 10_mia_evaluation.py           # Membership Inference Attack (Entropy-based)
    ├── 11_overforgetting_airplane.py  # Quantifies collateral damage on proximal class
    ├── 12_zrf_score.py                # Zero Retrain Forgetting (JS Divergence) score
    ├── 13_anamnesis_index.py          # Measures the "Rebound Effect" (AIN)
    ├── 14_confusion_matrix.py         # Generates pre/post unlearning confusion matrices
    ├── 15_time_benchmark.py           # Computational efficiency benchmarking (O(1) vs Retrain)
    │
    └── unlearning/                    # Core modular library
        ├── __init__.py
        ├── dataset.py                 # CIFAR-10 loaders and data splitting logic
        ├── model.py                   # CNN Architectures (ResNet-18, VGG-11_bn, MobileNetV2)
        ├── surgeon.py                 # The Engine: Task Vector computation, masking, & surgery
        └── utils.py                   # Evaluation loops, seeding, and utility functions
```

---

## Getting Started

### Dependencies
Install the required Python packages using the provided requirements file:
```bash
pip install -r requirements.txt
```
*Note: The pipeline has been developed and optimized using **PyTorch** running on **Apple Silicon (MPS)**. It also supports standard CUDA GPUs.*

### Hardware Note (Apple Silicon)
The code includes specific optimizations to handle MPS architecture quirks:
*   Aggressive memory management (`torch.mps.empty_cache()`) during heavy grid searches.
*   Environment variables set in `09_tsne_visualization.py` to prevent `Segmentation Faults` during parallel t-SNE computation.

---

## Execution Pipeline

To reproduce the experiments from scratch, run the scripts in numerical order. The pipeline is highly fault-tolerant: scripts will automatically skip training/inference if the corresponding `.pth` weights or `.json` results already exist in the directories.

**Phase 1: Training & Baseline Extraction**
```bash
python src/01_train_base_models.py
python src/02_train_experts.py
python src/03_test_original_performance.py
```

**Phase 2: Surgical Unlearning & Ablation**
```bash
python src/04_study_task_arithmetic.py   # Finds optimal Alpha and Drop Percentile
python src/05_study_baselines.py         # GA and RL baselines
python src/06_comprehensive_ablation.py  # Generalization test on all 10 classes
```

**Phase 3: Ground Truth Generation (Required for Privacy Metrics)**
```bash
python src/07_train_comparison_models.py # Models trained WITHOUT the forget class
```

**Phase 4: Cryptographic & Topological Evaluation**
```bash
python src/08_privacy_evaluation.py      # KL Divergence
python src/09_tsne_visualization.py      # Latent Space plotting
python src/10_mia_evaluation.py          # Membership Inference Attacks
python src/11_overforgetting_airplane.py # Collateral damage metric
python src/12_zrf_score.py               # Zero Retrain Forgetting metric
```

**Phase 5: Anamnesis & Benchmarking**
```bash
python src/13_anamnesis_index.py         # Rebound Effect measurement
python src/15_time_benchmark.py          # Green AI time savings
```

**Phase 6: Visualization Generation**
```bash
python src/14_confusion_matrix.py        # Final semantic shift plots
```

---

## Methodology Highlight: The Surgeon

The core unlearning logic resides in `src/unlearning/surgeon.py`. The unlearning intervention is formalized as:

$$\theta_{unl} = \theta_{pre} + \alpha \cdot \text{mask}(\tau, \rho)$$

Where:
*   $\tau = \theta_{ft} - \theta_{pre}$ is the Task Vector (Difference between Expert and Base).
*   $\alpha$ is the scaling factor (negative for ablation).
*   $\rho$ is the drop-percentile for magnitude-based pruning to isolate highly semantic weights.

---
