# Federated Self-Supervised Learning for Stress Recognition  

**Leveraging Federated Self-Supervised Approach for Stress Recognition to Mitigate Label Leakage**  
*Mohamed Benouis, Elisabeth André, and Yekta Said Can – Chair for Human-Centered Artificial Intelligence, University of Augsburg*  

!!! Note:** This paper is currently **under review** at *ACM Transactions on Accessible Computing*.  

---

## Table of Contents
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Usage](#usage)  
  - [Centralized Baseline](#centralized-baseline)  
  - [Federated Learning](#federated-learning)  
  - [Federated Learning: Ablation Studies](#federated-learning-ablation-studies)  
  - [Label Reconstruction Attack](#label-reconstruction-attack)  
- [Results](#results)  
- [Citation](#citation)  
- [License](#license)  

---

## Features
- **Centralized SSL Baselines**: SimCLR, BYOL, MoCo implementations.  
- **Federated Learning**: FedAvg, FedSimCLR, FedBYOL, FedU, and our proposed FSSL.  
- **Privacy-Preserving**: Mitigates **label leakage** risks in federated training.  
- **Transformer-based Encoder**: Captures intra- and cross-modal dependencies.  
- **Datasets**: DAPPER (pretraining), WESAD and VERBIO (evaluation).  
- **Ablation Studies**: Label scarcity, non-IID heterogeneity, and architecture variants (CNN vs Transformer).  
- **Attack Evaluation**: Label reconstruction attacks to assess leakage risks, including **LLG**, **ZLG**, and **iRLG** methods.

---

## Repository Structure
```
├── data/                     # Raw and processed datasets
│   ├── WESAD/                # Wearable Stress and Affect Detection
│   ├── VERBIO/               # Public speaking physiological responses
│   └── DAPPER/               # Ambulatory dataset for pretraining
├── src/                      # Source code
│   ├── centralized/          # Centralized SSL baselines (SimCLR, BYOL, MoCo)
│   ├── federated/            # Federated SSL (FedAvg, FedSimCLR, FedBYOL, FedU, our approach)
│   ├── attacks/              # Label reconstruction attack implementations
│   └── models/               # Transformer and CNN architectures
├── notebooks/                # Jupyter notebooks for analysis & visualization
├── results/                  # Trained models, logs, and evaluation outputs
├── figures/                  # Framework diagrams, ablations, and attack results
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## Requirements
- Python 3.8+  
- PyTorch  
- scikit-learn  
- pandas, numpy  
- matplotlib  
- tqdm  

Install via:  
```bash
pip install -r requirements.txt
```

---

## Data Preparation
1. **Download datasets**:  
   - WESAD: [Link](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)  
   - VERBIO: [Link](https://hubbs.engr.tamu.edu/resources/verbio-dataset/)  
   - DAPPER: [Link](https://www.synapse.org/#!Synapse:syn22418021/files/)  

2. **Place** raw data under `data/<dataset_name>/raw/`.  

3. **Preprocess**:  
```bash
python src/data_preprocessing.py --dataset WESAD
python src/data_preprocessing.py --dataset VERBIO
python src/data_preprocessing.py --dataset DAPPER
```  

---

## Usage

### Centralized Baseline
```bash
python src/centralized/train.py --dataset WESAD --method simclr --epochs 50 --batch_size 64
### method: BYOL, SimSiam, MoCo
```

### Federated Learning
```bash
python src/federated/fedavg.py --dataset WESAD --clients 15 --rounds 50 --local_epochs 1 --lr 1e-4
```

Proposed FSSL:  
```bash
python src/federated/fssl.py --public_dataset DAPPER --private_dataset WESAD --clients 15 --rounds 50 --lr 1e-4
```

### Federated Learning: Ablation Studies
```bash
python src/federated/ablation_study.py --dataset VERBIO --clients 55 --rounds 50 --scenario non-iid --labels 0.2
```

### Label Reconstruction Attack
Evaluate label leakage risk by reconstructing labels from gradients. This repository implements three methods:

- **LLG**: Logit-based Label Gradient attack  
- **ZLG**: Zero-shot Label Gradient attack  
- **iRLG**: Improved Representation-based Label Gradient attack  

Example usage:
```bash
python src/attacks/label_reconstruction.py \
    --dataset WESAD \
    --model transformer \
    --attack_method LLG
```
You can replace `LLG` with `ZLG` or `iRLG` to evaluate the other attacks.  

> **Note:** These attacks allow assessing privacy risks of federated self-supervised learning models beyond generic gradient inversion.

---

## Results
- **FSSL outperforms** FedAvg, FedSimCLR, FedBYOL, and FedU.  
- **Label reconstruction attacks**: FSSL reduces attack success rate by **>40%** compared to FedAvg.  
- **High performance** even with only 5–20% labeled data.  
- **Ablations** confirm robustness to non-IID splits, few labels,  and modality missing choices.  

---

## Citation
```bibtex
@article{benouis2025fssl,
  title={Leveraging Federated Self-Supervised Approach for Stress Recognition to Mitigate Label Leakage},
  author={Benouis, Mohamed and Andr{'e}, Elisabeth and Can, Yekta Said},
  journal={ACM Transactions on Accessible Computing},
  year={2025},
  note={Under review}
}
```

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

