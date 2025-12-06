# MTQ-Eval: Multilingual Text Quality Evaluation

Paper: MTQ-Eval: [Multilingual Text Quality Evaluation for Language Models](https://arxiv.org/abs/2511.09374), accepted at IJCNLP-AACL 2025

This repository contains the implementation of **MTQ-Eval**, a framework for evaluating text quality (Coherence, Fluency, Simplicity, Linguistic Acceptability) across 115+ languages using Large Language Models aligned via Direct Preference Optimization.

## Abstract

The use of large language models (LLMs) for evaluating outputs is becoming an increasingly effective and scalable approach. However, it remains uncertain whether this capability extends beyond task-specific evaluations to more general assessments of text quality, particularly in multilingual contexts. In this study, we introduce, MTQ-Eval, a novel framework for multilingual text quality evaluation that learns from examples of both high- and low-quality texts, adjusting its internal representations. To develop MTQ-Eval, we first automatically generate text quality preference data and then use it to train open-source base LLMs to align with ratings of high- and low-quality text. Our comprehensive evaluation across 115 languages demonstrates the improved performance of the proposed model. Upon further analysis, we find that this enhanced evaluation capability also leads to notable improvements in downstream tasks.

## Quick Start

### 1. Installation

```bash
git clone [https://github.com/yourusername/MTQ-Eval.git](https://github.com/yourusername/MTQ-Eval.git)
cd MTQ-Eval
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py
```

### 3. Fine-tune a base model using LoRA and DPO

```bash
python scripts/train_dpo.py
```

### 4. Evaluation

Run the trained model on input texts to get quality ratings (0 or 1).
Bash

```bash
python scripts/evaluate.py
```

# Project Structure

```text
MTQ-Eval/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── data/                   # Directory for processed datasets
├── scripts/
│   ├── prepare_data.py     # Downloads Belebele and generates DPO pairs
│   ├── train_dpo.py        # Runs DPO training loop
│   └── evaluate.py         # Inference script for text scoring
└── src/
    ├── deformation.py      # Functions for text corruption (word swapping)
    └── utils.py            # Prompt templates and helper functions
```

# Citation

```bibtex
@article{pokharel2025mtqeval,
  title={MTQ-Eval: Multilingual Text Quality Evaluation for Language Models},
  author={Pokharel, Rhitabrat and Agrawal, Ameeta},
  journal={arXiv preprint},
  year={2025}
}
```
