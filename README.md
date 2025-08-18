# DCFT: Deconvolution Fine-Tuning

PyTorch implementation of "Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace".

## Overview

DCFT overcomes LoRA's rank-one decomposition bottleneck using deconvolution to reconstruct features in subspace incremental matrices. Achieves 8× parameter reduction compared to LoRA with superior performance.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.model.dcft import DCFT
from src.train.train import load_base_model, get_dcft_mode

# Load base model
label_map = {'label2id': {'acceptable': 1, 'unacceptable': 0}, 'id2label': {0: 'unacceptable', 1: 'acceptable'}}
tokenizer, model = load_base_model('microsoft/deberta-v3-base', label_map)

# Apply DCFT
dcft_model = get_dcft_mode(model, d=8, k=1, dropout_rate=0.1)
```

### Training

```bash
python -m src.train.train
```

## Key Parameters

- `d`: Kernel size (2=accuracy, 8=efficiency, 12=minimal params)
- `k`: Rank of low-rank approximation
- `alpha`: Orthogonal loss weight (0.1-0.2)
- `dropout_rate`: Dropout probability

## Parameter Efficiency

- **Total model**: 184M parameters  
- **DCFT trainable**: 9,252 parameters (0.005%)
- **LoRA trainable**: ~170,000 parameters (0.09%)
- **Reduction**: ~18x fewer trainable parameters

## Results

DCFT vs LoRA on CoLA (DeBERTa-v3-base):

| Method | Params | CoLA |
|--------|--------|------|
| LoRA | 0.17M | 68.6 |
| DCFT | 0.024M | 71.1 |

## Project Structure

```
src/
├── model/dcft.py      # DCFT implementation
├── train/train.py     # Training script  
└── utils/loss.py      # Loss function
```

## Citation

```bibtex
@article{zhang2025dcft,
  title={Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace},
  author={Zhang, Jia-Chen and Xiong, Yu-Jie and Xia, Chun-Ming and Zhu, Dong-Hai and Qiu, Xi-He},
  journal={arXiv preprint arXiv:2503.01419},
  year={2025}
}
```