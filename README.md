# (LVI) CIFAR-100 Conditional Image Generation with SNGAN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3117/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This project implements conditional image generation for the CIFAR-100 dataset using Spectral Normalization GAN (SNGAN). It was developed as part of the Learning Vision Intelligence (LVI) course project.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Random Seed](#random-seed)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results based on 20 superclasses](#results-based-on-20-superclasses)
- [Team Members](#team-members)
- [License](#license)

## Overview

The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes. This project focuses on conditional image generation using SNGAN architecture, developed as part of the Learning Vision Intelligence (LVI) course.

## Getting Started

### Prerequisites

- Python 3.11.7
- PyTorch 2.1.2+cu121
- CUDA-capable GPU (recommended)
- Key dependencies:
  - torch==2.1.2+cu121
  - torchvision==0.16.2+cu121
  - numpy==1.26.4
  - pandas==2.0.3
  - matplotlib==3.7.5
  - jupyter==1.0.0
  - wandb==0.18.6
  - pytorch-fid==0.3.0

For a complete list of dependencies, please refer to requirements.txt.

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Bigeco/lvi-cifar100-conditional-SNGAN.git
   cd lvi-cifar100-conditional-SNGAN
   ```
2. Set up the Conda environment:
   ```sh
   conda create -n sngan python=3.11.7
   conda activate sngan
   ```
   
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up Jupyter environment:
   ```sh
   # Create a new kernel
   python3 -m ipykernel install --user --name sngan --display-name "Python (lvi-cifar100-sngan)"
   ```

4. In Jupyter Notebook:
   - Open your notebook
   - In the top-right corner of the notebook, select the kernel "Python (lvi-cifar100-sngan)" manually.

### Random Seed

The project includes code for fixing random seeds, but by default, it runs without fixed seeds. To enable seed fixing, set the `FIX_SEED` variable to True in the code:

```python
import os
import torch
import numpy as np
import random

FIX_SEED = False  # Set to True to fix seeds

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if FIX_SEED:
    seed_everything(SEED)
```

## Usage

Simply run the provided Jupyter notebooks in sequence. Make sure you've selected the correct kernel as described in the Installation section.

## Training Details

Our best model configuration:

- **Architecture**: ResNet-based conditional SNGAN
- **Generator Optimizer**: Adam (β1=0, β2=0.999)
- **Discriminator Optimizer**: Adam (β1=0, β2=0.999)
- **Learning Rate**: 0.0002 with LambdaLR scheduler
- **Batch Size**: 64
- **Training Iterations**: Up to 1,000,000 (early stopped based on conditions)
- **Loss Function**: Hinge Loss
- **Note**: No data augmentation or gradient penalty used

## Results based on 20 superclasses

Best performance metrics:
| Model      | FID Score | Intra-FID | Inception Score |
|------------|-----------|-----------|-----------------|
| SNGAN      |   19.92   |   64.13   |   6.77 ± 0.09   |

## Team Members

This project was developed collaboratively by:

### Team Leader
- **Park Jihye**
  - GitHub: [@park-ji-hye](https://github.com/park-ji-hye)
  - Role: TBD

- **Song Daeun**
  - GitHub: [@Song-Daeun](https://github.com/Song-Daeun)
  - Role: TBD

### Team Member
- **Lee Songeun**
  - GitHub: [@bigeco](https://github.com/bigeco)
  - Role: TBD

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
