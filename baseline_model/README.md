# Baseline model for 1st COG-MHEAR Audio-Visual Speech Enhancement Challenge (AVSEC)

[![Challenge link](https://img.shields.io/badge/arXiv-2111.09642-green.svg)](https://challenge.cogmhear.org/) 

## Requirements
* Python >= 3.5 (3.6 recommended)

You can install all requirements using 

```bash
pip install -r requirements.txt
```

## Usage
Update config.py with your dataset path
Try `python train.py --log_dir ./logs --a_only False --gpu 1 --max_epochs 15 --loss stoi` to run code.

