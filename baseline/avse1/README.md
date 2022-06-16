# Baseline model for 1st COG-MHEAR Audio-Visual Speech Enhancement Challenge (AVSE)

[Challenge link](https://challenge.cogmhear.org/)

## Requirements
* Python >= 3.5 (3.6 recommended)

You can install all requirements using

```bash
pip install -r requirements.txt
```

## Usage
Update config.py with your dataset path
Try `python train.py --log_dir ./logs --a_only False --gpu 1 --max_epochs 15 --loss stoi` to run code.

