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

### Train
```bash
python train.py --log_dir ./logs --a_only False --gpu 1 --max_epochs 15 --loss l1
```

### Model evaluation - dev set
```bash
python test.py --ckpt_path MODEL_CKPT_PATH --save_root SAVE_ROOT --model_uid baseline --dev_set True --test_set False --cpu False
```

### Model evaluation - test set
Extract `avse1_evalset.tar` to `$DATA_ROOT/test/scenes`

```bash
python test.py --ckpt_path MODEL_CKPT_PATH --save_root SAVE_ROOT --model_uid baseline --dev_set False --test_set True --cpu False
```

