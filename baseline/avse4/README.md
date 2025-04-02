## Baseline model for 4th COG-MHEAR Audio-Visual Speech Enhancement Challenge

[Challenge link](https://challenge.cogmhear.org/)

## Requirements
* [Python >= 3.6](https://www.anaconda.com/docs/getting-started/miniconda/install)
* [PyTorch](https://pytorch.org/)
* [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
* [Decord](https://github.com/dmlc/decord)
* [Hydra](https://hydra.cc)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [TQDM](https://github.com/tqdm/tqdm)

## Usage

```bash
# Expected folder structure for the dataset
data_root
|-- train
|   `-- scenes
|-- dev
|   `-- scenes
|-- eval
|   `-- scenes
```

### Train
```bash
python train.py data.root="./avsec4" data.num_channels=2 trainer.log_dir="./logs" data.batch_size=8 trainer.accelerator gpu trainer.gpus 1

more arguments in conf/train.yaml
```

### Test
```bash
python test.py data.root=./avsec4 data.num_channels=2 ckpt_path=pretrained.ckpt save_dir="./eval" model_uid="./avse4" 

more arguments in conf/eval.yaml
```
  