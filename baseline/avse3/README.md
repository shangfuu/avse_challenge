
# Baseline model for the 3rd COG-MHEAR Audio-Visual Speech Enhancement Challenge

[![Challenge Registration](https://img.shields.io/badge/Challenge-%20Registration-blue.svg)](https://challenge.cogmhear.org/#/)
[![Try In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17EEK6Q5hbCwf1rNwaZAytdAiC5aK32vI?usp=sharing)


## Requirements
* Python 3.6+
* [PyTorch 2.0+](https://pytorch.org/get-started/locally/) or [Tensorflow 2.0+](https://www.tensorflow.org/install)
* [Keras 3.0+](https://keras.io/getting_started/)
* [Decord](https://github.com/dmlc/decord)
* [Librosa](https://librosa.org/doc/main/install.html)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Numpy](https://numpy.org/install/)
* [Soundfile](https://pypi.org/project/SoundFile/)
* [TQDM](https://pypi.org/project/tqdm/)

## Usage

```text
# Expected directory structure
avsec3_data_root
├── dev
│   ├── lips
│   │   └── S37890_silent.mp4
│   └── scenes
│       ├── S37890_interferer.wav
│       ├── S37890_mixed.wav
│       ├── S37890_silent.mp4
│       └── S37890_target.wav
└── train
    ├── lips
    │   └── S34526_silent.mp4
    └── scenes
        ├── S34526_interferer.wav
        ├── S34526_mixed.wav
        ├── S34526_silent.mp4
        └── S34526_target.wav
```
- Change KERAS_BACKEND to 'torch' or 'tensorflow' in `config.py` based on the backend you are using.
### Train
To train the model, run the following command:
```bash
python train.py --data_root <avsec3_root>
```
where `<model_name>` is the name of the model to be trained, `<path_to_train_data>` is the path to the training data, `<path_to_val_data>` is the path to the validation data, and `<path_to_output>` is the path to save the trained model.

### Test
To test the model, run the following command:
```bash
python test.py --data_root <avsec3_root> --weight_path <trained_model_weights> --save_root <path_to_save_enhanced_utterances>
```



