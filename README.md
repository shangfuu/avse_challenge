## Scripts and Baseline model for generating data for 1st COG-MHEAR Audio-Visual Speech Enhancement Challenge

Human performance in everyday noisy situations is known to be dependent upon both aural and visual senses that are contextually combined by the brain’s multi-level integration strategies. The multimodal nature of speech is well established, with listeners known to unconsciously lip read to improve the intelligibility of speech in a real noisy environment. Studies in neuroscience have shown that the visual aspect of speech has a potentially strong impact on the ability of humans to focus their auditory attention on a particular stimulus.

Over the last few decades, there have been major advances in machine learning applied to speech technology made possible by Machine Learning related Challenges including CHiME, REVERB, Blizzard, Clarity and Hurricane. However, the aforementioned challenges are based on single and multi-channel audio-only processing and have not exploited the multimodal nature of speech. The aim of this first audio visual (AV) speech enhancement challenge is to bring together the wider computer vision, hearing and speech research communities to explore novel approaches to multimodal speech-in-noise processing.

In this repository, you will find code to support AVSE Challenges, including baselines, toolkits, and systems from participants. 

## Installation

```bash
# First clone the repo
git clone git@github.com:cogmhear/avse_challenge.git
cd avse_challenge

# Second create & activate environment with conda, see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
conda create --name avse_challenge python=3.8
conda activate avse_challenge

# Third install requirements
pip install -r requirements.txt

# Last install with pip
pip install -e .
```

## Preparing the scenes:
```
cd recipes/avse1/data_preparation
python prepare_avse1_data.py 
```

This script will use the AVSE1 scene render (audio from video, additive noise, one channel, 16kHz, 16bits): `clarity/data/scene_renderer_avse1.py`

The data paths are defined here: `recipes/avse1/data_preparation/data_config.yaml`

The scripts expect the following data directory structure:
```
cd /disk/scratch6/cvbotinh/av2022/av2022_data
mkdir -p metadata {train,dev}/{targets,targets_video,interferers{/noise,/speech},scenes}
```

To create the scenes json files:
```
cd recipes/avse1/data_preparation
python build_scenes.py
```
This script requires the following files in metadata:
```
target_speech_list.json
masker_noise_list.json
masker_speech_list.json
```

With the following format per file entry:
- target_speech_list.json
```
  {
    "wavfile": "T037_EFJ_01324",
    "dataset": "train",
    "nsamples": 158760  
  }
```
- masker_noise_list.json
```
  {
    "ID": "CIN_dishwasher_001",
    "class": "Dishwasher",
    "dataset": "train",
    "type": "noise",
    "nsamples": 6103440
  }
```
- masker_speech_list.json
```
  {
    "speaker": "irm_02484",
    "dataset": "train",
    "type": "speech",
    "nsamples": 31397509
  }
```

## Creating target and speech masker json files for LRS3:
```
python create_metadata.py
```
see in script for required files and paths.

## Preparing mixes from LRS3 data

Get noise data and scene files from server and set up working root directory  
(see EDIT_THIS in get_avse1_data.sh to point to the location of LRS3 data and set root directory location):
```
cd recipes/avse1/data_preparation
./get_avse1_data.sh 
```

Set the root in recipes/avse1/data_preparation/data_config.yaml to match the root defined in get_avse1_data.sh

Prepare mixes:
```
python prepare_avse1_data.py 
```

## License
The code in this repository is CC BY-SA 4.0 licensed, as found in the LICENSE file.

## Acknowledgements
Part of the code is adapted from: [Clarity Challenge](https://github.com/claritychallenge/clarity)