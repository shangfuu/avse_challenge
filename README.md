## Scripts and Baseline model for generating data for 1st COG-MHEAR Audio-Visual Speech Enhancement Challenge

The aim of the first AVE Challenge is to bring together the wider computer vision, hearing and speech research communities to explore novel approaches to multimodal speech-in-noise processing. Both raw and pre-processed AV datasets – derived from TED talk videos – will be made available to participants for training and development of audio-visual models to address to perform speech enhancement and speaker separation at SNR levels that will be significantly more challenging then typically used in audio-only scenarios.  Baseline models will be provided along, with scripts for objective evaluation. Challenge evaluation will utilise established objective measures such as STOI and PESQ as well as subjective intelligibility tests with human subjects. 

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


## License
The code in this repository is CC BY-SA 4.0 licensed, as found in the LICENSE file.

## Acknowledgements
Part of the code is adapted from: [Clarity Challenge](https://github.com/claritychallenge/clarity)