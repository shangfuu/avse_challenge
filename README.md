## Scripts and Baseline model for generating data for 1st COG-MHEAR Audio-Visual Speech Enhancement Challenge

Human performance in everyday noisy situations is known to be dependent upon both aural and visual senses that are contextually combined by the brain’s multi-level integration strategies. The multimodal nature of speech is well established, with listeners known to unconsciously lip read to improve the intelligibility of speech in a real noisy environment. Studies in neuroscience have shown that the visual aspect of speech has a potentially strong impact on the ability of humans to focus their auditory attention on a particular stimulus.

Over the last few decades, there have been major advances in machine learning applied to speech technology made possible by Machine Learning related Challenges including CHiME, REVERB, Blizzard, Clarity and Hurricane. However, the aforementioned challenges are based on single and multi-channel audio-only processing and have not exploited the multimodal nature of speech. The aim of this first audio visual (AV) speech enhancement challenge is to bring together the wider computer vision, hearing and speech research communities to explore novel approaches to multimodal speech-in-noise processing.

In this repository, you will find code to support AVSE Challenges, including baselines, toolkits, and systems from participants. 

## Installation

```bash
# Clone repository
git clone https://github.com/cogmhear/avse-challenge.git
cd avse-challenge

# Create & activate environment with conda, see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
conda create --name avse python=3.8
conda activate avse

# Install requirements
pip install -r requirements.txt

## Challenges

Current challenge

- The 1st Audio-Visual Speech Enhancement Challenge (AVSE1)  
[data_preparation](./data_preparation/avse1/)  
[baseline](./baseline/avse1/)  

## Data preparation

1) Download necessary data:
- target videos:  
Lip Reading Sentences 3 (LRS3) Dataset  
[https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)
- noise maskers and metadata:  
[https://data.cstr.ed.ac.uk/cogmhear/protected/avse1_data.tar]
(https://data.cstr.ed.ac.uk/cogmhear/protected/avse1_data.tar)


2) Set up data structure and create speech maskers (see EDIT_THIS to change local paths):
```bash
cd data_preparation/avse1
./setup_avse1_data.sh 
```

3) Change root path defined in [data_preparation/avse1/data_config.yaml](data_preparation/avse1/data_config.yaml) to the location of the data.

4) Prepare noisy data:
```bash
cd data_preparation/avse1
python prepare_avse1_data.py 
```

## License
The code in this repository is CC BY-SA 4.0 licensed, as found in the LICENSE file.

Interferers were derived from three different datasets:
- Clarity Challenge  
- DEMAND dataset  
- DNS Challenge second edition  
- TED data

Data preparation scripts were adapted from original code by [Clarity Challenge](https://github.com/claritychallenge/clarity). Modifications include: target audio is extracted from video, different sampling rate (16kHz) and number of channels (one channel), and scenes simulation (additive noise only, no room impulse responses and room simulation).
