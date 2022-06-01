#!/bin/bash

# working directory where data will be stored
root=/tmp/avse1_data/ # EDIT_THIS

# path to LRS3 data (pretrain and trainval directories should be located there)
LRS3=/tmp/LRS3/ # EDIT_THIS

# path to AVSE1 data
# wget https://data.cstr.ed.ac.uk/cogmhear/protected/avse1_data.tar
avse1data=/tmp/avse1_data.tar # EDIT_THIS

###########################################################
# Set up working directory structure and data
###########################################################

mkdir -p ${root}
tar -xvf ${avse1data} --directory ${root}/
masker_noise=${root}/maskers_noise/
masker_speech=${root}/maskers_speech/

mkdir -p ${root}/{train,dev}/{targets,interferers,scenes}

ln -s ${LRS3} ${root}/train/targets_video
ln -s ${LRS3} ${root}/dev/targets_video

ln -s ${masker_noise} ${root}/train/interferers/noise
ln -s ${masker_noise} ${root}/dev/interferers/noise

# Create speech masker data from LRS3 videos
python create_speech_maskers.py ${LRS3} ${root}/metadata/masker_speech_list.json ${masker_speech}

ln -s ${masker_speech} ${root}/train/interferers/speech
ln -s ${masker_speech} ${root}/dev/interferers/speech

