#!/bin/bash

# working directory where data will be stored
root=/tmp/ # EDIT_THIS

# path to LRS3 data (pretrain and trainval directories should be located there)
LRS3=/tmp/LRS3/  # EDIT_THIS
#path to Clarity data and OlHeaD-HRTF Database
# wget https://data.cstr.ed.ac.uk/cogmhear/protected/clarity_cec2_data.tar
clarity_data=/tmp/clarity_cec2_data.tar # EDIT_THIS
# path to AVSE4 data
# wget https://data.cstr.ed.ac.uk/cogmhear/protected/avsec4_data.tar
avsec4data=/tmp/avsec4_data.tar # EDIT_THIS

###########################################################
# Set up working directory structure and data
###########################################################

mkdir -p ${root}

tar -xvf ${avsec4data} --directory ${root}/

masker_music=${root}/maskers_music/
masker_noise=${root}/maskers_noise/
masker_speech=${root}/maskers_speech/

mkdir -p ${root}/{train,dev}/{targets,interferers,scenes}

#Extract impulse responses and room simulation info to dev/train folders and extract hrir folder
tar -xvf ${clarity_data} --directory ${root}/

ln -s ${LRS3} ${root}/train/targets_video
ln -s ${LRS3} ${root}/dev/targets_video

ln -s ${masker_music} ${root}/train/interferers/music
ln -s ${masker_music} ${root}/dev/interferers/music
ln -s ${masker_noise} ${root}/train/interferers/noise
ln -s ${masker_noise} ${root}/dev/interferers/noise

# Create speech masker data from LRS3 videos
python create_speech_maskers.py ${LRS3} ${root}/metadata/masker_speech_list.json ${masker_speech}

ln -s ${masker_speech} ${root}/train/interferers/speech
ln -s ${masker_speech} ${root}/dev/interferers/speech

