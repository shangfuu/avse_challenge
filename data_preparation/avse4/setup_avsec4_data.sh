#!/bin/bash

# working directory where data will be stored
root=/disk/data1/aaldana/avsec/ #/disk/data1/aaldana/avsec_LRS3 #/tmp/avsec4/ # EDIT_THIS

# path to LRS3 data (pretrain and trainval directories should be located there)
LRS3=/disk/data1/aaldana/LRS3/ #/tmp/LRS3/ # EDIT_THIS

#path to Clarity data
#clarity_data=/disk/data1/aaldana/ClarityData/clarity_CEC2_data/
# wget https://data.cstr.ed.ac.uk/cogmhear/protected/clarity_cec2_data.tar
clarity_data=/tmp/clarity_cec2_data.tar
# path to AVSE4 data
# wget https://data.cstr.ed.ac.uk/cogmhear/protected/avsec4_data.tar
#avsec4data=/tmp/avse1_data.tar # EDIT_THIS
avsec4data=/disk/data1/aaldana/avsec4_data.tar


###########################################################
# Set up working directory structure and data
###########################################################

mkdir -p ${root}

tar -xvf ${avsec4data} --directory ${root}/

masker_music=${root}/avsec4/maskers_music/
masker_noise=${root}/avsec4/maskers_noise/
masker_speech=${root}/avsec4/maskers_speech/

mkdir -p ${root}/avsec4/{train,dev}/{targets,interferers,scenes}

#Extract HOA_ir and rooms info to dev train folders
tar -xvf ${clarity_data} --directory ${root}/avsec4/

ln -s ${LRS3} ${root}/avsec4/train/targets_video
ln -s ${LRS3} ${root}/avsec4/dev/targets_video

ln -s ${masker_music} ${root}/avsec4/train/interferers/music
ln -s ${masker_music} ${root}/avsec4/dev/interferers/music
ln -s ${masker_noise} ${root}/avsec4/train/interferers/noise
ln -s ${masker_noise} ${root}/avsec4/dev/interferers/noise

#sym links clarity data
ln -s ${clarity_data}/clarity_data/train/rooms ${root}/avsec4/train/rooms
ln -s ${clarity_data}/clarity_data/dev/rooms ${root}/avsec4/dev/rooms
ln -s ${clarity_data}/clarity_data/hrir ${root}/avsec4/hrir

# Create speech masker data from LRS3 videos
python create_speech_maskers.py ${LRS3} ${root}/avsec4/metadata/masker_speech_list.json ${masker_speech}

ln -s ${masker_speech} ${root}/avsec4_data/train/interferers/speech
ln -s ${masker_speech} ${root}/avsec4_data/dev/interferers/speech

