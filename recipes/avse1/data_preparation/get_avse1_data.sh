#!/bin/bash

# working directory where data will be stored -- EDIT_THIS
root=/tmp/workdir/

# path to LRS3 data (where pretrain and trainval directories are located) -- EDIT_THIS
LRS3=/tmp/LRS3/

# wget https://data.cstr.ed.ac.uk/cogmhear/protected/av_challenge_data.tar
datafile=/tmp/av_challenge_data.tar

###########################################################
# Set up working directory structure and data
###########################################################

mkdir -p ${root}
cd ${root}

tar -xvf ${datafile}
masker_speech=maskers_speech
masker_noise=maskers_noise
metadata=metadata

mkdir -p {train,dev}/{targets,interferers,scenes}

ln -s ${LRS3} train/targets_video
ln -s ${LRS3} dev/targets_video

ln -s ${masker_noise} train/interferers/noise
ln -s ${masker_noise} dev/interferers/noise

ln -s ${masker_speech} train/interferers/speech
ln -s ${masker_speech} dev/interferers/speech

