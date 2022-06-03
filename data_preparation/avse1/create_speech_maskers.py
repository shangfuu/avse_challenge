# -*- coding: utf-8 -*-
'''
Create speech maskers data
'''
import sys
import numpy as np
import os
import glob
import json
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

fs = 16000

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_speech_maskers(datadir, metafile, wavdir):

    with open(metafile, "r") as f:
        maskers = json.load(f)

    futures  = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for masker in maskers:
            futures.append(executor.submit(create_masker_for_spk, datadir, wavdir, masker['speaker']))
        proc_list = [future.result() for future in tqdm(futures)]

def create_masker_for_spk(datadir, wavdir, spk):

    create_dir(f"{wavdir}/{spk}/")

    # Extract audio from videos and join them into one long masker file
    y = np.array([])
    for file in glob.iglob(f'{datadir}/*train*/{spk}/*.mp4'):
        basename = os.path.basename(file).split('.')[0]
        target_fn = f"{wavdir}/{spk}/{basename}.wav"
        command = ("ffmpeg -v 8 -y -i %s -vn -acodec pcm_s16le -ar %s -ac 1 %s < /dev/null" % (file, str(fs), target_fn))
        os.system(command)
        x = sf.read(target_fn)[0]
        y = np.concatenate((y, x), axis=-1)
    sf.write(f"{wavdir}/{spk}.wav", y, fs)

    command = ("rm -r %s/%s" % (wavdir,spk))
    os.system(command)

if __name__ == '__main__':
    
    datadir   = sys.argv[1] # '/group/corpora/public/lipreading/LRS3/'
    metafile  = sys.argv[2] # '/disk/scratch6/cvbotinh/av2022/av2022_data/metadata/masker_speech_list.json'
    wavdir    = sys.argv[3] # '/disk/scratch6/cvbotinh/av2022/av2022_data/maskers_speech/'
    
    # Create speech masker files
    create_speech_maskers(datadir, metafile, wavdir)

