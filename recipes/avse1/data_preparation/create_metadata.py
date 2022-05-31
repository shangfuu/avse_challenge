import numpy as np
import pandas as pd
import os
import glob
import json
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

DEBUG = True
EXTRACT_AUDIO = False
long_dursecs = 60*9
mid_dursecs  = 60*5
train_dev_split = 0.8
fs = 16000
random_state =5

def read_file(file):
    file1 = open(file, 'r')
    data = {}
    for line in file1.readlines():
        data[line.split(' ')[0]] = float(line.split(' ')[1].strip())
    df = pd.DataFrame.from_dict(data, orient='index',columns=['dursecs'])
    df['speakers'] = df.index
    return df

def select_speakers(df):
    
    divide = 1 # 1: set all speakers with more data to target/tr, mid speakers are tgt/dev and msk/tr,dev
    # divide = 2 # 2: set all speakers with more data to target, the rest is masker

    if divide==1:

        # Classify speakers according to amount of data
        df['amount'] = df['dursecs'].map(lambda x: "long" if x>long_dursecs else ( "mid" if (x>mid_dursecs) & (x<long_dursecs) else "short" ) )    
        
        # All speakers with long amounts of data are TARGET/TRAIN
        tgt_tr = df[df["amount"]=="long"]
        tgt_tr["group"]   = "target"
        tgt_tr["dataset"] = "train"

        # Select TARGET/DEV from mid speakers
        df_mid = df[df["amount"]=="mid"]
        tgt_dev = df_mid.sample(frac=0.05,random_state=random_state)
        tgt_dev["group"]   = "target"
        tgt_dev["dataset"] = "dev"
        
        # Select MASKER/TRAIN,DEV from remaining mid speakers
        df_mid = df_mid.drop(tgt_dev.index)
        df_mid["group"] = "masker"
        msk_tr  = df_mid.sample(frac=0.25,random_state=random_state)
        msk_tr["dataset"] = "train"
        msk_dev = df_mid.drop(msk_tr.index)
        msk_dev = msk_dev.sample(frac=0.025,random_state=random_state)
        msk_dev["dataset"] = "dev"

        df = pd.concat([tgt_tr, tgt_dev, msk_tr, msk_dev])

    if divide==2:

        # Divide into target/masker (group)
        df['group'] = df['dursecs'].map(lambda x: "target" if x>long_dursecs else "masker")    

        # Divide into train/dev (dataset)
        train = df.groupby("group").sample(frac=train_dev_split,random_state=random_state)
        dev   = df.drop(train.index)
        train["dataset"] = "train"
        dev["dataset"] = "dev"
        df = pd.concat([train, dev])

    if DEBUG:
        for group in ['target','masker']:
            print(f"{group}: --------")
            for dataset in ['train','dev']:
                nspks = len(df.loc[(df["group"]==group) & (df["dataset"]==dataset)])
                hrs   = df.loc[(df["group"]==group) & (df["dataset"]==dataset),"dursecs"].sum() / (60*60)
                print("{}: {} speakers / {:.1f} hrs".format(dataset, nspks, hrs))
    return df

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_entries(entries, filename):
    json.dump(entries, open(filename, "w"), indent=2)

def extract_audio(wavdir, spk):
    create_dir(f"{wavdir}/{spk}/")
    for file in glob.iglob(f'{datadir}/*train*/{spk}/*.mp4'):
        basename = os.path.basename(file).split('.')[0]
        target_video_fn = file
        target_fn = f"{wavdir}/{spk}/{basename}.wav"
        command = ("ffmpeg -v 8 -y -i %s -vn -acodec pcm_s16le -ar %s -ac 1 %s" % (target_video_fn, str(fs), target_fn))
        os.system(command)

def create_json_files(df, datadir, wavdir, metadadir, mskdir):

    tgtfile = f'{metadadir}/target_speech_list.json'
    mskfile = f'{metadadir}/masker_speech_list.json'
    create_dir(wavdir)
    create_dir(metadadir)
    create_dir(mskdir)

    if EXTRACT_AUDIO:
        # Extract audio to get nsamples
        ncores   = 10
        executor = ProcessPoolExecutor(max_workers=ncores)    
        futures  = []
        for spk in df["speakers"].tolist():
            futures.append(executor.submit(extract_audio, wavdir, spk))
        proc_list = [future.result() for future in tqdm(futures)]

        # Concatenate files of masker speakers
        for spk in tqdm(df[df["group"] == "masker"]["speakers"].tolist()):
            y = np.array([])
            for file in sorted(glob.iglob(f'{wavdir}/{spk}/*.wav')):
                x = sf.read(file)[0]
                y = np.concatenate((y, x), axis=-1)
            sf.write(f"{mskdir}/{spk}.wav", y, fs)

    # Creating msk json file
    entries = []
    for spk in df[df["group"] == "masker"]["speakers"].tolist():
        dataset = df[df["speakers"] == spk]["dataset"].tolist()[0]
        
        # Get number of samples
        target_fn = f"{mskdir}/{spk}.wav"
        x = sf.read(target_fn)[0]
        nsamples = len(x)

        # Write entry
        entry = {}
        entry["speaker"]  = spk
        entry["dataset"]  = dataset
        entry["nsamples"] = nsamples
        entry["fs"] = fs
        entry["type"] = "speech"
        entries.append(entry)
    save_entries(entries, mskfile)

    # Creating target json file
    entries = []
    for spk in df[df["group"] == "target"]["speakers"].tolist():
        dataset = df[df["speakers"] == spk]["dataset"].tolist()[0]
        entries_spk = []
        skip_spk = False
        for file in sorted(glob.iglob(f'{datadir}/*train*/{spk}/*.mp4')):
            basename = os.path.basename(file).split('.')[0]
            lrsset = file.split('/')[-3]

            # Get nnumber of samples
            target_fn = f"{wavdir}/{spk}/{basename}.wav"
            x = sf.read(target_fn)[0]
            nsamples = len(x)

            if nsamples > fs*60*2:
                print(f"Skipped {lrsset}/{spk}/{basename} {nsamples/(fs*60)}mins")
                skip_spk = True
                break

            # Write entry
            entry = {}
            entry["wavfile"]  = f"{lrsset}/{spk}/{basename}"
            entry["speaker"]  = spk
            entry["dataset"]  = dataset
            entry["nsamples"] = nsamples
            entry["fs"] = fs
            entries_spk.append(entry)

        if not skip_spk:
            entries.append(entries_spk)

    entries = list(np.concatenate(entries).flat)
    save_entries(entries, tgtfile)

if __name__ == '__main__':
    
    # local paths
    # file = '/Users/cvbotinh/Documents/Cog-mhear/data/spk2dur_copy.txt'

    # zamora paths
    file      = '/afs/inf.ed.ac.uk/group/cstr/projects/cog-mhear/cvbotinh/LRS3/spk2dur_copy.txt'
    datadir   = '/group/corpora/public/lipreading/LRS3/'
    wavdir    = '/disk/scratch6/cvbotinh/av2022/LRS3/audio/'
    mskdir    = '/disk/scratch6/cvbotinh/av2022/LRS3/maskers/'
    metadadir = '/disk/scratch6/cvbotinh/av2022/LRS3/metadata/'

    # Read into dataframe
    df = read_file(file)

    # Divide speakers into target/masker train/dev
    df = select_speakers(df)

    pd.DataFrame.to_csv(df,"speakers.csv",index=None)

    # Create target and masker json files
    create_json_files(df, datadir, wavdir, metadadir, mskdir)
