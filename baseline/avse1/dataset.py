#  Copyright (c) 2021 Mandar Gogate, All rights reserved.
import json
import logging
import os
import random
from os.path import join

import imageio
import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from decord import cpu
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

from config import *
from utils.generic import subsample_list


def get_images(mp4_file):
    data = [np.array(img)[np.newaxis, ...] for img in imageio.mimread(mp4_file)]
    return np.concatenate(data, axis=0)


def get_transform():
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


test_transform = get_transform()


class TEDDataset(Dataset):
    def __init__(self, scenes_root, shuffle=True, seed=SEED, subsample=1, mask_type="IRM",
                 add_channel_dim=True, a_only=True, return_stft=False,
                 meta_data=None, clipped_batch=True, sample_items=True):
        self.meta = {}
        with open(meta_data) as f:
            data = json.load(f)
            for d in data:
                self.meta[d["scene"]] = d["target"]["name"]
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.return_stft = return_stft
        self.a_only = a_only
        self.add_channel_dim = add_channel_dim
        self.files_list = self.build_files_list
        self.mask_type = mask_type.lower()
        self.rgb = True if nb_channels == 3 else False
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False
        self.sample_items = sample_items

    @property
    def build_files_list(self):
        files_list = []
        for file in os.listdir(self.scenes_root):
            if file.endswith("target.wav"):
                files_list.append((join(self.scenes_root, file),
                                   join(self.scenes_root, file.replace("target", "interferer")),
                                   join(self.scenes_root, file.replace("target", "mixed")),
                                   join(self.scenes_root, file.replace("_target.wav", "_silent.mp4")),
                                   ))
        return files_list

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        data = {}
        if self.sample_items:
            clean_file, noise_file, noisy_file, mp4_file = random.sample(self.files_list, 1)[0]
        else:
            clean_file, noise_file, noisy_file, mp4_file = self.files_list[idx]
        if self.a_only:
            if self.return_stft:
                data["noisy_audio_spec"], data["mask"], data["clean"], data["noisy_stft"] = self.get_data(clean_file,
                                                                                                          noise_file,
                                                                                                          noisy_file,
                                                                                                          mp4_file)
            else:
                data["noisy_audio_spec"], data["mask"] = self.get_data(clean_file, noise_file, noisy_file, mp4_file)
        else:
            if self.return_stft:
                data["noisy_audio_spec"], data["mask"], data["clean"], data["noisy_stft"], data[
                    "lip_images"] = self.get_data(clean_file,
                                                  noise_file,
                                                  noisy_file,
                                                  mp4_file)
            else:
                data["noisy_audio_spec"], data["mask"], data["lip_images"] = self.get_data(clean_file, noise_file,
                                                                                           noisy_file, mp4_file)

        data['scene'] = clean_file.replace(self.scenes_root,"").replace("_target.wav","").replace("/","")

        return data

    def get_noisy_features(self, noisy):
        audio_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                  window=self.window, center=True).T
        if self.add_channel_dim:
            return np.abs(audio_stft).astype(np.float32)[np.newaxis, ...]
        else:
            return np.abs(audio_stft).astype(np.float32)

    def load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)

    def get_data(self, clean_file, noise_file, noisy_file, mp4_file):
        noisy = self.load_wav(noisy_file)
        clean = self.load_wav(clean_file)
        # noise, _ = librosa.load(noise_file, sr=None)
        if self.clipped_batch:
            if clean.shape[0] > 48000:
                clip_idx = random.randint(0, clean.shape[0] - 48000)
                video_idx = max(int((clip_idx / 16000) * 25) - 2, 0)
                clean = clean[clip_idx:clip_idx + 48000]
                noisy = noisy[clip_idx:clip_idx + 48000]
                # noise = noise[clip_idx:clip_idx + 48000]
            else:
                video_idx = -1
                clean = np.pad(clean, pad_width=[0, 48000 - clean.shape[0]], mode="constant")
                noisy = np.pad(noisy, pad_width=[0, 48000 - noisy.shape[0]], mode="constant")
                # noise = np.pad(noise, pad_width=[0, 48000 - noise.shape[0]], mode="constant")
        if not self.a_only:
            vr = VideoReader(mp4_file, ctx=cpu(0))

            if not self.clipped_batch:
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            else:
                if len(vr) < 75:
                    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
                    frames = np.concatenate((frames, np.zeros((75 - len(vr), 224, 224, 3)).astype(frames.dtype)), axis=0)
                else:
                    frames = vr.get_batch(list(range(video_idx, video_idx + 75))).asnumpy()
            frames = np.moveaxis(frames, -1, 0)
        if self.return_stft:
            clean_audio = clean
            noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                      window=self.window, center=True).T
            if self.a_only:
                return self.get_noisy_features(noisy), self.get_noisy_features(
                    clean), clean_audio, noisy_stft
            else:
                return self.get_noisy_features(noisy), self.get_noisy_features(
                    clean), clean_audio, noisy_stft, frames
        else:
            if self.a_only:
                return self.get_noisy_features(noisy), self.get_noisy_features(clean)
            else:
                return self.get_noisy_features(noisy), self.get_noisy_features(clean), frames


class TEDDataModule(LightningDataModule):
    def __init__(self, batch_size=16, mask="IRM", add_channel_dim=True, a_only=False):
        super(TEDDataModule, self).__init__()
        self.train_dataset = TEDDataset(join(DATA_ROOT, "train/scenes"), mask_type=mask,
                                        add_channel_dim=add_channel_dim, a_only=a_only
                                        , meta_data=join(METADATA_ROOT, "scenes.train.json"))
        self.val_dataset = TEDDataset(join(DATA_ROOT, "dev/scenes"), mask_type=mask,
                                      add_channel_dim=add_channel_dim, a_only=a_only,
                                      meta_data=join(METADATA_ROOT, "scenes.dev.json"))
        self.test_dataset = TEDDataset(join(DATA_ROOT, "dev/scenes"), mask_type=mask,
                                       add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True,
                                       meta_data=join(METADATA_ROOT, "scenes.dev.json"), clipped_batch=False, sample_items=False)
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True,
                                           persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':

    dataset = TEDDataset(scenes_root=join(DATA_ROOT, "train/scenes"),
                         mask_type="mag", meta_data=join(METADATA_ROOT, "scenes.train.json"),
                         a_only=False, return_stft=True)
    print(dataset.files_list[:2])
    for i in tqdm(range(len(dataset)), ascii=True):
        data = dataset[i]
