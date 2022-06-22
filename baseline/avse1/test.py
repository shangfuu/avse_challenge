from argparse import ArgumentParser
from os import makedirs
from os.path import isfile, join

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from torch.nn import functional as F

from config import sampling_rate, window_shift, window_size

from dataset import TEDDataModule
from model import AVNet, FusionNet, build_audiofeat_net, build_visualfeat_net
from utils.generic import str2bool


def main(args):
    clean_root = join(args.save_root, "clean")
    noisy_root = join(args.save_root, "noisy")
    enhanced_root = join(args.save_root, args.model_uid)
    makedirs(args.save_root, exist_ok=True)
    makedirs(clean_root, exist_ok=True)
    makedirs(noisy_root, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)
    datamodule = TEDDataModule(batch_size=args.batch_size, mask=args.mask, a_only=args.a_only)
    test_dataset = datamodule.test_dataset
    print(args.oracle, not args.oracle)
    if not args.oracle:
        audiofeat_net = build_audiofeat_net(a_only=args.a_only)
        if not args.a_only:
            visual_net = build_visualfeat_net(extract_feats=True)
        else:
            visual_net = None
        fusion_net = FusionNet(a_only=args.a_only, mask=args.mask)
        print("Loading model components", args.ckpt_path)
        if args.ckpt_path.endswith("ckpt") and isfile(args.ckpt_path):
            model = AVNet.load_from_checkpoint(args.ckpt_path, nets=(visual_net, audiofeat_net, fusion_net),
                                               loss=args.loss, args=args,
                                               a_only=args.a_only)
            print("Model loaded")
        else:
            raise FileNotFoundError("Cannot load model weights: {}".format(args.ckpt_path))
        model.to("cuda:0")
        model.eval()
    i = 0
    loss = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):

            data = test_dataset[i]

            filename = f"{data['scene']}.wav"
            # filename = f"{str(i).zfill(5)}.wav"
            clean_path = join(clean_root, filename)
            noisy_path = join(noisy_root, filename)
            enhanced_path = join(enhanced_root, filename)

            if not isfile(clean_path):
                sf.write(clean_path, data["clean"], samplerate=sampling_rate)
            if not isfile(noisy_path):
                noisy = librosa.istft(data["noisy_stft"].T, win_length=window_size, hop_length=window_shift,
                                      window="hann")
                sf.write(noisy_path, noisy, samplerate=sampling_rate)
            if not isfile(enhanced_path):
                if args.oracle:
                    pred_mag = np.abs(data["noisy_stft"]) * data["mask"].T
                    i += 1
                else:
                    inputs = {"noisy_audio_spec": torch.from_numpy(data["noisy_audio_spec"][np.newaxis, ...]).to(
                        model.device)}
                    if not args.a_only:
                        inputs["lip_images"] = torch.from_numpy(data["lip_images"][np.newaxis, ...]).to(model.device)
                    pred = model(inputs).cpu()
                    pred_mag = pred.numpy()[0][0]
                    loss += F.l1_loss(pred[0], torch.from_numpy(data["mask"]))
                noisy_phase = np.angle(data["noisy_stft"])
                estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                estimated_audio = librosa.istft(estimated.T, win_length=window_size, hop_length=window_shift,
                                                window="hann")
                sf.write(enhanced_path, estimated_audio, samplerate=sampling_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--oracle", type=str2bool, required=False)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model_uid", type=str, required=True)
    parser.add_argument("--mask", type=str, default="mag")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss", type=str, default="l1")
    args = parser.parse_args()
    main(args)
