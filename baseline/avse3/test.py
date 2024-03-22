from config import *
from argparse import ArgumentParser
from os import makedirs
from os.path import isfile, join

import soundfile as sf
from tqdm import tqdm

from config import sampling_rate
from dataset import AVSEChallengeDataModule
from model import AVSE
from utils import *


def main(args):
    datamodule = AVSEChallengeDataModule(data_root=args.data_root, batch_size=1, time_domain=True)
    # can be changed to test_dataset
    test_dataset = datamodule.dev_dataset
    # test_dataset = datamodule.test_dataset

    makedirs(args.save_root, exist_ok=True)

    model = AVSE(64, 40800, batch_size=1)
    model.load_weights(args.weight_path)
    for i in tqdm(range(len(test_dataset))):
        data = test_dataset[i][0]
        filename = data["scene"] + ".wav"
        enhanced_path = join(args.save_root, filename)
        if not isfile(enhanced_path):
            estimated_audio = get_enhanced(model, data)
            estimated_audio /= np.max(np.abs(estimated_audio))
            sf.write(enhanced_path, estimated_audio, samplerate=sampling_rate)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--weight_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--save_root", type=str, default="./enhanced", help="Root directory to save enhanced audio")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset")
    args = parser.parse_args()
    main(args)
