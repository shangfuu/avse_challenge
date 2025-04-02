from os.path import isfile
from os import makedirs
from os.path import join

import soundfile as sf
import torch
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

from dataset import AVSE4DataModule
from model import AVSE4BaselineModule

SAMPLE_RATE = 16000

@hydra.main(config_path="conf", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    enhanced_root = join(cfg.save_dir, cfg.model_uid)
    makedirs(cfg.save_dir, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)
    datamodule = AVSE4DataModule(data_root=cfg.data.root,batch_size=1,rgb=cfg.data.rgb,
                                  num_channels=cfg.data.num_channels, audio_norm=cfg.data.audio_norm)
    if cfg.data.dev_set and cfg.data.eval_set:
        raise RuntimeError("Select either dev set or test set")
    elif cfg.data.dev_set:
        dataset = datamodule.dev_dataset
    elif cfg.data.eval_set:
        dataset = datamodule.eval_dataset
    else:
        raise RuntimeError("Select one of dev set and test set")
    try:
        model = AVSE4BaselineModule.load_from_checkpoint(cfg.ckpt_path)
        print("Model loaded")
    except Exception as e:
        raise FileNotFoundError("Cannot load model weights: {}".format(cfg.ckpt_path))
    if not cfg.cpu:
        model.to("cuda:0")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            filename = f"{data['scene']}.wav"
            enhanced_path = join(enhanced_root, filename)
            if not isfile(enhanced_path):
                clean, noisy, estimated_audio = model.enhance(data)
                sf.write(enhanced_path, estimated_audio.T, samplerate=SAMPLE_RATE)


if __name__ == '__main__':
    main()
