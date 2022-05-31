import os
import json
import logging
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor

from clarity.data.scene_renderer_avse1 import Renderer, check_scene_exists

def run_renderer(renderer, scene, scene_folder):

    if check_scene_exists(scene, scene_folder):
        logging.info(f"Skipping processed scene {scene['scene']}.")
    else:
        renderer.render(
            dataset=scene["dataset"],
            target=scene["target"]["name"],
            noise_type=scene["interferer"]["type"],
            interferer=scene["interferer"]["name"],
            scene=scene["scene"],
            offset=scene["interferer"]["offset"],
            snr_dB=scene["SNR"],
        )

def prepare_data(
    root_path, metafile_path, scene_folder, num_channels, fs,
):
    """
    Generate scene data given dataset (train or dev)
    Args:
        root_path: Clarity root path
        metafile_path: scene metafile path
        scene_folder: folder containing generated scenes
        num_channels: number of channels
        fs: sampling frequency (Hz)
    """
    with open(metafile_path, "r") as f:
        scenes = json.load(f)

    os.makedirs(scene_folder, exist_ok=True)

    renderer = Renderer(input_path=root_path, output_path=scene_folder, num_channels=num_channels,fs=fs)
    
    # for scene in scenes:
    #     run_renderer(renderer, scene, scene_folder)

    futures  = []
    ncores = 20
    executor = ProcessPoolExecutor(max_workers=ncores)
    for scene in scenes:
        futures.append(executor.submit(run_renderer,renderer, scene, scene_folder))
    proc_list = [future.result() for future in tqdm(futures)]

@hydra.main(config_path=".", config_name="data_config")
def run(cfg: DictConfig) -> None:
    for dataset in cfg["datasets"]:
        prepare_data(
            cfg["input_path"],
            cfg["datasets"][dataset]["metafile_path"],
            cfg["datasets"][dataset]["scene_folder"],
            cfg["num_channels"],
            cfg["fs"],
        )


if __name__ == "__main__":
    run()
