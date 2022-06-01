# -*- coding: utf-8 -*-
'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

import os
import logging

import hydra
from omegaconf import DictConfig
from scene_builder_avse1 import SceneBuilder, set_random_seed

logger = logging.getLogger(__name__)

def instantiate_scenes(cfg):
    set_random_seed(cfg.random_seed)
    for dataset in cfg.scene_datasets:
        scene_file = os.path.join(cfg.metadata_dir, f"scenes.{dataset}.json")
        if not os.path.exists(scene_file):
            logger.info(f"instantiate scenes for {dataset} set")
            sb = SceneBuilder(
                scene_datasets=cfg.scene_datasets[dataset],
                target=cfg.target,
                interferer=cfg.interferer,
                snr_range=cfg.snr_range[dataset],
            )
            sb.instantiate_scenes(dataset=dataset)
            sb.save_scenes(scene_file)
        else:
            logger.info(f"scenes.{dataset}.json exists, skip")


@hydra.main(config_path=".", config_name="data_config")
def run(cfg: DictConfig) -> None:
    logger.info("Instantiating scenes")
    instantiate_scenes(cfg)


if __name__ == "__main__":
    run()
