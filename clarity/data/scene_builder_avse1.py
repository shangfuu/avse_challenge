"""Code for building the scenes.json files."""
import itertools
import json
import logging
import math
import random
import re
from enum import Enum

import numpy as np
from tqdm import tqdm

# A logger for this file
log = logging.getLogger(__name__)


# Get json output to round to 4 dp
json.encoder.c_make_encoder = None


class RoundingFloat(float):
    """Round a float to 4 decimal places."""

    __repr__ = staticmethod(lambda x: format(x, ".4f"))


json.encoder.float = RoundingFloat

# rpf file Handling

N_SCENES = 10000  # Number of scenes to expect
N_INTERFERERS = 1  # Default number of interferers to expect


def set_random_seed(random_seed):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

def add_this_target_to_scene(target, scene):
    """Add the target details to the scene dict.
    Adds given target to given scene. Target details will be taken
    from the target dict but the start time will be
    according to the AVSE1 target start time specification.
    Args:
        target (dict): target dict read from target metadata file
        scene (dict): complete scene dict
    """
    scene_target = {}
    scene_target["name"] = target["wavfile"]
    scene["target"] = scene_target
    scene["duration"] = target["nsamples"]

# SNR handling
def generate_snr(snr_range):
    """Generate a random SNR."""
    return random.uniform(*snr_range)


# Interferer handling
class InterfererType(Enum):
    """Enum for interferer types."""

    SPEECH = "speech"
    NOISE = "noise"


def select_interferer_types(allowed_n_interferers):
    """Select the interferer types to use.
    The number of interferer is drawn randomly
    Args:
        allowed_n_interferers (list): list of allowed number of interferers
    Returns:
        list(InterfererType): list of interferer types to use
    """

    n_interferers = random.choice(allowed_n_interferers)
    selection = None
    while selection is None:
        selection = random.choices(list(InterfererType), k=n_interferers)
    return selection


def select_random_interferer(interferers, dataset, required_samples):
    """Randomly select an interferer.
    Interferers stored as list of list. First randomly select a sublist
    then randomly select an item from sublist matching constraints.
    Args:
        interferers (list(list)): interferers as list of lists
        dataset (str): desired data [train, dev, eval]
        required_samples (int): required number of samples
    Raises:
        ValueError: if no suitable interferer is found
    Returns:
        dict: the interferer dict
    """
    interferer_not_found = True
    while interferer_not_found:
        interferer_group = random.choice(interferers)
        filtered_interferer_group = [
            i
            for i in interferer_group
            if i["dataset"] == dataset and i["nsamples"] >= required_samples
        ]

        if filtered_interferer_group:
            interferer = random.choice(filtered_interferer_group)
            interferer_not_found = False
        else:
            if interferer_group[0]['type']=="noise":
                print(f"No suitable interferer found in class {interferer_group[0]['class']} for required samples {required_samples}")
            else:
                print(f"No suitable interferer found in class {interferer_group[0]['speaker']} for required samples {required_samples}")

    return interferer


def get_random_interferer_offset(interferer, required_samples):
    """Generate a random offset sample for interferer.
    The offset sample is the point within the masker signal at which the interferer
    segment will be extracted. Randomly selected but with care for it not to start
    too late, i.e. such that the required samples would overrun the end of the masker
    signal will be used is taken.
    Args:
        interferer (dict): the interferer metadata
        required_samples (int): number of samples that is going to be required
    Returns:
        int: a valid randomly selected offset
    """
    masker_nsamples = interferer["nsamples"]
    latest_start = masker_nsamples - required_samples
    if latest_start < 0:
        log.error(f"Interferer {interferer['ID']} does not has enough samples.")

    assert (
        latest_start >= 0
    )  # This should never happen - mean masker was too short for the scene
    return random.randint(0, latest_start)


def add_interferer_to_scene_inner(
    scene, interferers, number, start_time_range, end_early_time_range
):
    """Randomly select interferers and add them to the given scene.
    A random number of interferers is chosen, then each is given a random type
    selected from the possible speech, nonspeech, music types.
    Interferers are then chosen from the available lists according to the type
    and also taking care to match the scenes 'dataset' field, ie. train, dev, test.
    The interferer data is supplied as a dictionary of lists of lists. The key
    being "speech", "nonspeech", or "music", and the list of list being a partitioned
    list of interferers for that type.
    The idea of using a list of lists is that interferers can be split by
    subcondition and then the randomization draws equally from each subcondition,
    e.g. for nonspeech there is "washing machine", "microwave" etc. This ensures that
    each subcondition is equally represented even if the number of exemplars of
    each subcondition is different.
    Note, there is no return. The scene is modified in place.
    Args:
        scene (dict): the scene description
        interferers (dict): the interferer metadata
        number: number of interferers
        start_time_range: when to start
        end_early_time_range: when to end
    """
    dataset = scene["dataset"]
    selected_interferer_types = select_interferer_types(number)
    n_interferers = len(selected_interferer_types)
    
    
    scene["interferer"] = [{"type": scene_type.value} for scene_type in selected_interferer_types]

    # Randomly instantiate each interferer in the scene
    for scene_interferer, scene_type in zip(
        scene["interferer"], selected_interferer_types
    ):
        desired_start_time = random.randint(*start_time_range)

        scene_interferer["time_start"] = min(scene["duration"], desired_start_time)
        desired_end_time = scene["duration"] - random.randint(*end_early_time_range)

        scene_interferer["time_end"] = max(
            scene_interferer["time_start"], desired_end_time
        )

        required_samples = scene_interferer["time_end"] - scene_interferer["time_start"]
        interferer = select_random_interferer(
            interferers[scene_type], dataset, required_samples
        )
        # scene_interferer["type"] = scene_type.value
        scene_interferer["name"] = interferer["ID"]
        scene_interferer["offset"] = get_random_interferer_offset(
            interferer, required_samples
        )

    scene["interferer"] = scene["interferer"][0]

class SceneBuilder:
    """Functions for building a list scenes."""

    def __init__(
        self,
        scene_datasets,
        target,
        interferer,
        snr_range,
    ):
        self.scenes = []
        self.scene_datasets = scene_datasets
        self.target = target
        self.interferer = interferer
        self.snr_range = snr_range

    def save_scenes(self, filename):
        """Save the list of scenes to a json file."""
        scenes = [s for s in self.scenes]
        # Replace the room structure with the room ID
        # for scene in scenes:
        #     scene["room"] = scene["room"]["name"]
        json.dump(self.scenes, open(filename, "w"), indent=2)

    def instantiate_scenes(self, dataset):
        print(f"Initialise {dataset} scenes")
        self.initialise_scenes(dataset, **self.scene_datasets)
        print("adding targets to scenes")
        self.add_target_to_scene(dataset, **self.target)
        print("adding interferers to scenes")
        self.add_interferer_to_scene(**self.interferer)
        print("assigning an SNR to each scene")
        self.add_SNR_to_scene(self.snr_range)

    def initialise_scenes(self, dataset, n_scenes, scene_start_index):
        """
        Initialise the scenes for a given dataset.
        Args:
            dataset: train, dev, or eval set
            n_scenes: number of scenes to generate
            scene_start_index: index to start for scene IDs
        """

        # Construct the scenes adding the room and dataset label
        self.scenes = []
        scenes = [{"dataset": dataset} for _ in range(n_scenes)]

        # Set the scene ID
        for index, scene in enumerate(scenes, scene_start_index):
            scene["scene"] = f"S{index:05d}"
        self.scenes.extend(scenes)

    def add_target_to_scene(
        self,
        dataset,
        target_speakers,
        target_selection,
    ):
        """Add target info to the scenes.
        Target speaker file set via config.
        Raises:
            Exception: _description_
        """
        targets = json.load(open(target_speakers, "r"))

        targets_dataset = [t for t in targets if t["dataset"] == dataset]
        scenes_dataset = [s for s in self.scenes if s["dataset"] == dataset]

        random.shuffle(targets_dataset)

        if target_selection == "SEQUENTIAL":
            # Sequential mode: Cycle through targets sequentially
            for scene, target in zip(scenes_dataset, itertools.cycle(targets_dataset)):
                add_this_target_to_scene(
                    target, scene
                )
        elif target_selection == "RANDOM":
            # Random mode: randomly select target with replacement
            for scene in scenes_dataset:
                add_this_target_to_scene(
                    random.choice(targets_dataset),
                    scene,
                )
        else:
            assert False, "Unknown target selection mode"

    def add_SNR_to_scene(self, snr_range):
        """Add the SNR info to the scenes."""
        for scene in tqdm(self.scenes):
            scene["SNR"] = generate_snr(snr_range[scene["interferer"]["type"]])
            scene["pre_samples"] = 0
            scene["post_samples"] = 0

    def add_interferer_to_scene(
        self,
        speech_interferers,
        noise_interferers,
        number,
        start_time_range,
        end_early_time_range,
    ):
        """Add interferer to the scene description file."""
        # Load and prepare speech interferer metadata
        interferers_speech = json.load(open(speech_interferers, "r"))
        for interferer in interferers_speech:
            interferer["ID"] = (
                interferer["speaker"] # + ".wav"
            )  # selection require a unique "ID" field
        # Selection process requires list of lists
        interferers_speech = [interferers_speech]

        # Load and prepare noise (i.e. noise) interferer metadata
        interferers_noise = json.load(open(noise_interferers, "r"))
        # for interferer in interferers_noise:
        #     interferer["ID"] += ".wav"
        interferer_by_type = dict()
        for interferer in interferers_noise:
            interferer_by_type.setdefault(interferer["class"], []).append(interferer)
        interferers_noise = list(interferer_by_type.values())

        interferers = {
            InterfererType.SPEECH: interferers_speech,
            InterfererType.NOISE: interferers_noise,
        }

        for scene in tqdm(self.scenes):
            add_interferer_to_scene_inner(
                scene, interferers, number, start_time_range, end_early_time_range
            )
