'''
Adapted from original code by Clarity Enhancement Challenge 2
https://github.com/claritychallenge/clarity/tree/main/recipes/cec2

Clarity ambisonic scene rendering."
'''

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Final

import librosa
import numpy as np

# from clarity_core import signal as ccs
# from clarity_core.signal import SPEECH_FILTER
from scipy.io import loadmat, wavfile
from scipy.signal import convolve
from tqdm import tqdm #tqdm0

import clarity.data.HOA_tools_cec2 as hoa
from clarity.data.HOA_tools_cec2 import HOARotator
from clarity.data.utils import SPEECH_FILTER, better_ear_speechweighted_snr

#avsec
import os

logger = logging.getLogger(__name__)

SPEED_SOUND: Final = 344  # in m/s at 21 degrees C
SAMPLE_RATE: Final = 16000 #44100

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def two_point_rotation(rotation: dict, origin: np.ndarray, duration: int) -> np.ndarray:
    """Perform rotation defined by two control points.

    Args:
        rotation (dict): rotation object from scene definition
        origin (ndarray): origin view vector
        duration (int): total number of samples to generate for

    Returns:
        np.ndarray: sequence of theta values per sample
    """
    angle_origin = np.arctan2(origin[1], origin[0])
    angles = [math.radians(r["angle"]) - angle_origin for r in rotation]
    logger.info("angles=%s", angles)
    theta = hoa.compute_rotation_vector(
        angles[0], angles[1], duration, rotation[0]["sample"], rotation[1]["sample"]
    )
    return theta


def pad_signal_start_end(signal: np.ndarray, delay: int, duration: int) -> np.ndarray:
    """Pad signal at start and end.

    Args:
        signal (array-like): ambisonic signals
        delay (int): number of zeros to pad at start
        duration (int): desired duration after start and end padding

    Returns:
        array-like: padded signals
    """
    time_after_signal = duration - delay - signal.shape[0]
    if time_after_signal < 0:
        # Signal is too long, so we need to truncate it
        signal = signal[:time_after_signal, :]
        time_after_signal = 0

    signal_pad_front = np.zeros([delay, signal.shape[1]])
    signal_pad_back = np.zeros([time_after_signal, signal.shape[1]])

    padded_signal = np.concatenate([signal_pad_front, signal, signal_pad_back], axis=0)
    return padded_signal


class SceneRenderer:
    """Ambisonic scene rendering class.

    Contains methods for generating signals from pseudorandom datasets for CEC2
    """

    def __init__(
        self,
        paths,
        metadata,
        ambisonic_order,
        equalise_loudness,
        reference_channel,
        channel_norms,
        binaural_render,
        monoaural_render
    ):
        """Initialise SceneRenderer.

        Args:
            paths ():
            metadata ():
            ambisonic_order ():
            equalise_loudness ():
            reference_channel ():
            channel_norms ():
        """
        self.paths = paths
        self.metadata = metadata
        self.ambisonic_order = ambisonic_order
        self.equalise_loudness = equalise_loudness
        self.reference_channel = reference_channel
        self.channel_norms = channel_norms
        self.binaural_render = binaural_render
        self.monoaural_render = monoaural_render

        # Build the HOA rotator object with precomputed rotation matrices
        self.hoa_rotator = HOARotator(self.ambisonic_order, resolution=0.1)

        # Build dictionary for looking up room data
        with open(f"{self.metadata.room_definitions}", encoding="utf-8") as f:
            rooms = json.load(f)
        self.room_dict = {room["name"]: room for room in rooms}

        # Fixed hrir metadata for hrir sets being used
        with open(self.metadata.hrir_metadata, encoding="utf-8") as f:
            self.metadata.hrir_metadata = json.load(f)

    def make_interferer_filename(self, interferer: dict, dataset) -> str:
        """Construct filename for an interferer.

        Args:
            interferer (dict):

        Returns:
            str: Filename for an interferer.
        """
        data_type = interferer["type"]
        stem = self.paths.interferers.format(dataset=dataset, type=data_type)
        return f"{stem}/{interferer['name']}"

    def prepare_interferer_paths(self, scene):
        """Make list of full path filenames for interferers in scene.

        Args:
            scene ():

        Returns:
            list: List of full path filenames for interferers in scene.
        """
        logger.info("number of interferers: %s", len(scene["interferers"]))
        interferer_filenames = [
            self.make_interferer_filename(interferer, scene["dataset"])
            for interferer in scene["interferers"]
        ]
        return interferer_filenames

    def load_interferer_hoairs(self, scene, sample_rate):
        """Loads and returns the interferer hoa irs for given scene.

        Args:
            scene ():

        Returns:
            list: List of inferior hoa irs for the given scene.
        """
        room_id = scene["room"]
        n_interferers = len(scene["interferers"])

        hoair_path = self.paths.hoairs.format(dataset=scene["dataset"])
        interferer_hoairs = [
            f"{hoair_path}/HOA_{room_id}_i{n}.wav" for n in range(1, n_interferers + 1)
        ]
        irs = [wavfile.read(filename)[1] for filename in interferer_hoairs]
        return irs

    def load_interferer_signals(self, scene):
        """Loads and returns interferer signals for given scene.

        Args:
            scene ():

        Returns:
            list: List of signals.
        """
        interferer_audio_paths = self.prepare_interferer_paths(scene)

        # NOTE: all interferer signals are assumed to by at the LRS3 sr = 16000 Hz
        with warnings.catch_warnings():
            # Suppress annoying warning generated by librosa when reading mp3 files
            warnings.simplefilter("ignore")
            signals = [
                librosa.load(signal_path, sr=SAMPLE_RATE)[0]
                for signal_path in interferer_audio_paths
            ]

        signal_starts = [interferer["offset"] for interferer in scene["interferers"]]
        signal_lengths = [
            interferer["time_end"] - interferer["time_start"]
            for interferer in scene["interferers"]
        ]

        signals = [
            signal[ss : (ss + sl)]
            for (signal, ss, sl) in zip(signals, signal_starts, signal_lengths)
        ]

        return signals

    def make_hoa_target_anechoic(self, target, room):
        """Make the HOA anechoic target.

        Applies an anechoic HOA IR that models a source straight in front of the
        listener. The signal is delayed to match the propagation delay of the room.

        Args:
            target ():
            room (dict):
        """
        # TODO: The delay does not appear to correctly align the signals as expected
        t_pos = np.array(room["target"]["position"])
        l_pos = np.array(room["listener"]["position"])
        distance = np.linalg.norm(t_pos - l_pos)
        samples_delay = int(distance / SPEED_SOUND * SAMPLE_RATE)
        logger.info("%s, %s", room["target"]["position"], room["listener"]["position"])
        logger.info("target signal delay = %s samples %s m)", samples_delay, distance)

        # Values below for a source directly ahead of the listener
        anechoic_ir = np.array(
            [
                [1.0, 0.0, 0.0, 1.7320509, 0.0, 0.0, -1.1180342]
                + [0.0, 1.9364915, 0.0, 0.0, 0.0, 0.0, -1.6201853]
                + [0.0, 2.09165, 0.0, 0.0, 0.0, 0.0, 1.125]
                + [0.0, -1.6770511, 0.0, 2.2185302, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                + [
                    1.6056539,
                    0.0,
                    -1.7343045,
                    0.0,
                    2.326814,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                + [-1.1267347, 0.0, 1.6327935, 0.0, -1.7886358, 0.0, 2.4209614, 0.0]
                + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                + [0.0, -1.601086, 0.0, 1.6638975, 0.0, -1.839508, 0.0, 20.63472]
            ]
        )
        n_chans = (self.ambisonic_order + 1) ** 2
        anechoic_ir = anechoic_ir[:n_chans]
        hoa_target_anechoic = hoa.ambisonic_convolve(
            target, anechoic_ir, self.ambisonic_order
        )
        # prepend with zeros to match the propagation delay
        hoa_target_anechoic = np.vstack(
            (np.zeros((samples_delay, n_chans)), hoa_target_anechoic)
        )
        logger.info(
            "%s, %s, %s", anechoic_ir.shape, target.shape, hoa_target_anechoic.shape
        )

        return hoa_target_anechoic

    def generate_hoa_signals(self, scene: dict) -> tuple:
        """Generates HOA signals.

        Args:
            scene (dict): scene definitions
        """
        room_id = scene["room"]
        room = self.room_dict[room_id]

        # Make HOA target signal
        # NOTE: Target signals in AVSEC are at 16000 Hz
        target_audio_path = self.paths.targets.format(dataset=scene["dataset"]) #avsec4_data/train/targets/

        #video filename
        target_video_path = self.paths.videos.format(dataset=scene["dataset"])

        target_video_fn = f"{target_video_path}/{scene['target']['name']}.mp4"
        target_audio_fn = f"{target_audio_path}/{scene['target']['name']}.wav"

        # create parent directory
        target_fn_dir = os.path.dirname(target_audio_fn)
        create_dir(target_fn_dir)

        #extract audio from video to interferers/{dataset}
        command = ("ffmpeg -v 8 -y -i %s -vn -acodec pcm_s16le -ar %s -ac 1 %s" % (target_video_fn, str(SAMPLE_RATE), target_audio_fn))
        os.system(command)

        # Write video file without audio stream to scenes folder
        # output_video_fn = f"{out_path}/{scene['scene']}_silent.mp4"

        # target, _sample_rate = librosa.load(target_audio_path, sr=None) #AVSEC data is at 16000 kHz
        target, _sample_rate = librosa.load(target_audio_fn, sr=SAMPLE_RATE)  # AVSEC data is at 16000 kHz

        # TODO: set target to a fixed reference level??
        target_filt = convolve(target, SPEECH_FILTER, mode="full", method="fft")
        # rms of the target after speech weighted filter
        target_rms = np.sqrt(np.mean(target_filt**2))
        logger.info("target rms: %s", target_rms)
        target_hoair = self.paths.hoairs.format(dataset=scene["dataset"])
        target_hoair = f"{target_hoair}/HOA_{room_id}_t.wav"
        _sample_rate, ir = wavfile.read(target_hoair)

        hoa_target = hoa.ambisonic_convolve(target, ir, self.ambisonic_order)
        hoa_target = pad_signal_start_end(
            hoa_target, scene["target"]["time_start"], scene["duration"]
        )

        # Make anechoic HOA target signal
        hoa_target_anechoic = self.make_hoa_target_anechoic(target, room)

        hoa_target_anechoic = pad_signal_start_end(
            hoa_target_anechoic, scene["target"]["time_start"], scene["duration"]
        )

        # Make HOA interferer signals
        irs = self.load_interferer_hoairs(scene, SAMPLE_RATE) #these are resampled to 16000 for avsec
        interferers = self.load_interferer_signals(scene)

        hoa_interferers = [
            hoa.ambisonic_convolve(signal, ir, order=self.ambisonic_order)
            for (signal, ir) in zip(interferers, irs)
        ]

        padded_interferers = [
            pad_signal_start_end(signal, interferer["time_start"], scene["duration"])
            for signal, interferer in zip(hoa_interferers, scene["interferers"])
        ]

        if self.equalise_loudness:
            padded_interferers = hoa.equalise_rms_levels(padded_interferers)

        #sum interferers
        flat_hoa_interferers = sum(padded_interferers)

        logger.info(
            "hoa_target.shape=%s; flat_hoa_interferers.shape=%s",
            hoa_target.shape,
            flat_hoa_interferers.shape,
        )

        # head rotation not used in AVSEC
        # th = two_point_rotation(
        #     scene["listener"]["rotation"],
        #     room["listener"]["view_vector"],
        #     scene["duration"],
        # )

        # hoa_target_anechoic_rotated = self.hoa_rotator.rotate(hoa_target_anechoic, th)
        # hoa_target_rotated = self.hoa_rotator.rotate(hoa_target, th)
        # flat_hoa_interferers_rotated = self.hoa_rotator.rotate(flat_hoa_interferers, th)

        # return (
        #     hoa_target_rotated,
        #     flat_hoa_interferers_rotated,
        #     hoa_target_anechoic_rotated,
        #     th,
        # )

        return (
            hoa_target,
            flat_hoa_interferers,
            hoa_target_anechoic,
        )

    def save_signal_16bit(
        self, filename: str, signal: np.ndarray, norm: float = 1.0
    ) -> None:
        """Saves a signal to a 16 bit wav file.

        Args:
            filename (string): filename
            signal (np.array): signal
            norm (float): normalisation factor
        """
        signal /= norm
        n_clipped = np.sum(np.abs(signal) > 1.0)
        if n_clipped > 0:
            logger.warning("Writing %s: %s samples clipped", filename, n_clipped)
            np.clip(signal, -1.0, 1.0, out=signal)
        signal_16 = (32767 * signal).astype(np.int16)
        wavfile.write(filename, SAMPLE_RATE, signal_16)

    def generate_binaural_signals(
        self,
        scene: dict,
        hoa_target: np.ndarray,
        hoa_interferer: np.ndarray,
        hoa_target_anechoic: np.ndarray,
        out_path: str,
    ) -> None:
        """Generate and write binaural signals.

        Args:
            scene (dict): scene definitions
            hoa_target (ndarray): target signal in HOA domain
            hoa_interferer (ndarray): interferer signal in HOA domain
            hoa_target_anechoic (ndarray): anechoic target signal in HOA domain
            out_path (string): output path
        """
        ref_chan = self.reference_channel  # channel at which SNR is computed #In AVSEC is ch 0


        # Load all hrirs - one for each microphone pair
        hrir_filenames = [
            f"{self.paths.hrirs}/{name}.mat"
            for name in scene["listener"]["hrir_filename"]
        ]

        hrirs = [loadmat(hrir_filename) for hrir_filename in hrir_filenames]

        # Target and (flattened) interferer mixed down to binaural using each
        # set of hrirs
        targets = [
            hoa.binaural_mixdown(hoa_target, hrir, self.metadata.hrir_metadata)
            for hrir in hrirs
        ]
        interferers = [
            hoa.binaural_mixdown(hoa_interferer, hrir, self.metadata.hrir_metadata)
            for hrir in hrirs
        ]

        target_anechoic = hoa.binaural_mixdown(
            hoa_target_anechoic,
            hrirs[ref_chan],  # Uses the reference channel's HRIR
            self.metadata.hrir_metadata,
        )

        # Measure pre-scaled SNR at reference channel
        start_time = scene["target"]["time_start"]
        end_time = scene["target"]["time_end"]
        sw_snr = better_ear_speechweighted_snr(
            targets[ref_chan][start_time:end_time, :],
            interferers[ref_chan][start_time:end_time, :],
        )

        # Scale interferers to achieve desired SNR at reference channel
        desired_snr = scene["SNR"]
        for interferer in interferers:
            interferer *= sw_snr * hoa.dB_to_gain(-desired_snr)

        # Make binaural mixture by summing target and interferers
        mix = [t + i for t, i in zip(targets, interferers)]
        # norm = np.max(mix)

        all_signals = np.concatenate((targets, interferers, mix))
        norm_scene = np.max(np.abs(all_signals))

        # Save all signal types for all channels
        out_path = out_path.format(dataset=scene["dataset"])
        file_stem = f"{out_path}/{scene['scene']}"

        #export mixes
        # normalisation is done by scene

        if self.binaural_render:
            # export binaural mixes
            for channel, (t, i, m) in enumerate(
                    zip(targets, interferers, mix)
            ):
                for sig, sig_type in zip([t, i, m], ["target", "interferer", "mix"]):
                    self.save_signal_16bit(
                        f"{file_stem}_{sig_type}.wav", sig, norm_scene
                    )

        # save monoaural target and interferer signals. Level normalised by scene
        if self.monoaural_render:
            mono_target = [(ch1 + ch2) / 2 for ch1, ch2 in zip(targets[0][:,0], targets[0][:,1])]
            mono_interferers = [(ch1 + ch2) / 2 for ch1, ch2 in zip(interferers[0][:, 0], interferers[0][:, 1])]
            mono_mix = [t + i for t, i in zip(mono_target, mono_interferers)]
            mono_signals = np.concatenate((mono_target, mono_interferers, mono_mix))
            # normalisation is done by scene
            norm_mono_scene = np.max(np.abs(mono_signals))

            # save monoaural target
            self.save_signal_16bit(
                f"{file_stem}_target_mono.wav", mono_target, norm_mono_scene
            )
            #save monoaural interferers
            self.save_signal_16bit(
                f"{file_stem}_interferer_mono.wav", mono_interferers, norm_mono_scene
            )
            #save mono mix
            self.save_signal_16bit(
                f"{file_stem}_mono_mix.wav", mono_mix, norm_mono_scene
            )

        #save anechoic signals to compute objective metrics:

        # Save the anechoic binaural reference signal. Level normalised to abs max 1.0
        if self.binaural_render:
            norm_anechoic = np.max(np.abs(target_anechoic))
            self.save_signal_16bit(
                f"{file_stem}_target_anechoic.wav", target_anechoic, norm_anechoic
            )

        #save monoaural signal from anechoic signal
        if self.monoaural_render:
            mono_anechoic = [(ch1+ch2)/2 for ch1,ch2 in zip(target_anechoic[:, 0], target_anechoic[:, 1])]
            # save monoaural target anechoic. Level normalised to abs max 1.0
            norm_anechoic = np.max(np.abs(mono_anechoic))
            self.save_signal_16bit(
                f"{file_stem}_target_mono_anechoic.wav", mono_anechoic, norm_anechoic
            )

        # Export video
        logger.info("generating scene: %s in %s dataset", scene["scene"],scene["dataset"])
        # Write video file without audio stream to scenes folder
        target_video_path = self.paths.videos.format(dataset=scene["dataset"])
        target_video_fn = f"{target_video_path}/{scene['target']['name']}.mp4"
        #path to output silent video
        output_video_fn = f"{out_path}/{scene['scene']}_silent.mp4"
        command = f"ffmpeg -v 8 -i {target_video_fn} -c:v copy -an {output_video_fn} < /dev/null"
        os.system(command)


    def render_scenes(self, scenes: dict):
        """Renders scenes.

        Args:
            scenes (dict): scene definitions
        """
        # Make all necessary output directories
        output_path = self.paths.scenes
        for dataset in {s["dataset"] for s in scenes}:
            full_output_path = Path(output_path.format(dataset=dataset))
            if not full_output_path.exists():
                full_output_path.mkdir(parents=True, exist_ok=True)

        for scene in tqdm(scenes):
            # Stage 1: Generate the rotated signals in the HOA domain
            # target, interferers, anechoic, head_turn = self.generate_hoa_signals(scene)
            target, interferers, anechoic = self.generate_hoa_signals(scene)

            #bypassing head rotation (not used in AVSEC)
            # Save the head rotation signal
            # head_turn = (np.mod(head_turn, 2 * np.pi) - np.pi) / np.pi
            # file_stem = (
            #     output_path.format(dataset=scene["dataset"]) + "/" + scene["scene"]
            # )
            # wavfile.write(f"{file_stem}_hr.wav", 44100, head_turn)

            # Stage 2: Mix down to the binaural domain and export video files
            self.generate_binaural_signals(
                scene, target, interferers, anechoic, output_path
            )