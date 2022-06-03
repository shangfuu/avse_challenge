# -*- coding: utf-8 -*-
'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

import os
import math
import logging
import numpy as np
import soundfile

from soundfile import SoundFile
from scipy.signal import convolve

from utils import speechweighted_snr, sum_signals, pad

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Renderer:
    """
    SceneGenerator of AVSE1 training and development sets. The render() function generates all simulated signals for each
    scene given the parameters specified in the metadata/scenes.train.json or metadata/scenes.dev.json file.
    """

    def __init__(
        self,
        input_path,
        output_path,
        num_channels=1,
        fs=44100,
        ramp_duration=0.5,
        tail_duration=0.2,
        test_nbits=16,
    ):

        self.input_path = input_path
        self.output_path = output_path
        self.fs = fs
        self.ramp_duration = ramp_duration
        self.n_tail = int(tail_duration * fs)
        self.test_nbits = test_nbits
        self.floating_point = False

        self.channels = list(range(num_channels))

    def read_signal(
        self, filename, offset=0, nsamples=-1, nchannels=0, offset_is_samples=False
    ):
        """Read a wavefile and return as numpy array of floats.
        Args:
            filename (string): Name of file to read
            offset (int, optional): Offset in samples or seconds (from start). Defaults to 0.
            nchannels: expected number of channel (default: 0 = any number OK)
            offset_is_samples (bool): measurement units for offset (default: False)
        Returns:
            ndarray: audio signal
        """
        try:
            wave_file = SoundFile(filename)
        except:
            # Ensure incorrect error (24 bit) is not generated
            raise Exception(f"Unable to read {filename}.")

        if nchannels != 0 and wave_file.channels != nchannels:
            raise Exception(
                f"Wav file ({filename}) was expected to have {nchannels} channels."
            )

        if wave_file.samplerate != self.fs:
            raise Exception(f"Sampling rate is not {self.fs} for filename {filename}.")

        if not offset_is_samples:  # Default behaviour
            offset = int(offset * wave_file.samplerate)

        if offset != 0:
            wave_file.seek(offset)

        x = wave_file.read(frames=nsamples)
        return x

    def write_signal(self, filename, x, fs, floating_point=True):
        """Write a signal as fixed or floating point wav file."""

        if fs != self.fs:
            logging.warning(f"Sampling rate mismatch: {filename} with sr={fs}.")
            # raise ValueError("Sampling rate mismatch")

        if floating_point is False:
            if self.test_nbits == 16:
                subtype = "PCM_16"
                # If signal is float and we want int16
                x *= 32768
                x = x.astype(np.dtype("int16"))
                assert np.max(x) <= 32767 and np.min(x) >= -32768
            elif self.test_nbits == 24:
                subtype = "PCM_24"
        else:
            subtype = "FLOAT"

        soundfile.write(filename, x, fs, subtype=subtype)

    def save_signal_16bit(self, filename, signal, fs, norm=1.0):
        """Saves a signal to a 16 bit wav file.
        Args:
            filename (string): filename
            signal (np.array): signal
            norm (float): normalisation factor
        """
        signal /= norm
        n_clipped = np.sum(np.abs(signal) > 1.0)
        if n_clipped > 0:
            print("CLIPPED {} {} {}".format(norm,np.max(signal),np.min(signal)))
            logging.warning(f"Writing {filename}: {n_clipped} samples clipped")
            np.clip(signal, -1.0, 1.0, out=signal)
        signal_16 = (32767 * signal).astype(np.int16)

        # wavfile.write(filename, FS, signal_16)
        soundfile.write(filename, signal_16, fs, subtype="PCM_16")

    def apply_ramp(self, x, dur):
        """Apply half cosine ramp into and out of signal

        dur - ramp duration in seconds
        """
        ramp = np.cos(np.linspace(math.pi, 2 * math.pi, int(self.fs * dur)))
        ramp = (ramp + 1) / 2
        y = np.array(x)
        y[0 : len(ramp)] *= ramp
        y[-len(ramp) :] *= ramp[::-1]
        return y

    def compute_snr(self, target, noise):
        """Return the SNR.
        Take the overlapping segment of the noise and get the speech-weighted
        better ear SNR. (Note, SNR is a ratio -- not in dB.)
        """
        segment_target = target
        segment_noise = noise
        assert len(segment_target) == len(segment_noise)

        snr = speechweighted_snr(segment_target, segment_noise)

        return snr

    def render(
        self,
        target,
        noise_type,
        interferer,
        scene,
        offset,
        snr_dB,
        dataset,
    ):

        target_video_fn = f"{self.input_path}/{dataset}/targets_video/{target}.mp4"
        target_fn = f"{self.input_path}/{dataset}/targets/{target}.wav"

        target_fn_dir = os.path.dirname(target_fn)
        create_dir(target_fn_dir)

        command = ("ffmpeg -v 8 -y -i %s -vn -acodec pcm_s16le -ar %s -ac 1 %s < /dev/null" % (target_video_fn, str(self.fs), target_fn))
        os.system(command)

        interferer_fn = (
            f"{self.input_path}/{dataset}/interferers/{noise_type}/{interferer}.wav"
        )

        target = self.read_signal(target_fn)

        interferer = self.read_signal(
            interferer_fn, offset=offset, nsamples=len(target), offset_is_samples=True
        )

        if len(target) != len(interferer):
            logging.debug("Target and interferer have different lengths")

        # Apply 500ms half-cosine ramp
        interferer = self.apply_ramp(interferer, dur=self.ramp_duration)

        prefix = f"{self.output_path}/{scene}"

        snr_ref = None

        target_at_ear = target
        interferer_at_ear = interferer

        # Scale interferer to obtain SNR specified in scene description
        logging.info(f"Scaling interferer to obtain mixture SNR = {snr_dB} dB.")

        if snr_ref is None:
            # snr_ref computed for first channel in the list and then
            # same scaling applied to all
            snr_ref = self.compute_snr(
                target_at_ear,
                interferer_at_ear,
            )

        if snr_ref == np.Inf:
             print(f"Scene {scene} was skipped")
             return

        # Apply snr_ref reference scaling to get 0 dB and then scale to target snr_dB
        interferer_at_ear = interferer_at_ear * snr_ref
        interferer_at_ear = interferer_at_ear * 10 ** ((-snr_dB) / 20)

        # Sum target and scaled and ramped interferer
        signal_at_ear = sum_signals([target_at_ear, interferer_at_ear])
        outputs = [
                (f"{prefix}_mixed.wav", signal_at_ear),
                (f"{prefix}_target.wav", target_at_ear),
                (f"{prefix}_interferer.wav", interferer_at_ear),
        ]
        all_signals = np.concatenate((signal_at_ear,target_at_ear,interferer_at_ear))
        norm = np.max(np.abs(all_signals))

        # Write all audio output files
        for (filename, signal) in outputs:
            self.save_signal_16bit(filename, signal, self.fs, norm=norm)

        # Write video file without audio stream
        output_video_fn = f"{prefix}_silent.mp4"
        command = f"ffmpeg -v 8 -i {target_video_fn} -c:v copy -an {output_video_fn} < /dev/null"
        os.system(command)

def check_scene_exists(scene, output_path):
    """Checks correct dataset directory for full set of pre-existing files.

    Args:
        scene (dict): dictionary defining the scene to be generated.

    Returns:
        status: boolean value indicating whether scene signals exist
            or do not exist.

    """

    pattern = f"{output_path}/{scene['scene']}"
    files_to_check = [
        f"{pattern}_mixed.wav",
        f"{pattern}_target.wav",
        f"{pattern}_interferer.wav",
    ]

    scene_exists = True
    for filename in files_to_check:
        scene_exists = scene_exists and os.path.exists(filename)
    return scene_exists


def main():
    import json

    scene = json.load(
        open(
            "/tmp/avse1_data/metadata/scenes.train.json",
            "r",
        )
    )[0]

    renderer = Renderer(
        input_path="/tmp/avse1_data/",
        output_path=".",
        num_channels=1,
    )
    renderer.render(
        dataset=scene["dataset"],
        target=scene["target"]["name"],
        noise_type=scene["interferer"]["type"],
        interferer=scene["interferer"]["name"],
        scene=scene["scene"],
        offset=scene["interferer"]["offset"],
        snr_dB=scene["SNR"],
    )

