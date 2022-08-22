from scipy.signal import windows as w

SEED = 999999
dB_levels = [0, 3, 6, 9]
sampling_rate = 16000
img_rows, img_cols = 224, 224
windows = w.hann
# windows = w.hamming

max_frames = 75
stft_size = 512
window_size = 512
window_shift = 128
window_length = None
fading = False

max_utterance_length = 48000
num_frames = int(25 * (max_utterance_length / 16000))
num_stft_frames = 376#int((max_utterance_length - window_size + window_shift) / window_shift)

nb_channels, img_height, img_width = 1, img_rows, img_cols
DATA_ROOT = "/home/mgo/Documents/data/avse1_data/"
METADATA_ROOT = "/home/mgo/Documents/data/avse1_data/metadata/"