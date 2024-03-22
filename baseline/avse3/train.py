from config import *
import argparse
from loss import si_snr_loss
from model import AVSE
from config import *
import keras
import time
from os.path import join

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard


from argparse import ArgumentParser
from dataset import AVSEChallengeDataModule


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    dataset = AVSEChallengeDataModule(data_root=args.data_root, batch_size=args.batch_size, time_domain=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=10 ** (-10), cooldown=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=5, mode='auto')
    checkpointer = ModelCheckpoint(join(args.log_dir, '{epoch:03d}_{val_loss:04f}.weights.h5'),
                                   monitor='val_loss', save_best_only=False, save_weights_only=True,
                                   mode='auto', save_freq='epoch')
    tensorboard = TensorBoard(log_dir=args.log_dir)

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model = AVSE(batch_size=args.batch_size, video_frames=max_video_length, audio_frames=max_audio_length)
    model.summary()
    model.compile(optimizer=optimizer, loss=si_snr_loss)
    if args.checkpoint is not None:
        model.load_weights(args.checkpoint)
    start = time.time()
    model.fit(dataset.train_dataloader(), epochs=args.max_epochs,
              validation_data=dataset.val_dataloader(),
              callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard])
    print(f"Time taken {time.time() - start} sec")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Path to log directory")
    args = parser.parse_args()
    main(args)
