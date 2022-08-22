import numpy as np
import torch
from dataset import TEDDataModule
from model import AVNet, FusionNet, build_audiofeat_net, build_visualfeat_net

SEED = 1143
# fix random seeds for reproducibility
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.generic import str2bool


def main(args):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch")
    datamodule = TEDDataModule(batch_size=args.batch_size, mask=args.mask, a_only=args.a_only)
    audiofeat_net = build_audiofeat_net(a_only=args.a_only)
    visual_net = build_visualfeat_net(extract_feats=True)
    fusion_net = FusionNet(a_only=args.a_only, mask=args.mask)

    if args.a_only:
        model = AVNet((None, audiofeat_net, fusion_net), args.loss, a_only=args.a_only,
                      val_dataset=datamodule.dev_dataset)
    else:
        model = AVNet((visual_net, audiofeat_net, fusion_net), args.loss, a_only=args.a_only,
                      val_dataset=datamodule.dev_dataset)
    trainer = Trainer.from_argparse_args(args, default_root_dir=args.log_dir, callbacks=[checkpoint_callback])
    if args.tune:
        trainer.tune(model, datamodule)
    else:
        trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, default=False)
    parser.add_argument("--tune", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00158)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--mask", type=str, default="mag")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
