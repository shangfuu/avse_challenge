import random
import time

import librosa
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import num_frames, num_stft_frames, stft_size, window_shift, window_size
from utils.nn import MultiscaleMultibranchTCN, TCN, threeD_to_2D_tensor
from utils.resnet import BasicBlock, ResNet


class VisualFeatNet(nn.Module):
    def __init__(self, hidden_dim=256, backbone_type='resnet', num_classes=500,
                 relu_type='prelu', tcn_options=None, extract_feats=False):
        super(VisualFeatNet, self).__init__()
        if tcn_options is None:
            tcn_options = {}
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(3, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                      bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
        self.tcn = tcn_class(input_size=self.backend_out,
                             num_channels=[hidden_dim * len(tcn_options['kernel_size']) * tcn_options[
                                 'width_mult']] *
                                          tcn_options['num_layers'],
                             num_classes=num_classes,
                             tcn_options=tcn_options,
                             dropout=tcn_options['dropout'],
                             relu_type=relu_type,
                             dwpw=tcn_options['dwpw'],
                             )

    def forward(self, x, lengths):
        B, C, T, H, W = x.size()
        if (type(lengths) == int):
            lengths = [lengths] * B
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        # return x if self.extract_feats else self.tcn(x, lengths, B)
        return self.tcn(x, lengths, B, self.extract_feats).squeeze(2).permute(0, 2, 1)


def build_audiofeat_net(filters=64, input_nc=1, output_nc=1, visual_feat_dim=1280, weights='', a_only=False,
                        activation="Sigmoid"):
    net = AudioFeatNet(filters=filters)

    if len(weights) > 0:
        print('Loading weights for UNet')
        net.load_state_dict(torch.load(weights))
    return net


def build_visualfeat_net(weights='', extract_feats=True):
    net = VisualFeatNet(tcn_options=dict(num_layers=4, kernel_size=[3], dropout=0.2, dwpw=False, width_mult=2),
                        relu_type="prelu",
                        extract_feats=extract_feats)
    if len(weights) > 0:
        print('Loading weights for lipreading stream')
        net.load_state_dict(torch.load(weights))
    return net


class FusionNet(nn.Module):
    def __init__(self, a_only=False, mask="ibm"):
        super(FusionNet, self).__init__()
        if a_only:
            visfeat_size = 0
        else:
            visfeat_size = 512
        self.lstm_conv = nn.LSTM(visfeat_size + 1028, stft_size // 2 + 1, num_layers=1, batch_first=True)
        self.time_distributed_1 = nn.Linear(in_features=stft_size // 2 + 1, out_features=stft_size // 2 + 1)
        torch.nn.init.xavier_uniform_(self.time_distributed_1.weight)
        self.activation = F.sigmoid

    def forward(self, input):
        x = self.lstm_conv(input)[0]
        pred_mask = self.activation(self.time_distributed_1(x))
        return pred_mask


class AVNet(LightningModule):
    def __init__(self, nets, loss="bce", lr=0.001, val_dataset=None, a_only=False):
        super(AVNet, self).__init__()
        self.a_only = a_only
        self.net_visualfeat, self.net_audiofeat, self.net_fusion = nets
        if loss.lower() == "l1":
            self.loss = F.l1_loss
        elif loss.lower() == "l2":
            self.loss = F.mse_loss
        else:
            raise NotImplementedError(
                "{} is currently unavailable as loss function. Select one of l1, l2 and bce".format(loss))
        self.lr = lr
        self.val_dataset = val_dataset

    def forward(self, input):
        noisy_audio_spec = input['noisy_audio_spec']
        _, _, num_aud_feat, _ = noisy_audio_spec.shape
        if self.a_only:
            combined = self.net_audiofeat(noisy_audio_spec)
        else:
            lip_images = input['lip_images']
            _, _, num_vis_feat, _, _ = lip_images.shape
            visual_feat = self.net_visualfeat(lip_images.float(), 75).unsqueeze(1)
            upsampled_visual_feat = F.interpolate(visual_feat, size=(num_aud_feat, 512)).reshape(-1, num_aud_feat,
                                                                                                    512)
            audio_feat = self.net_audiofeat(noisy_audio_spec)
            combined = torch.cat((upsampled_visual_feat, audio_feat), dim=-1)
        mask = self.net_fusion(combined)
        return torch.mul(noisy_audio_spec, mask.unsqueeze(1))

    def training_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                for index in range(5):
                    rand_int = random.randint(0, len(self.val_dataset))
                    data = self.val_dataset[rand_int]
                    inputs = {
                        "noisy_audio_spec": torch.from_numpy(data["noisy_audio_spec"][np.newaxis, ...]).to(self.device)}
                    if not self.a_only:
                        inputs["lip_images"] = torch.from_numpy(data["lip_images"][np.newaxis, ...]).to(self.device)
                    pred_mag = self(inputs)[0][0].cpu().numpy()
                    noisy_phase = np.angle(data["noisy_stft"])
                    estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                    estimated_audio = librosa.istft(estimated.T, win_length=window_size, hop_length=window_shift,
                                                    window="hann", center=True)
                    noisy = librosa.istft(data["noisy_stft"].T, win_length=window_size, hop_length=window_shift,
                                          window="hann", center=True)
                    tensorboard.add_audio("{}/{}_clean".format(self.current_epoch, index), data["clean"][np.newaxis, ...],
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_noisy".format(self.current_epoch, index), noisy[np.newaxis, ...], sample_rate=16000)
                    tensorboard.add_audio("{}/{}_enhanced".format(self.current_epoch, index), estimated_audio[np.newaxis, ...],
                                          sample_rate=16000)

    def cal_loss(self, batch_inp):
        mask = batch_inp["mask"]
        pred_mask = self(batch_inp)
        loss = self.loss(pred_mask, mask)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.8, patience=2),
                "monitor": "val_loss_epoch",
            },
        }


class AudioFeatNet(nn.Module):

    def __init__(self, num_conv=5, kernel_size=5, filters=64, last_filter=4, dilation=True, batch_norm=True,
                 fc_layers=0, lstm_layers=0, lr=0.0003):
        super(AudioFeatNet, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.last_filter = last_filter
        self.batch_norm = batch_norm
        self.dilation = dilation
        self.num_conv = num_conv
        self.lstm_layers = lstm_layers
        self.fc_layers = fc_layers
        self.lr = lr
        self.embed_size = stft_size // 2 + 1

        if batch_norm:
            setattr(self, "bn0", nn.BatchNorm2d(1))
        for i in range(num_conv):
            if i == 0:
                inp_filter = 1
                out_filter = self.filters
            else:
                inp_filter, out_filter = self.filters, self.filters
            if self.dilation:
                dilation_size = 2 ** i
                padding = (self.kernel_size - 1) * dilation_size
            else:
                padding = self.kernel_size - 1
                dilation_size = 1
            setattr(self, "conv{}".format(i + 1),
                    nn.Conv2d(inp_filter, out_filter, (self.kernel_size, self.kernel_size), padding=padding // 2,
                              dilation=dilation_size))
            if batch_norm:
                setattr(self, "bn{}".format(i + 1), nn.BatchNorm2d(out_filter))
        if num_conv == 0:
            inp_filter = 2
        else:
            inp_filter = self.filters
        self.convf = nn.Conv2d(inp_filter, self.last_filter, (1, 1), padding=0, dilation=(1, 1))
        if batch_norm:
            self.bn_last = nn.BatchNorm2d(self.last_filter)
        last_conv = True
        for i in range(lstm_layers):
            if i == 0 and not last_conv and num_conv == 0:
                input_size = 2 * self.embed_size
            elif last_conv and num_conv == 0:
                input_size = self.last_filter * self.embed_size
            elif not last_conv and num_conv != 0:
                input_size = self.filters * self.embed_size
            elif i == 0 and last_conv:
                input_size = self.last_filter * self.embed_size
            else:
                input_size = self.embed_size
            setattr(self, "lstm{}".format(i + 1), nn.LSTM(input_size, self.embed_size))

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, x):
        _, _, num_aud_feat, _ = x.shape
        if self.batch_norm:
            x = getattr(self, "bn0")(x)
        for i in range(self.num_conv):
            x = getattr(self, "conv{}".format(i + 1))(x)
            if self.batch_norm:
                x = getattr(self, "bn{}".format(i + 1))(x)
            x = F.relu(x)
        x = self.convf(x)
        if self.batch_norm:
            x = self.bn_last(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1, 3).reshape(-1, num_aud_feat, self.embed_size * self.last_filter)
        for i in range(self.lstm_layers):
            if i == 0:
                x = x.transpose(1, 0)
            getattr(self, "lstm{}".format(i + 1)).flatten_parameters()
            x = getattr(self, "lstm{}".format(i + 1))(x)
            x = x[0]
        if self.lstm_layers > 0:
            x = x.transpose(1, 0)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    audiofeat_net = build_audiofeat_net(a_only=False)
    visual_net = build_visualfeat_net(extract_feats=True)
    fusion_net = FusionNet(a_only=False)
    net = AVNet((visual_net, audiofeat_net, fusion_net), "l2", a_only=False)
    net.eval()
    audiofeat_net.eval()
    visual_net.eval()
    # test_audio_data = torch.rand((1, num_stft_frames, stft_size // 2 + 1))
    # test_visual_data = torch.rand([1, 1, num_frames, 88, 88])
    test_audio_data = torch.rand((1, 1, num_stft_frames, stft_size // 2 + 1))
    test_visual_data = torch.rand([1, 3, num_frames, 224, 224])

    with torch.no_grad():
        start_time = time.time()
        pred_mask = audiofeat_net(test_audio_data).detach().numpy()
        print(time.time() - start_time)
        print("Audio-only Feat", pred_mask.shape)
        # print(np.min(pred_mask), np.max(pred_mask))
        start_time = time.time()
        visual_feat = visual_net(test_visual_data, 75)
        print(time.time() - start_time)
        print("Visual feat", visual_feat.shape)
        # print(net)
        warmup = net({'noisy_audio_spec': test_audio_data, "lip_images": test_visual_data})
        start_time = time.time()
        print("Audio-visual Net", net({'noisy_audio_spec': test_audio_data, "lip_images": test_visual_data}).shape)
        print(time.time() - start_time)
