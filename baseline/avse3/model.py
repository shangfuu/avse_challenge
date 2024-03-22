from config import *
from keras import Input
from model_utils import *


@keras.saving.register_keras_serializable(name="VisualFeatNet")
class VisualFeatNet(Layer):
    def __init__(self, tcn_options=None, hidden_dim=256):
        super().__init__(name="visual_feat_extract")
        if tcn_options is None:
            self.tcn_options = dict(num_layers=4, kernel_size=[3], dropout=0.2, width_mult=2)
        self.frontend_nout = 64
        self.backend_out = 512
        self.hidden_dim = hidden_dim
        self.trunk = ResNet18()
        self.frontend3D = Sequential([
            nn.Conv3D(self.frontend_nout, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding="same", use_bias=False),
            nn.BatchNormalization(),
            nn.ReLU(),
            nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding="valid")])
        self.tcn = TCN([self.hidden_dim * len(self.tcn_options['kernel_size']) * self.tcn_options['width_mult']] *
                       self.tcn_options['num_layers'],
                       self.tcn_options["kernel_size"],
                       self.tcn_options["num_layers"],
                       dilations=[1, 2, 4, 8], return_sequences=True, activation="relu", use_batch_norm=True,
                       padding="same",
                       dropout_rate=self.tcn_options["dropout"])

    def call(self, x):
        x = self.frontend3D(x)
        B, T, H, W, C = x.shape
        if B is None:
            B = 1
        x = ops.reshape(x, (-1, H, W, C))
        x = self.trunk(x)
        x = ops.reshape(x, (B, T, -1))
        x = self.tcn(x)
        return ops.reshape(x, (B, 1, T, -1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, input_shape[1], 512


@keras.saving.register_keras_serializable(name="UNet")
class UNet(Layer):
    def __init__(self, filters=64, output_nc=2, av_embedding=1024, a_only=True, activation='sigmoid'):
        super().__init__(name="audio_separator")
        self.a_only = a_only
        self.filters = filters
        self.output_nc = output_nc
        self.av_embedding = av_embedding
        self.activation = activation
        self.conv1 = unet_conv(self.filters)
        self.conv2 = unet_conv(self.filters * 2)
        self.conv3 = conv_block(self.filters * 4)
        self.conv4 = conv_block(self.filters * 8)
        self.conv5 = conv_block(self.filters * 8)
        self.conv6 = conv_block(self.filters * 8)
        self.conv7 = conv_block(self.filters * 8)
        self.conv8 = conv_block(self.filters * 8)
        self.frequency_pool = nn.MaxPool2D([2, 1])
        if not self.a_only:
            self.upconv1 = up_conv(self.filters, self.filters * 8)
        else:
            self.upconv1 = up_conv(self.filters * 8)
        self.upconv2 = up_conv(self.filters * 8)
        self.upconv3 = up_conv(self.filters * 8)
        self.upconv4 = up_conv(self.filters * 8)
        self.upconv5 = up_conv(self.filters * 4)
        self.upconv6 = up_conv(self.filters * 2)
        self.upconv7 = unet_upconv(self.filters)
        self.upconv8 = unet_upconv(self.output_nc, True)
        self.activation = nn.Activation(self.activation.lower())

    def call(self, mix_spec, visual_feat=None):
        noisy_stft_real, noisy_stft_imag = ops.stft(mix_spec, sequence_length=window_size, sequence_stride=window_shift,
                                                    fft_length=stft_size)
        noisy_stft_real = ops.expand_dims(noisy_stft_real, axis=-1)
        noisy_stft_imag = ops.expand_dims(noisy_stft_imag, axis=-1)
        noisy_stft = ops.concatenate((noisy_stft_real, noisy_stft_imag), axis=-1)#** 0.3
        feat, pads = pad(noisy_stft, 32)
        conv1feat = self.conv1(feat)
        conv2feat = self.conv2(conv1feat)
        conv3feat = self.conv3(conv2feat)
        conv3feat = self.frequency_pool(conv3feat)
        conv4feat = self.conv4(conv3feat)
        conv4feat = self.frequency_pool(conv4feat)
        conv5feat = self.conv5(conv4feat)
        conv5feat = self.frequency_pool(conv5feat)
        conv6feat = self.conv6(conv5feat)
        conv6feat = self.frequency_pool(conv6feat)
        conv7feat = self.conv7(conv6feat)
        conv7feat = self.frequency_pool(conv7feat)
        conv8feat = self.conv8(conv7feat)
        conv8feat = self.frequency_pool(conv8feat)
        if self.a_only:
            av_feat = conv8feat
        else:
            B, H, W, C = conv8feat.shape
            upsample_visuals = ops.image.resize(visual_feat, (H, W))
            av_feat = ops.concatenate((conv8feat, upsample_visuals), axis=-1)
        upconv1feat = self.upconv1(av_feat)
        upconv2feat = self.upconv2(ops.concatenate((upconv1feat, conv7feat), axis=-1))
        upconv3feat = self.upconv3(ops.concatenate((upconv2feat, conv6feat), axis=-1))
        upconv4feat = self.upconv4(ops.concatenate((upconv3feat, conv5feat), axis=-1))
        upconv5feat = self.upconv5(ops.concatenate((upconv4feat, conv4feat), axis=-1))
        upconv6feat = self.upconv6(ops.concatenate((upconv5feat, conv3feat), axis=-1))
        upconv7feat = self.upconv7(ops.concatenate((upconv6feat, conv2feat), axis=-1))
        predicted_mask = self.upconv8(ops.concatenate((upconv7feat, conv1feat), axis=-1))
        pred_mask = self.activation(predicted_mask)
        pred_mask = unpad(pred_mask, pads)
        enhanced_stft = ops.multiply(pred_mask, noisy_stft) #** (1/0.3)
        enhanced_audio = ops.istft((enhanced_stft[:, :, :, 0], enhanced_stft[:, :, :, 1]),
                                   sequence_length=window_size, sequence_stride=window_shift,
                                   fft_length=stft_size)
        return enhanced_audio

    def compute_output_shape(self, input_shape):
        return input_shape


def AVSE(video_frames=64, audio_frames=40800, batch_size=1):
    visual_input = Input(batch_shape=(batch_size, video_frames, video_frame_size[0], video_frame_size[1], 1),
                         name="video_frames")
    audio_input = Input(batch_shape=(batch_size, audio_frames), name="noisy_audio")
    visual_feat = VisualFeatNet()(visual_input)
    output = UNet(a_only=False)(audio_input, visual_feat)
    return Model(inputs=[audio_input, visual_input], outputs=output)


if __name__ == '__main__':
    model = AVSE(batch_size=1)
    print(model.predict({"noisy_audio": ops.ones((1, 40800)), "video_frames": ops.ones((1, 64, 88, 88, 1))}).shape)
    model.summary()
