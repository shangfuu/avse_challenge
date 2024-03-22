import keras.ops as ops
import keras.layers as nn
from keras import Sequential


def unet_conv(output_nc, norm_layer=nn.BatchNormalization):
    unet_conv = Sequential(
        [nn.Conv2D(output_nc, kernel_size=4, strides=(2, 2), padding="same"),
         norm_layer(),
         nn.LeakyReLU(0.2)]
    )
    return unet_conv


def unet_upconv(output_nc, outermost=False, norm_layer=nn.BatchNormalization, kernel_size=4):
    upconv = nn.Conv2DTranspose(output_nc, kernel_size=kernel_size, strides=2, padding="same")
    uprelu = nn.ReLU(True)
    upnorm = norm_layer()
    if not outermost:
        return Sequential([upconv, upnorm, uprelu])
    else:
        return Sequential([upconv])


def conv_block(ch_out):
    block = Sequential(
        [nn.Conv2D(ch_out, kernel_size=3, strides=1, padding="same", use_bias=True),
         nn.BatchNormalization(),
         nn.LeakyReLU(0.2),
         nn.Conv2D(ch_out, kernel_size=3, strides=1, padding="same", use_bias=True),
         nn.BatchNormalization(),
         nn.LeakyReLU(0.2)]
    )
    return block


def up_conv(ch_out, outermost=False):
    if not outermost:
        up = Sequential(
            [nn.UpSampling2D(size=(2, 1)),
             nn.Conv2D(ch_out, kernel_size=3, strides=1, padding="same", use_bias=True),
             nn.BatchNormalization(),
             nn.ReLU()]
        )
    else:
        up = Sequential(
            [nn.UpSampling2D(size=(2, 1)),
             nn.Conv2D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=True),
             nn.Activation(activation="sigmoid")]
        )
    return up


if __name__ == '__main__':
    model = up_conv(1)
    print(model(ops.ones((1, 128, 128, 1))).shape)