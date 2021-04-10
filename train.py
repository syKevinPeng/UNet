import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import preprocessing

# define a convolution block which consist of Conv2d -> BatchNorm -> Relu
class ConvBlock(nn.module):
    def __init__(self, in_channels, out_channels, filter_size =3):
        super().__init__()
        self.conv_block = nn.Sequential(
            # we are suppose to do "same" padding
            nn.Conv2d(in_channels,out_channels, kernel_size=filter_size, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)

# Define encoder block, which include max poll-> convBlock -> convBlock
class EncoderBlock(nn.module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels,out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder_block(x)


# Define a deconvolution block, which is used to up-sampling the feature map.
class DecoderBlock(nn.module):
    def __init__(self, in_channels, out_channels):
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels //2 , kernel_size = 2, stride = 2),
            ConvBlock(in_channels, out_channels, filter_size=3),
            ConvBlock(out_channels, out_channels, filter_size=3)
        )

    def forward(self, x1, x2):
        pass


# Define last output layer: a conv layer with 1x1 filter size.
class OutputLayer(nn.module):
    def __init__(self, in_channels, out_channels):
        super(OutputLayer, self).__init__()
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.output(x)



class UNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(UNet, self).__init__()
        self.n_channels = input_channel
        self.n_classes = n_classes
        # define different layers
        self.in_layer_1 = ConvBlock(self.n_channels, 64)
        self.in_layer_2 = ConvBlock(64, 64)
        self.encoder1 = EncoderBlock(64, 128)
        self.encoder2 = EncoderBlock(128, 256)
        self.encoder3 = EncoderBlock(256, 512)
        self.encoder4 = EncoderBlock(512, 1024)
        # concat prev encoder layer. i.e. skip connection
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        self.out_layer = OutputLayer(64, self.n_classes)


    def forward(self, input):
        pass


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = preprocessing()
    util.dataloader_tester(train_dataloader, val_dataloader, test_dataloader)