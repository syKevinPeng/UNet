import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
'''
------- Unet Component -------
'''
# define a convolution block which consist of Conv2d -> BatchNorm -> Relu
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride = 1):
        super().__init__()
        self.conv_block = nn.Sequential(
            # we are suppose to do "same" padding
            nn.Conv2d(in_channels,out_channels, kernel_size=filter_size, padding=1, padding_mode='replicate', stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x.float())


# Define encoder block, which include max poll-> convBlock -> convBlock
class EncoderBlock(nn.Module):
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
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_filter_size=3, conv_stride=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels //2, kernel_size=2, stride=2)
        self.two_convs = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def padding(self, x1, x2):
        diff_vertical = x2.size()[2] - x1.size()[2]
        diff_horizontal = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_horizontal // 2, diff_horizontal - diff_horizontal // 2, diff_vertical // 2, diff_vertical - diff_vertical // 2])
        return x1

    def forward(self, x1, x2):
        # pipeline: calculate up-sampling -> concat with prev layer -> 2 x conv
        x1 = self.deconv(x1)
        x1 = self.padding(x1, x2)
        concat = torch.cat([x2, x1], 1)  # concat on dimension 1
        result = self.two_convs(concat)
        return result


# Define last output layer: a conv layer with 1x1 filter size.
class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputLayer, self).__init__()
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.output(x)

'''
------- ResUnet Component -------
This architecture is based on paper road Extraction by Deep Residual U-Net: https://arxiv.org/pdf/1711.10684.pdf
'''

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class UpsampleResidual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleResidual, self).__init__()
        self.upsample = nn.Upsample(in_channels,in_channels, kernel=2, stride=2)
        self.residual = ResidualConv(in_channels+out_channels, out_channels, 1, 1)


    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        concat = torch.cat([x1, x2], dim=1)
        return self.residual(concat)


class residual_input(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(residual_input, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
    def forward(self,x):
        return self.input(x)


class double_residual(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_chaneels):
        super(double_residual, self).__init__()
        self.double_residual = nn.Sequential(
            ResidualConv(in_channels, intermediate_channels, 2, 1),
            ResidualConv(intermediate_channels, out_chaneels, 2, 1)
        )
    def forward(self, x):
        self.double_residual(x)

'''
------- Define Architecture -------
'''
class ResUnet(nn.Module):
    def __init__(self, input_channel, n_classes, channel_list=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.n_channels = input_channel
        self.n_classes = n_classes
        # Input layers
        self.in_layer = residual_input(input_channel, channel_list[0])
        self.in_layer_skip = nn.Conv2d(input_channel, channel_list[0], 3, 1)
        # Encoder
        self.encoder_1 = ResidualConv(channel_list[0], channel_list[1], 2, 1)
        self.encoder_2 = ResidualConv(channel_list[1], channel_list[2], 2, 1)
        self.skip = ResidualConv(channel_list[2], channel_list[3], 2, 1)
        # Decoder
        self.upsample_residual_1 = UpsampleResidual(channel_list[3], channel_list[2])
        self.upsample_residual_2 = UpsampleResidual(channel_list[2], channel_list[1])
        self.upsample_residual_3 = UpsampleResidual(channel_list[1], channel_list[0])
        # Output Layer
        self.output = OutputLayer(channel_list[0], n_classes)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        # Encode
        x2 = self.encoder_1(x1)
        x3 = self.encoder_2(x2)
        x4 = self.skip(x3)
        # Decode
        x5 = self.upsample_residual_1(x4, x3)
        x6 = self.upsample_residual_2(x5, x2)
        x7 = self.upsample_residual_3(x6, x1)

        output = self.output(x7)
        return output



class UNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(UNet, self).__init__()
        # define different layers
        self.n_channels = input_channel
        self.n_classes = n_classes
        # define input layers
        self.in_layer_1 = ConvBlock(self.n_channels, 64)
        self.in_layer_2 = ConvBlock(64, 64)
        # define encoder layers
        self.encoder1 = EncoderBlock(64, 128)
        self.encoder2 = EncoderBlock(128, 256)
        self.encoder3 = EncoderBlock(256, 512)
        self.encoder4 = EncoderBlock(512, 1024)
        # define decoder layers
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        # define output layers
        self.out_layer = OutputLayer(64, self.n_classes)


    def forward(self, x):
        # input layer
        x = self.in_layer_1(x)
        i = self.in_layer_2(x)
        # downsampling / conv
        e1 = self.encoder1(i)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # concat prev encoder layer. i.e. skip connection
        # name the output as x to save graphic memory
        x = self.decoder1(e4, e3)
        x = self.decoder2(x, e2)
        x = self.decoder3(x, e1)
        x = self.decoder4(x, i)
        logits = self.out_layer(x)
        return logits

class MiniUnet(nn.Module):
    # Mini Unet reduce the number encoder and decoder
    def __init__(self, input_channel, n_classes):
        super(MiniUnet, self).__init__()
        # define different layers
        self.n_channels = input_channel
        self.n_classes = n_classes
        # define input layers
        self.in_layer_1 = ConvBlock(self.n_channels, 64)
        self.in_layer_2 = ConvBlock(64, 64)
        # define encoder layers
        self.encoder1 = EncoderBlock(64, 128)
        self.encoder2 = EncoderBlock(128, 256)
        self.encoder3 = EncoderBlock(256, 512)
        # define decoder layers
        self.decoder1 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder3 = DecoderBlock(128, 64)
        # define output layers
        self.out_layer = OutputLayer(64, self.n_classes)


    def forward(self, x):
        # input layer
        x = self.in_layer_1(x)
        i = self.in_layer_2(x)
        # downsampling / conv
        e1 = self.encoder1(i)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        # concat prev encoder layer. i.e. skip connection
        # name the output as x to save graphic memory
        x = self.decoder1(e3, e2)
        x = self.decoder2(x, e1)
        x = self.decoder3(x, i)
        logits = self.out_layer(x)
        return logits


if __name__ == "__main__":
    # display module
    model = ResUnet(input_channel=3, n_classes=1)
    model = model.cuda()
    summary(model, (3, 256, 256))