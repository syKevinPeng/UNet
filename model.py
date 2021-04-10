import torch
import torch.nn as nn
import torch.nn.functional as F


# define a convolution block which consist of Conv2d -> BatchNorm -> Relu
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels //2, kernel_size=2, stride=2)
        self.two_convs = nn.Sequential(
            ConvBlock(in_channels, out_channels, filter_size=3),
            ConvBlock(out_channels, out_channels, filter_size=3)
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
        self.output = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.output(x)