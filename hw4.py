import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import preprocessing

class ConvBlock(nn.module):
    def __init__(self, in_channels, out_channels, filter_size):
        super().__init__()
        self.conv_block = nn.Sequential(
            # we are suppose to do "same" padding
            nn.Conv2d(in_channels,out_channels, kernel_size=filter_size, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)


class EncoderBlock(nn.module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.MaxPool2d()
        )

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

    def forward(self, input):
        pass


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = preprocessing()
    util.dataloader_tester(train_dataloader, val_dataloader, test_dataloader)