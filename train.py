import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import preprocessing
from model import ConvBlock, EncoderBlock, DecoderBlock, OutputLayer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
DEVICE = 'cuda:0'


class UNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(UNet, self).__init__()
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
        e3 = self.decoder3(e2)
        e4 = self.decoder4(e3)
        # concat prev encoder layer. i.e. skip connection
        # name the output as x to save graphic memory
        x = self.decoder1(e4, e3)
        x = self.decoder2(x, e2)
        x = self.decoder3(x, e1)
        x = self.decoder4(x, i)
        logits = self.out_layer(x)
        return logits


def training(train_dataloader, val_dataloader):
    # Hyper-parameters:
    num_epochs = 100
    lr = 1e-4
    batch_size = 32

    # model initialization
    model = UNet(input_channel=3, n_classes=10)
    model = model.cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_{datetime.now().strftime("%H-%M-%S")}')

    # Run your training / validation loops
    epoch_loss = 0
    for epoch in range(num_epochs):
        model.train()
        # training loop
        epoch_loss = 0
        with tqdm(train_dataloader, unit='batch') as tepoch:
            for batch_num, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                img, label = batch
                img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                prediction = model(img)
                loss = criterion(prediction, label)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), batch_num)
                tepoch.set_postfix(loss=epoch_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # with torch.no_grad():
        #     model.eval()

if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = preprocessing()
    # util.dataloader_tester(train_dataloader, val_dataloader, test_dataloader)
    training(train_dataloader,val_dataloader)



