import util
import torch
import torch.nn as nn
import numpy as np
from dataloader import preprocessing
from model import ConvBlock, EncoderBlock, DecoderBlock, OutputLayer, UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchsummary import summary
DEVICE = 'cuda:0'



def training(train_dataloader, val_dataloader):
    # Hyper-parameters:
    num_epochs = 100
    lr = 1e-4
    batch_size = 32

    # model initialization
    model = UNet(input_channel=3, n_classes=10)
    model = model.cuda()
    summary(model, (3, 256, 256))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    # Run your training / validation loops
    with tqdm(range(num_epochs), total=num_epochs, unit='Epoch') as pbar:
        for epoch in pbar:
            model.train()
            # training loop
            train_epoch_loss = 0
            valid_epoch_loss = 0
            # with tqdm(train_dataloader, unit='batch') as tepoch:
            for batch_num, batch in enumerate(train_dataloader):
                img, label = batch
                img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                prediction = model(img)
                loss = criterion(prediction, label.type(torch.int64).squeeze(1))
                train_epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # validation loop
            with torch.no_grad():
                model.eval()
                for batch_num, batch in enumerate(val_dataloader):
                    img, label = batch
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    prediction = model(img)
                    loss = criterion(prediction, label.type(torch.int64).squeeze(1))
                    valid_epoch_loss += loss.item()


            pbar.set_postfix(train_loss=train_epoch_loss)
            pbar.set_postfix(val_loss=valid_epoch_loss)
            # write to tensorboard
            writer.add_scalar('Loss/train', train_epoch_loss.item(), epoch)
            writer.add_scalar('Loss/validation', valid_epoch_loss.item(), epoch)

if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = preprocessing()
    # util.dataloader_tester(train_dataloader, val_dataloader, test_dataloader)
    training(train_dataloader,val_dataloader)



