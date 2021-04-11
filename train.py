import util
import torch
import torch.nn as nn
import numpy as np
from dataloader import preprocessing
from model import UNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchsummary import summary
import os
DEVICE = 'cuda:0'



def training(train_dataloader, val_dataloader,
             num_epochs=100,
             lr=1e-4,
             weight_decay=1e-8,
             momentum=0.9,
             batch_size=32):

    # model initialization
    number_of_class = 1
    model = UNet(input_channel=3, n_classes=number_of_class)
    model = model.cuda()
    summary(model, (3, 256, 256))
    criterion = nn.BCEWithLogitsLoss() if number_of_class == 1 else nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    # Run your training / validation loops
    with tqdm(range(num_epochs), total=num_epochs, unit='Epoch') as pbar:
        for epoch in pbar:
            model.train()
            # training loop
            train_epoch_loss = 0
            valid_epoch_loss = 0
            train_epoch_acc = 0
            valid_epoch_acc = 0
            for batch_num, batch in enumerate(train_dataloader):
                img, label = batch
                img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                prediction = model(img)
                if number_of_class != 1:
                    train_loss = criterion(prediction, label.type(torch.int64).squeeze(1))
                    # TODO calculate acc for instance segmentation
                else:
                    train_loss = criterion(prediction, label)
                    train_acc = util.batch_accuracy(prediction, label)
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc
                train_loss.backward()
                optimizer.step()

            # validation loop
            with torch.no_grad():
                model.eval()
                for batch_num, batch in enumerate(val_dataloader):
                    img, label = batch
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    validation_pred = model(img)
                    if number_of_class != 1:
                        val_loss = criterion(validation_pred, label.type(torch.int64).squeeze(1))
                        # TODO calculate acc for instance segmentation
                    else:
                        val_loss = criterion(validation_pred, label)
                        val_acc = util.batch_accuracy(validation_pred, label)

                    valid_epoch_loss += val_loss.item()
                    valid_epoch_acc  += val_acc
            # Average accuracy
            train_epoch_acc /= len(train_dataloader)
            valid_epoch_acc /= len(train_dataloader)
            pbar.set_postfix(train_loss=train_epoch_loss, train_acc=train_epoch_acc, val_loss=valid_epoch_loss, val_acc=valid_epoch_acc)
            # write to tensorboard
            writer.add_scalar('Loss/train', train_epoch_loss, epoch)
            writer.add_scalar('Loss/validation', valid_epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', train_epoch_acc, epoch)
            writer.add_scalar('Accuracy/validation', valid_epoch_acc, epoch)
        # save last epoch
        if not os.path.isdir("weights"):
            os.mkdir("weights")
        torch.save(model.state_dict(), os.path.join("weights", f'{datetime.strftime(datetime.now(), "%M-%H-%d")}.pt'))

if __name__ == "__main__":
    # define Hyper-parameters:
    num_epochs = 100
    lr = 1e-4
    weight_decay = 1e-8
    momentum = 0.9
    batch_size = 32

    train_dataloader, val_dataloader, test_dataloader = preprocessing()
    # util.dataloader_tester(train_dataloader, val_dataloader, test_dataloader)
    training(train_dataloader, val_dataloader,
             num_epochs=num_epochs,
             lr=lr,
             weight_decay=weight_decay,
             momentum=momentum,
             batch_size=batch_size)




