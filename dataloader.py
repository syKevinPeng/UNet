import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import util
import numpy as np
import glob
msrc_directory = 'SegmentationDataset'
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SegmentationData(data.Dataset):
    def __init__(self, transform, mode='train'):
        if mode not in ['train','test','val']:
            raise ValueError('Invalid Split %s' % mode)
        self.mode = mode
        self.transform = transform
        self.img_list_train_val = [x.split('.')[-2].split('/')[-1][:-3] for x in glob.glob(msrc_directory+'/train/*') if 'GT' in x]
        self.img_list_train_val.sort()
        self.img_list_test = [x.split('.')[-2].split('/')[-1] for x in glob.glob(msrc_directory+'/test/*')]
        self.img_list_test.sort()

        self.x={}
        self.y={}
        self.x['train'] = ['%s/%s.bmp' % (msrc_directory, x) for x in self.img_list_train_val[:168]]
        self.y['train'] = ['%s/%s_GT.bmp' % (msrc_directory, x) for x in self.img_list_train_val[:168]]
        self.x['val'] = ['%s/%s.bmp' % (msrc_directory, x) for x in self.img_list_train_val[168:]]
        self.y['val'] = ['%s/%s_GT.bmp' % (msrc_directory, x) for x in self.img_list_train_val[168:]]
        self.x['test'] = ['%s/%s.bmp' % (msrc_directory, x) for x in self.img_list_test]

    def __len__(self):
        return len(self.x[self.mode])

    def __getitem__(self, index):
      if self.mode in ['train', 'val']:
          img = Image.open(self.x[self.mode][index]).convert('RGB')
          mask = util.get_binary_seg(np.array(Image.open(self.y[self.mode][index]).convert('RGB')))#.astype(np.int)
          mask = np.squeeze(mask.astype(np.uint8), axis=2)
          mask = Image.fromarray(mask)
          tensor_img = self.transform(img)
          tensor_mask = self.transform(mask)
          return tensor_img,tensor_mask
      else:
          img = Image.open(self.x[self.mode][index]).convert('RGB')
          tensor_img = self.transform(img)
          return tensor_img

def preprocessing():
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_set = SegmentationData(transform=transform, mode='train')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_set = SegmentationData(transform=transform, mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)
    test_set = SegmentationData(transform=transform, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader