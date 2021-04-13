import glob
import numpy as np
import torch.utils.data as data
from PIL import Image
import util
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import torch

DEVICE = 'cuda:0'
msrc_directory = 'SegmentationDataset'

class SegmentationData(data.Dataset):
    def __init__(self, img_transform, mask_transform, img_aug_transform, mode='train', img_aug=False):
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Invalid Split %s' % mode)
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.img_aug_transform = img_aug_transform
        self.img_list_train_val = [x.split('.')[-2].split('/')[-1][:-3] for x in glob.glob(msrc_directory + '/train/*')
                                   if 'GT' in x]
        self.img_list_train_val.sort()
        self.img_list_test = [x.split('.')[-2].split('/')[-1] for x in glob.glob(msrc_directory + '/test/*')]
        self.img_list_test.sort()
        self.img_aug = img_aug

        self.x = {}
        self.y = {}
        self.x['train'] = ['%s/%s.bmp' % (msrc_directory, x) for x in self.img_list_train_val[:168]]
        self.y['train'] = ['%s/%s_GT.bmp' % (msrc_directory, x) for x in self.img_list_train_val[:168]]
        self.x['val'] = ['%s/%s.bmp' % (msrc_directory, x) for x in self.img_list_train_val[168:]]
        self.y['val'] = ['%s/%s_GT.bmp' % (msrc_directory, x) for x in self.img_list_train_val[168:]]
        self.x['test'] = ['%s/%s.bmp' % (msrc_directory, x) for x in self.img_list_test]


    def __len__(self):
        return len(self.x[self.mode])

    def __getitem__(self, index):
        if self.mode in ['train', 'val']:
            img = plt.imread(self.x[self.mode][index])
            mask = util.get_binary_seg(plt.imread(self.y[self.mode][index]))
            img, mask = np.moveaxis(img,-1,0), np.moveaxis(mask, -1, 0)
            img, mask = torch.as_tensor(np.array(img), dtype=float).to(DEVICE), \
                        torch.as_tensor(np.array(mask), dtype=float).to(DEVICE)
            if self.img_aug:
                # in order to apply same data augmentation to both image and mask, we combine them together and
                # apply image augmentation on combined item.
                combined = torch.cat((img, mask), axis=0)
                combined_aug = self.img_aug_transform(combined)
                img = combined_aug[0:3, :, :]
                mask = combined_aug[3:4, :, :]
            img = self.img_transform(img)
            mask = self.mask_transform(mask)
            # tensor_mask[tensor_mask > 0] = 1
            return img, mask
        else:
            # TODO modify testing dataloader
            img = plt.imread(self.x[self.mode][index])
            img = np.moveaxis(img, -1, 0)
            img = torch.as_tensor(np.array(img), dtype=float).to(DEVICE)
            img = self.img_transform(img)
            return img


def preprocessing(batch_size, is_img_aug = True):
    mean, std = util.dataset_stats()
    img_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.Normalize(mean,std)
         ])
    mask_transform = transforms.Compose([transforms.Resize((256, 256))])
    img_aug = transforms.Compose([transforms.RandomApply([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(256, padding=0, pad_if_needed=True, padding_mode='constant'),
                transforms.RandomVerticalFlip(),
                transforms.GaussianBlur((5,5)),
                transforms.RandomRotation(25)
                ])])

    train_set = SegmentationData(img_transform=img_transform,
                                 mask_transform=mask_transform,
                                 img_aug_transform=img_aug,
                                 mode='train',
                                 img_aug =is_img_aug)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = SegmentationData(img_transform=img_transform,
                               mask_transform=mask_transform,
                               img_aug_transform=img_aug,
                               mode='val')
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_set = SegmentationData(img_transform=img_transform,
                                mask_transform=mask_transform,
                                img_aug_transform=img_aug,
                                mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    preprocessing(32)
    # augmentation_test()
    # augmentation_test()
    # augmentation_test()