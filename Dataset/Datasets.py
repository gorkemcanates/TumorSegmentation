__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import Dataset
from transforms.transforms import Transforms
import cv2
import matplotlib.pyplot as plt

class LITSDataset:
    def __init__(self,
                 im_path,
                 mask_path,
                 train_transform,
                 val_transform,
                 num_classes=3,
                 split_size=0.2,
                 shuffle=True,
                 debug=False):

        self.train_transform = train_transform
        self.val_transform = val_transform


        if debug:
            train_ids, test_ids = train_test_split(np.arange(16),
                                                   test_size=split_size,
                                                   random_state=42,
                                                   shuffle=shuffle)

        else:
            train_ids, test_ids = train_test_split(np.arange(os.listdir(im_path).__len__()),
                                                   test_size=split_size,
                                                   random_state=42,
                                                   shuffle=shuffle)

        self.train_dataset = LITS(image_dir=im_path,
                                  mask_dir=mask_path,
                                  num_classes=num_classes,
                                  indexes=train_ids,
                                  transform=self.train_transform
                                  )
        self.test_dataset = LITS(image_dir=im_path,
                                 mask_dir=mask_path,
                                 num_classes=num_classes,
                                 indexes=test_ids,
                                 transform=self.val_transform
                                 )

        print('Data load completed.')

class LITS(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        im_list = os.listdir(image_dir)
        self.im_list = [im_list[i] for i in indexes]


    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

            mask[mask > self.num_classes-1] = self.num_classes-1
            mask[mask < 0] = 0
            image = image / 255.
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)

if __name__ == '__main__':
    transform = Transforms()
    dataset = LITSDataset(im_path='C:\GorkemCanAtes\PycharmProjects\TumorSegmentation\Dataset\images/',
                          mask_path='C:\GorkemCanAtes\PycharmProjects\TumorSegmentation\Dataset\masks/',
                          num_classes=3,
                          train_transform=transform.train_transform,
                          val_transform=transform.val_transform)

