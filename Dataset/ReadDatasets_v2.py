__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
import cv2
import numpy as np
import pandas as pd
import scipy
import nibabel as nib
import pydicom as dicom
from skimage import morphology
from PIL import Image
import matplotlib.pyplot as plt

class RAW_PET:
    def __init__(self,
                 datapath,
                 labelpath
                 ):
        self.datapath = datapath
        self.labelpath = labelpath
        dnames = os.listdir(self.datapath)
        med_images = [dicom.read_file(datapath + f) for f in dnames]
        images = [im.pixel_array for im in med_images]
        hu_images = [self.hu_transform(med_images[i], images[i]) for i in range(med_images.__len__())]
        windowed_images = [self.window_image(image=hu_images[i],
                                             window_width=1500,
                                             window_length=-600)
                           for i in range(hu_images.__len__())]

        clean_images = [self.remove_noise(windowed_images[i])
                        for i in range(windowed_images.__len__())]
        normalized_images = [self.normalize(windowed_images[i]) for i in range(windowed_images.__len__())]
        k = 29
        self.plot([images[k],
                   hu_images[k],
                   windowed_images[k],
                   normalized_images[k]],
                  color='gray')

    def normalize(self, image, scale=1):
        return (image - image.min()) * (scale / (image.max() - image.min()))

    def hu_transform(self, med_image, image):
        intercept = med_image.RescaleIntercept
        slope = med_image.RescaleSlope
        return image * slope + intercept

    def window_image(self, image, window_width, window_length):
        ## HEAD and NECK
        # brain W:80 L:40
        # subdural W:130-300 L:50-100
        # stroke W:8 L:32 or W:40 L:40 3
        # temporal bones W:2800 L:600
        # soft tissues: W:350–400 L:20–60 4
        ## CHEST
        # lungs W:1500 L:-600
        # mediastinum W:350 L:50
        ## ABDOMEN
        # soft tissues W:400 L:50
        # liver W:150 L:30
        ## SPINEq
        # soft tissues W:250 L:50
        # bone W:1800 L:400

        img_min = window_length - window_width // 2
        img_max = window_length + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        return window_image

    def remove_noise(self, image):
        segmentation = morphology.dilation(image, np.ones((5, 5)))
        labels, label_nb = scipy.ndimage.label(segmentation)
        label_count = np.bincount(labels.ravel().astype(np.int))
        label_count[0] = 0
        mask = labels == label_count.argmax()
        mask = morphology.dilation(mask, np.ones((5, 5)))
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))
        masked_image = mask * image
        return masked_image

    def crop_image(self, image):
        mask = image == 0
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        croped_image = image[top_left[0]:bottom_right[0],
                       top_left[1]:bottom_right[1]]

        return croped_image

    def add_pad(self, image, new_height=512, new_width=512):
        height, width = image.shape
        final_image = np.zeros((new_height, new_width))
        pad_left = int((new_width - width) / 2)
        pad_top = int((new_height - height) / 2)
        final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
        return final_image

    def plot(self, images, color=None):
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(images[0], cmap=color)
        axarr[0, 1].imshow(images[1], cmap=color)
        axarr[1, 0].imshow(images[2], cmap=color)
        axarr[1, 1].imshow(images[3], cmap=color)






if __name__ == '__main__':
    dataset = RAW_PET(datapath='C:\GorkemCanAtes\PycharmProjects\TumorSegmentation\PET\Lung-PET-CT-Dx\Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805/',
                   labelpath='C:\GorkemCanAtes\PycharmProjects\TumorSegmentation\liver_tumor\segmentations/'
                      )
    dum = dataset.save_images()

