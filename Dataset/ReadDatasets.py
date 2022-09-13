__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.testing import data_path
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *
from tqdm import tqdm

class RAW_LITS:
    def __init__(self,
                 datapath,
                 labelpath
                 ):
        self.datapath = datapath
        self.labelpath = labelpath
        self.create_df()
        self.preprocess()


    def read_nii(self, fpath):
        ct_scan = nib.load(fpath)
        array = ct_scan.get_fdata()
        array = np.rot90(np.array(array))
        return (array)

    def create_df(self):
        file_list = []
        for dirname, _, filenames in os.walk(self.datapath):
            for filename in filenames:
                file_list.append((dirname, filename))

        for dirname, _, filenames in os.walk(self.labelpath):
            for filename in filenames:
                file_list.append((dirname, filename))

        self.df_files = pd.DataFrame(file_list, columns=['dirname', 'filename'])
        self.df_files.sort_values(by=['filename'], ascending=True)
        self.df_files["mask_dirname"] = ""
        self.df_files["mask_filename"] = ""

        for i in range(131):
            ct = f"volume-{i}.nii"
            mask = f"segmentation-{i}.nii"

            self.df_files.loc[self.df_files['filename'] == ct, 'mask_filename'] = mask
            self.df_files.loc[self.df_files['filename'] == ct, 'mask_dirname'] = self.labelpath
        self.df_files = self.df_files[self.df_files.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True)

    def preprocess(self, liverx=150, livery=30):

        self.dicom_windows = types.SimpleNamespace(
            brain=(80, 40),
            subdural=(254, 100),
            stroke=(8, 32),
            brain_bone=(2800, 600),
            brain_soft=(375, 40),
            lungs=(1500, -600),
            mediastinum=(350, 50),
            abdomen_soft=(400, 50),
            liver=(liverx, livery),
            spine_soft=(250, 50),
            spine_bone=(1800, 400),
            custom=(200, 60)
        )

    def save_images(self):

        path = Path(".")

        os.makedirs('train_images', exist_ok=True)
        os.makedirs('train_masks', exist_ok=True)
        for ii in tqdm(range(0, len(self.df_files), 3)):  # take 1/3 nii files for training
            curr_ct = self.read_nii(self.df_files.loc[ii, 'dirname'] + "/" + self.df_files.loc[ii, 'filename'])
            curr_mask = self.read_nii(self.df_files.loc[ii, 'mask_dirname'] + "/" + self.df_files.loc[ii, 'mask_filename'])
            curr_file_name = str(self.df_files.loc[ii, 'filename']).split('.')[0]
            curr_dim = curr_ct.shape[2]  # 512, 512, curr_dim

            for curr_slice in range(0, curr_dim, 2):  # export every 2nd slice for training
                data = tensor(curr_ct[..., curr_slice].astype(np.float32))
                mask_slice = curr_mask[..., curr_slice]
                # dum = mask_slice.reshape(-1).tolist()
                # if dum.count(3) != 0:
                #     print('error')

                mask = Image.fromarray(mask_slice.astype('uint8'), mode="L")
                data.save_jpg(f"train_images/{curr_file_name}_slice_{curr_slice}.jpg",
                              [self.dicom_windows.liver, self.dicom_windows.custom])
                mask.save(f"train_masks/{curr_file_name}_slice_{curr_slice}.jpg")





@patch
def windowed(self: Tensor, w, l):
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)

class TensorCTScan(TensorImageBW): _show_args = {'cmap': 'bone'}

@patch
def freqhist_bins(self: Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                   tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()


@patch
def hist_scaled(self: Tensor, brks=None):
    "Scales a tensor using `freqhist_bins` to values between 0 and 1"
    if self.device.type == 'cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0: res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))


@patch
def save_jpg(x: (Tensor), path, wins, bins=None, quality=120):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    im.save(fn, quality=quality)




if __name__ == '__main__':
    dataset = RAW_LITS(datapath='C:\GorkemCanAtes\PycharmProjects\TumorSegmentation\liver_tumor/volumes/',
                   labelpath='C:\GorkemCanAtes\PycharmProjects\TumorSegmentation\liver_tumor\segmentations/'
                      )
    dum = dataset.save_images()

