# Added by Alex Showalter-Bucher(alex@darkpumpkin.ai)
# - Derived code from the dataloaders provided in https://github.com/blackjack2015/IRS


from __future__ import print_function, division
import os
import numpy as np
import OpenEXR
import Imath
from skimage import io
from torch.utils.data import Dataset
from PIL import Image
from utilities.preprocess import *
from albumentations import Compose
from dataset.preprocess import augment
from dataset.stereo_albumentation import RGBShiftStereo, RandomBrightnessContrastStereo


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if CNum > 1:
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if CNum == 1:
        hdr[:, :, 0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:, :, 0] = np.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:, :, 1] = np.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:, :, 2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr


class IRSDataset(Dataset):

    def __init__(self, root_dir, phase='train', load_disp=True, load_norm=True, to_angle=False,
                 scale_size=(576, 960)):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """

        if phase == 'train':
            txt_file = os.path.join(root_dir, 'lists', 'IRSDataset_TRAIN.list')
        elif phase == 'test':
            txt_file = os.path.join(root_dir, 'lists', 'IRSDataset_TEST.list')

        with open(txt_file, "r") as f:
            self.imgPairs = f.readlines()

        self.root_dir = root_dir
        self.phase = phase
        self.load_disp = load_disp
        self.load_norm = load_norm
        self.to_angle = to_angle
        self.scale_size = scale_size
        self.fx = 480.0
        self.fy = 480.0
        self.sx = 540
        self.sy = 960
        self._augmentation()

    def get_img_size(self):
        return self.sx, self.sy

    def _augmentation(self):
        if self.phase == 'train':
            self.transformation = Compose([
                RGBShiftStereo(always_apply=True, p_asym=0.5),
                RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
            ])
        else:
            self.transformation = None

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):

        img_names = self.imgPairs[idx].rstrip().split()

        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])
        if self.load_disp:
            gt_disp_name = os.path.join(self.root_dir, img_names[2])

        def load_exr(filename):
            hdr = exr2hdr(filename)
            h, w, c = hdr.shape
            if c == 1:
                hdr = np.squeeze(hdr)
            return hdr

        def load_rgb(filename):

            img = None
            if filename.find('.npy') > 0:
                img = np.load(filename)
            else:
                img = io.imread(filename)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.pad(img, ((0, 0), (0, 0), (0, 2)), 'constant')
                    img[:, :, 1] = img[:, :, 0]
                    img[:, :, 2] = img[:, :, 0]
                h, w, c = img.shape
                if c == 4:
                    img = img[:, :, :3]
            return img

        def load_disp(filename):
            gt_disp = None
            if gt_disp_name.endswith('pfm'):
                gt_disp, scale = load_pfm(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('npy'):
                gt_disp = np.load(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('exr'):
                gt_disp = load_exr(filename)
            else:
                gt_disp = Image.open(gt_disp_name)
                gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32) / 256

            return gt_disp

        input_data = {}

        input_data['left'] = load_rgb(img_left_name)
        input_data['right'] = load_rgb(img_right_name)

        # disp
        input_data['disp'] = load_disp(gt_disp_name)
        input_data['occ_mask'] = np.zeros_like(input_data['disp']).astype(np.bool)

        input_data = augment(input_data, self.transformation)

        return input_data
