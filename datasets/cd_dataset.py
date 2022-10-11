import os
from typing import Dict, Sequence, Tuple, Optional, List

from PIL import Image
import numpy as np

import torch
from torch.utils import data

from datasets.transforms import get_transforms, get_mask_transforms
from datasets.transforms import get_cd_augs

"""
some basic data loader
for example:
bitemporal image loader, change detection folder

data root
├─A
├─B
├─label
└─list
"""


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


class BiImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int =256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list'):
        super(BiImageDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split  # train | train_aug | val
        self.list_path = os.path.join(self.root_dir, list_folder_name, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.img_folder_names = img_folder_names
        self.img_size = img_size
        assert len(img_folder_names) == 2
        self.basic_transforms = get_transforms(norm=norm, img_size=img_size)

    def _get_bi_images(self, name):
        imgs = []
        for img_folder_name in self.img_folder_names:
            A_path = os.path.join(self.root_dir, img_folder_name, name)
            img = np.asarray(Image.open(A_path).convert('RGB'))
            imgs.append(img)
        if self.basic_transforms is not None:
            imgs = [self.basic_transforms(img) for img in imgs]

        return imgs

    def __getitem__(self, index):
        name = self.img_name_list[index % self.A_size]
        imgs = self._get_bi_images(name)
        return {'A': imgs[0],  'B': imgs[1], 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(BiImageDataset):
    '''
    注意：这里仅应用基础的transforms，即tensor化，resize等
        其他transforms在外部的augs中应用
    '''
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list',
                 label_transform: str = 'norm',
                 label_folder_name: str = 'label',):
        super(CDDataset, self).__init__(root_dir, split=split,
                                        img_folder_names=img_folder_names,
                                        list_folder_name=list_folder_name,
                                        img_size=img_size,
                                        norm=norm)
        self.basic_mask_transforms = get_mask_transforms(img_size=img_size)
        self.label_folder_name = label_folder_name
        self.label_transform = label_transform

    def _get_label(self, name):
        mask_path = os.path.join(self.root_dir, self.label_folder_name, name)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            mask = mask // 255
        elif self.label_transform == 'ignore0_sub1':
            mask = mask - 1
            # 原来label==0的部分变为255，自动被ignore
        if self.basic_mask_transforms is not None:
            mask = self.basic_mask_transforms(mask)
        return mask

    def __getitem__(self, index):
        name = self.img_name_list[index]
        imgs = self._get_bi_images(name)
        mask = self._get_label(name)
        # img_concat = torch.concat(imgs, dim=0)
        # return img_concat, mask
        return {'A': imgs[0], 'B': imgs[1], 'mask': mask, 'name': name}


class SCDDataset(BiImageDataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list',
                 label_folder_names: Tuple[str, ] = ('label1_gray', 'label2_gray'),
                 label_transform: str = 'norm',
                 ):
        super(SCDDataset, self).__init__(root_dir, split=split,
                                        img_folder_names=img_folder_names,
                                        list_folder_name=list_folder_name,
                                        img_size=img_size,
                                        norm=norm)
        self.basic_mask_transforms = get_mask_transforms(img_size=img_size)
        self.label_folder_names = label_folder_names
        self.label_transform = label_transform

    def _get_labels(self, name):
        masks = []
        for label_folder_name in self.label_folder_names:
            mask_path = os.path.join(self.root_dir, label_folder_name, name)
            mask = np.array(Image.open(mask_path), dtype=np.uint8)
            masks.append(mask)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            masks = [mask // 255 for mask in masks]
        if self.basic_mask_transforms is not None:
            masks = [self.basic_mask_transforms(mask) for mask in masks]
        return masks

    def __getitem__(self, index):
        name = self.img_name_list[index]
        imgs = self._get_bi_images(name)
        masks = self._get_labels(name)

        return {'A': imgs[0], 'B': imgs[1],
                'mask1': masks[0], 'mask2': masks[1], 'name': name}


from misc.torchutils import visualize_tensors


if __name__ == '__main__':
    is_train = True
    root_dir = r'E:\cddataset\LEVIR-CD256'
    split = 'train'
    label_transform = 'norm'
    dataset = CDDataset(root_dir=root_dir, split=split,
                         label_transform=label_transform)
    augs = get_cd_augs(imgz_size=256)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(0),
        drop_last=True)

    for i, batch in enumerate(dataloader):
        A0 = batch['A']
        B0 = batch['B']
        A, B, L = augs(batch['A'], batch['B'], batch['mask'])
        # A = batch['A']
        # L = batch['mask']
        print(A.max())
        print(L.max())
        # A2 = batch['A2']
        # L2 = batch['L2']
        # mask = batch['seg_mask']
        visualize_tensors(A0[0], A[0], B[0], L[0])
