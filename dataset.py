import os
import numpy as np
import torch
import random
import sys
from torch.utils import data
from skimage import io
from torchvision import transforms
import PIL
from PIL import Image

root = '/mnt/DADOS_PONTOISE_1/matheusp/imbalanced/'

# Class that reads a sequence of image paths from a text file and creates a data.Dataset with them.
class ListDataset(data.Dataset):

    def __init__(self, mode, dataset, fold, new_data_size, transform, A_transform=None):

        # Initializing variables.
        self.mode = mode
        self.dataset = dataset
        self.fold = fold
        self.new_data_size = new_data_size
        self.A_transform = A_transform
        self.transform = transform
        # Creating list of paths.
        self.imgs = self.make_dataset()


    def make_dataset(self):
        items = []

        if self.fold is not None:
            data_list = [l.strip('\n') for l in open(os.path.join(root, self.dataset, self.mode+'_fold'+str(self.fold)+'.txt')).readlines()]
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, self.dataset, self.mode+'.txt')).readlines()]

        # Creating list containing image and ground truth paths.
        new_data = 0
        for it in data_list:
            if it[0].isdigit():
                new_data += 1
                if new_data <= self.new_data_size:
                    item = (os.path.join(root, self.dataset, 'images', it), os.path.join(root, self.dataset, 'masks', it.replace('.tif','.png')))
                    items.append(item)
            else:
                item = (os.path.join(root, self.dataset, 'images', it), os.path.join(root, self.dataset, 'masks', it.replace('.tif','.png')))
                items.append(item)

        # Returning list.
        return items


    def old__getitem__(self, index):

        # Reading items from list.
        img_path, msk_path = self.imgs[index]

        # Reading images.
        img = io.imread(img_path)
        msk = io.imread(msk_path)

        # Casting images to the appropriate dtypes.
        img = img.astype(np.uint8)
        msk = msk.astype(np.int64)
        
        msk = msk-1
        # Splitting path.
        spl = img_path.split("/")
        
        # Turning to tensors.
        img = Image.fromarray(img)
        msk = torch.from_numpy(msk)

        img = self.transform(img)
        print(img)
        print(msk)
        sys.exit()
        # Returning to iterator.
        return img, msk, spl[-1]
    



    def __getitem__(self, index):

        # Reading items from list.
        img_path, msk_path = self.imgs[index]

        # Reading images.
        img = io.imread(img_path)
        msk = io.imread(msk_path)

        # Casting images to the appropriate dtypes.
        img = img.astype(np.float32)/255.0
        msk = msk.astype(np.int64)
        
        msk = msk-1
        # Splitting path.
        spl = img_path.split("/")
        
        # Turning to tensors.
        if self.A_transform is not None:
            augmented = self.A_transform(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']
        img = self.transform(img)
        msk = torch.from_numpy(msk.astype(np.int64))

        # Returning to iterator.
        return img, msk, spl[-1]
    

    def __len__(self):

        return len(self.imgs)
