import glob
import random
import os
import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            valid_image_files_B = [file for file in self.files_B if self.is_valid_image(file)]
            item_B = self.transform(Image.open(valid_image_files_B[random.randint(0, len(valid_image_files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def is_valid_image(self, file_path):
        try:
            # Attempt to open the file as an image
            Image.open(file_path).verify()
            return True
        except (IOError, OSError, PIL.UnidentifiedImageError):
            return False

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
