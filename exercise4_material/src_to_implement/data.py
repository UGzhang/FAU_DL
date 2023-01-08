from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose(
            [tv.transforms.ToPILImage(), tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std)]
        )

    def __getattr__(self, index):
        filename, crack, inactive = self.data.iloc[index]
        img = imread(filename, as_gray=True)
        img = gray2rgb(img)
        img = self._transform(img)
        return img, torch.tensor([crack, inactive], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

