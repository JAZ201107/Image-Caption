import os
import h5py
import json

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import train_transform, test_transform


class CaptionDataset(Dataset):
    def __init__(self, data_dir, data_name, split, transform=None):
        """
        the data loader init function
        """
        self.split = split
        assert self.split in {"train", "val", "test"}

        self.h = h5py.File(
            os.path.join(data_dir, self.split + "_IMAGES_" + data_name + ".hdf5"), "r"
        )
        self.imgs = self.h["images"]

        # Captions per image
        self.cpi = self.h.attrs["captions_per_image"]

        # Load encoded captions (completely into memory)
        with open(
            os.path.join(data_dir, self.split + "_CAPTIONS_" + data_name + ".json"),
            "r",
        ) as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(
            os.path.join(data_dir, self.split + "_CAPLENS_" + data_name + ".json"),
            "r",
        ) as j:
            self.caplens = json.load(j)

        self.transform = transform

        self.dataset_size = len(self.captions)

    def ___getitem__(self, idx):
        """
        Standard Implement
        """
        img = torch.FloatTensor(self.imgs[idx // self.cpi] / 255.0)

        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[idx])
        caplen = torch.LongTensor([self.caplens[idx]])

        if self.split == "train":
            return img, caption, caplen

        else:
            all_captions = torch.LongTensor(
                self.captions[
                    ((idx // self.cpi) * self.cpi) : (
                        ((idx // self.cpi) * self.cpi) + self.cpi
                    )
                ]
            )
            return img, caption, caplen, all_captions

    def __len__(self):
        """
        The length of dataloader
        """
        return self.dataset_size


def fetch_dataloader(types, data_dir, data_name, params):
    dataloaders = {}
    for split in ["train", "test", "val"]:
        if split in types:

            if split == "train":
                dl = DataLoader(
                    CaptionDataset(
                        data_dir, data_name, "train", transform=train_transform
                    ),
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                )
            else:
                df = DataLoader(
                    CaptionDataset(
                        data_dir, data_name, "val", transform=test_transform
                    ),
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                )

        dataloaders[split] = dl

    return dataloaders
