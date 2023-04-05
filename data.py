import nibabel as nib
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import pandas as pd
import PIL
import numpy as np
import torch
from munch import Munch
import os
import random

from torch.utils.data.sampler import WeightedRandomSampler


class Dataset_2d(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.transforms = transforms
        self.targets = self.df["Domain"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        subject_id = self.df["Path"][index]
        labels = self.df["Domain"][index]

        img = PIL.Image.open(os.path.join(self.data_dir, subject_id)).convert("L")
        img = torch.from_numpy(np.array(img)).float()
        # img = (img/ img.mean())/img.std()
        img = torch.unsqueeze(img, 0)
        Y = np.array(labels, dtype=np.float32)

        if self.transforms:
            img = self.transforms(img)

        Y = torch.from_numpy(Y).float()
        return img, Y


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


class ReferenceDataset(data.Dataset):
    def __init__(self, df, root, transform=None):
        self.transform = transform
        self.samples, self.targets = self._make_dataset(df)
        self.data_dir = root

    def _make_dataset(self, df):
        domains = df.Domain.unique()
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            cls_fnames = df[df["Domain"] == domain]["Path"].values.tolist()
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]

        img = PIL.Image.open(os.path.join(self.data_dir, fname)).convert("L")
        img = torch.from_numpy(np.array(img)).float()
        # img = (img/ img.mean())/img.std()
        img = torch.unsqueeze(img, 0)

        img2 = PIL.Image.open(os.path.join(self.data_dir, fname2)).convert("L")
        img2 = torch.from_numpy(np.array(img2)).float()
        # img2 = (img2/ img2.mean())/img2.std()
        img2 = torch.unsqueeze(img2, 0)
        label = torch.from_numpy(np.array(label)).long()

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def get_train_loader(
    df, root, which="source", img_size=256, batch_size=8, prob=0.5, num_workers=4
):
    print(
        "Preparing DataLoader to fetch %s images "
        "during the training phase..." % which
    )

    transform = transforms.Compose(
        [
            transforms.CenterCrop([img_size, img_size]),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    if which == "source":
        dataset = Dataset_2d(df, root, transform)
    elif which == "reference":
        dataset = ReferenceDataset(df, root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )


def get_eval_loader(
    root,
    img_size=256,
    batch_size=32,
    imagenet_normalize=True,
    shuffle=True,
    num_workers=4,
    drop_last=False,
):
    print("Preparing DataLoader for the evaluation phase...")

    transform = transforms.Compose(
        [
            transforms.CenterCrop([img_size, img_size]),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    dataset = Dataset_2d(root, transform=transform)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def get_test_loader(df, root, img_size=256, batch_size=32, shuffle=True, num_workers=4):
    print("Preparing DataLoader for the generation phase...")

    transform = transforms.Compose(
        [
            transforms.CenterCrop([img_size, img_size]),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    dataset = Dataset_2d(df, root, transform)
    return data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=""):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    # def _fetch_inputs_with_covar(self):
    #     try:
    #         x, y, covar = next(self.iter)
    #     except (AttributeError, StopIteration):
    #         self.iter = iter(self.loader)
    #         x, y, covar = next(self.iter)
    #     return x, y, covar

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        if self.mode == "train":
            x, y = self._fetch_inputs()
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(
                x_src=x,
                y_src=y,
                y_ref=y_ref,
                x_ref=x_ref,
                x_ref2=x_ref2,
                z_trg=z_trg,
                z_trg2=z_trg2,
            )
        elif self.mode == "val":
            x, y = self._fetch_inputs()

            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref)
        elif self.mode == "test":
            x, y = self._fetch_inputs()
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
