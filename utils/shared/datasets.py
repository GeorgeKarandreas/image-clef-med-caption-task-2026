"""Helper file for Custom Dataset classes"""

import os
from pathlib import Path

from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.caption.helpers import clean_caption, preprocess_to_tensor

class ConceptsDataset(Dataset):
    """Dataset Class for Concept Task Data"""
    def __init__(self, csv_file, img_dir, label_dict, split, transform=None):
        allowed_splits = {"train", "valid", "all"}

        if split not in allowed_splits:
            raise ValueError(
                f"Invalid split='{split}'. Must be one of {allowed_splits}"
            )

        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_dictionary = label_dict
        self.num_of_classes = len(self.label_dictionary)

        # train/val/test split
        if split != "all":
            self.data = self.data[
                self.data["ID"].str.contains(rf"_{split}_")
            ].reset_index(drop=True)

        self.targets = torch.zeros((len(self.data), self.num_of_classes), dtype=torch.float32)

        for row_index, cuis in enumerate(self.data["CUIs"]):
            for label in cuis.split(";"):
                self.targets[row_index, self.label_dictionary[label]] = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["ID"] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx].clone()


class CaptionsDataset(Dataset):
    """Dataset Class for Caption Task Data"""
    def __init__(
            self,
            csv_file,
            img_dir,
            split,
            transform=None,
            biomed_preprocess=None,
            swin_preprocess=None,
            tokenizer=None,
            max_length=128
        ):
        allowed_splits = {"train", "valid", "all"}

        if split not in allowed_splits:
            raise ValueError(
                f"Invalid split='{split}'. Must be one of {allowed_splits}"
            )

        if tokenizer is None:
            raise ValueError("tokenizer must not be None.")

        if biomed_preprocess is None:
            raise ValueError("biomed_preprocess must not be None.")

        if swin_preprocess is None:
            raise ValueError("swin_preprocess must not be None.")

        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.transform = transform
        self.biomed_preprocess = biomed_preprocess
        self.swin_preprocess = swin_preprocess

        self.tokenizer = tokenizer
        self.max_length = max_length

        # train/val/test split
        if split != "all":
            self.data = self.data[
                self.data["ID"].str.contains(rf"_{split}_")
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_id = str(row["ID"])
        img_path = os.path.join(self.img_dir, image_id + ".jpg")
        image = Image.open(img_path).convert("RGB")

        # Shared image-level transform first.
        if self.transform:
            image = self.transform(image)
            
        # Encoder-specific preprocessing.
        biomed_pixels = preprocess_to_tensor(self.biomed_preprocess, image)
        swin_pixels = preprocess_to_tensor(self.swin_preprocess, image)


        caption = clean_caption(row["Caption"])

        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100 # ignore padding tokens
        
        return {
            "biomed_pixels": biomed_pixels,
            "swin_pixels": swin_pixels,
            "labels": labels,
            "caption": caption,
            "image_id": image_id,
        }


class TestImageDataset(Dataset):
    """Dataset class for unlabeled concept and caption test images."""

    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.image_paths = sorted(
            self.img_dir.glob("*.jpg"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1])
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image_id = image_path.stem
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_id
