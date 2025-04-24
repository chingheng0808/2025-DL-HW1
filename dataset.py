from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from scipy.ndimage import convolve
import torchvision
import cv2


def read_image_paths_and_labels(filepath):
    image_paths = []
    labels = []
    data_dir = os.path.dirname(filepath)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Ignore empty lines
                parts = line.split()
                image_paths.append(os.path.join(data_dir, parts[0]))
                labels.append(int(parts[1]))
    return image_paths, labels


def sobel_numpy(img: np.ndarray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    img = img.astype(float)
    Gx = convolve(img, Kx, mode="reflect")
    Gy = convolve(img, Ky, mode="reflect")

    G = np.hypot(Gx, Gy)
    G /= G.max()
    return G


def histogramlize_numpy(img: np.ndarray, bins: int = 256) -> np.ndarray:
    assert img.dtype == np.uint8, "Input image must be uint8 type, range [0,255]"
    flat = img.flatten()

    # calculate histogram
    hist = np.bincount(flat, minlength=bins)
    # CDF
    cdf = hist.cumsum().astype(np.float64)
    cdf_min = cdf[cdf > 0][0]
    # normalize CDF to [0, bins-1]
    cdf_norm = (cdf - cdf_min) / (cdf[-1] - cdf_min) * (bins - 1)
    cdf_norm = np.clip(cdf_norm, 0, bins - 1).astype(np.uint8)
    # mapping
    img_eq = cdf_norm[flat].reshape(img.shape)

    return img_eq


def randn_crop(image, size=(224, 224)):
    h, w = image.shape[:2]
    new_h, new_w = size
    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    return image[top : top + new_h, left : left + new_w]


def augment_image(image):

    if np.random.rand() > 0.5:
        image = np.fliplr(image).copy()
    if np.random.rand() > 0.5:
        image = np.rot90(image, k=1, axes=(0, 1)).copy()
    if np.random.rand() > 0.5:
        image = np.rot90(image, k=3, axes=(0, 1)).copy()
    return image


class MyDataset(Dataset):
    def __init__(
        self, data_dir, mode="train", size=(256, 256), gamma=1.2, file_name=False
    ):
        self.image_dir = os.path.join(data_dir, "images")
        self.data_path_list, self.label_list = read_image_paths_and_labels(
            os.path.join(data_dir, f"{mode}.txt")
        )
        self.mode = mode
        self.train = mode == "train"
        self.size = size
        self.file_name = file_name

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
        ).astype("uint8")
        self.gamma_lut = table

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        img_path = self.data_path_list[idx]
        label = self.label_list[idx]
        fn = img_path

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Reading file error: {img_path}")

        if img.ndim == 2:
            gray = img
        else:  # transform to gray by averaging channels
            gray = img.mean(axis=2).astype(np.uint8)
        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_LINEAR)

        clahe_ch = self.clahe.apply(gray)
        gamma_ch = cv2.LUT(gray, self.gamma_lut)
        lap_ch = cv2.Laplacian(gray, ddepth=cv2.CV_8U, ksize=3)
        merged = np.stack([clahe_ch, gamma_ch, lap_ch], axis=-1)

        if self.train:
            merged = augment_image(merged)
        merged = merged.transpose(2, 0, 1).astype(np.float32) / 255.0
        if self.file_name:
            return torch.from_numpy(merged), label, fn
        return torch.from_numpy(merged), label


class ChannelSelectDataset(Dataset):
    def __init__(
        self, data_dir, mode="train", size=(256, 256), gamma=1.2, channels=[0, 1, 2]
    ):
        self.image_dir = os.path.join(data_dir, "images")
        self.data_path_list, self.label_list = read_image_paths_and_labels(
            os.path.join(data_dir, f"{mode}.txt")
        )
        self.mode = mode
        self.train = mode == "train"
        self.size = size
        self.channels = channels

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        inv_gamma = 1.0 / gamma
        self.gamma_lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
        ).astype("uint8")

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        img_path = self.data_path_list[idx]
        label = self.label_list[idx]

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Reading file error: {img_path}")

        if img.ndim == 2:
            gray = img
        else:
            selected = img[:, :, self.channels]
            gray = selected.mean(axis=2).astype(np.uint8)

        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_LINEAR)

        clahe_ch = self.clahe.apply(gray)
        gamma_ch = cv2.LUT(gray, self.gamma_lut)
        lap_ch = cv2.Laplacian(gray, ddepth=cv2.CV_8U, ksize=3)
        merged = np.stack([clahe_ch, gamma_ch, lap_ch], axis=-1)

        if self.train:
            merged = augment_image(merged)

        merged = merged.transpose(2, 0, 1).astype(np.float32) / 255.0

        return torch.from_numpy(merged), label


class OrgDataset(Dataset):
    def __init__(self, data_dir, mode="train", size=(256, 256)):
        self.image_dir = os.path.join(data_dir, "images")
        self.data_path_list, self.label_list = read_image_paths_and_labels(
            os.path.join(data_dir, f"{mode}.txt")
        )
        self.mode = mode
        self.train = mode == "train"
        self.size = size

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        img_path = self.data_path_list[idx]
        label = self.label_list[idx]

        image = Image.open(img_path)
        image = image.resize(self.size, Image.BILINEAR)
        image = np.array(image)
        image = np.clip(image, 0, 255)
        if self.train:
            image = augment_image(image)
        if len(image.shape) == 2:  # deal with gray image
            image = np.concatenate(
                [
                    image[:, :, np.newaxis],
                    image[:, :, np.newaxis],
                    image[:, :, np.newaxis],
                ],
                axis=-1,
            )
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255.0

        return image, label
