import os
import itertools
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# ========== Configuration flags ==========
USE_CROP = False   # True: resize + center crop; False: direct resize

# ========== Normalization parameters ==========
NORMALIZE_MEAN = 128.0
NORMALIZE_SCALE = 255.0

# ========== Basic transform functions ==========
def simple_resize(img, resolution):
    return img.resize((resolution, resolution), Image.BILINEAR)

def resize_center_crop(img, resolution):
    w, h = img.size
    if w / h > 4 / 3:
        new_w = int(4 / 3 * h)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
        w = new_w
    elif h / w > 4 / 3:
        new_h = int(4 / 3 * w)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
        h = new_h
    scale = resolution / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    return img.crop((left, top, left + resolution, top + resolution))

def obtainClassNames(root_dir: Path):
    root = str(root_dir)

    # ---------- ① Known datasets: keep legacy logic ----------
    if 'IntelImage' in root:
        return ['buildings','forest','glacier','mountain','sea','street']
    if 'TB' in root:
        return ['Normal','TB']
    if 'ChestX-Ray_Pneumonia' in root:
        return ['NORMAL','PNEUMONIA']
    if 'NCT-CRC-HE-100K' in root:
        return ['ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM']
    if 'Retino' in root:
        return ['0','1','2','3','4']
    if 'NeoJaundice' in root:
        return ['NORMAL', 'NeoJaundice']
    if 'ColonPath' in root:
        return ['0','1']
    if 'MNIST' in root or 'CIFAR10' in root:
        return [str(i) for i in range(10)]
    if 'Dog_Emotions' in root:
        return ['angry','happy','relaxed','sad']

    # ---------- ② Generic fallback: scan subdirectories ----------
    subdirs = [d.name for d in root_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No class folders found under {root_dir}")

    # Check if names have numeric prefix
    with_idx = [d for d in subdirs if '_' in d and d.split('_', 1)[0].isdigit()]
    if len(with_idx) == len(subdirs):          # All have numeric prefixes
        # e.g., '0_Alarm_Clock' → (0, 'Alarm_Clock')
        parsed = sorted(
            [(int(d.split('_', 1)[0]), d.split('_', 1)[1]) for d in subdirs],
            key=lambda x: x[0]
        )
        return [name for _, name in parsed]
    else:                                      # Directory name itself is the class
        return sorted(subdirs)


class BaseImageDataset(Dataset):
    def __init__(self, resolution, use_augment=False):
        self.resolution = resolution
        self.use_augment = use_augment
        # Define torchvision augmentation pipeline
        if use_augment:
            self.augment = T.Compose([
                T.RandomHorizontalFlip(p=0.2),
                # # ① First randomly crop a region with aspect ratio between 0.75–1.33 and resize to target resolution
                # T.RandomResizedCrop(
                #     resolution,
                #     scale=(0.9, 1.1),
                #     ratio=(0.75, 1.33)
                # ),
                # ② Additional affine transforms (±15° rotation, ±10% translation, ±10% scale, ±10° shear)
                T.RandomApply([T.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10
                )], p=0.5),
                T.RandomPerspective(distortion_scale=0.5, p=0.8),
                # (Optional) T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.augment = None

    def preprocess(self, img: Image.Image) -> Image.Image:
        # resize / crop
        if USE_CROP:
            img = resize_center_crop(img, self.resolution)
        else:
            img = simple_resize(img, self.resolution)
        # apply augmentation
        if self.augment is not None:
            # Apply augmentation with probability
            if random.random() < 0.4:
                img = self.augment(img)
        return img

    def finalize(self, img: Image.Image) -> np.ndarray:
        # Legacy normalization (commented out):
        # arr = np.array(img, dtype=np.float32)
        # arr = (arr - NORMALIZE_MEAN) / NORMALIZE_SCALE
        # return np.transpose(arr, (2, 0, 1))  # HWC → CHW

        arr = np.array(img, dtype=np.float32) / 255.0
        # ImageNet mean/std normalization
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        arr = (arr - mean) / std
        return arr.astype(np.float32)


class CenDataset(BaseImageDataset):
    def __init__(self, root_dirs, N_CLASSES, resolution, N_GDATA=None, use_augment=False):
        super().__init__(resolution, use_augment)
        self.num_classes = N_CLASSES
        self.samples = []

        for root_dir in root_dirs:
            p = Path(root_dir)
            names = obtainClassNames(p)
            for i in range(N_CLASSES):
                dir_i = p / f"{i}_{names[i]}"
                if not dir_i.exists():
                    continue
                files = list(dir_i.glob('*'))
                if N_GDATA and 'diffusion' in str(dir_i):
                    files = files[:min(N_GDATA, len(files))]
                for f in files:
                    if f.is_file():
                        self.samples.append((f, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
        arr = self.finalize(img)
        return arr, label

class Dataset(BaseImageDataset):
    def __init__(self, root_dir, N_CLASSES, resolution, use_augment=False, return_filename=False):
        super().__init__(resolution, use_augment)
        self.num_classes = N_CLASSES
        self.samples = []
        self.return_filename = return_filename
        p = Path(root_dir)
        names = obtainClassNames(p)
        for i in range(N_CLASSES):
            dir_i = p / f"{i}_{names[i]}"
            if not dir_i.exists():
                continue
            for f in dir_i.iterdir():
                if f.is_file():
                    self.samples.append((f, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
        arr = self.finalize(img)

        if self.return_filename:
            return arr, label, path.name  # Return original filename (e.g. xxx.png)
        return arr, label


# ========== Utility functions ==========
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        return self.transform(inp), self.transform(inp)

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
