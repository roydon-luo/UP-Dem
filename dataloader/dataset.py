import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ANGLES = ("0", "45", "90", "135")
RAW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_tensor(path, mode):
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode), dtype=np.float32).copy()
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array.transpose(2, 0, 1)) / 255.0


class PairedPolarizationDataset(Dataset):
    def __init__(self, roots, crop_size=None, augment=False, return_name=False):
        roots = [roots] if isinstance(roots, (str, Path)) else roots
        self.crop_size = crop_size
        self.augment = augment
        self.return_name = return_name
        self.scenes = []
        for root in map(Path, roots):
            if not root.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {root}")
            for folder in chain((root,), root.rglob("*")):
                if not folder.is_dir():
                    continue
                files = tuple(folder / f"{angle}.png" for angle in ANGLES)
                if all(path.is_file() for path in files):
                    name = folder.relative_to(root).as_posix()
                    self.scenes.append((files, name if name != "." else folder.name))
                angle_dirs = tuple(folder / f"gt_{angle}" for angle in ANGLES)
                if all(path.is_dir() for path in angle_dirs):
                    for first in angle_dirs[0].glob("*_0.png"):
                        stem = first.stem[:-2]
                        files = tuple(directory / f"{stem}_{angle}.png" for directory, angle in zip(angle_dirs, ANGLES))
                        if all(path.is_file() for path in files):
                            parent = folder.relative_to(root).as_posix()
                            name = stem if parent == "." else f"{parent}/{stem}"
                            self.scenes.append((files, name))
        self.scenes.sort(key=lambda scene: scene[1].lower())
        if not self.scenes:
            raise ValueError("No complete four-angle scenes found under: " + ", ".join(map(str, roots)))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        files, name = self.scenes[index]
        images = [image_tensor(path, "RGB") for path in files]
        sizes = {image.shape[-2:] for image in images}
        if len(sizes) != 1:
            raise ValueError(f"Four-angle images have different sizes: {files}")
        tensor = torch.cat(images, dim=0)
        if self.crop_size:
            height, width = tensor.shape[-2:]
            if min(height, width) < self.crop_size:
                raise ValueError(f"Scene is smaller than crop size {self.crop_size}: {files}")
            top = random.randrange(height - self.crop_size + 1)
            left = random.randrange(width - self.crop_size + 1)
            tensor = tensor[:, top:top + self.crop_size, left:left + self.crop_size]
        if self.augment:
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            if random.random() < 0.5:
                tensor = tensor.flip(-2)
            tensor = torch.rot90(tensor, random.randrange(4), dims=(-2, -1))
        return (tensor, name) if self.return_name else tensor


class RawPolarizationDataset(Dataset):
    def __init__(self, root, return_name=False):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        self.files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS)
        if not self.files:
            raise ValueError(f"No raw images found in: {root}")
        self.return_name = return_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        tensor = image_tensor(path, "L")
        return (tensor, path.stem) if self.return_name else tensor
