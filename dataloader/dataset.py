import random
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
            for dataset_dir in root.iterdir():
                if not dataset_dir.is_dir():
                    continue
                for scene_dir in dataset_dir.iterdir():
                    if not scene_dir.is_dir():
                        continue
                    files = tuple(scene_dir / f"{angle}.png" for angle in ANGLES)
                    present = [path.is_file() for path in files]
                    if any(present) and not all(present):
                        raise ValueError(f"Incomplete four-angle scene: {scene_dir}")
                    if all(present):
                        self.scenes.append((files, f"{dataset_dir.name}/{scene_dir.name}"))
        self.scenes.sort(key=lambda scene: scene[1].lower())
        if not self.scenes:
            raise ValueError("Expected dataset/scene/{0,45,90,135}.png under: " + ", ".join(map(str, roots)))

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
