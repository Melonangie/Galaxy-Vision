"""`torch.utils.data.Dataset` para elíptica vs. espiral, con imágenes en RAM.

Cada imagen se lee de disco, se recorta al centro (`crop_size`) y se cachea
como array en memoria **una sola vez** en `__init__`. El data augmentation
(rotación, flips, traslación, brillo/contraste) se aplica de nuevo en cada
`__getitem__`, así que sigue siendo distinto en cada época, pero sin pagar el
costo de E/S de disco + decodificación JPEG en cada época de entrenamiento.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop

from src.config import CNN_CROP_SIZE, LABEL_TO_INT, TRAIN_IMAGES_DIR


class InMemoryGalaxyDataset(Dataset):
    """Dataset de galaxias con las imágenes (recortadas) cacheadas en RAM."""

    def __init__(
        self,
        df,
        images_dir: Path = TRAIN_IMAGES_DIR,
        crop_size: int = CNN_CROP_SIZE,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            df: DataFrame con columnas `GalaxyID` y `label`
                (`elliptical`/`spiral`).
            images_dir: carpeta con las imágenes `<GalaxyID>.jpg`.
            crop_size: lado (px) del recorte centrado aplicado una sola vez
                al cachear (antes de cualquier resize/augmentation).
            transform: transform de torchvision aplicado sobre cada imagen
                cacheada en cada `__getitem__` (ver `src/data/augmentation.py`).
        """
        self.labels = df["label"].map(LABEL_TO_INT).to_numpy()
        self.transform = transform
        self.images = np.stack(
            [
                self._load_and_crop(images_dir / f"{galaxy_id}.jpg", crop_size)
                for galaxy_id in df["GalaxyID"]
            ]
        )

    @staticmethod
    def _load_and_crop(path: Path, crop_size: int) -> np.ndarray:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = center_crop(img, [crop_size, crop_size])
            return np.asarray(img, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
