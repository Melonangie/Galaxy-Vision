"""Transforms reproducibles para los cutouts RGB de MaNGA."""

import random
from typing import Literal

from PIL import Image
from torchvision import transforms


ImageNormalization = Literal["none", "imagenet"]


class RandomRotate90:
    """Rota una imagen PIL un múltiplo aleatorio de 90 grados."""

    _ROTATIONS = (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    )

    def __call__(self, image: Image.Image) -> Image.Image:
        rotation = random.choice(self._ROTATIONS)
        return image if rotation is None else image.transpose(rotation)


def build_image_transform(
    image_size: int = 256,
    train: bool = False,
    normalization: ImageNormalization = "none",
) -> transforms.Compose:
    """Construye transforms para entrenamiento o evaluación de cutouts RGB."""
    if image_size <= 0:
        raise ValueError("image_size debe ser positivo.")
    if normalization not in {"none", "imagenet"}:
        raise ValueError("normalization debe ser 'none' o 'imagenet'.")

    steps: list[object] = []
    if train:
        steps.extend(
            [
                RandomRotate90(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    steps.extend([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    if normalization == "imagenet":
        steps.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms.Compose(steps)