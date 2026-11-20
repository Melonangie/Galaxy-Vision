"""Transforms de preprocesamiento y data augmentation para la CNN.

Las galaxias no tienen una orientación "correcta" en el cielo — es la lección
más consistente de la literatura revisada (ver README.md, sección
Preprocesamiento de imágenes; Dieleman, Willett & Dambre 2015):
rotación libre 0-360°, flips y pequeñas traslaciones/variaciones de brillo
son obligatorios en entrenamiento. En validación/test no se aumenta: solo se
reescala, para medir desempeño sobre la imagen "real".
"""

from torchvision import transforms

from src.config import CNN_IMAGE_SIZE


def get_train_transforms(image_size: int = CNN_IMAGE_SIZE) -> transforms.Compose:
    """Transforms de entrenamiento: augmentation agresivo en rotación + flips.

    Args:
        image_size: lado (px) de la imagen ya cacheada en memoria (ver
            `src/data/torch_dataset.py`) sobre la que se aplica el augmentation.

    Returns:
        `transforms.Compose` listo para usar en un `Dataset` de PyTorch.
    """
    return transforms.Compose(
        [
            # RandomRotation(180) muestrea de forma uniforme en [-180, 180]:
            # cubre las 360° posibles sin duplicar el extremo.
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # también normaliza a [0, 1]
        ]
    )


def get_eval_transforms(image_size: int = CNN_IMAGE_SIZE) -> transforms.Compose:
    """Transforms de validación/test: sin augmentation, solo reescalado.

    Args:
        image_size: lado (px) de salida.

    Returns:
        `transforms.Compose` listo para usar en un `Dataset` de PyTorch.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
