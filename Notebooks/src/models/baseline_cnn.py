"""CNN simple entrenada desde cero — baseline de la fase 1 (ver README.md, sección Arquitectura del modelo).

4 bloques conv+pooling seguidos de un head denso con dropout. Sirve para
validar el pipeline de datos con una arquitectura real antes de invertir en
transfer learning con ResNet50/ResNet50V2.
"""

from torch import nn


class BaselineCNN(nn.Module):
    """CNN de 4 bloques conv (Conv2d + BatchNorm + ReLU + MaxPool) + head denso."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        """
        Args:
            num_classes: número de clases de salida (2: elíptica/espiral).
            dropout: probabilidad de dropout antes de las capas densas finales.
        """
        super().__init__()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        # AdaptiveAvgPool2d hace el head independiente del tamaño de entrada:
        # no hay que recalcular a mano la dimensión aplanada si cambia
        # CNN_IMAGE_SIZE.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)
