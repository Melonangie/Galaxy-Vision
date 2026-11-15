# Fusión multimodal por tokens

`src.models.multimodal_token_fusion` implementa tres variantes experimentales que consumen
directamente el batch del DataLoader multimodal (`image`, `spectrum`, `spectrum_mask`) y devuelven
logits `[B]`, compatibles con `BCEWithLogitsLoss`.

| `fusion_variant` | Variante | Fusión |
|---|---|---|
| `independent_mean` | M0 | Mean pooling por modalidad y concatenación |
| `independent_attention` | M1 | Attention pooling enmascarado por modalidad y concatenación |
| `joint_attention` | M2 | Attention pooling sobre todos los tokens concatenados |

Para imágenes de `256 x 256`, el ResNet-18 entrega `[B, 64, 512]` antes del pooling global.
Para `FLUX_FoV` de longitud 4096, `patch_size=32` y `gru_hidden=64`, Patch+BiGRU entrega
`[B, 128, 128]`. Ambas ramas se proyectan a `token_dim` (128 por defecto), se normalizan
opcionalmente y reciben embeddings aprendibles de modalidad. M0 y M1 proyectan la concatenación
`[B, 2D]` a `[B, D]`; M2 ya produce `[B, D]`; las tres usan el mismo MLP final.

```python
from src.models import build_multimodal_token_fusion

config = {
    "model": {
        "name": "multimodal_token_fusion",
        "fusion_variant": "joint_attention",
        "token_dim": 128,
        "classifier_hidden_dim": 128,
        "dropout": 0.3,
        "freeze_visual_encoder": True,
        "freeze_spectral_encoder": True,
    }
}
model = build_multimodal_token_fusion(config, sample_batch)
logits = model(sample_batch)
weights = model.get_attention_weights(sample_batch)
```

`load_visual_encoder_checkpoint` y `load_spectral_encoder_checkpoint` cargan estrictamente,
respectivamente, los prefijos `image_encoder.features.*` y `spectrum_encoder.*` de los checkpoints
preexistentes. Devuelven un `CheckpointLoadReport`; cualquier clave ausente o incompatible provoca
un error explícito. Las capas de proyección de tokens, los embeddings de modalidad, pooling y
clasificador son nuevos y permanecen entrenables al congelar los encoders.

La máscara espectral se reduce a una máscara por patch. Attention pooling asigna peso cero a los
tokens inválidos y rechaza una fila completamente enmascarada. La GRU conserva el comportamiento
original: no usa `pack_padded_sequence`, así que una máscara interior no evita que el estado
recurrente de una posición inválida afecte contextualmente posiciones válidas posteriores; no se
alteró ese mecanismo para preservar la comparabilidad con los experimentos previos.
