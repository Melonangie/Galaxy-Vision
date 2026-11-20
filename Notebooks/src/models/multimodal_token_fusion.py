"""Token-level image/spectrum fusion models for binary MaNGA morphology experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


FusionVariant = Literal["independent_mean", "independent_attention", "joint_attention"]
FUSION_VARIANTS: frozenset[str] = frozenset(
    {"independent_mean", "independent_attention", "joint_attention"}
)


def _validate_spectrum_inputs(spectrum: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if spectrum.ndim != 3 or mask.shape != spectrum.shape[:1] + spectrum.shape[2:]:
        raise ValueError("Se esperaba spectrum=[B, C, L] y spectrum_mask=[B, L].")
    return spectrum * mask.bool().unsqueeze(1), mask.bool()


def _masked_mean_max(tokens: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    mean = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    maximum = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
    return torch.cat([mean, torch.nan_to_num(maximum, neginf=0.0)], dim=-1)


class ResNet18TokenEncoder(nn.Module):
    """ResNet-18 feature map extractor compatible with the existing image checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        # This preserves the parameter names under the former ImageEncoder.features.
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.output_dim = backbone.fc.in_features

    def forward_tokens(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Se esperaba image=[B, 3, H, W].")
        feature_map = self.features(image)
        batch_size, channels, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        mask = torch.ones((batch_size, height * width), dtype=torch.bool, device=image.device)
        return tokens, mask


class PatchBiGRUEncoder(nn.Module):
    """Patch+BiGRU encoder retaining the original global embedding forward path."""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        *,
        patch_size: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch = patch_size
        if seq_len % self.patch:
            raise ValueError("La longitud espectral debe ser múltiplo de patch_size.")
        self.embed = nn.Sequential(
            nn.Linear(in_channels * self.patch, gru_hidden * 2),
            nn.LayerNorm(gru_hidden * 2),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            gru_hidden * 2,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden * 4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.token_dim = gru_hidden * 2

    def forward_tokens(self, spectrum: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        spectrum, mask = _validate_spectrum_inputs(spectrum, mask)
        batch_size, channels, length = spectrum.shape
        patch_count = length // self.patch
        patches = (
            spectrum.view(batch_size, channels, patch_count, self.patch)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, patch_count, channels * self.patch)
        )
        token_mask = mask.view(batch_size, patch_count, self.patch).any(dim=-1)
        states, _ = self.gru(self.embed(patches) * token_mask.unsqueeze(-1))
        return states, token_mask

    def forward(self, spectrum: Tensor, mask: Tensor) -> Tensor:
        states, token_mask = self.forward_tokens(spectrum, mask)
        return self.projection(_masked_mean_max(states, token_mask))


class MaskedAttentionPooling(nn.Module):
    """Learn a scalar score per token while excluding invalid positions exactly."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score_layer = nn.Linear(dim, 1)

    def attention_weights(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("Se esperaba tokens=[B, T, D].")
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if mask.shape != tokens.shape[:2]:
            raise ValueError("La máscara de tokens debe tener shape [B, T].")
        mask = mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("Attention pooling recibió al menos una muestra sin tokens válidos.")
        scores = self.score_layer(tokens).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Los pesos de attention pooling no son finitos.")
        return weights

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        weights = self.attention_weights(tokens, mask)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)


def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    if (~mask.any(dim=1)).any():
        raise ValueError("Mean pooling recibió al menos una muestra sin tokens válidos.")
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1)


def _sinusoidal_position_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    position = torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10_000.0, device=device, dtype=dtype)) / max(dim, 1))
    )
    embedding = torch.zeros((length, dim), device=device, dtype=dtype)
    embedding[:, 0::2] = torch.sin(position * frequencies)
    if dim > 1:
        embedding[:, 1::2] = torch.cos(position * frequencies[: embedding[:, 1::2].shape[1]])
    return embedding


def _visual_position_encoding(
    token_count: int, dim: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    side = int(token_count**0.5)
    if side * side != token_count:
        raise ValueError("Los tokens de ResNet deben formar una cuadrícula espacial cuadrada.")
    y_dim = dim // 2
    x_dim = dim - y_dim
    y = _sinusoidal_position_1d(side, y_dim, device, dtype).repeat_interleave(side, dim=0)
    x = _sinusoidal_position_1d(side, x_dim, device, dtype).repeat(side, 1)
    return torch.cat((y, x), dim=1).unsqueeze(0)


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded_keys: tuple[str, ...]
    new_keys: tuple[str, ...]
    incompatible_keys: tuple[str, ...]


def _load_prefixed_encoder_checkpoint(
    encoder: nn.Module, checkpoint_path: str | Path, source_prefix: str, target_prefix: str = ""
) -> CheckpointLoadReport:
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("El checkpoint debe contener un mapping de state_dict.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("model_state_dict debe ser un mapping.")
    target = encoder.state_dict()
    source = {
        f"{target_prefix}{key.removeprefix(source_prefix)}": value
        for key, value in state.items()
        if key.startswith(source_prefix)
    }
    if not source:
        raise ValueError(f"El checkpoint no contiene parámetros con prefijo {source_prefix!r}.")
    missing = tuple(sorted(set(target) - set(source)))
    incompatible = tuple(
        sorted(
            key
            for key, value in source.items()
            if key not in target or not isinstance(value, Tensor) or value.shape != target[key].shape
        )
    )
    if missing or incompatible:
        raise ValueError(
            "Checkpoint incompatible: "
            f"faltantes={list(missing)}, incompatibles={list(incompatible)}."
        )
    encoder.load_state_dict(source, strict=True)
    return CheckpointLoadReport(tuple(sorted(target)), (), ())


def load_visual_encoder_checkpoint(
    encoder: ResNet18TokenEncoder, checkpoint_path: str | Path
) -> CheckpointLoadReport:
    return _load_prefixed_encoder_checkpoint(
        encoder, checkpoint_path, "image_encoder.features.", target_prefix="features."
    )


def load_spectral_encoder_checkpoint(
    encoder: PatchBiGRUEncoder, checkpoint_path: str | Path
) -> CheckpointLoadReport:
    return _load_prefixed_encoder_checkpoint(encoder, checkpoint_path, "spectrum_encoder.")


class MultimodalTokenClassifier(nn.Module):
    """M0/M1/M2 token-fusion classifier; ``forward(batch)`` returns logits ``[B]``."""

    def __init__(
        self,
        spectrum_channels: int,
        spectrum_length: int,
        *,
        fusion_variant: FusionVariant = "joint_attention",
        token_dim: int = 128,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.3,
        patch_size: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        visual_positional_encoding: bool = False,
        layer_norm: bool = True,
        freeze_visual_encoder: bool = False,
        freeze_spectral_encoder: bool = False,
    ) -> None:
        super().__init__()
        if fusion_variant not in FUSION_VARIANTS:
            raise ValueError(f"fusion_variant no soportada: {fusion_variant!r}.")
        self.fusion_variant: FusionVariant = fusion_variant
        self.visual_positional_encoding = visual_positional_encoding
        self.visual_encoder = ResNet18TokenEncoder()
        self.spectral_encoder = PatchBiGRUEncoder(
            spectrum_channels,
            spectrum_length,
            patch_size=patch_size,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        self.visual_projection = nn.Linear(self.visual_encoder.output_dim, token_dim)
        self.spectral_projection = nn.Linear(self.spectral_encoder.token_dim, token_dim)
        self.visual_normalization = nn.LayerNorm(token_dim) if layer_norm else nn.Identity()
        self.spectral_normalization = nn.LayerNorm(token_dim) if layer_norm else nn.Identity()
        self.visual_modality_embedding = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.spectral_modality_embedding = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.visual_attention_pool = MaskedAttentionPooling(token_dim)
        self.spectral_attention_pool = MaskedAttentionPooling(token_dim)
        self.joint_attention_pool = MaskedAttentionPooling(token_dim)
        self.fusion_projection = (
            nn.Linear(token_dim * 2, token_dim)
            if fusion_variant in {"independent_mean", "independent_attention"}
            else nn.Identity()
        )
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )
        self.set_encoder_frozen("visual", freeze_visual_encoder)
        self.set_encoder_frozen("spectral", freeze_spectral_encoder)

    def set_encoder_frozen(self, modality: Literal["visual", "spectral"], frozen: bool) -> None:
        encoder = self.visual_encoder if modality == "visual" else self.spectral_encoder
        for parameter in encoder.parameters():
            parameter.requires_grad = not frozen

    def _project_tokens(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required = {"image", "spectrum", "spectrum_mask"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"Faltan entradas multimodales: {sorted(missing)}.")
        visual_tokens, visual_mask = self.visual_encoder.forward_tokens(batch["image"])
        spectral_tokens, spectral_mask = self.spectral_encoder.forward_tokens(
            batch["spectrum"], batch["spectrum_mask"]
        )
        visual_tokens = self.visual_normalization(self.visual_projection(visual_tokens))
        if self.visual_positional_encoding:
            visual_tokens = visual_tokens + _visual_position_encoding(
                visual_tokens.shape[1],
                visual_tokens.shape[2],
                visual_tokens.device,
                visual_tokens.dtype,
            )
        visual_tokens = visual_tokens + self.visual_modality_embedding
        spectral_tokens = self.spectral_normalization(self.spectral_projection(spectral_tokens))
        spectral_tokens = spectral_tokens + self.spectral_modality_embedding
        return visual_tokens, visual_mask, spectral_tokens, spectral_mask

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        visual_tokens, visual_mask, spectral_tokens, spectral_mask = self._project_tokens(batch)
        if self.fusion_variant == "independent_mean":
            fusion = torch.cat(
                (_masked_mean(visual_tokens, visual_mask), _masked_mean(spectral_tokens, spectral_mask)),
                dim=1,
            )
        elif self.fusion_variant == "independent_attention":
            fusion = torch.cat(
                (
                    self.visual_attention_pool(visual_tokens, visual_mask),
                    self.spectral_attention_pool(spectral_tokens, spectral_mask),
                ),
                dim=1,
            )
        else:
            fusion = self.joint_attention_pool(
                torch.cat((visual_tokens, spectral_tokens), dim=1),
                torch.cat((visual_mask, spectral_mask), dim=1),
            )
        return self.classifier(self.fusion_projection(fusion)).squeeze(1)

    @torch.no_grad()
    def get_attention_weights(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        visual_tokens, visual_mask, spectral_tokens, spectral_mask = self._project_tokens(batch)
        if self.fusion_variant == "independent_attention":
            return {
                "visual": self.visual_attention_pool.attention_weights(visual_tokens, visual_mask),
                "spectral": self.spectral_attention_pool.attention_weights(spectral_tokens, spectral_mask),
            }
        if self.fusion_variant == "joint_attention":
            return {
                "joint": self.joint_attention_pool.attention_weights(
                    torch.cat((visual_tokens, spectral_tokens), dim=1),
                    torch.cat((visual_mask, spectral_mask), dim=1),
                ),
                "visual_token_count": torch.tensor(visual_tokens.shape[1], device=visual_tokens.device),
            }
        raise ValueError("independent_mean no utiliza attention pooling.")


def build_multimodal_token_fusion(config: Mapping[str, Any], batch: Mapping[str, Tensor]) -> MultimodalTokenClassifier:
    """Build the model from the repository's ``config['model']`` convention."""
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise TypeError("config.model debe ser un mapping.")
    if batch["spectrum"].ndim != 3:
        raise ValueError("batch['spectrum'] debe tener shape [B, C, L].")
    fusion_variant = str(model_config.get("fusion_variant", "joint_attention"))
    if fusion_variant not in FUSION_VARIANTS:
        raise ValueError(f"fusion_variant no soportada: {fusion_variant!r}.")
    return MultimodalTokenClassifier(
        spectrum_channels=batch["spectrum"].shape[1],
        spectrum_length=batch["spectrum"].shape[2],
        fusion_variant=cast(FusionVariant, fusion_variant),
        token_dim=int(model_config.get("token_dim", 128)),
        classifier_hidden_dim=int(model_config.get("classifier_hidden_dim", 128)),
        dropout=float(model_config.get("dropout", 0.3)),
        patch_size=int(model_config.get("patch_size", 32)),
        gru_hidden=int(model_config.get("gru_hidden", 64)),
        gru_layers=int(model_config.get("gru_layers", 2)),
        visual_positional_encoding=bool(model_config.get("visual_positional_encoding", False)),
        layer_norm=bool(model_config.get("layer_norm", True)),
        freeze_visual_encoder=bool(model_config.get("freeze_visual_encoder", False)),
        freeze_spectral_encoder=bool(model_config.get("freeze_spectral_encoder", False)),
    )
