"""Utilidades de datos y procesamiento de cutouts."""


def analyze_image_quality(*args, **kwargs):
    """Carga el análisis de calidad solo cuando se solicita."""
    from src.data.image_quality import analyze_image_quality as _analyze_image_quality

    return _analyze_image_quality(*args, **kwargs)


__all__ = ["analyze_image_quality"]
