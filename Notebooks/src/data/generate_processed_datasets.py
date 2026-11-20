"""CLI para crear datasets JPEG con una transformación por directorio."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.data.image_transforms import (
    DEFAULT_TRANSFORM_PARAMETERS,
    TRANSFORMS,
    apply_transform,
    percentile_saturation_statistics,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ProcessingResult:
    method: str
    source: Path
    status: str
    error: str | None = None
    details: dict[str, object] | None = None


def discover_images(input_dir: Path) -> list[Path]:
    """Encuentra JPEGs recursivamente y los ordena para resultados reproducibles."""
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )


def load_rgb(path: Path) -> np.ndarray:
    """Abre y valida un JPEG como RGB de 8 bits."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"No se pudo abrir como imagen RGB: {error}") from error


def output_is_valid(source: Path, destination: Path) -> bool:
    """Comprueba que una salida existente conserva tamaño y puede decodificarse."""
    if not destination.is_file():
        return False
    try:
        source_image = load_rgb(source)
        output_image = load_rgb(destination)
    except ValueError:
        return False
    return source_image.shape == output_image.shape and output_image.dtype == np.uint8


def validate_sample(images: list[Path], methods: list[str], sample_size: int = 3) -> int:
    """Comprueba invariantes sobre una muestra de JPEGs válidos antes del lote."""
    validated = 0
    for source in images:
        try:
            image = load_rgb(source)
            for method in methods:
                transformed = apply_transform(method, image)
                if transformed.shape != image.shape or transformed.dtype != np.uint8:
                    raise ValueError(f"{method} no conservó la forma RGB de 8 bits.")
        except ValueError:
            continue
        validated += 1
        if validated == sample_size:
            break
    if validated == 0:
        raise ValueError("No se pudo validar ninguna imagen RGB para la muestra inicial.")
    return validated


def process_image(
    method: str, source: Path, input_dir: Path, output_dir: Path, jpeg_quality: int, skip_existing: bool
) -> ProcessingResult:
    """Transforma una imagen y conserva su ruta relativa bajo el método indicado."""
    destination = output_dir / method / source.relative_to(input_dir)
    try:
        if skip_existing and output_is_valid(source, destination):
            details = (
                percentile_saturation_statistics(load_rgb(source)) if method == "percentile" else None
            )
            return ProcessingResult(method, source, "skipped", details=details)
        image = load_rgb(source)
        transformed = apply_transform(method, image)
        if transformed.shape != image.shape or transformed.dtype != np.uint8:
            raise ValueError("La transformación no conservó la forma RGB de 8 bits.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(transformed, mode="RGB").save(destination, format="JPEG", quality=jpeg_quality)
        if not output_is_valid(source, destination):
            raise ValueError("El archivo guardado no es una imagen RGB válida del tamaño esperado.")
        details = percentile_saturation_statistics(image) if method == "percentile" else None
        return ProcessingResult(method, source, "processed", details=details)
    except (OSError, ValueError) as error:
        return ProcessingResult(method, source, "error", str(error))


def create_preview(
    images: list[Path], input_dir: Path, output_dir: Path, methods: list[str], count: int
) -> Path | None:
    """Crea una cuadrícula reproducible de originales y resultados ya generados."""
    if count <= 0:
        return None
    import matplotlib.pyplot as plt

    selection = random.Random(0).sample(images, k=min(count, len(images)))
    columns = ["Original", *[method.upper() for method in methods]]
    figure, axes = plt.subplots(len(selection), len(columns), squeeze=False, figsize=(3 * len(columns), 3 * len(selection)))
    for row, source in enumerate(selection):
        axes[row, 0].imshow(load_rgb(source))
        axes[row, 0].set_title("Original")
        for column, method in enumerate(methods, start=1):
            axes[row, column].imshow(load_rgb(output_dir / method / source.relative_to(input_dir)))
            axes[row, column].set_title(method.upper())
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    destination = output_dir / "processing_preview.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True, help="gamma grayscale percentile o all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preview-count", type=int, default=0)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> list[str]:
    """Valida rutas, métodos y límites antes de crear cualquier salida."""
    input_dir = arguments.input_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"El directorio de entrada no existe: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("La salida no puede ser el directorio de entrada ni uno de sus padres.")
    if arguments.workers < 1:
        raise ValueError("--workers debe ser al menos 1.")
    if not 1 <= arguments.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality debe estar entre 1 y 100.")
    if arguments.preview_count < 0:
        raise ValueError("--preview-count no puede ser negativo.")
    requested = arguments.methods
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("'all' no puede combinarse con métodos individuales.")
        return list(TRANSFORMS)
    invalid = sorted(set(requested) - set(TRANSFORMS))
    if invalid:
        raise ValueError(f"Métodos no válidos: {invalid}.")
    return list(dict.fromkeys(requested))


def main() -> int:
    arguments = parse_arguments()
    try:
        methods = validate_arguments(arguments)
        input_dir, output_dir = arguments.input_dir.resolve(), arguments.output_dir.resolve()
        images = discover_images(input_dir)
        if not images:
            raise ValueError(f"No se encontraron archivos JPG/JPEG en {input_dir}.")
        sample_images_validated = validate_sample(images, methods)
    except ValueError as error:
        print(f"Error de validación: {error}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(method, image) for method in methods for image in images]
    results: list[ProcessingResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = [
            executor.submit(
                process_image, method, image, input_dir, output_dir, arguments.jpeg_quality,
                not arguments.no_skip_existing,
            )
            for method, image in jobs
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="image"):
            results.append(future.result())

    errors = [
        {"method": result.method, "path": result.source.relative_to(input_dir).as_posix(), "error": result.error}
        for result in results if result.status == "error"
    ]
    results_by_source: dict[Path, list[ProcessingResult]] = {image: [] for image in images}
    for result in results:
        results_by_source[result.source].append(result)
    metadata = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "methods": methods,
        "parameters": {method: DEFAULT_TRANSFORM_PARAMETERS[method] for method in methods},
        "images_found": len(images),
        "sample_images_validated": sample_images_validated,
        "images_processed_successfully": sum(
            all(result.status == "processed" for result in image_results)
            for image_results in results_by_source.values()
        ),
        "images_skipped": sum(
            all(result.status == "skipped" for result in image_results)
            for image_results in results_by_source.values()
        ),
        "outputs_processed": sum(result.status == "processed" for result in results),
        "outputs_skipped": sum(result.status == "skipped" for result in results),
        "errors_count": len(errors),
        "failed_files": errors,
        "percentile_saturation": [
            {
                "path": result.source.relative_to(input_dir).as_posix(),
                "status": result.status,
                **result.details,
            }
            for result in results
            if result.method == "percentile" and result.details is not None
        ],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
    }
    if arguments.preview_count and all(result.status != "error" for result in results):
        metadata["preview"] = str(create_preview(images, input_dir, output_dir, methods, arguments.preview_count))
    (output_dir / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=list) + "\n", encoding="utf-8"
    )
    print(
        f"Finalizado: {metadata['outputs_processed']} procesadas, "
        f"{metadata['outputs_skipped']} omitidas, {metadata['errors_count']} con error."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
