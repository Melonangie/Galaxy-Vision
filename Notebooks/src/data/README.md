# Generación de datasets procesados

Instale las dependencias del proyecto:

```bash
uv pip install -r requirements.txt
```

Ejecute el generador desde la raíz del proyecto:

```bash
python -m src.data.generate_processed_datasets \
  --input-dir /ruta/dataset_original \
  --output-dir /ruta/datasets_procesados \
  --methods all --workers 4 --preview-count 5
```

También se pueden elegir transformaciones independientes, por ejemplo
`--methods gamma percentile`.
El resultado contiene un directorio por método, conserva nombres y subdirectorios
relativos, y guarda `processing_metadata.json` con parámetros, versiones, conteos
y errores. Las salidas JPEG válidas se omiten en ejecuciones posteriores; use
`--no-skip-existing` para regenerarlas. `--jpeg-quality` controla la calidad
(95 por defecto).

Los parámetros iniciales están centralizados en
`DEFAULT_TRANSFORM_PARAMETERS` de `image_transforms.py`: `gamma` aplica
gamma 0.85 a RGB normalizado; `grayscale` replica la luminancia en los tres
canales; y `percentile` normaliza la luminancia entre los percentiles 1 y
99.5, conservando las proporciones de color cuando es posible. El metadata
incluye, por imagen percentil, los umbrales y el número/proporción de píxeles
saturados en cada límite.

La opción `--preview-count` guarda `processing_preview.png` con originales y
los métodos seleccionados, usando una selección reproducible. Las tres
operaciones trabajan sobre JPEG RGB renderizados y no son procesamiento
fotométrico de bandas FITS.
