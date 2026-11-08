
# Guidelines — Infraestructura de experimentación multimodal para clasificación morfológica de galaxias

## 1. Propósito del proyecto

Construir una infraestructura modular, reproducible y extensible para experimentar con arquitecturas de redes neuronales orientadas a la clasificación morfológica de galaxias MaNGA.

La tarea principal es una clasificación binaria:

- `temprana`
- `tardia`

El proyecto dispone de tres posibles fuentes de información:

1. Imágenes RGB de galaxias.
2. Espectros astronómicos MaNGA.
3. Metadatos físicos y observacionales del catálogo.

El objetivo científico es investigar cuánto aporta cada modalidad y qué estrategias de representación y fusión permiten mejorar la clasificación.

No buscamos inicialmente la arquitectura más sofisticada. Buscamos establecer un sistema experimental confiable para comparar hipótesis.

### Principios generales

- PyTorch como framework principal.
- Arquitectura modular con responsabilidades separadas.
- Configuraciones declarativas mediante YAML.
- Un mismo pipeline de datos y entrenamiento para diferentes modelos.
- Experimentos reproducibles.
- Prevención estricta de data leakage.
- Comparaciones controladas entre arquitecturas.
- Código Python claro, tipado cuando aporte valor y documentado.
- Evitar dependencias, abstracciones y servicios innecesarios.
- Desarrollar incrementalmente y validar cada módulo.
- No modificar los datos originales.
- No construir todos los modelos simultáneamente.

El sistema debe permitir incorporar nuevas arquitecturas sin reescribir el DataLoader ni el training loop.

---

# 2. Datos disponibles

## 2.1. Catálogo morfológico

Ruta:

`data/morfologia_curada.csv`

Contiene las galaxias curadas para el proyecto.

Columnas conocidas:

| Columna | Descripción |
|---|---|
| `name` | ID MaNGA, formato `manga-<plate>-<ifudesign>` |
| `objra` | Ascensión recta en grados |
| `objdec` | Declinación en grados |
| `Type` | Tipo morfológico detallado de Hubble |
| `Ttype` | Tipo numérico de Hubble |
| `nsa_z` | Redshift espectroscópico |
| `LogMass` | Logaritmo de masa estelar en masas solares |
| `incl` | Inclinación del disco en grados |
| `PETRO_TH90` | Radio de Petrosian que contiene el 90 % de la luz |
| `label` | Clasificación binaria temprana/tardía |

La columna `label` ya está curada y deriva de la clasificación morfológica.

No modificar las etiquetas ni volver a calcularlas.

Verificar el esquema real antes de implementar.

## 2.2. Espectros MaNGA

Directorio:

`data/MPL-11/`

Los espectros se almacenan como archivos FITS:

`manga-7443-1901-1D.fits`

Cada archivo corresponde a una galaxia.

Estructura observada:

- HDU 0: PRIMARY, sin datos.
- HDU 1: Spectra.

Columnas:

- `WAVE`
- `FLUX_1RE`
- `FLUX_FoV`
- `ERROR_1RE`
- `ERROR_FoV`

En la exploración inicial se observaron espectros de 4487 muestras espectrales.

No asumir que todos los archivos tienen obligatoriamente la misma longitud sin validarlo.

Interpretación preliminar:

- WAVE: longitud de onda.
- FLUX_1RE: flujo integrado dentro de la apertura 1Re.
- FLUX_FoV: flujo integrado sobre el campo de visión.
- ERROR_1RE: incertidumbre de FLUX_1RE.
- ERROR_FoV: incertidumbre de FLUX_FoV.

Verificar mediante headers y documentación del producto las unidades, definición de aperturas, significado de ERROR y marco espectral.

Los archivos se leen con Astropy durante la construcción de la caché; no se abren durante cada época de entrenamiento.

En el EDA anterior se observó una estructura bastante homogénea en la muestra inspeccionada.

El manifest actual registra `fits_available=True` para 10,013 de las 10,014 filas. La existencia del archivo no garantiza que el espectro sea elegible: la caché registra por separado el resultado del preprocesamiento.

## 2.3. Imágenes RGB

El proyecto utiliza cutouts RGB de galaxias.

Estado implementado:

- Imágenes RGB derivadas de datos de Legacy Survey.
- Campo de visión adaptado al tamaño angular de la galaxia.
- Regla conceptual: `FOV = k * PETRO_TH90`.
- Se descargaron los factores `+3`, `+2`, `+1`, `0`, `-1`, `-2` y `-3`; `0` es la variante de referencia.
- Cada conjunto está en `data/datasets/morfologia_curada_fov_<factor>/` y contiene imágenes JPEG, `download_index.csv` y `morfologia_curada_with_fov.csv`.
- `download_index.csv` es la fuente de verdad para la descarga: incluye `name`, `local_path`, `status`, `fov_factor`, `fov_requested_arcsec`, `fov_effective_arcsec`, `pixscale`, `size_px`, `layer` y `bands`.
- Los JPEG registrados tienen tamaño solicitado de 256 × 256 píxeles, capa `ls-dr11` y bandas `grz`.

El manifest centralizado, no el orden de los archivos ni el orden de las filas de los índices, debe utilizarse para seleccionar un recorte. La disponibilidad requiere simultáneamente un estado `downloaded` o `cached` y que el archivo en `local_path` exista en disco.

La función `src.data.analyze_image_quality` evalúa cada JPEG disponible y registra una bandera `is_suspicious`. Esta es una heurística de revisión para artefactos lineales o contaminación brillante; no constituye por sí sola un criterio automático de descarte.

Utilizar el pipeline existente de imágenes como fuente de verdad.

No implementar un downloader nuevo salvo que se solicite explícitamente.

## 2.4. Síntesis de poblaciones estelares STARLIGHT

Directorio de entrada: `data/starlight/` con un `.out` de STARLIGHT por galaxia (`manga-<plate>-<ifudesign>-1D.out`). Hay 10,243 archivos; 230 no tienen fila en el manifest y se omiten, de modo que 10,013 galaxias del manifest tienen salida STARLIGHT.

Los ajustes se hicieron sobre el espectro `FLUX_FoV` de MPL-11 llevado a reposo con `nsa_z` (el `f_obs` de STARLIGHT reescalado por `fobs_norm` coincide con `FLUX_FoV` de MPL-11 a reposo con razón mediana 1.00001; no coincide con `FLUX_1RE`; comprobado en una galaxia). Los resultados no son observaciones nuevas: derivan del mismo espectro FoV, con la base BC03 de 45 poblaciones (`N_base=45`).

El extractor `src.data.starlight_extractor` convierte cada `.out` en tres insumos por galaxia, todos nombrados con la columna `name` del manifest (sin el sufijo `-1D` de los `.out`):

    python -m src.data.starlight_extractor data/starlight --output data/processed/starlight

| Insumo | Ruta (bajo `data/processed/starlight/`) | Contenido |
|---|---|---|
| Espectro modelado | `espectros_fits/<name>-1D.fits` | FITS compatible con `src.dataloader.spectra`; ver más abajo. |
| Poblaciones estelares | `poblaciones_estelares/<name>.npz` | `matriz` `[2, 45]` (fila 0: fracción en luz `x_j` %, fila 1: fracción en masa actual %), `edades_yr`, `metalicidades_z` y `canales`. |
| Propiedades | `propiedades/<name>.json` | Parámetros del ajuste y propiedades derivadas. |

También se generan `espectros_starlight/<name>.npz` (espectro de STARLIGHT en reposo: `lambda_angstrom`, `f_obs`, `f_syn`, `residuo`, `peso`, `mascara_valida`; flujos normalizados por `fobs_norm`), `propiedades/indice.csv` (una fila por galaxia, llave `name`) y `poblaciones_estelares/base_poblaciones.npz` (cuadrícula común de edad y metalicidad). El extractor falla para una galaxia si su cuadrícula edad/Z difiere de la de las demás.

**Espectro modelado (FITS).** Tiene el mismo esquema que los de MaNGA: extensión `Spectra` con las columnas `WAVE` y `FLUX_FoV`, más `FLUX_FoV_OBS` (espectro observado) y `WEIGHT_FoV` (pesos de STARLIGHT). La cabecera guarda `Z` (el `nsa_z` usado), `FOBSNORM` y `SPECKIND`.

- `FLUX_FoV` es el **modelo de mejor ajuste** (`f_syn * fobs_norm`), en las unidades del espectro de entrada, no el espectro observado. No hay columnas `FLUX_1RE` ni `ERROR_*`: el dataloader solo se puede usar con `--channels FLUX_FoV`.
- STARLIGHT trabaja en reposo y el dataloader divide `WAVE` entre `1 + nsa_z` (sección 6.2). Por eso `WAVE` se escribe en marco observado: `lambda_reposo * (1 + nsa_z)`. La corrección sigue aplicándose una sola vez, en el dataloader.
- Los píxeles sin dato observado (`wei = -2`) se escriben como NaN para que la máscara de cobertura no incluya modelo extrapolado.
- Cobertura: muestreo de 1 Å en 3400--8900 Å de reposo; el primer píxel con dato está en ~3488 Å (mediana) y no se observaron huecos internos (muestra de 513 archivos, uno de cada 20). Con la malla por defecto de 3600--9000 Å la fracción válida es 0.981; los últimos ~100 Å quedan enmascarados.

**Propiedades (`.json`).** Entre otras: `name`, `nsa_z`, `source_file`, calidad del ajuste (`chi2_nl_eff`, `adev_pct`, `n_lambda_eff`, `n_chains`), S/N (`sn_window`, `sn_norm`, `sn_err_window`), cinemática y extinción (`v0_kms`, `vd_kms`, `av_mag`, `yav_mag`), normalización (`fobs_norm`, `lambda_norm_angstrom`), masas y flujo (`flux_tot`, `mini_tot`, `mcor_tot`), rango espectral y derivados de las poblaciones: `age_mean_light_gyr`, `age_mean_mass_gyr`, `z_mean_light`, `z_mean_mass`, `light_young_lt_100myr_pct` y `mass_old_ge_1gyr_pct`. Las unidades de masa (`mini_tot`, `mcor_tot`) no están verificadas (`mass_units_verified: false`); no usarlas como masa física sin confirmarlas. La suma de `x_j` no es exactamente 100 % (p. ej. 103 % en `manga-10001-12701`); los derivados en luz se normalizan por esa suma.

Estos insumos son derivados de espectros: su uso como entrada tiene las restricciones de las secciones 11.2 y 22.6.

## 2.5. Integración mediante identificadores

El identificador `name` será la llave principal.

Ejemplo:

Catálogo:
`manga-7443-1901`

Espectro:
`manga-7443-1901-1D.fits`

Construir una correspondencia determinista.

La asociación con las imágenes debe validarse según su esquema real.

Insumos STARLIGHT:
`manga-7443-1901.json`, `manga-7443-1901.npz` y `manga-7443-1901-1D.fits`

Los `.out` originales usan el sufijo `-1D`; el extractor lo elimina de los `.json` y `.npz` para que `name` coincida con el manifest, y lo conserva en el FITS para que `spectrum_file_for_name` lo encuentre.

No confiar en el orden de los archivos ni en índices posicionales para relacionar modalidades.

---

# 3. Diseño general de infraestructura

La arquitectura del software debe separar cinco responsabilidades:

1. Preparación e indexación de datos.
2. Datasets, transforms y DataLoaders.
3. Arquitecturas neuronales.
4. Entrenamiento y evaluación.
5. Configuración, seguimiento y análisis experimental.

Flujo:

    Catálogo + RGB + FITS
              |
              v
       Manifest maestro
              |
              v
        Split persistente
              |
              v
       GalaxyDataset
              |
              v
         DataLoader
              |
       +------+------+
       |      |      |
      RGB  Espectro Metadata
       |      |      |
     CNN2D  CNN1D   MLP
       |      |      |
       +------+------+
              |
              v
        Fusion Module
              |
              v
          Classifier
              |
              v
          Logits
              |
              v
       Trainer / Evaluator
              |
              v
    Métricas + Checkpoints

Los modelos pueden utilizar una, dos o tres modalidades.

El training loop no debe depender de la arquitectura interna.

---

# 4. Estructura del repositorio

Propuesta:

    project/
    |
    |-- data/
    |   |-- morfologia_curada.csv
    |   |-- MPL-11/
    |   |   `-- manga-*-1D.fits.gz
    |   |-- images/
    |   |   `-- ...
    |   `-- processed/
    |       |-- manifest.csv
    |       |-- splits.csv
    |       `-- spectra/
    |
    |-- src/
    |   |-- data/
    |   |   `-- prepare.py
    |   |
    |   |-- dataloader/
    |   |   |-- dataset.py
    |   |   |-- spectra.py
    |   |   |-- transforms.py
    |   |   `-- datamodule.py
    |   |
    |   |-- models/
    |   |   |-- image_encoder.py
    |   |   |-- spectrum_encoder.py
    |   |   |-- metadata_encoder.py
    |   |   |-- fusion.py
    |   |   |-- classifier.py
    |   |   `-- factory.py
    |   |
    |   |-- training/
    |   |   |-- trainer.py
    |   |   |-- metrics.py
    |   |   `-- callbacks.py
    |   |
    |   `-- utils/
    |       |-- reproducibility.py
    |       `-- config.py
    |
    |-- configs/
    |   |-- base.yaml
    |   |-- e1_image.yaml
    |   |-- e2_spectrum.yaml
    |   |-- e3_concat.yaml
    |   |-- e4_attention.yaml
    |   |-- e5_gated.yaml
    |   |-- e6_cross_attention.yaml
    |   `-- ...
    |
    |-- scripts/
    |   |-- prepare_data.py
    |   |-- train.py
    |   `-- evaluate.py
    |
    |-- notebooks/
    |   |-- eda_espectros_manga.ipynb
    |   `-- experiment_analysis.ipynb
    |
    |-- tests/
    |   |-- test_data.py
    |   |-- test_models.py
    |   `-- test_training.py
    |
    |-- runs/
    |   `-- ...
    |
    |-- requirements.txt
    |-- README.md
    `-- EXPERIMENT_GUIDELINES.md

Adaptar la estructura si existen convenciones previas en el repositorio.

No duplicar módulos existentes que ya resuelvan correctamente una responsabilidad.

Evitar crear frameworks internos innecesarios.

---

# 5. Preparación e indexación de datos

## 5.1. Manifest maestro

Implementado:

`data/galaxy_manifest.csv`

El notebook `notebook/julio/generar_manifest_galaxias.ipynb` genera este archivo con una fila por cada galaxia de `data/morfologia_curada.csv`, unida de forma determinista por `name` con los siete índices de FOV. La versión actual contiene 10,014 filas; en cada FOV hay 10,006 JPEG disponibles. Estas cantidades deben regenerarse y reportarse si se actualizan los índices o los archivos de imágenes.

Conserva las columnas del catálogo curado:

- `name`
- `label`
- `Ttype`
- `nsa_z`
- `LogMass`
- `incl`
- `PETRO_TH90`
- `objra`
- `objdec`

Para cada factor `F` en `+3`, `+2`, `+1`, `0`, `-1`, `-2` y `-3`, agrega:

- `fov_F`: disponibilidad real del JPEG (`True` solo si el estado es descargado/en caché y la ruta existe).
- `fov_F_path`: ruta del JPEG relativa a la raíz del repositorio.
- `fov_F_is_suspicious`: bandera nullable de `analyze_image_quality`; es nula cuando el JPEG no está disponible o no puede analizarse.
- `fov_F_quality_status` y `fov_F_quality_error`: trazabilidad del análisis visual.

También agrega:

- `fits_available`: `True` si existe `data/MPL-11/<name>-1D.fits`.
- `fits_path`: ruta relativa del FITS cuando existe; nula en caso contrario.

Para cada insumo STARLIGHT (sección 2.4), una celda del notebook agrega una pareja disponibilidad/ruta:

- `starlight_spectrum_available` y `starlight_spectrum_path`: `data/processed/starlight/espectros_fits/<name>-1D.fits`.
- `starlight_populations_available` y `starlight_populations_path`: `data/processed/starlight/poblaciones_estelares/<name>.npz`.
- `starlight_properties_available` y `starlight_properties_path`: `data/processed/starlight/propiedades/<name>.json`.

La celda recarga el manifest, reemplaza estas columnas si ya existían y lo guarda de nuevo. Falla si una galaxia tiene solo parte de los tres insumos, si no existe `propiedades/indice.csv`, si este no coincide con los FITS disponibles o, opcionalmente, si el `Z` de la cabecera de un FITS difiere de `nsa_z`. Informa sin fallar cuántos archivos no tienen fila en el manifest.

El generador falla explícitamente si el catálogo o un índice de FOV contiene IDs duplicados, si faltan las columnas requeridas del índice o si no existe un índice esperado. `fits_available` y las columnas `starlight_*_available` solo validan presencia en disco; la validez científica/técnica de una apertura se determina durante la construcción de la caché espectral y no se persiste en el manifest.

El manifest es el registro central de disponibilidad de datos.

## 5.2. Splits persistentes

Implementado:

`data/processed/splits_seed42_obj-label.csv`

El split binario usa `label`, estratificación y semilla 42. La cohorte se define antes de dividir: conserva únicamente galaxias con imágenes disponibles y no sospechosas en los FOV `+1`, `0` y `-1`. Una bandera de calidad ausente o un error de análisis excluye la galaxia de esta cohorte. Esta decisión produce un baseline RGB limpio, no una eliminación permanente de las filas del manifest.

La partición persistida contiene 9,351 galaxias y conserva la proporción de clases en los tres conjuntos:

| Split | Total | Tardía | Temprana |
|---|---:|---:|---:|
| Train | 6,545 | 4,078 | 2,467 |
| Validation | 1,403 | 874 | 529 |
| Test | 1,403 | 875 | 528 |

Cada fila contiene `name`, `label` y `split`. No hay IDs duplicados ni etiquetas de split fuera de `train`, `validation` y `test`.

Los splits deben generarse una sola vez y reutilizarse en todos los experimentos comparables.

Si hay observaciones repetidas de una misma galaxia física, utilizar un identificador de grupo para evitar que aparezcan en conjuntos diferentes.

Investigar duplicados por identificador de objeto y, cuando sea necesario, utilizando coordenadas u otros identificadores catalográficos.

La cohorte multimodal activa es la misma partición de 9,351 IDs: la única galaxia sin FITS se retiró antes de persistir el split. El archivo alternativo `data/processed/splits_seed42_obj-label_FoV.csv` contiene los mismos IDs y la misma asignación; `data/processed/splits_seed42_obj-label.csv` es la referencia operativa.

Para cada experimento espectral, la caché seleccionada debe contener un registro `ready` para todos los IDs del split. Si una caché registra errores de preprocesamiento, crear y persistir una cohorte alineada antes de entrenar; nunca eliminar filas silenciosamente desde `GalaxyDataset`.

Esto garantiza una comparación justa entre modalidades.

Se pueden realizar experimentos secundarios con muestras más amplias, pero deben reportarse por separado.

No utilizar el conjunto de test para seleccionar arquitecturas o ajustar hiperparámetros.

No modificar los splits entre experimentos.

---

# 6. Preprocesamiento espectral

Esta parte requiere especial atención científica.

## 6.1. Lectura

Implementado en `src.dataloader.spectra`.

Responsabilidades:

- Abrir archivos `.fits`.
- Leer la extensión Spectra.
- Extraer WAVE, FLUX y ERROR.
- Verificar tipos, dimensiones y unidades.
- Detectar valores no finitos.
- Cerrar correctamente los archivos.
- Reportar problemas con mensajes claros.

El lector debe funcionar independientemente del Dataset.

## 6.2. Corrección por redshift

Tenemos `nsa_z` en el catálogo curado.

Para longitudes de onda observadas:

    lambda_rest = lambda_obs / (1 + z)

El EDA en `notebook/julio/eda_espectros_manga.ipynb` confirmó que `WAVE` describe longitudes de onda observadas y que la información de `nsa_z` está disponible para la muestra. Por tanto, el preprocesador debe aplicar esta conversión una sola vez antes de interpolar en la malla común. La malla y sus límites se expresarán en marco de reposo.

No aplicar una segunda corrección a espectros ya convertidos.

Los espectros de STARLIGHT ya están en reposo. Para reutilizar el dataloader sin cambios, sus FITS se escriben con `WAVE = lambda_reposo * (1 + nsa_z)` (sección 2.4); el preprocesador los devuelve a reposo con el mismo `nsa_z` del manifest. Si un FITS de STARLIGHT se generó con otro `nsa_z`, el espectro queda desplazado; por eso su cabecera guarda `Z` y el manifest lo valida.

Si el marco espectral no puede confirmarse, detener esa parte del preprocesamiento o dejarla explícitamente pendiente.

La corrección debe quedar registrada en la configuración.

La transformación del eje espectral no debe confundirse con una calibración física completa de las densidades de flujo.

## 6.3. Malla espectral común

Los modelos necesitan entradas de longitud consistente.

Crear una malla de longitudes de onda compartida.

Parámetros configurables:

- Longitud de onda mínima.
- Longitud de onda máxima.
- Número de muestras.
- Marco espectral.

Se puede comenzar evaluando una longitud de 4096 muestras.

No fijar el intervalo definitivo sin revisar la cobertura en reposo de la muestra.

Interpolar los espectros cuando sea necesario.

Restricciones:

- No extrapolar valores fuera de cobertura.
- Mantener máscara de regiones válidas.
- No interpretar valores rellenados como observaciones reales.
- Documentar el impacto de la interpolación.
- Preservar suficiente resolución para las líneas relevantes.

La máscara debe acompañar al espectro durante su uso por el modelo.

## 6.4. Canales espectrales

La caché determina el número de canales, por lo que el tensor puede ser `[B, 1, L]` para una apertura o `[B, 2, L]` para ambas. El batch siempre incluye `spectrum_mask` con forma `[B, L]`.

Cachés disponibles:

- `spectra_cache_1RE`: `FLUX_1RE`, normalización `per_channel`.
- `spectra_cache_FoV`: `FLUX_FoV`, normalización `per_channel`.
- `spectra_cache_1RE_FoV`: `FLUX_1RE` y `FLUX_FoV`, normalización `shared`.

Las tres usan malla de reposo $3600$--$9000$ Å, longitud 4,096, ventana de continuo $5400$--$5500$ Å y al menos 5 muestras de continuo.

Caché del espectro modelado por STARLIGHT (solo `FLUX_FoV`, ver 2.4). STARLIGHT corta en 8900 Å, así que la malla se recorta para que quede cubierta por completo:

    python -m src.dataloader.cache_spectra \
      --fits-dir data/processed/starlight/espectros_fits \
      --channels FLUX_FoV --rest-wave-max 8900 \
      --output-dir data/processed/spectra_starlight --all-manifest

Esta caché tiene longitud 4,096 sobre 3600--8900 Å (paso ≈1.29 Å), por lo que su huella de configuración difiere de las anteriores y no es comparable punto a punto con ellas. Con `--all-manifest`, las galaxias sin FITS quedan con `status=error` en `index.csv`. Para comparar con MPL-11 usar el mismo rango en ambas cachés (regenerar la de MPL-11 con `--rest-wave-max 8900`); con la malla de 3600--9000 Å, un modelo entrenado con MPL-11 vería una cola de ceros en 8900--9000 Å que no existe en MPL-11.

Explorar posteriormente:

- Solo 1RE.
- Solo FoV.
- Ambos canales.
- Incertidumbres como canales adicionales.
- Estrategias de enmascaramiento informadas por errores.

No incorporar columnas ERROR como canales hasta verificar su significado físico.

## 6.5. Normalización

La normalización espectral debe ser explícita y configurable.

Objetivo principal:

Permitir comparar formas espectrales sin que la amplitud absoluta domine necesariamente el aprendizaje.

La normalización inicial por muestra usará la mediana de flujo finito en la ventana de continuo de reposo `CONTINUUM_WINDOW = (5400, 5500)`. Si una apertura no contiene suficientes muestras válidas en esa ventana o su mediana no es finita/no nula, el espectro debe marcarse como inválido con una razón explícita.

Evitar utilizar máximos dominados por líneas de emisión o ruido.

Mantener configurables las dos variantes:

1. Normalización independiente de cada apertura.
2. Normalización compartida de ambas aperturas.

La segunda puede preservar diferencias relativas de amplitud entre 1RE y FoV.

No asumir de antemano que ambas estrategias son equivalentes. La primera caché debe guardar el modo seleccionado y la escala aplicada por muestra para que las ablaciones sean reproducibles.

La normalización implementada es física y por muestra; no hay una segunda estandarización global del espectro. Si se añade una, sus estadísticas por canal y longitud de onda se deben ajustar exclusivamente con train, ignorando posiciones con `spectrum_mask=False`, y reutilizarse sin reajuste en validation y test.

Las transformaciones por muestra deben estar completamente documentadas.

## 6.6. Tratamiento de valores inválidos

Implementar un mecanismo consistente para:

- NaN.
- Infinito.
- Valores de error inválidos.
- Regiones fuera de cobertura.
- Espectros insuficientemente válidos.

Mantener máscaras de validez.

Los valores de relleno pueden establecerse en cero después de la normalización, pero el modelo debe recibir o respetar la máscara.

Evitar que valores artificiales dominen el pooling o los mecanismos de atención.

Los criterios para excluir espectros de baja calidad deben definirse y registrarse; no eliminar silenciosamente ejemplos.

## 6.7. Caché de espectros

No abrir y descomprimir todos los FITS durante cada época de entrenamiento.

Implementado: `src.dataloader.cache_spectra` crea cachés de tensores `float32` mediante `python -m src.dataloader.cache_spectra`.

Formato recomendado inicialmente:

- NumPy `.npy`.
- Memoria mapeada cuando resulte útil.
- Archivo o índice de correspondencia con IDs.
- Máscaras de validez.
- Metadatos del preprocesamiento.

Por ejemplo, 10,000 galaxias con 2 canales y 4096 valores requieren aproximadamente 328 MB para los flujos en float32, sin contar máscaras y metadatos.

Cada caché contiene `metadata.json`, `index.csv`, `tensors/<name>.npy` y `masks/<name>.npy`. El índice registra por ID `status`, error, ruta fuente, rutas de tensor/máscara, fracción válida y escalas de normalización. `metadata.json` guarda la configuración y las huellas de configuración/cohorte.

Seleccionar explícitamente `data.spectrum.cache_dir` cuando existan varias cachés; no inferir una caché por orden de directorio.

No almacenar tensores en memoria innecesariamente si el acceso desde disco es suficientemente rápido.

---

# 7. Preprocesamiento de imágenes

Utilizar los archivos RGB existentes.

Formato implementado del tensor:

    [3, 256, 256]

Verificar tamaño y canales reales de las imágenes.

Pipeline implementado:

- Carga de imagen.
- Conversión explícita a RGB.
- Resize cuando sea necesario.
- Conversión a tensor.
- Normalización `none` para tensores en `[0, 1]`, o `imagenet` para encoders que requieran esas estadísticas.

Si se utiliza una CNN preentrenada, respetar su preprocesamiento esperado.

## 7.1. Augmentations

Durante entrenamiento:

- Rotaciones de múltiplos de 90 grados.
- Flips horizontales.
- Flips verticales.
- Opcional: selección aleatoria de FOV (`data.image.train_fov_columns`, ver sección 8.3). Se aplica antes de las transformaciones geométricas.

Estas transformaciones deben preservar la etiqueta binaria.

No aplicar transformaciones geométricas ni selección aleatoria de FOV a validación o test; estos splits usan siempre `data.image.fov_column`.

Evitar inicialmente cambios de color o brillo agresivos que puedan destruir información astronómica.

Registrar las augmentations utilizadas.

## 7.2. Variables de escala

Diferenciar:

- `PETRO_TH90`: tamaño angular catalogado.
- `fov_arcsec`: campo de visión angular del cutout.
- `requested_pixscale`: escala solicitada al servicio de descarga.
- `final_pixscale`: escala angular efectiva de la imagen final.

Si el ancho del FOV es conocido:

    final_pixscale = fov_arcsec / image_width_px

Esta relación se aplica cuando FOV describe efectivamente el ancho angular final de la imagen.

No asumir que requested_pixscale y final_pixscale son iguales.

El redimensionamiento de imágenes puede modificar la escala efectiva.

La escala angular tampoco equivale por sí sola a la resolución óptica efectiva, que depende del PSF y otros factores.

Conservar el origen de estos parámetros.

---

# 8. GalaxyDataset y DataLoader

Implementado en `src.dataloader.multimodal`: un `GalaxyDataset` único y `build_dataloaders(config)` para las cinco modalidades (`image`, `spectrum`, `metadata`, `properties`, `populations`). Las dos últimas usan los insumos STARLIGHT de la sección 2.4 y están implementadas en `src.dataloader.starlight`.

Interfaz conceptual:

    GalaxyDataset(
        manifest,
        split,
        modalities,
        image_transform,
        spectrum_config,
        metadata_config
    )

Modalidades:

- `image`
- `spectrum`
- `metadata` (columnas del manifest)
- `properties` (propiedades STARLIGHT, `propiedades/indice.csv`)
- `populations` (matriz de poblaciones STARLIGHT)

Permitir cualquier combinación válida.

Ejemplo de muestra:

    {
        "id": str,
        "image": Tensor[3, 256, 256],
        "spectrum": Tensor[2, L],
        "spectrum_mask": Tensor[L],
        "metadata": Tensor[M],
        "metadata_mask": Tensor[M],
        "properties": Tensor[P],
        "properties_mask": Tensor[P],
        "populations": Tensor[C, 45],
        "label": Tensor
    }

Las claves opcionales pueden omitirse cuando la modalidad no esté activa.

No cargar una modalidad que no se utilizará.

El mapeo de etiquetas será fijo:

    temprana = 0
    tardia   = 1

Usar un tipo numérico compatible con BCEWithLogitsLoss.

Los metadatos adicionales destinados a evaluación, como Ttype, pueden conservarse fuera de los predictores.

## 8.1. DataLoaders

La función común implementada es:

    build_dataloaders(config)

Devuelve:

- train_loader
- val_loader
- test_loader

Configuración inicial:

- batch_size: 32
- num_workers: configurable
- pin_memory: cuando resulte conveniente
- persistent_workers: solo si `num_workers > 0`
- shuffle: únicamente para train

Adaptarse al hardware disponible.

No asumir que existe CUDA.

## 8.2. Contrato de batch

Los modelos reciben un diccionario con `id`, `label` y las claves de las modalidades activadas.

Contrato:

    logits = model(batch)

Salida:

    logits.shape == [batch_size]

El smoke test multimodal validado produjo `image: [32, 3, 256, 256]`, `metadata: [32, 4]`, `metadata_mask: [32, 4]`, `spectrum: [32, 1, 4096]`, `spectrum_mask: [32, 4096]` y `label: [32]` de tipo `float32`. La salida del modelo debe contener logits sin sigmoid.

El training loop no necesita conocer los detalles de los encoders.

## 8.3. Referencia operativa para experimentos

Esta es la interfaz vigente para crear experimentos. Importar siempre
`build_dataloaders` desde `src.dataloader.multimodal`; no reconstruir CSVs, transforms
ni datasets a mano dentro de un notebook.

```python
from src.dataloader.multimodal import build_dataloaders

config = {
    "data": {
        "project_root": ROOT,
        "manifest": "data/galaxy_manifest.csv",
        "splits": "data/processed/splits_seed42_obj-label.csv",
        "modalities": ["image", "metadata", "spectrum"],
        "image": {
            "fov_column": "fov_0",
            "size": 256,
            "normalization": "none",  # o "imagenet" para un encoder compatible
        },
        "metadata": {"features": ["nsa_z", "LogMass"]},
        "spectrum": {
            "cache_dir": "data/processed/spectra/spectra_cache_FoV",
        },
    },
    "training": {
        "batch_size": 32,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    },
}
train_loader, validation_loader, test_loader = build_dataloaders(config)
```

`project_root`, `manifest`, `splits` y `cache_dir` aceptan rutas relativas a
`project_root` o rutas absolutas. El manifest y el archivo de splits se unen por
`name`; el loader falla si hay IDs duplicados, IDs ausentes o si las etiquetas de
ambas fuentes no coinciden. Los únicos splits admitidos son `train`,
`validation` y `test`.

### Modalidad `image`

Activar con `data.modalities = ["image"]` o junto con otras modalidades. Requiere
`data.image.fov_column`; el valor habitual es `fov_0`, que hace que el Dataset
use `fov_0_path` del manifest. Cada fila solicitada debe marcar ese FOV como
disponible y su JPEG debe existir: no se omiten ejemplos ausentes.

Opcionalmente, `data.image.train_fov_columns` (por ejemplo
`["fov_-1", "fov_0", "fov_+1"]`) actúa como una augmentation más:

```python
"image": {
    "fov_column": "fov_0",                                  # FOV fijo de validation/test
    "train_fov_columns": ["fov_-1", "fov_0", "fov_+1"],     # pool aleatorio solo en train
    "size": 256,
    "normalization": "none",
},
```

- En **train**, cada vez que se accede a una galaxia (`__getitem__`) se elige un
  FOV de la lista con `random.choice`: distribución uniforme (1/N por FOV),
  independiente entre accesos y con el RNG global de Python, por lo que
  `set_seed` la hace reproducible con `num_workers=0`. Los duplicados de la
  lista se eliminan, así que repetir un FOV no aumenta su probabilidad.
- La selección ocurre antes de las transformaciones geométricas.
- Validation y test usan siempre `fov_column`, sin aleatoriedad.
- No cambia el tamaño del dataset (cada época sigue teniendo una muestra por
  galaxia) ni la forma del tensor.
- Todos los FOV de la lista deben estar disponibles (`fov_X == True`) y tener
  JPEG existente para todas las filas de train; si no, `build_dataloaders` lanza
  `ValueError`. Sin esta clave, el comportamiento es el de un único `fov_column`.
- `GalaxyDataset` recibe la lista mediante `random_image_fov_columns`.
- Al generar predicciones de train tras entrenar, usar una copia del dataset con
  `image_transform = build_image_transform(size, train=False, ...)` y
  `random_image_path_columns = ()`; de lo contrario las predicciones incluyen
  augmentations y FOV aleatorio. El notebook
  `notebook/julio/experiments_fov/experimento_binario_imagenes_gpu_acc_fov.ipynb`
  lo implementa en `predict_split`.

Transformaciones implementadas en `src.dataloader.multimodal_transforms`:

| Split | Transformaciones, en orden |
|---|---|
| Train | Conversión PIL a RGB; rotación aleatoria en $0$, $90$, $180$ o $270$ grados; flip horizontal aleatorio; flip vertical aleatorio; `Resize((size, size))`; `ToTensor()`; normalización opcional. |
| Validation/Test | Conversión PIL a RGB; `Resize((size, size))`; `ToTensor()`; la misma normalización opcional, sin augmentations geométricas. |

`normalization="none"` produce `float32` en $[0, 1]$. `normalization="imagenet"`
aplica media `(0.485, 0.456, 0.406)` y desviación `(0.229, 0.224, 0.225)`;
usarla únicamente cuando el encoder lo requiera. El batch aporta:

```text
image: float32 [B, 3, size, size]
```

### Modalidad `metadata`

Activar con `"metadata"` y declarar `data.metadata.features`. El
`MetadataPreprocessor` se ajusta exclusivamente con el frame `train`:

1. Convierte cada feature a numérica; valores no convertibles se consideran
   ausentes.
2. Calcula mediana, media y desviación estándar en train.
3. Imputa ausentes con la mediana de train.
4. Estandariza como `(valor_imputado - media_train) / escala_train`.

El batch aporta:

```text
metadata:      float32 [B, M]
metadata_mask: bool    [B, M]  # True cuando el valor original era finito
```

Nunca incluir `label`, `Ttype`, `Type`, `name`, `objra` ni `objdec` en
`metadata.features`: el loader los rechaza explícitamente por leakage. Tampoco
usar estadísticas de validation o test para imputación o escalado.

### Modalidad `spectrum`

Activar con `"spectrum"` y una ruta explícita en `data.spectrum.cache_dir`. El
Dataset lee tensores y máscaras desde la caché seleccionada; no abre FITS durante
las épocas. La caché debe tener estado `ready` para cada ID solicitado, de lo
contrario el loader falla antes de entrenar. No escoger automáticamente una
caché cuando existen varias aperturas.

El preprocesamiento ya almacenado en las cachés vigentes aplica corrección a
marco de reposo una vez, normalización por mediana del continuo en $5400$--$5500$
Å y reinterpolación sin extrapolar a una malla de $3600$--$9000$ Å con 4,096
posiciones. Las regiones no observadas se rellenan con cero y siempre deben
acompañarse de su máscara. El batch aporta:

```text
spectrum:      float32 [B, C, 4096]  # C depende de la caché: 1RE, FoV o ambas
spectrum_mask: bool    [B, 4096]     # True en longitudes observadas/válidas
```

Los modelos espectrales deben respetar `spectrum_mask` en pooling, atención o
cualquier agregación sobre la longitud de onda.

### Modalidad `properties`

Activar con `"properties"` y declarar `data.properties.features`. Lee `propiedades/indice.csv` (por defecto `data/processed/starlight/propiedades`, configurable con `data.properties.dir`) y lo une por `name`, sin modificar el manifest. Usa el mismo `MetadataPreprocessor` que `metadata` (mediana, media y desviación ajustadas solo con train). El loader falla si algún ID del split no tiene propiedades. El batch aporta:

```text
properties:      float32 [B, P]
properties_mask: bool    [B, P]  # True cuando el valor original era finito
```

Se rechazan como features, con un error explícito:

- Identificadores y no numéricas: `name`, `source_file`, `flux_units`, `mass_units_verified`.
- Calidad del ajuste y normalización (guía 11.2): `chi2_nl_eff`, `adev_pct`, `fobs_norm` y cualquier columna que empiece por `sn_` o `n_`.
- `mini_tot` y `mcor_tot`, salvo `data.properties.allow_unverified_mass: true`, porque sus unidades no están verificadas.

### Modalidad `populations`

Activar con `"populations"`. Lee `poblaciones_estelares/<name>.npz` (por defecto `data/processed/starlight/poblaciones_estelares`, configurable con `data.populations.dir`). Opciones:

- `channels`: subconjunto de `["luz_pct", "masa_actual_pct"]` (por defecto ambos).
- `renormalize` (por defecto `true`): cada canal se divide entre su suma, de modo que son fracciones que suman 1. Necesario porque la fracción de luz no suma exactamente 100 % (p. ej. 103 % en `manga-10001-12701`).
- `flatten` (por defecto `false`): devuelve `[B, C*45]` en lugar de `[B, C, 45]`.

Validaciones: la cuadrícula de edad y metalicidad de cada galaxia debe coincidir con `base_poblaciones.npz`, el formato y los canales deben ser los esperados, y no se admiten valores no finitos, negativos ni sumas nulas. Falla si falta el archivo de una galaxia. El batch aporta:

```text
populations: float32 [B, C, 45]   # [B, C*45] con flatten
```

Los insumos de ambas modalidades derivan del mismo espectro FoV (secciones 11.2 y 22.6): reportar los experimentos que las usen como de contexto físico y comparar con ablaciones.

### Claves comunes y contrato de entrenamiento

Todo batch siempre contiene:

```text
id:    secuencia de B strings MaNGA
label: float32 [B]  # temprana=0.0, tardia=1.0
```

La etiqueta `float32` existe para `BCEWithLogitsLoss`. Al calcular métricas de
scikit-learn, convertir las etiquetas y predicciones de umbral a enteros binarios
unidimensionales; las probabilidades permanecen `float`. El contrato común es:

```python
logits = model(batch)                  # float [B], sin sigmoid
loss = BCEWithLogitsLoss()(logits, batch["label"])
probabilities = torch.sigmoid(logits)  # solo para métricas o inferencia
```

Las pruebas de contrato de las modalidades STARLIGHT están en `tests/test_starlight_modalities.py`.

Antes de proponer una arquitectura, inspeccionar un batch real y construir el
modelo solo con las claves de modalidad declaradas en `config["data"]["modalities"]`.
Mantener el mismo manifest, split persistente, FOV y caché al comparar modelos;
un cambio en cualquiera de esos insumos define una cohorte o experimento nuevo.

---

# 9. Arquitectura neuronal

Implementar encoders independientes e intercambiables.

## 9.1. ImageEncoder

Primera implementación:

ResNet-18.

Preferentemente inicializada con pesos preentrenados cuando estén disponibles y su uso esté autorizado.

Si los pesos no están almacenados localmente, no realizar descargas inesperadas.

Permitir:
- Inicialización aleatoria.
- Backbone congelado.
- Backbone parcialmente congelado.
- Fine-tuning completo.

Utilizar global average pooling.

Evitar flatten directo de mapas espaciales grandes.

Salida inicial:

    image_embedding: [B, 128]

La dimensión debe ser configurable.

## 9.2. SpectrumEncoder

Primera implementación:

CNN 1D pequeña con varios bloques convolucionales.

Posibles componentes:

- Conv1d.
- Activación ReLU/GELU.
- Normalización apropiada.
- Downsampling controlado.
- Bloques residuales opcionales.
- Pooling global consciente de máscaras.
- Proyección final.

Entrada:

    spectrum: [B, 2, L]

Salida:

    spectrum_embedding: [B, 128]

Diseñar una interfaz que permita reemplazar este encoder posteriormente.

## 9.3. Encoder espectral con atención

Segunda arquitectura espectral:

    Spectrum
       |
       v
    CNN 1D
       |
       v
    Downsampling / Tokens
       |
       v
    Positional Encoding
       |
       v
    Self-Attention
       |
       v
    Pooling
       |
       v
    Embedding

No comenzar con atención cuadrática sobre los 4487 puntos originales.

Reducir primero la secuencia mediante convoluciones o patches.

Se puede explorar una tokenización de aproximadamente 32 muestras por patch, ajustándola a la resolución espectral y al hardware.

La codificación posicional es necesaria para conservar la información de ubicación espectral.

Las máscaras deben propagarse correctamente hasta la atención.

No interpretar automáticamente los pesos de atención como explicaciones causales.

## 9.4. MetadataEncoder

Implementar un MLP pequeño para variables numéricas.

Ejemplo conceptual:

    Linear(M, 32)
    ReLU()
    Linear(32, 32)

Salida:

    metadata_embedding: [B, 32]

No utilizar arquitecturas innecesariamente complejas para pocas variables tabulares.

## 9.5. FusionModule

Diseñar una interfaz flexible para diferentes estrategias.

Primera estrategia:

Concatenación.

    z = concat(z_image, z_spectrum)

Segunda estrategia:

Gated Fusion.

Un pequeño módulo aprende pesos o mecanismos de combinación para los embeddings.

Tercera estrategia:

Cross-Attention.

Esta requiere conservar tokens o mapas de características antes del pooling final, por lo que debe contar con un contrato específico.

No forzar cross-attention dentro de una interfaz que solamente acepte vectores finales.

## 9.6. Classifier

Clasificador MLP configurable.

Propuesta inicial:

    Linear(input_dim, 128)
    ReLU()
    Dropout(0.3)
    Linear(128, 1)

Salida binaria mediante un logit.

Usar:

`BCEWithLogitsLoss`

No incluir sigmoid dentro del modelo durante entrenamiento.

---

# 10. Model Factory

Implementar una función:

    build_model(config)

La factory debe construir el modelo solicitado mediante YAML.

Ejemplos:

- image_only
- spectrum_only
- multimodal_concat
- multimodal_attention
- multimodal_gated
- multimodal_cross_attention
- metadata_only
- multimodal_with_metadata

Utilizar un registro sencillo de modelos cuando resulte útil.

No introducir un sistema excesivamente abstracto.

Cada modelo debe cumplir:

    forward(batch) -> logits

Los modelos unimodales deben ignorar modalidades innecesarias.

El trainer debe permanecer idéntico al cambiar de modelo.

---

# 11. Metadatos: selección y prevención de leakage

No todas las columnas del catálogo deben utilizarse como features.

## 11.1. Variables prohibidas como entradas

Nunca introducir:

- `label`
- `Ttype`
- `Type`
- `name`

Razones:

- label es el objetivo.
- Ttype interviene directamente en la definición de la etiqueta.
- Type contiene clasificación morfológica.
- name es un identificador, no una propiedad predictiva.

Ttype y Type pueden utilizarse para análisis, visualización y evaluación, pero no como features del clasificador.

## 11.2. Variables experimentales

Candidatas:

- `nsa_z`
- `LogMass`
- `PETRO_TH90`
- `incl`
- `final_pixscale`

Evaluar cuidadosamente la procedencia de incl, ya que podría codificar información morfológica dependiente de otro proceso de clasificación.

Candidatas derivadas de STARLIGHT (sección 2.4), solo para experimentos de contexto (11.4): `age_mean_light_gyr`, `age_mean_mass_gyr`, `z_mean_light`, `z_mean_mass`, `av_mag`, `vd_kms`, `light_young_lt_100myr_pct`, `mass_old_ge_1gyr_pct` y la matriz de poblaciones `[2, 45]`. Excluir de las features los parámetros de calidad del ajuste y de normalización (`chi2_nl_eff`, `adev_pct`, `sn_*`, `fobs_norm`, `n_*`), que describen el procedimiento y no la galaxia. `mini_tot` y `mcor_tot` quedan fuera mientras sus unidades no estén verificadas. Ajustar cualquier estandarización solo con train, como el resto de variables tabulares.

No utilizar inicialmente objra y objdec como entradas.

Estas coordenadas pueden conservarse para detectar duplicados y analizar sesgos de selección.

## 11.3. Preprocesamiento tabular

Implementar:

- Selección de features desde YAML.
- Estandarización.
- Imputación documentada de valores faltantes.
- Máscara de features ausentes cuando proceda.
- Transformaciones justificadas para variables asimétricas.

Los parámetros de imputación y estandarización deben ajustarse únicamente con train.

Aplicar las mismas transformaciones a validation y test.

Guardar el estado del preprocesamiento junto con los modelos.

## 11.4. Experimentos de contexto

Separar conceptualmente:

A. Clasificación visual/espectroscópica.

B. Clasificación enriquecida con información física y observacional.

Esta distinción es importante para interpretar los resultados.

Una mejora por LogMass, redshift o pixscale no demuestra necesariamente que la red aprendió mejores características morfológicas.

---

# 12. Training Loop universal

Implementar una clase sencilla:

    Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config
    )

Debe ofrecer:

    trainer.fit()

Responsabilidades:

- Entrenamiento por épocas.
- Evaluación en validación.
- Forward y backward.
- Optimización.
- Scheduler.
- Early stopping.
- Guardado de checkpoints.
- Registro de métricas.
- Soporte opcional de precisión mixta.
- Control de semillas.
- Manejo de CPU y GPU.

No utilizar PyTorch Lightning inicialmente.

Preferimos un loop explícito y fácil de modificar.

## 12.1. Configuración inicial

| Parámetro | Valor propuesto |
|---|---|
| Optimizer | AdamW |
| Loss | BCEWithLogitsLoss |
| Batch size | 32 |
| Max epochs | 50 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 8 |
| Scheduler | ReduceLROnPlateau |
| Seed | 42 |

Estos valores son iniciales, no hiperparámetros definitivos.

Para fine-tuning de encoders preentrenados permitir learning rates diferentes entre backbone y capas nuevas.

Si existe desbalance de clases, evaluar mecanismos de ponderación basados exclusivamente en train.

## 12.2. Precisión mixta

Soportar AMP cuando el dispositivo lo permita.

Usar las APIs modernas de torch.amp.

Desactivar automáticamente cuando no sea compatible.

## 12.3. Checkpoints

Guardar:

- Model state_dict.
- Optimizer state_dict.
- Scheduler state_dict.
- Época.
- Mejor métrica de validación.
- Configuración.
- Estado de early stopping.
- Metadatos de reproducibilidad.
- Estados RNG cuando se requiera reanudación exacta.

Permitir reanudar entrenamientos interrumpidos.

Distinguir entre:
- best_model
- last_checkpoint

---

# 13. Métricas y evaluación

Métricas principales:

- Macro-F1.
- Balanced accuracy.
- Precision por clase.
- Recall por clase.
- ROC-AUC.
- Matriz de confusión.

Registrar también:

- Train loss.
- Validation loss.
- Tiempo de entrenamiento.
- Número de parámetros.
- Tiempo de inferencia, cuando sea útil.

La métrica primaria comparativa propuesta es macro-F1.

Para early stopping y selección inicial del checkpoint utilizar validation loss, salvo que un experimento justifique explícitamente otro criterio.

## 13.1. Umbral binario

Utilizar inicialmente 0.5.

Si se optimiza el umbral, realizar la selección únicamente con validación.

No ajustar umbrales utilizando test.

## 13.2. Evaluación científica

Analizar resultados por:

- label.
- Rangos de Ttype.
- Rangos de redshift.
- Rangos de masa estelar.
- Calidad espectral.
- Variables de escala angular.

Investigar si hay subgrupos donde el rendimiento se degrada.

Reportar tamaños de muestra por subgrupo.

## 13.3. Evaluación final

El conjunto de test debe permanecer reservado para las arquitecturas seleccionadas.

Durante desarrollo, comparar experimentos con validación.

No utilizar repetidamente test para decidir cambios.

Cuando sea posible, repetir los experimentos finalistas con varias semillas y reportar media y dispersión.

No interpretar diferencias pequeñas de una sola ejecución como evidencia concluyente.

---

# 14. Matriz de experimentos

El sistema debe permitir los siguientes experimentos.

## Experimentos principales

| ID | Arquitectura | Objetivo |
|---|---|---|
| E1 | CNN 2D, solo RGB | Baseline visual |
| E2 | CNN 1D, solo espectro | Baseline espectroscópico |
| E3 | CNN 2D + CNN 1D + concatenación | Fusión multimodal básica |
| E4 | CNN 2D + encoder espectral con atención | Evaluar atención espectral |
| E5 | Encoders + gated fusion | Evaluar fusión adaptativa |
| E6 | Encoders + cross-attention | Evaluar interacción entre modalidades |

## Experimentos con metadatos

| ID | Arquitectura | Objetivo |
|---|---|---|
| E7 | MLP tabular | Baseline de metadatos |
| E8 | RGB + espectro + metadatos físicos | Complementariedad física |
| E9 | RGB + espectro + pixscale | Impacto del contexto de escala |
| E10 | RGB + espectro + metadatos permitidos | Fusión trimodal completa |

El diseño de E7 debe documentar exactamente qué variables utiliza.

Para E8 comenzar con un subconjunto como LogMass y nsa_z, sujeto a disponibilidad y validación.

Agregar ablaciones cuando aporten información:

- Solo 1RE.
- Solo FoV.
- Ambos espectros.
- Normalización compartida vs independiente.
- RGB + metadatos.
- Espectro + metadatos.
- Sin pixscale vs con pixscale.

No ejecutar toda la matriz durante la primera fase del desarrollo.

---

# 15. Configuración declarativa

Cada experimento se define mediante YAML.

Ejemplo:

    experiment:
      name: e3_multimodal_concat
      seed: 42

    data:
      manifest: data/processed/manifest.csv
      splits: data/processed/splits.csv

      modalities:
        - image
        - spectrum

      image_size: 256

      spectrum:
        channels:
          - FLUX_1RE
          - FLUX_FoV
        length: 4096
        use_mask: true

    model:
      name: multimodal_concat

      image_encoder:
        name: resnet18
        pretrained: true
        embedding_dim: 128

      spectrum_encoder:
        name: cnn1d
        embedding_dim: 128

      fusion:
        name: concat

      classifier:
        hidden_dim: 128
        dropout: 0.3

    training:
      epochs: 50
      batch_size: 32
      optimizer: adamw
      learning_rate: 0.0001
      weight_decay: 0.0001
      early_stopping_patience: 8

    evaluation:
      primary_metric: macro_f1

Este YAML ilustra la interfaz prevista; las opciones deben validarse contra la implementación real.

No duplicar toda la configuración entre experimentos si se puede utilizar un archivo base y overrides simples.

Evitar introducir Hydra salvo que la complejidad lo justifique.

---

# 16. CLI para experimentación

Implementar comandos reproducibles.

Preparación:

    python -m scripts.prepare_data --config configs/base.yaml

Entrenamiento:

    python -m scripts.train --config configs/e1_image.yaml

Evaluación:

    python -m scripts.evaluate --run runs/<experiment>/<run_id>

Permitir sobreescritura de algunos parámetros mediante argumentos cuando sea sencillo.

Los scripts deben funcionar desde la raíz del proyecto.

Resolver rutas de manera consistente.

No depender del directorio de trabajo de un notebook.

---

# 17. Tracking de experimentos

Cada ejecución debe generar una carpeta única.

Ejemplo:

    runs/
    `-- e3_multimodal_concat/
        `-- seed_42_<run_id>/
            |-- config.yaml
            |-- best_model.pt
            |-- last_checkpoint.pt
            |-- metrics.json
            |-- history.csv
            |-- predictions.csv
            `-- confusion_matrix.png

Guardar información suficiente para reconstruir una ejecución:

- Configuración completa.
- Semilla.
- Versión de código cuando esté disponible.
- Identificador de splits.
- Identificador de caché espectral.
- Versiones relevantes de dependencias.
- Modelo y parámetros.
- Métricas por época.
- Métricas finales.

No sobrescribir ejecuciones anteriores accidentalmente.

Utilizar archivos locales y opcionalmente TensorBoard.

No introducir servicios de tracking externos inicialmente.

## 17.1. Predictions

El archivo predictions.csv debe permitir investigar errores:

- ID de galaxia.
- Split.
- Label real.
- Probabilidad predicha.
- Clase predicha.
- Ttype.
- nsa_z.
- Otros metadatos relevantes para evaluación.

No incorporar estas variables al modelo solamente por aparecer en el archivo de predicciones.

---

# 18. Reproducibilidad

Implementar una función:

    set_seed(seed)

Controlar, en la medida de lo posible:

- random.
- numpy.
- torch.
- CUDA.
- DataLoader workers.

Documentar las operaciones que no sean completamente deterministas.

Guardar configuraciones resueltas y versiones.

No modificar el conjunto de validación durante la búsqueda de hiperparámetros.

Para comparaciones finales, utilizar varias semillas cuando sea viable.

---

# 19. Testing y validación técnica

Antes de entrenar modelos completos deben pasar pruebas básicas.

## 19.1. Tests de datos

Verificar:

- Asociación correcta ID/FITS.
- Asociación correcta ID/imagen.
- Labels válidos.
- No duplicados indebidos.
- Splits sin solapamientos.
- Espectros con dimensiones válidas.
- Máscaras con shape correcto.
- Metadatos con transformaciones consistentes.
- Ausencia de NaN en tensores de entrada.
- No utilizar información de validation/test para ajustar preprocesadores.

## 19.2. Tests de modelos

Para cada arquitectura:

- Crear batch sintético.
- Ejecutar forward.
- Comprobar shape de salida.
- Comprobar logits finitos.
- Ejecutar backward.
- Confirmar gradientes.
- Comprobar funcionamiento en CPU.
- Comprobar que el modelo ignora modalidades inactivas.

## 19.3. Smoke test

Entrenar sobre una muestra pequeña durante unas pocas épocas.

Validar:

- Lectura de batches.
- Forward.
- Backward.
- Actualización de parámetros.
- Evaluación.
- Guardado de checkpoint.
- Recuperación de checkpoint.
- Generación de métricas.

También realizar una prueba de sobreajuste sobre un conjunto diminuto para detectar fallos evidentes en el pipeline.

No confundir esta prueba con un resultado científico.

---

# 20. Plan de implementación

Desarrollar por fases.

## Fase 0 — Auditoría del repositorio

Antes de programar:

1. Inspeccionar estructura actual.
2. Leer catálogo morfológico.
3. Revisar notebook EDA existente.
4. Inspeccionar FITS representativos.
5. Identificar imágenes RGB y su convención de nombres.
6. Localizar el pipeline de descarga de imágenes.
7. Investigar cómo se definieron FOV y pixscale.
8. Confirmar unidades y marco espectral.
9. Identificar código que se pueda reutilizar.
10. Revisar hardware y dependencias disponibles.

Entregable:

Resumen técnico de hallazgos, decisiones confirmadas y dudas pendientes.

No inventar respuestas cuando falten datos.

## Fase 1 — Infraestructura de datos

Implementar:

- Manifest maestro.
- Validación de correspondencias.
- Splits persistentes.
- Lector FITS.
- Preprocesador espectral.
- Caché de espectros.
- GalaxyDataset.
- DataLoaders.
- Transforms de imágenes.
- Tests de datos.

Entregable:

Poder inspeccionar batches multimodales correctamente alineados.

## Fase 2 — Baselines neuronales

Implementar:

- ImageEncoder ResNet-18.
- SpectrumEncoder CNN 1D.
- ConcatenationFusion.
- BinaryClassifier.
- ModelFactory.

No implementar todavía Transformers o cross-attention.

Entregable:

Forward y backward funcional para E1, E2 y E3.

## Fase 3 — Entrenamiento y evaluación

Implementar:

- Trainer.
- Métricas.
- Early stopping.
- Checkpoints.
- Logging.
- YAML.
- CLI.
- Evaluator.

Realizar smoke tests.

Entregable:

Pipeline completo que pueda entrenar y evaluar E1, E2 y E3.

## Fase 4 — Primera campaña experimental

Entrenar:

- E1: solo RGB.
- E2: solo espectro.
- E3: RGB + espectro.

Usar los mismos splits.

Analizar:

- Rendimiento.
- Curvas de aprendizaje.
- Matrices de confusión.
- Sobreajuste.
- Diferencias por Ttype.
- Diferencias por redshift.

No asumir que la fusión multimodal mejorará necesariamente los resultados.

## Fase 5 — Atención y fusión avanzada

Solamente después de contar con baselines confiables:

- Encoder CNN 1D + attention.
- Gated fusion.
- Cross-attention.

Mantener el mismo pipeline de entrenamiento.

Implementar E4, E5 y E6.

## Fase 6 — Metadatos

Implementar:

- MetadataEncoder.
- Preprocesador tabular.
- Selección de features mediante YAML.
- Fusión trimodal.

Ejecutar E7-E10 y ablaciones.

Analizar especialmente el efecto de LogMass, redshift y pixscale.

---

# 21. Preguntas científicas a responder

La infraestructura debe facilitar investigar:

### H1. Complementariedad multimodal

¿La información espectroscópica mejora la clasificación frente a usar solamente imágenes?

Comparar E1, E2 y E3.

### H2. Atención espectral

¿Modelar relaciones entre diferentes regiones espectrales produce mejores embeddings que una CNN 1D convencional?

Comparar E3 y E4.

### H3. Estrategias de fusión

¿Una fusión adaptativa o con cross-attention mejora la concatenación simple?

Comparar E3, E5 y E6.

### H4. Contexto astrofísico

¿La masa estelar, el redshift y otras propiedades aportan información complementaria?

Comparar E3, E7, E8 y E10.

### H5. Información de escala angular

¿Conocer la escala de la imagen mejora la interpretación morfológica de la CNN?

Comparar E3 y E9.

Controlar especialmente las diferencias de escala introducidas por el pipeline de cutouts.

### H6. Contribución de las aperturas espectrales

¿Utilizar FLUX_1RE y FLUX_FoV conjuntamente aporta ventajas frente a utilizar solamente uno?

Realizar ablaciones con el encoder espectral.

---

# 22. Limitaciones metodológicas importantes

## 22.1. Morfología vs propiedades físicas

Las etiquetas representan clases morfológicas.

Las propiedades espectroscópicas pueden estar correlacionadas con esas clases, pero no son equivalentes a la estructura morfológica.

No interpretar un buen clasificador espectral como una medición directa de morfología.

## 22.2. Variables derivadas del target

No permitir el ingreso de Ttype o Type al clasificador.

Deben existir validaciones que detecten estas columnas si aparecen por error en la configuración de features.

## 22.3. Sesgos del catálogo

Evaluar relaciones entre label y:

- Redshift.
- Masa estelar.
- Calidad espectral.
- Tamaño angular.
- Disponibilidad de imágenes.

Un modelo podría explotar correlaciones de selección en lugar de aprender características generalizables.

## 22.4. Modalidad dominante

Es posible que la red ignore una modalidad cuando una de ellas sea más predictiva.

Investigar mediante ablaciones:

- Imagen solamente.
- Espectro solamente.
- Ambas modalidades.
- Perturbación o eliminación de una modalidad durante evaluación controlada.

Evitar concluir que hay complementariedad únicamente porque la arquitectura tiene dos ramas.

## 22.5. Disponibilidad durante inferencia

Definir claramente qué entradas estarán disponibles para galaxias nuevas.

Si una arquitectura requiere imágenes y FITS, no puede utilizarse directamente en objetos sin espectro.

Como línea de investigación futura se podría estudiar distillation o teacher-student para aprovechar espectros durante entrenamiento y utilizar solo imágenes en inferencia.

Esta idea está fuera del alcance de la implementación inicial.

## 22.6. Insumos derivados de STARLIGHT

- El espectro de STARLIGHT es un modelo de mejor ajuste (combinación de 45 plantillas BC03), no el espectro observado: es más suave, sin el ruido ni las líneas de emisión que la base no reproduce. Un modelo entrenado con él no es directamente comparable con uno entrenado con MPL-11 y no debe presentarse como el mismo experimento.
- Las poblaciones y propiedades derivan del mismo espectro FoV que ya usa la modalidad espectral. Combinarlos como entradas adicionales no aporta información independiente; la mejora, si la hay, puede deberse a una representación más compacta de lo que el espectro ya contiene.
- Edad, metalicidad y fracción de población joven correlacionan con la morfología; es una asociación astrofísica, no una medición de estructura (22.1). Reportar estos experimentos como de contexto físico, separados de la clasificación visual/espectroscópica (11.4).
- Disponibilidad: 10,013 de las 10,014 filas del manifest tienen salida STARLIGHT. Si se usa como cohorte de un experimento, la caché (o el índice de propiedades) debe tener registro para todos los IDs del split; de lo contrario crear y persistir una cohorte alineada, sin eliminar filas silenciosamente.
- El FITS de STARLIGHT depende de `nsa_z` (sección 6.2); regenerarlo si cambia el manifest.

---

# 23. Criterios de aceptación

La primera versión funcional de infraestructura estará completa cuando:

- [ ] Exista un manifest reproducible y validado.
- [ ] Los splits estén guardados y sean reutilizables.
- [ ] Se puedan leer imágenes y FITS correctamente.
- [ ] Exista un preprocesamiento espectral documentado.
- [ ] Se pueda generar y reutilizar la caché espectral.
- [ ] GalaxyDataset funcione en modo image, spectrum y multimodal.
- [ ] Los batches estén correctamente alineados.
- [ ] ResNet-18 y CNN 1D produzcan embeddings configurables.
- [ ] Exista un modelo de fusión por concatenación.
- [ ] El trainer funcione con todos los baselines.
- [ ] Se puedan guardar y recuperar checkpoints.
- [ ] Se registren pérdidas y métricas.
- [ ] Se puedan ejecutar configuraciones YAML desde CLI.
- [ ] Los smoke tests funcionen.
- [ ] Los tests de datos y modelos pasen.
- [ ] Se puedan ejecutar E1, E2 y E3 sin modificar el training loop.
- [ ] Los resultados sean reproducibles dentro de las limitaciones documentadas.

No se exige implementar E4-E10 para considerar completa la primera versión.

---

# 24. Instrucciones de trabajo para el agente Codex

Actúa como ingeniero de machine learning y desarrollador científico.

Debes construir una infraestructura confiable, no simplemente un conjunto de scripts que produzcan resultados.

Reglas de implementación:

1. Inspecciona primero el código existente.
2. Reutiliza componentes correctos del proyecto.
3. No modifiques archivos de datos originales.
4. Trabaja por fases.
5. Prioriza contratos claros entre componentes.
6. No introduzcas complejidad innecesaria.
7. Escribe funciones pequeñas y legibles.
8. Documenta decisiones científicas relevantes.
9. Escribe tests para los contratos críticos.
10. No realices entrenamientos largos sin autorización.
11. No descargues grandes recursos inesperadamente.
12. No asumas disponibilidad de GPU.
13. No introduzcas datos no permitidos al modelo.
14. No utilices test durante desarrollo experimental.
15. Reporta explícitamente incertidumbres y decisiones pendientes.
16. Mantén las configuraciones fáciles de modificar.
17. No avances a arquitecturas complejas hasta validar los baselines.

Antes de implementar una fase:

- Revisa sus dependencias.
- Identifica decisiones pendientes.
- Define sus interfaces.
- Establece criterios de validación.

Después de implementar:

- Ejecuta las pruebas correspondientes.
- Resume los archivos creados o modificados.
- Explica cómo ejecutar el componente.
- Reporta qué funciona y qué falta.
- Indica los siguientes pasos recomendados.

No ocultes errores mediante fallbacks silenciosos.

Si falta una ruta, metadata esencial o una decisión científica que impida proceder correctamente, consúltala o documenta claramente el bloqueo.

---

# 25. Prioridad inmediata

La prioridad actual NO es implementar las diez arquitecturas.

El objetivo inmediato es construir una infraestructura mínima y sólida que permita:

1. Preparar la muestra multimodal.
2. Cargar imágenes RGB y espectros alineados.
3. Construir tres modelos baselines.
4. Entrenarlos con el mismo Trainer.
5. Evaluarlos bajo un protocolo común.
6. Guardar resultados reproducibles.

Los primeros modelos serán:

E1 — Image-only ResNet-18.

E2 — Spectrum-only CNN 1D.

E3 — ResNet-18 + CNN 1D + concatenación + MLP.

Una vez validados, extender el sistema hacia atención espectral, mecanismos avanzados de fusión y metadatos.

---

# 26. Resultado esperado

La infraestructura debe permitir ejecutar:

    python -m scripts.train --config configs/e1_image.yaml

    python -m scripts.train --config configs/e2_spectrum.yaml

    python -m scripts.train --config configs/e3_concat.yaml

Y obtener experimentos comparables, reproducibles y analizables.

La meta final es contar con una plataforma de investigación para responder:

- ¿Qué aprende el modelo a partir de las imágenes?
- ¿Qué aprende a partir de los espectros?
- ¿Qué información es complementaria?
- ¿Mejora la atención la representación espectral?
- ¿Qué método de fusión funciona mejor?
- ¿Qué aportan los metadatos físicos?
- ¿Cómo influye la escala angular de las imágenes?
- ¿En qué tipos de galaxias falla cada arquitectura?

El proyecto debe favorecer la experimentación científica rigurosa sobre la complejidad arquitectónica.
