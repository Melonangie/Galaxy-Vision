"""Rutas, hiperparámetros y constantes de configuración del proyecto.

Ninguna ruta ni umbral debe hardcodearse fuera de este módulo (ver README.md,
sección Convenciones de código).
"""

from pathlib import Path

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Dataset Galaxy Zoo - The Galaxy Challenge (Kaggle), aislado en su propia
# subcarpeta para dejar espacio en data/ a futuros datasets (p. ej. Galaxy
# Zoo DECaLS, ver README.md sección Roadmap).
GALAXY_ZOO_DIR = DATA_DIR / "galaxy_zoo_challenge"
PROCESSED_DATA_DIR = GALAXY_ZOO_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_SOLUTIONS_CSV = GALAXY_ZOO_DIR / "training_solutions_rev1.csv"
TRAIN_IMAGES_DIR = GALAXY_ZOO_DIR / "images_training_rev1"
TEST_IMAGES_DIR = GALAXY_ZOO_DIR / "images_test_rev1"

# Reproducibilidad 
RANDOM_SEED = 42

# Etiquetado binario (elíptica vs. espiral) ---
# Umbral de vote fraction "confiable", igual al usado por Walmsley et al. (2022)
# y consistente con el subconjunto de alto acuerdo de Dieleman et al. (2015).
CLEAN_LABEL_THRESHOLD = 0.8

ELLIPTICAL_LABEL = "elliptical"
SPIRAL_LABEL = "spiral"
AMBIGUOUS_LABEL = "ambigua"

# Codificación numérica de la etiqueta binaria (para métricas de sklearn que
# requieren y numérico, p. ej. roc_auc_score): espiral = clase positiva (1).
LABEL_TO_INT = {ELLIPTICAL_LABEL: 0, SPIRAL_LABEL: 1}

# Splits
TRAIN_FRAC = 0.7 # Dataset entrenamiento 70%
VAL_FRAC = 0.15 # Dataset validacion 15%
TEST_FRAC = 0.15 # Dataset test 15% (No se usa para entrenar ni validar el modelo)

IMAGE_SIZE_RAW = 424  # px, tal como vienen las imágenes originales (RGB)

# --- Baselines de sklearn (fase previa a la CNN/ResNet, ver README.md sección Arquitectura del modelo) ---
# Tamaño al que se reescala cada imagen antes de aplanarla a vector de píxeles.
# 64x64x3 = 12288 features de entrada; suficiente para que un baseline clásico
# capture forma/textura gruesa sin disparar el costo de memoria/cómputo.
FEATURE_IMAGE_SIZE = 64

# Componentes de PCA aplicados sobre los píxeles aplanados (tras escalado) antes
# de cada clasificador clásico: reduce dimensionalidad (12288 -> 150), acelera
# KNN/SVM y mitiga el sobreajuste típico de estos modelos en alta dimensión.
PCA_N_COMPONENTS = 150

# Submuestreo estratificado del set de entrenamiento para mantener tiempos de
# entrenamiento razonables (SVM/KNN escalan mal con el número de muestras).
# Decisión de diseño explícita, documentada aquí y en train_baselines.py.
BASELINE_TRAIN_SAMPLE_SIZE = 6000
BASELINE_VAL_SAMPLE_SIZE = 1500

# --- CNN baseline entrenada desde cero (ver README.md, sección Arquitectura del modelo) ---
# Recorte centrado (elimina fondo sobrante de los 424x424 originales) antes
# de reescalar a CNN_IMAGE_SIZE. 160/424 ~ 38% de la imagen original: deja
# margen para que RandomRotation/RandomAffine no dejen esquinas vacías tras
# el resize final, sin disparar el costo de las transforms por época.
CNN_CROP_SIZE = 160
# Tamaño final de entrada a la CNN. Más pequeño que el 224/256 sugerido en
# README.md, sección Preprocesamiento de imágenes: decisión de cómputo para
# que el baseline entrene en minutos, no horas, en esta fase de validación
# de pipeline.
CNN_IMAGE_SIZE = 96

CNN_BATCH_SIZE = 64
CNN_EPOCHS = 8
CNN_LEARNING_RATE = 1e-3
CNN_DROPOUT = 0.5
# 0 = carga de imágenes en el proceso principal. Evita los problemas de
# DataLoader multi-worker (spawn) en Windows/Jupyter; viable porque el
# dataset cachea las imágenes reescaladas en RAM una sola vez (ver
# src/data/torch_dataset.py), no las relee de disco en cada época.
CNN_NUM_WORKERS = 0

# --- Dataset 2: proyecto integrador MaNGA + Legacy Survey (colaboración con el
# Instituto de Astronomía, UNAM; ver README.md sección Dataset — MaNGA + Legacy
# Survey y archives/email_1.pdf) ---
MORFOLOGIA_CSV = DATA_DIR / "morfologia.csv"
# morfologia.csv tras aplicar los cortes de calidad acordados y con la columna
# `label` temprana/tardía ya derivada (generado por
# src.data.morfologia.build_curated_morfologia; no editar a mano).
MORFOLOGIA_CURATED_CSV = DATA_DIR / "morfologia_curada.csv"

# Etiquetado binario temprana/tardía (early/late-type) a partir de "Ttype",
# criterio comunicado por J. A. Vázquez Mata (UNAM) el 2026-09-11.
TTYPE_EARLY_MAX = 0  # Ttype <= 0 -> temprana
TTYPE_LATE_MIN = 1  # Ttype >= 1 (y <= 10 según el correo) -> tardía

EARLY_TYPE_LABEL = "temprana"
LATE_TYPE_LABEL = "tardia"

# Etiqueta para las galaxias cuyo Ttype cae fuera de los dos rangos declarados
# en el correo (la muestra tiene 67 filas con Ttype == 11): no se les asigna
# clase, quedan marcadas y fuera de los splits. Confirmado en la junta del
# 2026-09-23: Ttype > 10 se ignora (ver README.md, sección Dataset — MaNGA +
# Legacy Survey).
TTYPE_OUT_OF_RANGE_LABEL = "fuera_de_rango"

# Cortes de calidad sobre morfologia.csv acordados en la junta con el IA-UNAM
# del 2026-09-23 (ver README.md, sección Retroalimentación de la junta con el
# IA-UNAM). Todos los rangos son inclusivos; un NaN no cumple ningún rango.
NSA_Z_SENTINEL = -9999.0  # valor centinela de "sin dato" en nsa_z
NSA_Z_MIN = 0.0
NSA_Z_MAX = 0.15
LOGMASS_MIN = 8.0
LOGMASS_MAX = 12.0
# incl == 0.0 probablemente es un error del catálogo, pero se revisa a mano
# contra las imágenes antes de descartarlo: es un corte candidato, no acordado.
INCL_SUSPECT_VALUE = 0.0

# --- Cutouts de imagen: servicio público de legacysurvey.org ---
# Endpoints y parámetros tomados del correo de J. A. Vázquez (archives/email_1.pdf);
# ver README.md, sección Imágenes: legacysurvey.org.
LEGACY_CUTOUT_BASE_URL = "https://www.legacysurvey.org/viewer"
# Variante activa (capa, bandas): la que usan por defecto la descarga, la
# lectura y la máscara cuando no se pide otra explícitamente. Ojo: ls-dr10/grz
# tiñe de morado las galaxias con dec < 32.4° (ver README.md, sección Variantes
# de capa y bandas); se mantiene hasta decidir el cambio con el IA-UNAM.
LEGACY_LAYER = "ls-dr10"  # legacy -> 9
LEGACY_BANDS = "grz"
# Escala de píxel nativa de DECam/Legacy Surveys (arcsec/px); es la que usa el
# ejemplo del correo. Pedir menos arcsec/px no añade información real, solo
# reamplía los píxeles nativos.
LEGACY_NATIVE_PIXSCALE = 0.262

# Formatos soportados por el descargador. "jpg" = composición RGB grz de 8 bits
# ya estirada por el visor (set de trabajo para la CNN); "fits" = cubo (3, H, W)
# float32 con el flujo calibrado de cada banda (set de análisis).
CUTOUT_FORMAT_JPG = "jpg"
CUTOUT_FORMAT_FITS = "fits"
CUTOUT_FORMAT = CUTOUT_FORMAT_JPG

# Lado del cutout en píxeles. 256 coincide con el tamaño de entrada sugerido en
# README.md, sección Preprocesamiento de imágenes, y evita reescalar al entrenar.
CUTOUT_SIZE_PX = 256

# Campo de visión pedido, en múltiplos de PETRO_TH90 (radio que encierra el 90%
# de la luz): FOV = CUTOUT_FOV_FACTOR * PETRO_TH90 arcsec de lado. Con 4.0 la
# galaxia ocupa ~la mitad del encuadre, dejando margen para que las rotaciones
# de augmentation no recorten estructura (ver src/data/augmentation.py).
CUTOUT_FOV_FACTOR = 4.0

# Límites del zoom resultante (arcsec/px). El mínimo evita pedir ampliaciones
# absurdas para galaxias diminutas; el máximo evita encuadres tan abiertos que
# la galaxia quede en un puñado de píxeles.
CUTOUT_PIXSCALE_MIN = 0.1
CUTOUT_PIXSCALE_MAX = 0.6

# Descarga: el servicio es público y gratuito, se consume con moderación.
CUTOUT_TIMEOUT_S = 60
CUTOUT_MAX_RETRIES = 3
CUTOUT_RETRY_BACKOFF_S = 2.0
CUTOUT_PAUSE_S = 0.2  # pausa entre peticiones consecutivas

# Tamaño por defecto de la muestra a descargar (balanceada temprana/tardía).
# 50 galaxias: suficiente para validar encuadre, formato y peso en disco
# sin castigar el servicio público ni el disco (~1.2 MB en .jpg).
CUTOUT_SAMPLE_SIZE = 50

# Estructura en disco: <dir de formato>/<layer>_<bands>/<name>.<ext>, p. ej.
# data/legacy_cutouts/jpg/ls-dr10_grz/manga-8589-12705.jpg (ver
# src.data.legacy_cutouts.cutout_path). Cada variante en su subcarpeta.
CUTOUTS_DIR = DATA_DIR / "legacy_cutouts"
CUTOUTS_JPG_DIR = CUTOUTS_DIR / CUTOUT_FORMAT_JPG
CUTOUTS_FITS_DIR = CUTOUTS_DIR / CUTOUT_FORMAT_FITS
# Índice de lo descargado, una fila por (galaxia, formato, variante):
# coordenadas, zoom aplicado, bytes en disco, estado y, para .jpg, cobertura y
# estadísticas de la máscara. Hace la descarga auditable y reanudable.
CUTOUTS_INDEX_CSV = CUTOUTS_DIR / "cutouts_index.csv"

# Render asinh para convertir un cubo .fits (flujo lineal por banda) en una
# imagen RGB de 8 bits comparable a la que sirve el visor. El estiramiento
# asinh es el estándar para imágenes astronómicas (Lupton et al. 2004): comprime
# el núcleo brillante sin borrar los brazos y el disco externo, que es
# justamente lo que distingue temprana de tardía.
FITS_RGB_ASINH_Q = 8.0
FITS_RGB_ASINH_STRETCH = 0.06

# --- Fotometría sobre los cubos .fits (ver src/data/fits_photometry.py,
# notebook/exploration_fits.ipynb y README.md, sección Exploración de los .fits) ---
# Los píxeles vienen en nanomaggies (AB): mag = 22.5 - 2.5 log10(flujo).
LEGACY_NANOMAGGY_ZEROPOINT = 22.5
# Sigma del recorte iterativo con que se estiman la mediana y el ruido del cielo.
FITS_SKY_SIGMA_CLIP = 3.0
# El cielo se mide fuera de este radio (múltiplos de PETRO_TH90), para que la
# galaxia no infle la mediana. Si fuera queda menos de FITS_SKY_MIN_FRAC de la
# imagen, se usa la imagen completa (el recorte sigma descarta la galaxia).
FITS_SKY_EXCLUDE_FACTOR = 1.5
FITS_SKY_MIN_FRAC = 0.1
# Apertura circular del flujo "total", de los colores y de la curva de
# crecimiento, en múltiplos de PETRO_TH90 (se recorta al borde del cutout).
FITS_APERTURE_FACTOR = 1.5
# Perfil radial de brillo superficial: anillos hasta este múltiplo de PETRO_TH90.
FITS_PROFILE_MAX_FACTOR = 2.0
FITS_PROFILE_N_BINS = 20
# Índice de concentración C = 5 log10(r80 / r20) (Kent 1985; Conselice 2003),
# medido en la banda r: las tempranas concentran más luz en el centro.
CONCENTRATION_INNER_FRAC = 0.2
CONCENTRATION_OUTER_FRAC = 0.8
FITS_CONCENTRATION_BAND = "r"

# --- Normalización de los cubos .fits para la red (ver
# src/data/fits_normalization.py, notebook/normalization_fits.ipynb y
# README.md, sección Normalización de los .fits) ---
# Esquemas de escala: "global" usa las mismas constantes para todo el dataset
# (conserva el brillo relativo entre galaxias); "por_imagen" lleva cada
# galaxia a su propio rango (borra el brillo absoluto).
FITS_NORM_SCHEME_GLOBAL = "global"
FITS_NORM_SCHEME_PER_IMAGE = "por_imagen"
FITS_NORM_SCHEME = FITS_NORM_SCHEME_GLOBAL
# Suavidad del asinh (beta) en múltiplos del ruido del cielo de la intensidad
# media de las bandas: por debajo de ~beta la respuesta es lineal (el ruido no
# se amplifica), por encima es logarítmica (el núcleo no satura).
FITS_NORM_SOFTENING_SIGMA = 3.0
# Flujo de referencia que se mapea a 1: percentil de la intensidad dentro de
# PETRO_TH90 (por imagen) o percentil de esos valores entre galaxias (global).
FITS_NORM_REF_PERCENTILE = 99.5
FITS_NORM_GLOBAL_REF_PERCENTILE = 90.0
# Recorte final de valores normalizados (protege de estrellas saturadas y de
# píxeles de cielo muy negativos).
FITS_NORM_CLIP_MIN = -0.5
FITS_NORM_CLIP_MAX = 3.0

# --- Máscara de color de la galaxia central (limpieza de estrellas) ---
# Ver src/data/galaxy_mask.py y README.md, sección Limpieza de imágenes. Todos
# los tamaños en píxeles asumen cutouts de CUTOUT_SIZE_PX (256) de lado.
# Los cutouts de salida (galaxia central sobre fondo negro) se guardan aquí,
# con las mismas subcarpetas <layer>_<bands>/ que CUTOUTS_JPG_DIR.
CUTOUTS_MASKED_JPG_DIR = CUTOUTS_DIR / "jpg_masked"

# Suavizado gaussiano (px) previo a medir colores: reduce el ruido de fondo
# píxel a píxel, que de otro modo fragmenta la máscara en el halo débil.
GALAXY_MASK_SMOOTH_SIGMA = 1.5
# Radio del disco central, como fracción del lado de la imagen, de donde se
# toma el "tono" de la galaxia. 0.12 (~31 px de radio en 256 px) cae dentro de
# PETRO_TH90 con el encuadre de CUTOUT_FOV_FACTOR = 4 y abarca bulbo + disco
# interno; un radio menor (p. ej. 0.06) solo ve el núcleo saturado y pierde el
# disco azulado de las espirales.
GALAXY_MASK_CENTER_FRAC = 0.12
# Ancho (px) del marco perimetral de donde se toma el color del fondo de cielo.
GALAXY_MASK_BORDER_PX = 12
# Se descarta el 5% más brillante del marco para que las estrellas que caen
# en el borde no contaminen el modelo de color del fondo.
GALAXY_MASK_BG_PERCENTILE = 95
# Regularización (unidades de intensidad^2) sumada a la diagonal de las
# covarianzas de color: evita matrices singulares cuando el núcleo está saturado.
GALAXY_MASK_COV_REG = 4.0
# Iteraciones de apertura morfológica antes de buscar la componente central:
# corta los "puentes" finos que conectan estrellas cercanas con la galaxia.
GALAXY_MASK_OPEN_ITER = 3
# Iteraciones de cierre morfológico final (tapa huecos y suaviza el contorno).
GALAXY_MASK_CLOSE_ITER = 2
# Semilanchura (px) del recuadro central donde se busca la componente conexa
# que corresponde a la galaxia objetivo.
GALAXY_MASK_SEED_HALF = 3
# Margen (px) que se añade alrededor de la máscara final con una dilatación
# circular. Sin margen la galaxia queda "encerrada" justo en su contorno
# (retroalimentación IA-UNAM 2026-09-30, ver README.md); 0 desactiva el margen.
GALAXY_MASK_MARGIN_PX = 6

# --- Tratamientos de imagen alternativos a la máscara dura (ver
# src/data/image_treatments.py, notebook/image-curation.ipynb, sección 7, y
# README.md, sección Limpieza de imágenes) ---
# Variantes en las que se comparan: las cuatro finalistas del linear probe
# (notebook/variant-classification-for-images-only.ipynb).
TREATMENT_VARIANTS = [
    ("ls-dr9", "grz"),
    ("ls-dr10-grz", "grz"),
    ("ls-dr11", "grz"),
    ("ls-dr11", "griz"),
]
# Semillas de las particiones de la validación cruzada: el AUC de 50 galaxias
# cambia ~0.01–0.02 según cómo caigan los folds, así que se promedian varias.
TREATMENT_CV_SEEDS = tuple(range(10))
# Detección de estrellas para `remove_stars`: fuentes con brillo por encima
# de la mediana del cielo + este múltiplo de su desviación robusta (MAD), que
# no tocan la galaxia central. Se dilatan unos píxeles para borrar su halo.
STAR_DETECT_SIGMA = 4.0
STAR_GROW_PX = 3
# Margen (px) de la galaxia central que protege `remove_stars`: lo que toque
# esta zona no se borra (estrellas encima del disco, brazos, vecinas fusionadas).
STAR_PROTECT_MARGIN_PX = GALAXY_MASK_MARGIN_PX
# Apertura circular suave: radio en múltiplos de PETRO_TH90 (con el encuadre de
# CUTOUT_FOV_FACTOR, PETRO_TH90 = CUTOUT_SIZE_PX / CUTOUT_FOV_FACTOR px) y sigma
# (px) de la caída gaussiana hacia el ruido de cielo.
SOFT_APERTURE_R90_FACTOR = 1.0
SOFT_APERTURE_EDGE_SIGMA_PX = 16
# Máscara de color suave: margen extra (px) y sigma (px) de la caída del borde.
SOFT_MASK_MARGIN_PX = 20
SOFT_MASK_EDGE_SIGMA_PX = 8
# Exponente de la corrección gamma (< 1 aclara el disco externo y el halo débil).
TREATMENT_GAMMA = 0.6

# --- Variantes de capa/bandas del visor (experimento de curación) ---
# Valor del argumento --variants que selecciona todas las de LEGACY_VARIANTS.
ALL_VARIANTS = "all"
# Misma muestra descargada con cada combinación (LEGACY_LAYER, LEGACY_BANDS)
# para comparar render, cobertura y comportamiento de la máscara de color (ver
# README.md, sección Variantes de capa y bandas). Solo se listan combinaciones
# que el servicio devuelve distintas entre sí, verificado el 2026-10-07:
# - ls-dr9 ignora `bands` (siempre grz).
# - ls-dr10 sin `bands` == griz; ls-dr11 sin `bands` == grz.
# - ls-dr10-grz ignora `bands` (grz con otro balance de color que ls-dr10/grz).
# - ls-dr11-grz devuelve una imagen gris vacía y ls-dr11-early un error 500,
#   por eso no se incluyen.
LEGACY_VARIANTS = [
    ("ls-dr9", "grz"),
    ("ls-dr10", "grz"),
    ("ls-dr10", "griz"),
    ("ls-dr10", "gri"),
    ("ls-dr10-grz", "grz"),
    ("ls-dr11", "grz"),
    ("ls-dr11", "griz"),
    ("ls-dr11", "gri"),
]

# Una zona sin cobertura del survey llega como color plano (varianza local ~0);
# el cielo real siempre tiene ruido. Ventana (px) y umbral de varianza
# (intensidad^2) con que se detectan esos píxeles "planos".
NO_COVERAGE_WINDOW_PX = 5
NO_COVERAGE_VAR_MAX = 0.5
# Un núcleo saturado también es plano (canal a 255); los píxeles con algún
# canal >= este valor no cuentan como "sin cobertura".
NO_COVERAGE_SATURATION = 250

# --- Filtro de calidad y atajos por variante (ver README.md, sección Elegir la
# variante de imágenes; notebook/image-curation.ipynb y
# notebook/variant-classification-for-images-only.ipynb) ---
# Codificación numérica temprana/tardía para métricas de sklearn: tardía =
# clase positiva (1), análogo a LABEL_TO_INT de la Fase 1.
MORFOLOGIA_LABEL_TO_INT = {EARLY_TYPE_LABEL: 0, LATE_TYPE_LABEL: 1}

# Frontera norte/sur de Legacy Surveys (DR9 en adelante): en el casquete
# galáctico norte, dec >= 32.375° se observó con BASS/MzLS ("norte") y el resto
# con DECam ("sur"); todo el casquete galáctico sur es DECam. Cada región tiene
# otro telescopio, otra profundidad y, en algunas capas, otro render de color.
LEGACY_NORTH_DEC_MIN = 32.375
LEGACY_REGION_NORTH = "norte"
LEGACY_REGION_SOUTH = "sur"

# Pruebas de "¿esta representación predice este atributo?" (fuga norte/sur,
# atajo de la máscara, clasificador temprana/tardía): AUC con validación
# cruzada estratificada repetida y p-valor por permutación de etiquetas.
SHORTCUT_CV_FOLDS = 5
SHORTCUT_CV_REPEATS = 10
SHORTCUT_N_PERMUTATIONS = 200
SHORTCUT_P_VALUE_MAX = 0.05
# AUC a partir de la cual (con p < SHORTCUT_P_VALUE_MAX) una representación
# "delata" el atributo: 0.5 = azar, 1.0 = lo predice perfecto.
SHORTCUT_AUC_ALERT = 0.70
# Regularización (C de LogisticRegression) para features tabulares de pocas
# columnas (colores, forma de la máscara).
SHORTCUT_LOGREG_C = 1.0

# Silueta de la máscara reducida a SxS px para el atajo de la máscara: conserva
# forma y tamaño del contorno sin nada de la galaxia.
MASK_SILHOUETTE_SIZE = 32

# Linear probe: backbone preentrenado en ImageNet y congelado (features de 512
# dimensiones) + regresión logística. Mide qué tan separable es cada variante
# sin que el ajuste de hiperparámetros de una CNN sesgue la comparación.
PROBE_BACKBONE = "resnet34"
PROBE_IMAGE_SIZE = 224
PROBE_BATCH_SIZE = 64
# Regularización fuerte: con cientos de features y pocas decenas de galaxias
# por fold, una logística sin regularizar memoriza el set de entrenamiento.
PROBE_LOGREG_C = 0.05
# Remuestreos bootstrap para los intervalos de confianza del AUC.
PROBE_BOOTSTRAP_N = 2000
# Variante de referencia contra la que se comparan las demás (la actual).
PROBE_REFERENCE_VARIANT = "ls-dr10_grz"

# --- Fusión imagen + .fits (ver src/models/fusion_probe.py,
# notebook/variant-classification-images-and-fits.ipynb y README.md, sección
# Imágenes + .fits: ¿aporta el .fits al clasificador?) ---
# Variantes (capa, bandas) en las que se compara el .jpg con el .fits de la
# misma capa y bandas, para que la única diferencia entre ambos sea el formato:
# las cuatro finalistas de notebook/variant-classification-for-images-only.ipynb
# y, al final, la variante activa (la de la primera corrida de esta comparación).
FUSION_VARIANTS = [
    ("ls-dr9", "grz"),
    ("ls-dr10-grz", "grz"),
    ("ls-dr11", "grz"),
    ("ls-dr11", "griz"),
    (LEGACY_LAYER, LEGACY_BANDS),
]
# Fotometría del .fits que entra como bloque tabular: colores y concentración.
# Sin magnitudes ni radios en arcsec, que dependen de la distancia y la masa.
FUSION_PHOTOMETRY_FEATURES = ("g_r", "r_z", "concentracion")
# Bandas del .fits que forman los canales R, G y B que recibe la red de
# ImageNet, por juego de bandas; un canal con varias bandas es su promedio
# (ya estandarizadas). Con grz es el mapeo del visor (R = z, G = r, B = g);
# con griz, i y z (las dos más rojas) se promedian en R para usar las 4 bandas
# con una red de 3 canales. Ojo: en el .fits de ls-dr11_griz la banda i viene
# en ceros en todo el norte (y en algunas galaxias del sur), así que ese canal
# cambia con el telescopio (ver README.md, sección Imágenes + .fits: ¿aporta el
# .fits al clasificador?).
FUSION_FITS_RGB_CHANNELS = {
    "grz": ("z", "r", "g"),
    "griz": ("iz", "r", "g"),
}
# Representaciones que se comparan: nombre -> bloques de features que se
# concatenan ("imagen" = ResNet sobre el .jpg, "fits" = ResNet sobre el .fits
# normalizado, "fotometria" = FUSION_PHOTOMETRY_FEATURES).
FUSION_REPRESENTATIONS = {
    "imagen": ("imagen",),
    "fits": ("fits",),
    "fotometria": ("fotometria",),
    "imagen + fits": ("imagen", "fits"),
    "imagen + fotometria": ("imagen", "fotometria"),
    "imagen + fits + fotometria": ("imagen", "fits", "fotometria"),
}
# Representación contra la que se comparan las demás (bootstrap pareado).
FUSION_REFERENCE = "imagen"

# Homogeneidad de color: una variante no pasa el filtro si la distancia entre
# la cromaticidad mediana del norte y la del sur (cielo o galaxia) supera este
# múltiplo de la mediana de esa distancia entre todas las variantes. La
# mediana recoge la diferencia física entre telescopios, presente en todas.
COLOR_GAP_MAX_FACTOR = 2.0

# --- Atajos en el catálogo completo (ver src/data/shortcuts.py,
# notebook/shortcuts.ipynb y README.md, sección Atajos: qué podría aprender el
# modelo en lugar de la morfología) ---
# Cosmología para pasar de arcsec a kpc (nombre de un objeto de astropy.cosmology).
COSMOLOGY = "Planck18"
# Bins por cuantiles de cada variable con que se reponderan las clases para
# que tengan la misma distribución (p. ej. AUC del redshift a masa fija).
SHORTCUT_MATCH_N_BINS = 10
# Dos entradas del catálogo a menos de esta distancia son la misma galaxia
# observada con otro plate-ifu (MaNGA tiene repeticiones).
DUPLICATE_MAX_SEP_ARCSEC = 3.0

# Autor: Alexander Carreño Correa (alexandercorreacorrea1997@gmail.com)

# Parámetros para la estimación del campo de visión (FOV) y la cobertura angular.
ESTIMATION_FOVS = [8, 7, 6, 5, 4, 3, 2, 1]
TARGET_RADIUS_PX = 75
RING_HALF_WIDTH_PX = 4
N_SECTORS = 72
MIN_ANGULAR_COVERAGE = 0.25
CENTER_RADIUS_PX = 20