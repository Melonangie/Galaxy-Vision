# Análisis de los experimentos de ablación (imágenes, espectros, poblaciones STARLIGHT)

Clasificación binaria `temprana (0)` vs `tardia (1)` en galaxias MaNGA. Este documento resume qué se ha ejecutado, qué muestran las métricas y las curvas de entrenamiento, cuánto overfitting hay y qué decisiones sugiere para la siguiente fase.

Fuentes: los notebooks de esta carpeta y los artefactos en `models/experiments/<experimento>/` (`history.csv`, `results.json`, `config.json`). Las curvas están en [figuras/curvas_entrenamiento.png](figuras/curvas_entrenamiento.png).

Estructura: §1–§7 cubren la **ablación** (imagen / espectro / poblaciones y sus fusiones); §8 resume el **estudio de complementariedad** ([analisis_complementariedad_multimodal.ipynb](../../analisis_complementariedad_multimodal.ipynb), solo analiza predicciones, no entrena); §9 documenta la **optimización del experto espectral** (RNN/atención), que *no* forma parte de la ablación pero se acumula aquí; §10 resume el **diagnóstico de confiabilidad y selección de modalidad** (primera fase del *gate*, [diagnostico_gate_multimodal.ipynb](../../diagnostico_gate_multimodal.ipynb), solo `validation`, entrena únicamente selectores tabulares pequeños y árboles); §11 documenta la **fusión por tokens con attention pooling** (M0/M1/M2, encoders preentrenados congelados) y la interpreta como complemento de §8; §12 documenta el **estudio de preprocesamiento de imagen** (escala de grises, gamma, normalización percentilar y NLM) sobre la CNN pequeña.

---

## 1. Resumen ejecutivo

1. **Las imágenes RGB dominan.** Cualquier modelo que incluya imagen llega a **AUC test ≈ 0.962–0.965** y accuracy ≈ 0.89–0.90. Sin imagen, el techo es **AUC ≈ 0.86 (espectro) y ≈ 0.90 (poblaciones)**.
2. **Añadir espectros o poblaciones a la imagen no mejora de forma demostrable.** Las diferencias entre las 6 configuraciones visuales (0.962–0.965) están dentro del ruido (error estándar del AUC en test ≈ ±0.005).
3. **Dos arquitecturas visuales sencillas ya son líneas base fuertes.** La CNN pequeña (AUC test 0.9623) y el control ResNet-18 (0.9633) empatan prácticamente. Extender la CNN a 46 épocas no mejoró su AUC de test ni su mejor AUC de validación.
4. **Las poblaciones STARLIGHT (90 números) superan al espectro con la CNN 1D** (0.895 vs 0.857–0.864) aunque derivan del mismo espectro. Esto indica que la CNN 1D actual **no está extrayendo bien la información espectral**; no que el espectro carezca de ella.
5. **ResNet-18 sobreajusta incluso con imagen sola** (loss final train/val: 0.159/0.317). Es el mismo patrón de sus fusiones, por lo que no se debe atribuir el sobreajuste a añadir modalidades. La CNN compacta solo muestra sobreajuste leve al final (0.237/0.287). Los modelos sin imagen **subajustan** (train ≈ val, plateau).
6. El control **ResNet-18 solo imagen ya cierra la ablación arquitectónica**: usa el mismo split FoV, tamaño, regularización, LR y selección que las fusiones. Queda la limitación central: **una sola semilla y un solo split**.
7. **Complementariedad (§8).** Hay complementariedad *potencial* (el oráculo RGB∪poblaciones sube la accuracy de 0.888 a 0.946; poblaciones corrige el 51.6 % de los errores RGB) pero **no aprovechable con las técnicas probadas**: la fusión tardía con α fijado en validación da ΔAUC = +0.0005 [−0.0012, +0.0022] en test. Escenario **B**: hay información distinta, falta saber *cuándo* confiar en cada modalidad.
8. **Los errores se concentran en la frontera morfológica** (Ttype 0–1: 16 % de las galaxias, 48 % de los errores RGB; error 33 % vs 7 % en el resto). Ninguna propiedad STARLIGHT distingue significativamente los grupos de error tras Bonferroni.
9. **El espectro sí tenía información que la CNN 1D no extraía (§9).** Con LR 3e-4, *gradient clipping* y encoders más capaces el AUC test sube de 0.857–0.864 a **0.90–0.917** (FoV: 0.901 atención, 0.912 CNN+BiGRU, 0.917 patch-BiGRU; STARLIGHT: 0.907 / 0.906 / 0.900). El **control con atención sin recurrencia** ya captura casi toda la mejora: la recurrencia aporta como mucho +0.016 (FoV) y nada en STARLIGHT. Esto matiza la jerarquía de §3: el espectro bien modelado **iguala o supera a las poblaciones** (0.895).
10. **Diagnóstico del gate (§10).** Entre las 257 galaxias de `validation` donde RGB y espectro (atención, STARLIGHT) discrepan, el espectro tiene razón en el 33.1 %. La señal para anticipar cuál acierta existe pero es **modesta y reside en la confianza** (`2·|p−0.5|`), no en el valor lineal de las probabilidades (selector con confianzas: AUC fuera de fold 0.676 [0.604, 0.738]); calidad STARLIGHT, observacionales y propiedades físicas no añaden señal detectable. **No se traduce en ganancia neta demostrada frente a RGB sola** (+5 galaxias [−9, +18] de 1403 en el mejor caso; el oráculo daría +85) y los árboles/XGBoost **no** aprovechan estructura no lineal adicional (empeoran por sobreajuste). Veredicto: señal sin ganancia neta; no justifica todavía un gate adaptativo.
11. **Fusión por tokens (§11).** Tres variantes con ResNet-18 y patch-BiGRU (FoV) preentrenados y **congelados** (M0 mean, M1 attention independiente, M2 attention conjunto) alcanzan **AUC test 0.9660 / 0.9661 / 0.9666** y accuracy 0.899–0.902, frente a 0.9633 (ResNet-18) y 0.9635 (concat previa). Es la primera fusión con una mejora pequeña y consistente sobre RGB (ΔAUC ≈ +0.003, IC95 % pareado cerca de excluir 0; McNemar p = 0.011–0.023), pero **las tres variantes son indistinguibles entre sí** (M2−M0 = +0.0006 [−0.0009, +0.0022]) y **no superan de forma demostrable a la concat previa** (IC incluye 0). Como M0 (sin atención) ya obtiene casi toda la mejora, el factor probable es usar encoders fuertes preentrenados con cabeza nueva, **no el attention pooling**. Aprovecha ≈ 18–23 % del margen del oráculo RGB∪espectro (+15 a +19 galaxias netas de +82 posibles).
12. **Preprocesamiento de imagen (§12).** Cuatro transformaciones globales sobre la CNN pequeña (base: AUC test 0.9623, acc 0.9009): percentil 0.9651 (acc 0.9059), escala de grises 0.9657 (0.8967), gamma 0.9627 (0.8874) y NLM 0.9553 (0.8952). **Ninguna mejora de forma demostrable**: los ΔAUC pareados incluyen 0 salvo NLM, que *empeora* (−0.007 [−0.012, −0.002]). Hay un indicio débil a favor de percentil y grises (+0.003) que requiere semillas; gamma solo desplaza el umbral y NLM no ayuda.

---

## 2. Qué se ha ejecutado

Todas las corridas comparten cohorte y split: `splits_seed42_obj-label*.csv` (train 6545 / val 1403 / test 1403; 4078 tardías en train, 875 en test). Se verificó que el split de imagen (`splits_seed42_obj-label.csv`) y el `_FoV` asignan **exactamente las mismas galaxias a cada partición**, por lo que **todas las métricas de test son directamente comparables**. Clase mayoritaria = tardía (62.3 % → accuracy trivial 0.623, AUC 0.5).

| Experimento | Insumo | Modelo | LR | Épocas máx. | Épocas reales | s/época |
|---|---|---|---|---|---|---|
| `experimento_binario_imagenes_cnn` | RGB `fov_0` 256² | CNN 4 bloques (32→256), desde cero | 3e-4 | 30 | 30 | 715 (CPU) |
| `experimento_binario_imagenes_cnn_gpu` | ídem | ídem (repetición en GPU, extendida) | 3e-4 | 50 | 46 (early stop) | 27 |
| `imagenes_resnet18_fov` | RGB `fov_0` 256² | ResNet-18 aleatorio + proyección/MLP | 1e-4 | 50 | 32 (early stop) | 18 |
| `espectros_fov_cnn1d` | espectro MPL-11 `FLUX_FoV` | CNN 1D con pooling enmascarado (emb. 128) | 1e-4 | 50 | 50 | 3 |
| `espectros_starlight_cnn1d` | espectro modelado STARLIGHT (3600–8900 Å) | ídem | 1e-4 | 50 | 50 | 3 |
| `poblaciones_starlight_mlp` | luz + masa actual por población (90 valores, renormalizados) | MLP (128) | 1e-4 | 50 | 50 | 1 |
| `multimodal_espectros_poblaciones_starlight_concat` | espectro SL + poblaciones | concat | 1e-4 | 50 | 50 | 4 |
| `multimodal_imagenes_espectros_fov_concat` | imagen + espectro FoV | ResNet-18 + CNN1D, concat | 1e-4 | 50 | 21 (early stop) | 19 |
| `multimodal_imagenes_espectros_starlight_concat` | imagen + espectro SL | ídem | 1e-4 | 50 | 21 (early stop) | 19 |
| `multimodal_imagenes_poblaciones_starlight_concat` | imagen + poblaciones | ResNet-18 + MLP | 1e-4 | 50 | 33 (early stop) | 18 |
| `multimodal_imagenes_espectros_poblaciones_starlight_concat` | imagen + espectro SL + poblaciones | 3 ramas | 1e-4 | 50 | 27 (early stop) | 19 |

Protocolo común: AdamW (wd 1e-4), BCE con logit, early stopping (paciencia 8) y selección del checkpoint por **ROC-AUC de validación**; test se evalúa una vez con el checkpoint elegido y umbral 0.5.

> El control `imagenes_resnet18_fov` hace comparable la rama visual: el mismo ResNet-18 aleatorio, proyección a 128, clasificador, dropout, LR 1e-4 y split `_FoV` que usan las fusiones.

---

## 3. Resultados en test

Ordenado por AUC. `Rec. T` = recall de temprana, `Rec. L` = recall de tardía (en la tabla: `T`=temprana, `L`=tardía).

| Experimento | AUC | Acc. | F1 macro | Rec. T | Rec. L | Pérdida |
|---|---|---|---|---|---|---|
| Img + Esp(SL) + Pob | **0.9650** | 0.8895 | 0.884 | 0.883 | 0.894 | 0.244 |
| Img + Pob | 0.9639 | 0.8988 | 0.892 | 0.869 | 0.917 | 0.255 |
| Img CNN (CPU) | 0.9634 | **0.9009** | 0.894 | 0.867 | 0.921 | 0.241 |
| Img ResNet-18 (control) | 0.9633 | 0.8881 | 0.881 | 0.860 | 0.905 | 0.248 |
| Img + Esp(FoV) | 0.9635 | 0.8874 | 0.879 | 0.837 | 0.918 | 0.237 |
| Img + Esp(SL) | 0.9632 | 0.8931 | 0.887 | 0.890 | 0.895 | 0.239 |
| Img CNN (GPU, 50 ép. configuradas) | 0.9623 | **0.9009** | **0.895** | 0.892 | 0.906 | 0.242 |
| Esp(SL) + Pob | 0.8960 | 0.8346 | 0.825 | 0.799 | 0.856 | 0.388 |
| Poblaciones (MLP) | 0.8949 | 0.8375 | 0.828 | 0.799 | 0.861 | 0.389 |
| Espectro SL (CNN1D) | 0.8642 | 0.7890 | 0.767 | 0.636 | 0.881 | 0.451 |
| Espectro FoV (CNN1D) | 0.8572 | 0.7947 | 0.777 | 0.682 | 0.863 | 0.457 |

Matrices de confusión (filas = real T/L; columnas = predicho T/L): Img CNN extendida `[[471,57],[82,793]]`; Img ResNet-18 `[[454,74],[83,792]]`; Img+Esp+Pob `[[466,62],[93,782]]`; Poblaciones `[[422,106],[122,753]]`; Espectro SL `[[336,192],[104,771]]`.

### Lecturas importantes

- **Tres escalones nítidos**: espectro solo (~0.86) < poblaciones (~0.90) < imagen (~0.96). Los saltos (≥0.03) son varias veces el error estándar; las diferencias dentro de cada escalón no lo son.
- **Control ResNet-18.** ResNet-18 solo imagen da AUC 0.9633, indistinguible de sus fusiones: Img+Esp(FoV) 0.9635, Img+Esp(SL) 0.9632, Img+Pob 0.9639 e Img+Esp+Pob 0.9650. Por tanto, ninguna fusión ha superado de forma demostrable a su propia rama visual.
- **Ruido de referencia.** El error estándar del AUC en test con n=1403 es ≈0.005 en AUC 0.96, ≈0.008 en 0.90 y ≈0.009 en 0.86 (aprox. Hanley–McNeil). La diferencia máxima ResNet solo vs fusión es 0.0017: **las diferencias de este orden no son interpretables** con una semilla.
- **El umbral 0.5 distorsiona la accuracy.** La corrida GPU tiene recall T 0.938 / L 0.842 y la CPU 0.867 / 0.921 con AUC casi igual: son el mismo ranking con distinto punto de operación. Para comparar modelos conviene **AUC, balanced accuracy y/o umbral optimizado en validación**, no accuracy a 0.5.
- **Los modelos sin imagen sesgan hacia la clase mayoritaria** (espectro: recall T 0.64–0.68). Las tempranas son la clase difícil para el espectro con la CNN actual.
- **Espectro modelado STARLIGHT vs observado FoV**: +0.007 AUC test (+0.007 en val), en el límite del ruido. Es plausible que el espectro modelado, sin ruido, ayude un poco, pero no está demostrado. Además no son insumos independientes (mismo ajuste).
- **Poblaciones + espectro ≈ poblaciones solas** (0.896 vs 0.895): el espectro no añade nada sobre las poblaciones, coherente con que ambas salen del mismo ajuste.

---

## 4. Evolución del entrenamiento y diagnóstico de overfitting

Criterios usados: brecha final `val_loss − train_loss`; época de mejor AUC y de mínima val_loss; si la val_loss sube mientras train_loss baja; si el modelo aún mejoraba al terminar.

| Experimento | Mejor ép. (AUC) | Train / Val loss en mejor ép. | Train / Val al final | Brecha final | Diagnóstico |
|---|---|---|---|---|---|
| Img CNN (CPU, histórico) | 30 / 30 | 0.257 / 0.251 | 0.257 / 0.251 | −0.006 | Sin overfitting en esa corrida |
| Img CNN (GPU, extendida) | 38 / 46 | 0.246 / 0.254 | 0.237 / 0.287 | 0.050 | Sobreajuste leve después de la mejor época |
| Img ResNet-18 (control) | 24 / 32 | 0.191 / 0.243 | 0.159 / 0.317 | 0.158 | **Overfitting** desde ~época 24 |
| Espectro FoV | 50 / 50 | 0.461 / 0.475 | 0.461 / 0.475 | 0.014 | **Subajuste** / plateau |
| Espectro SL | 48 / 50 | 0.458 / 0.473 | 0.453 / 0.470 | 0.016 | **Subajuste** / plateau |
| Poblaciones MLP | 50 / 50 | 0.382 / 0.403 | 0.382 / 0.403 | 0.021 | Subajuste leve, convergencia lenta |
| Esp + Pob | 49 / 50 | 0.378 / 0.402 | 0.374 / 0.401 | 0.027 | Subajuste leve |
| Img + Esp(FoV) | 13 / 21 | 0.225 / 0.237 | 0.194 / 0.385 | **0.191** | **Overfitting** desde ép. ~13 |
| Img + Esp(SL) | 13 / 21 | 0.229 / 0.236 | 0.196 / 0.291 | 0.095 | **Overfitting** desde ép. ~13 |
| Img + Pob | 25 / 33 | 0.176 / 0.261 | 0.139 / 0.319 | 0.180 | **Overfitting** progresivo |
| Img + Esp + Pob | 19 / 27 | 0.204 / 0.242 | 0.169 / 0.292 | 0.123 | **Overfitting** desde ép. ~12–19 |

### Patrones

- **Imagen sola (CNN pequeña):** al extender a 50 épocas configuradas, el entrenamiento se detuvo por paciencia en la época 46; el mejor AUC de validación fue 0.9587 en época 38 y el test fue 0.9623. La pérdida de validación subió de 0.254 (época elegida) a 0.287 mientras train siguió bajando a 0.237: **las épocas adicionales no ayudaron y revelan sobreajuste leve**, no falta de convergencia. La val_loss sigue siendo irregular, consistente con BatchNorm, batch 32 y LR fijo sin scheduler.
- **Espectros:** las primeras 6–7 épocas el modelo predice la clase mayoritaria (balanced acc 0.50, AUC 0.54–0.67), luego "despega" en la época 7–8 y se estanca con train≈val. Señales de un problema de **optimización/normalización de entrada** (LR bajo, escala del flujo, inicialización) y de **capacidad/representación insuficiente**, no de sobreajuste. AUC val sigue subiendo ~0.0001–0.0003 por época al final de las 50 épocas: más épocas ayudarían muy poco.
- **Poblaciones:** convergencia suave y estable; el AUC val aún sube lentamente en la época 50 (0.8899) con brecha pequeña. Es un problema "fácil" para un MLP de 90 entradas; probablemente el techo es la información (ver §5), no el modelo.
- **ResNet-18 solo y fusiones:** el control ResNet-18 muestra train/val 0.159/0.317 al final, casi el mismo patrón que las fusiones (0.139–0.196 / 0.291–0.385). El **early stopping funciona**: toma el checkpoint anterior al rebote y todos obtienen AUC test ~0.963–0.965. Pero ResNet-18 aleatorio, sin augmentations, está sobreparametrizado para 6545 galaxias. La fusión no empeora de manera detectable este patrón ni extrae una ganancia medible.
- **Val vs test:** el AUC en test es igual o ligeramente mayor que en val en todos los casos (p. ej. espectro SL 0.848 → 0.864; pob 0.890 → 0.895). No hay evidencia de sobreajuste de selección sobre validación ni de fuga de datos; las diferencias están dentro del ruido.

---

## 5. Qué aprendemos

### 5.1 ¿Puede una arquitectura sencilla aprender la tarea?
**Sí, con imágenes.** Tanto una CNN de ~1.2 M de parámetros como un ResNet-18 inicializado desde cero logran AUC ≈0.963 y ~89–90 % de accuracy. La CNN compacta es el baseline más eficiente (27 s/época), y ambos requieren regularización/augmentations si se entrenan más allá de su punto óptimo.

### 5.2 ¿Qué insumo aporta más?
Imagen ≫ poblaciones > espectro. Esto es esperable: la etiqueta es morfológica (Hubble T-type), y la forma/estructura visible (bulbo, disco, brazos) está directamente en la imagen. Espectros y poblaciones solo capturan la morfología **por correlación astrofísica** (edad, formación estelar, masa).

### 5.3 ¿Qué combinación es mejor?
Con la evidencia actual, **no hay combinación demostrablemente mejor que la imagen sola**. El control ResNet-18 da 0.9633 y la mejor fusión (Img+Esp+Pob) 0.9650: +0.0017, muy por debajo del error estándar. Img+Pob (+0.0007) y las fusiones espectrales (−0.0001 a +0.0002) también empatan al control. Hay una hipótesis razonable de que poblaciones aporten información distinta a la imagen, pero el experimento actual no la confirma (§7).

### 5.4 ¿Qué datos conviene conservar?
- **Imágenes: imprescindible.**
- **Poblaciones STARLIGHT: conservar.** Con solo 90 números dan AUC 0.895, rinden por encima de lo esperable del espectro con una CNN 1D, son baratas (1 s/época) e interpretables (luz/masa por edad y metalicidad). Son la candidata más prometedora para complementar a la imagen.
- **Espectro (observado o modelado) con la CNN 1D actual: baja prioridad.** No aporta sobre poblaciones ni sobre imagen. Solo tiene sentido retomarlo con una arquitectura/preprocesamiento mejor (ver §6), porque la información espectral relevante probablemente existe (las poblaciones, derivadas de ella, rinden 0.03 más).
- **Espectro FoV vs STARLIGHT modelado:** sin diferencia concluyente; preferir el observado si se quiere independencia entre insumos (el modelado y las poblaciones salen del mismo ajuste).

### 5.5 Advertencias metodológicas
- **Poblaciones = contexto físico, no estructura**: pueden correlacionar con la etiqueta por masa/edad (la guía lo marca en 11.4 y 22.6). Un AUC de 0.89 con 90 números sugiere que gran parte de la dicotomía temprana/tardía es "color/edad/masa". Conviene saber **cuánto de ese 0.89 lo explica solo `LogMass`** (u otros metadatos) para no sobreinterpretar.
- **Una semilla, un split**, sin intervalos de confianza → no se pueden ordenar los modelos con imagen.
- **Una semilla no cuantifica la variación de entrenamiento**: el control ResNet-18 resuelve la comparación arquitectónica, pero faltan repeticiones para separar cambios reales de la estocasticidad.
- **El AUC de validación elige el checkpoint y es ruidoso** en las corridas con imagen (val_loss oscila ±0.1); el modelo seleccionado puede variar de una repetición a otra.
- Metadatos (`LogMass`, `incl`, `nsa_z`, `PETRO_TH90`) aún no se han usado como insumo ni como baseline.

---

## 6. Diagnóstico por rama (dónde está el margen)

| Rama | Estado | Palanca principal |
|---|---|---|
| Imagen (CNN pequeña) | Sobreajuste leve después de época 38 | Augmentations, scheduler (cosine / ReduceLROnPlateau), EMA/SWA para estabilizar val; no simplemente más épocas |
| Imagen (ResNet-18, solo o en fusión) | Overfitting | Augmentations, mayor weight decay/dropout, congelar o reducir ramas, preentrenamiento (ImageNet o galáxicos) |
| Espectro (CNN1D) | Subajuste + arranque lento | Normalización por espectro (flujo mediano), enmascarar líneas de cielo, LR mayor/warm-up, más profundidad/campo receptivo, features de líneas (Hα, Dn4000, Hδ_A, Mg b) como baseline interpretable |
| Poblaciones (MLP) | Convergencia lenta, posiblemente limitado por información | Agrupar en bins edad × metalicidad, probar modelos de árbol (GBM) como referencia, regularizar |
| Fusión (concat) | Rama de imagen domina y memoriza | Dropout por modalidad, fusión tardía de logits, pérdidas auxiliares por rama |

---

## 7. Próximos pasos propuestos (priorizados)

### A. Cerrar los huecos de la ablación actual (bajo costo, alto valor)
1. **Repetir con 3–5 semillas** al menos ResNet-18 solo, Img+Pob e Img+Esp+Pob; reportar media ± desvío de AUC/balanced accuracy. Es la única manera de saber si 0.9650 vs 0.9633 significa algo. Complementar con **bootstrap pareado de test** y test de DeLong/McNemar sobre predicciones guardadas.
2. **Metadatos como baseline**: regresión logística / GBM con `LogMass`, `nsa_z`, `incl`, `PETRO_TH90` (y combinaciones con poblaciones). Responde cuánto de 0.895 de poblaciones es solo masa.
3. **Análisis de errores**: comparar predicciones de ResNet-18, CNN y poblaciones sobre test. Si los errores son distintos (poca correlación), un **ensamble por promedio de probabilidades** (late fusion) debería mejorar y justificaría fusiones más complejas; si coinciden, las poblaciones no aportan. Revisar también el error en función de `Ttype`, `LogMass`, `incl` y `nsa_z`.
4. **Reportar métricas independientes del umbral** (AUC, AUPRC, balanced accuracy, y umbral elegido en validación) y calibración (ECE).

### B. Mejorar el baseline de imagen (donde está la señal)
5. Introducir augmentations y scheduler en CNN y ResNet-18; comparar contra sus baselines actuales. La extensión sin estos cambios ya mostró que más épocas por sí solas no mejora la CNN.
7. Comparar **FoV** (`+3…−3`, ya descargados) con la CNN pequeña: ¿el contexto/zoom cambia la separación? Es un experimento de insumo barato y ya soportado.
8. Probar **transfer learning** (ResNet/EfficientNet preentrenados) y comparar contra la CNN desde cero.

### C. Decidir sobre espectros
9. Revisar normalización y estabilidad de entrenamiento de la CNN 1D (arranque con 6–7 épocas en la clase mayoritaria). Probar LR 3e-4–1e-3, normalización por flujo mediano, y un baseline con **índices espectrales** (Dn4000, Hδ_A, Hα EW) + GBM. Si con eso el espectro llega a ≥0.90, tiene sentido volver a fusionarlo; si no, se descarta frente a poblaciones.

### D. Fusión (solo si A.2–A.4 muestran complementariedad)
10. Fusión tardía (promedio/stacking de logits) antes que concat; luego dropout por modalidad y atención cruzada/gating si se justifica. Evaluar si Img+Pob mantiene su ventaja con regularización.

> **Actualización:** A.3 (análisis de errores) está hecho en §8 y C.9 parcialmente en §9: el espectro mejora a 0.90–0.917 con LR 3e-4, atención/BiGRU y clipping, por lo que ya **no** está "en pausa" como experto, aunque su aporte a la fusión sigue sin demostrarse. El gating/incertidumbre de §8.2 se exploró en §10: hay señal modesta en la confianza pero sin ganancia neta demostrada, por lo que el siguiente paso pasa a ser **obtener predicciones OOF de los encoders** antes de entrenar un gate.

### Decisión sugerida
**Seguir con imagen como insumo central y poblaciones STARLIGHT como complemento candidato; poner en pausa el espectro en bruto hasta mejorar su rama.** El control ResNet-18 confirma que las fusiones aún no ganan a la imagen. Los esfuerzos inmediatos: (1) semillas/intervalos de confianza de ResNet-18 y las dos fusiones candidatas, (2) baseline de metadatos y análisis de complementariedad de errores, (3) augmentations/scheduler para imagen y comparación de FoV. Solo después tiene sentido invertir en arquitecturas más complejas.

---

## 8. Complementariedad multimodal (análisis de predicciones)

Fuente: [analisis_complementariedad_multimodal.ipynb](../../analisis_complementariedad_multimodal.ipynb); salidas en `models/analysis/complementariedad_multimodal/` (tablas CSV y figuras). **No entrena**: reutiliza los `predictions.csv` de la ablación y de la rama espectral con atención (§9). Pregunta: ¿las modalidades contienen información predictiva *complementaria* que las arquitecturas actuales no aprovechan? Se distinguen tres tipos: **potencial** (una acierta donde otra falla), **aprovechable** (una combinación sin conocer la etiqueta mejora) y **física** (los desacuerdos se asocian a propiedades astrofísicas).

**Protocolo.** Comparación por `name` + `split` (cohortes idénticas, 0 duplicados, 0 galaxias en más de un split). `test` se usa solo de forma descriptiva: la fusión se explora **solo en validation** con una regla de elección fijada antes de ver resultados, y test se muestra después con el α ya fijado. Modelos usados: `rgb_cnn` (GPU), `rgb_resnet` (referencia), `esp_fov`, `esp_sl`, `esp_sl_attn` (`espectros_starlight_cnn_attn_control`, §9), `pob` y las fusiones concat. Una semilla, un solo split.

### 8.1 Hallazgos principales (test, n = 1403)

| Tema | Resultado |
|---|---|
| Ranking unimodal (AUC) | ResNet-18 0.963 ≈ CNN RGB 0.962 > CNN+atención espectro SL 0.907 > poblaciones 0.895 > CNN1D esp. SL 0.864 > CNN1D esp. FoV 0.857 |
| Rescate de errores RGB (ResNet-18: 157 errores, 11.2 %) | Poblaciones 51.6 % [43.7, 58.8]; esp. FoV 51.0 %; esp. SL 51.6 %; esp. SL atención 49.7 %. Al menos una de las 4 corrige el 64.3 %; las 4 a la vez, 38.2 % |
| Oráculo RGB∪poblaciones | Accuracy 0.946 vs 0.888 (+0.058 [0.046, 0.068]); **límite retrospectivo**, no un sistema implementable |
| Errores compartidos RGB∩pob. | 76 galaxias (48.4 % de los errores RGB); Jaccard 0.25. Con errores independientes se esperarían ≈26: hay **dificultad común** además de complementariedad. Rescate val 56.2 % vs test 51.6 % (consistente) |
| Fusión tardía `α·p_RGB + (1−α)·p_otra` | Poblaciones: α*=0.85 (región estable 0.70–1.00), AUC val 0.9644 → 0.9664, **test 0.9633 → 0.9637 (Δ=+0.0005 [−0.0012, +0.0022])**. Esp. FoV: α*=1.0 (la mejor fusión es no fusionar). El promedio simple (α=0.5) **empeora** (pob. test 0.9581; esp. FoV val 0.9512) |
| Concat entrenada vs RGB | Ninguna diferencia con IC95 % que excluya 0 (RGB+pob, RGB+esp FoV, RGB+esp SL+pob). McNemar vs ResNet: concat p = 0.14–1.0; fusión tardía p = 0.79 |
| Diferencias reales (ΔAUC, IC95 % bootstrap pareado) | ResNet − poblaciones +0.068 [0.054, 0.084]; poblaciones − esp. SL (CNN1D) +0.031 [0.019, 0.042]; esp. SL − esp. FoV (CNN1D) +0.007 [0.002, 0.013]; atención − CNN1D (esp. SL) y ResNet − atención también excluyen 0 |
| CNN vs ResNet RGB | ΔAUC −0.0009 [−0.0068, 0.0043] (empate), pero Δbalanced acc +0.017 [0.003, 0.030]: diferencia de **punto de operación**, no de ranking |

### 8.2 Escenario resultante

La regla documentada en el notebook (oráculo ≥ 4 pts y errores compartidos < 70 % sin mejora demostrable de la fusión) clasifica el caso como **Escenario B: alta cobertura del oráculo, fusión tardía sin mejora demostrable**. Interpretación: poblaciones y espectro *saben* algo distinto a la imagen en ~la mitad de los fallos RGB, pero un promedio fijo de probabilidades o una concatenación no sabe **cuándo** confiar en cada rama. La línea siguiente es *gated fusion*, fusión basada en incertidumbre/calibración o stacking entrenado con validación cruzada, no más arquitecturas concat. *(Actualización: la primera exploración de esa línea está en §10; encuentra señal modesta pero sin ganancia neta demostrada.)*

### 8.3 Confianza y calibración

- Las discrepancias no son solo de clase: |ΔP| mediana 0.12 (todas), **0.57** en desacuerdos de clase, 0.09 cuando ambos aciertan.
- ECE (10 bins) / Brier: CNN RGB 0.014 / 0.072; ResNet-18 0.038 / 0.077; esp. SL atención 0.024 / 0.117; poblaciones 0.027 / 0.122; CNN1D espectro 0.034–0.037 / 0.147. La CNN compacta está mejor calibrada que ResNet-18 con el mismo AUC; el ECE es sensible al binning.
- **35 galaxias (22.3 % de los errores RGB) son errores con |P−0.5| ≥ 0.40**: alta confianza no garantiza acierto. Hay 29 galaxias donde RGB duda (|P−0.5| ≤ 0.15) y poblaciones aciertan con |P−0.5| ≥ 0.3: candidatas naturales para gating.
- Las 9 galaxias con predicciones opuestas y ambas muy confiadas son mayoritariamente tempranas jóvenes (22 % tardías, edad media por luz ≈ 1.9 Gyr, `logM` ≈ 9.6): posibles morfologías peculiares o etiquetas dudosas.

### 8.4 Dónde se equivocan los modelos (Ttype)

- **La frontera Ttype 0–1 concentra los errores**: 16.1 % de las galaxias, 47.8 % de los errores RGB; error RGB 33.2 % (Ttype 0–1) vs 7.0 % (resto); poblaciones 46.5 % vs 10.5 %.
- Error por Ttype (ResNet / poblaciones): E (−5) 3.0 % / 12.2 %; S0 (−2) 21.5 % / 24.3 %; S0a (0) 34.6 % / 37.0 %; Ttype 1 32.4 % / **51.7 %**; Ttype 2 17.0 % / 19.2 %; Ttype ≥ 3 ≤ 4.5 % / ≤ 10 %. Las poblaciones casi no distinguen las espirales tempranas (Ttype 1) de las S0: la dicotomía física es continua, la etiqueta es una frontera arbitraria sobre ella.
- Mayor desacuerdo RGB–poblaciones: Ttype 1 (44.1 %, n = 145). El rescate por poblaciones varía por clase (S0: 68 %; Ttype 1: 38 %; muestras pequeñas).
- Ninguna propiedad física/STARLIGHT distingue 'solo poblaciones acierta' de 'ambos fallan' tras Bonferroni (n = 81 vs 76). Asociación ≠ causalidad y los grupos son de decenas de galaxias: **no hay una regla física simple para decidir qué modalidad usar**.
- La galería (cualitativa) sugiere hipótesis a revisar: morfologías peculiares, alta inclinación y compañeras. `rgb_falla_pob_acierta` (60 % tardías, Ttype mediano 1) y `rgb_acierta_pob_falla` (37 % tardías, Ttype mediano −1) son grupos complementarios.

### 8.5 Rol de `cnn_attn_control` en este estudio

`espectros_starlight_cnn_attn_control` (`esp_sl_attn`, AUC test 0.9066, recall T 0.81 / L 0.86) es la **mejor rama espectral usada en la comparación de complementariedad** y sustituye a la CNN1D en los análisis de rescate y fusión tardía (par `rgb_resnet` + `esp_sl_attn`: α*=0.85, región estable 0.65–1.00). Aun con ese espectro más capaz, rescata 49.7 % de los errores RGB, **sin mejorar la complementariedad** respecto a la CNN1D (51.6 %): mejorar la rama espectral por sí sola no cambia el diagnóstico de "información distinta pero no integrable por promedio".

### 8.6 Conclusiones acumuladas

1. Imagen domina y ninguna fusión (concat o tardía) la supera de forma demostrable. Pero **no** se debe afirmar que las modalidades secundarias "no aportan": corrigen ~50 % de los fallos RGB; lo que falla es el método de integración.
2. La complementariedad es real pero acotada por la **dificultad común en la frontera** (Ttype 0–1) y por los errores compartidos (76 galaxias): el techo realista de una fusión queda por debajo del 0.946 del oráculo.
3. El promedio 50/50 perjudica; el peso óptimo es ≈0.85 hacia RGB con ganancia nula. Cualquier fusión debe condicionarse por confianza/incertidumbre.
4. Comparar modelos con AUC + balanced accuracy, IC bootstrap pareado y McNemar; la accuracy a 0.5 confunde ranking con punto de operación.
5. Pendiente: varias semillas (los IC solo reflejan el muestreo de galaxias de test, no la variabilidad de entrenamiento), CV para stacking y revisión de las galaxias discordantes con un experto. El estudio de cuándo confiar en cada modalidad se abordó en §10 (solo con `validation`).

---

## 9. Optimización del experto espectral (RNN, atención) — fuera de la ablación

Notebooks: [experimento_espectros_fov_rnn.ipynb](experimento_espectros_fov_rnn.ipynb) (espectro observado MaNGA `FLUX_FoV`, 3600–9000 Å, 4096 muestras) y [experimento_espectros_starlight_rnn.ipynb](experimento_espectros_starlight_rnn.ipynb) (espectro modelado STARLIGHT). Artefactos: `models/experiments/espectros_{fov,starlight}_{patch_bigru,cnn_bigru,cnn_attn_control}/`. **No son parte de la ablación**: optimizan el *experto* espectral y se acumulan para decidir su rol futuro. Misma cohorte/split `_FoV` y protocolo de selección que la CNN1D de referencia.

**Hipótesis.** La CNN 1D con pooling global no integra el espectro completo (continuo, quiebre de 4000 Å, líneas); una arquitectura recurrente podría extraer más información. Cambia el encoder, pero también la optimización (ver 9.2).

| Variante | Qué prueba |
|---|---|
| `patch_bigru` | RNN "pura": 128 patches de 32 muestras + BiGRU (2 capas, 64) |
| `cnn_bigru` | CNN local (4096 → 256 tokens) + BiGRU global |
| `cnn_attn_control` | Control: CNN más profunda con pooling por atención, **sin recurrencia** (separa recurrencia de capacidad) |

Común: pooling media+máximo enmascarado, máscara propagada por token, embedding 128, dropout 0.3, AdamW wd 1e-4, BCE, **LR 3e-4 (la CNN1D usaba 1e-4)**, *gradient clipping* 1.0, 60 épocas máx., paciencia 10, ≈ 3.5–4 s/época.

### 9.1 Resultados en test

| Espectro | Modelo | AUC | Acc. | Rec. T | Rec. L | F1 macro | Mejor ép. / total | Train / Val loss al final |
|---|---|---|---|---|---|---|---|---|
| FoV | CNN1D (referencia) | 0.8572 | 0.7947 | 0.682 | 0.863 | 0.777 | 50 / 50 | 0.461 / 0.475 |
| FoV | `cnn_attn_control` | 0.9011 | 0.8418 | 0.822 | 0.854 | 0.833 | 8 / 18 | 0.246 / 0.587 |
| FoV | `cnn_bigru` | 0.9122 | 0.8468 | 0.824 | 0.861 | 0.838 | 6 / 16 | 0.232 / 0.509 |
| FoV | `patch_bigru` | **0.9170** | **0.8525** | 0.839 | 0.861 | 0.845 | 18 / 28 | 0.305 / 0.382 |
| SL | CNN1D (referencia) | 0.8642 | 0.7890 | 0.636 | 0.881 | 0.767 | 48 / 50 | 0.454 / 0.470 |
| SL | `cnn_attn_control` | **0.9066** | 0.8418 | 0.812 | 0.859 | 0.833 | 34 / 44 | 0.364 / 0.421 |
| SL | `cnn_bigru` | 0.9064 | 0.8382 | 0.775 | 0.877 | 0.827 | 40 / 50 | 0.361 / 0.405 |
| SL | `patch_bigru` | 0.8997 | 0.8211 | 0.686 | 0.903 | 0.803 | 27 / 37 | 0.394 / 0.409 |

AUC de validación del mejor checkpoint: FoV 0.9036 / 0.9046 / 0.9153 (attn / cnn_bigru / patch); SL 0.8985 / 0.9002 / 0.8912. Error estándar del AUC ≈ 0.008 en este rango (una semilla; sin IC bootstrap en esta tabla).

### 9.2 Lecturas

- **La CNN1D de referencia estaba infraajustada y mal optimizada**: las variantes nuevas parten de AUC val ≈ 0.88–0.89 ya en la época 1 (la CNN1D tardaba 7–8 épocas en "despegar" de la clase mayoritaria) y llegan a 0.90–0.92 de AUC test (+0.04 a +0.06, muy por encima del ruido). La información espectral existía; confirma el diagnóstico de §1.4 y §4.
- **Factores confundidos**: el cambio incluye LR 1e-4 → 3e-4, clipping y mayor capacidad *junto con* la recurrencia. La ganancia **no** puede atribuirse a la recurrencia: `cnn_attn_control` (sin recurrencia) logra 0.901 (FoV) y 0.907 (SL).
- **Recurrencia: efecto pequeño o nulo.** FoV: patch-BiGRU +0.016 y cnn_bigru +0.011 sobre atención (≈1–2 errores estándar, no concluyente). SL: atención ≥ cnn_bigru > patch-BiGRU (0.907 / 0.906 / 0.900). La RNN pura no supera a las híbridas en SL, coherente con que la CNN local ayuda a capturar líneas.
- **FoV observado ≈ STARLIGHT modelado con estos encoders** (mejor FoV 0.917 vs mejor SL 0.907; con atención SL gana 0.907 vs 0.901). Con la CNN1D, SL ganaba +0.007. Sin ventaja consistente del espectro modelado, conviene preferir el **observado** (insumo independiente de las poblaciones).
- **Techo ≈ 0.90–0.92.** Tras optimizar, el espectro iguala o supera a las poblaciones (0.895) pero queda ~0.05 AUC por debajo de la imagen; puede ser límite de información del espectro integrado FoV (no resuelve estructura espacial).
- **Sesgo hacia tardías**: `patch_bigru` SL tiene recall T 0.686 / L 0.903; las demás variantes están más equilibradas (recall T 0.78–0.84). Usar balanced accuracy/umbral de validación.
- **Sobreajuste**: FoV con atención o CNN+BiGRU sobreajusta rápido (mejor ép. 6–8, val_loss final 0.51–0.59 con train 0.23–0.25) y se detiene por paciencia; `patch_bigru` y las variantes SL son más estables. El early stopping selecciona bien, pero convienen LR menor/scheduler y más regularización (dropout, weight decay, augmentations espectrales como ruido o enmascarado).

### 9.3 Experimento relacionado aún no integrado: imagen + espectro con atención (SL)

`multimodal_imagenes_espectros_starlight_cnnattn_fusion` y su control `..._image_only_control` (LR 3e-4, clipping 1.0, mismo split `_FoV`; carpetas aún no mapeadas en el notebook de complementariedad):

| Experimento | AUC test | Acc. | Rec. T | Rec. L | Mejor ép. / total |
|---|---|---|---|---|---|
| Imagen sola (control, LR 3e-4) | **0.9667** | 0.8952 | 0.909 | 0.887 | 47 / 50 |
| Imagen + espectro SL (atención), concat | 0.9334 | 0.8617 | 0.902 | 0.838 | 4 / 12 |

La fusión queda **0.033 AUC por debajo** del control de imagen con la misma optimización y se detiene pronto (mejor en la época 4, val_loss al alza). Con LR 3e-4 la rama espectral más fuerte parece **perjudicar** la fusión concat (atajos/memorización), en línea con §8.2: añadir un experto mejor no basta sin una fusión controlada. El control de imagen con LR 3e-4 es además el mejor AUC visual (+0.003 sobre ResNet-18 LR 1e-4, dentro del ruido). Una sola semilla: señal a confirmar.

### 9.4 Siguientes pasos para el experto espectral

1. Reentrenar la CNN1D de referencia con LR 3e-4 + clipping para **separar optimización de arquitectura** (hoy confundidas).
2. Repetir `cnn_attn_control` y `patch_bigru` (FoV) con 3–5 semillas e IC bootstrap; decidir si la recurrencia justifica su costo.
3. Regularización/scheduler en FoV (sobreajuste rápido) y baseline interpretable con índices espectrales (Dn4000, Hδ_A, Hα EW).
4. Evaluar complementariedad de los nuevos expertos con imagen (rescate, fusión tardía, gating); `espectros_fov_patch_bigru` aún no entra a §8.
5. Incluir las corridas `..._cnnattn_*` en el estudio de complementariedad.

---

## 10. Diagnóstico de confiabilidad y selección de modalidad (Fase 1 del gate)

Fuente: [diagnostico_gate_multimodal.ipynb](../../diagnostico_gate_multimodal.ipynb); salidas en `models/analysis/diagnostico_gate_multimodal/` (tablas CSV, figuras y `config.json`; esa carpeta está ignorada por git). **No reentrena ninguna red**: reutiliza los `predictions.csv` de ResNet-18 RGB (`imagenes_resnet18_fov`), de la CNN con atención sobre espectro STARLIGHT (`espectros_starlight_cnn_attn_control`) y del MLP de poblaciones. Pregunta: ¿se puede anticipar, con información disponible en inferencia, cuándo el espectro es más fiable que RGB?

**Protocolo.** Alineación por `name` + `split` (1403 galaxias en `validation`, sin pérdidas). El desarrollo y el selector usan **solo `validation`**; `train` solo se audita y **`test` no se usa**. Se estudian los **desacuerdos** (RGB y espectro predicen clases distintas): 257 de 1403 (18.3 %); RGB acierta en 172 y el espectro en 85. El selector (regresión logística pequeña con imputación y escalado dentro de cada fold, `StratifiedKFold(5)`) predice `selector_target` (1 = gana el espectro) y se evalúa con predicciones *out-of-fold* **del selector**: las redes base no se reentrenaron con validación cruzada. Una semilla de las redes base y un solo split.

> **Sesgo conocido.** Los checkpoints de las redes base se eligieron con el ROC-AUC de `validation`; sus predicciones allí son algo favorables (por ejemplo, la accuracy de RGB es 0.896 y la del espectro 0.834 en `validation`). Las cifras de esta sección **no son comparables con las de test de §3–§9**.

### 10.1 Hallazgos principales (`validation`, n = 1403; 257 desacuerdos)

| Tema | Resultado |
|---|---|
| Quién gana en un desacuerdo | RGB 66.9 %; espectro **33.1 %** [27.6, 39.0]. Sin asimetría por dirección: el espectro gana 33.6 % cuando RGB dice tardía (n = 131) y 32.5 % cuando RGB dice temprana (n = 126) |
| Mapa `p_rgb × p_spectrum` (5×5, mín. 15 por celda) | 14 de 25 celdas interpretables; **ninguna** con ventaja del espectro (IC pareado > 0); 5 celdas con ventaja de RGB, todas con RGB muy confiado (p ≤ 0.2 o ≥ 0.8). 70 de los 257 desacuerdos caen en celdas con n < 15 |
| La confianza detecta los errores propios | AUC de `confidence_rgb` como detector de acierto de RGB 0.878 [0.854, 0.900]; del espectro 0.754 [0.721, 0.786] |
| …pero el rescate no depende claramente de la confianza de RGB | El espectro rescata 52.9 % de los 34 errores RGB cometidos con confianza ≥ 0.8 y 59.8 % de los 112 restantes (de 146 errores RGB). Entre desacuerdos, `uncertainty_rgb` anticipa que gane el espectro con AUC 0.693 [0.620, 0.758]: el espectro gana ≈ 51–55 % si `confidence_rgb` < 0.4 y 16 % si > 0.9 |
| Ambos confiados y discrepan | 8 desacuerdos con ambos con confianza ≥ 0.8; el espectro gana 25 % [7, 59] (muestra insuficiente) |
| Calibración | Brier / ECE: RGB 0.074 / 0.026; espectro 0.123 / 0.035; Δ Brier espectro − RGB +0.049 [+0.037, +0.061]. **La escala difiere** (desviación estándar del logit 5.4 en RGB vs 2.2 en el espectro). En los desacuerdos RGB asigna 0.79 a su predicción (accuracy real 0.67) y el espectro 0.72 (accuracy real 0.33): el espectro está **sobreconfiado justo donde discrepan**, lo que concuerda con que el promedio simple perjudique (§8.1) |
| Propiedades asociadas al rescate | **Ninguna de 14** variables (observacionales, calidad STARLIGHT, físicas) sobrevive a Holm; mayor efecto `sn_window` (δ de Cliff +0.17, p_Holm 0.34). Tampoco separando por dirección del desacuerdo |
| `Ttype` (solo interpretación) | El espectro gana más en la frontera: S0a (Ttype 0) 48.8 % (n = 41), Ttype 1 41.7 % (n = 60); E 15.6 % (n = 32), Ttype ≥ 3 23.1 % (n = 26). IC amplios |

### 10.2 Selector exploratorio (predicciones out-of-fold sobre los 257 desacuerdos)

Variantes fijadas a priori, todas disponibles en inferencia: **S1** regla de confianza (elegir el modelo más lejano de 0.5), **S2min** (`p_rgb`, `p_spectrum`), **S2** (+ `confidence_*`, `probability_gap`), **S3** (+ `nsa_z`, `PETRO_TH90`, `LogMass`), **S4** (+ `sn_window`, `adev_pct`, `chi2_nl_eff`), **S5** (+ `age_mean_light_gyr`, `light_young_lt_100myr_pct`, `vd_kms`). Se evalúan dos modos con los mismos rasgos y umbral 0.5: `balanced` (clases ponderadas; optimiza balanced accuracy) y `prior` (sin ponderar; coherente con maximizar la ganancia neta).

| Variante | Bal. acc. OOF | AUC OOF [IC95 %] | p permutación | Ganancia neta `balanced` | Ganancia neta `prior` |
|---|---|---|---|---|---|
| S0 siempre RGB | 0.500 | — | — | 0 | 0 |
| S1 confianza (regla) | 0.616 | 0.674 [0.614, 0.737] | — | −6 [−25, 13] | — |
| S2min | 0.424 | 0.398 [0.322, 0.467] | 0.955 | −60 [−81, −39] | 0 |
| **S2** | **0.629** [0.563, 0.691] | **0.676** [0.604, 0.738] | **0.005** | −11 [−32, 11] | **+5 [−9, 18]** |
| S3 observacional | 0.374 | 0.357 | 0.995 | −75 [−98, −54] | +1 |
| S4 calidad | 0.546 | 0.528 | 0.259 | −27 [−49, −6] | −5 [−10, −1] |
| S5 física | 0.345 | 0.311 | 1.000 | −81 [−105, −58] | −1 |
| *Referencias (mismas 1403 galaxias)* | | | | Fusión fija α=0.85: +3 [−6, 12]; espectro solo: −87; **oráculo: +85 [68, 102]** | |

Ganancia neta = errores de RGB corregidos − aciertos de RGB perdidos, sobre las 1403 galaxias. S2 `prior`: 49 cambios a espectro, 27 corregidos y 22 perdidos (accuracy total 0.8995 vs 0.8959 de RGB sola, Δ = +0.0036 [−0.0064, +0.0128]).

Lecturas:

- **La señal está en la confianza, no en la probabilidad lineal.** S1 y S2 superan el azar (S2: AUC 0.676, p de permutación 0.005, AUC > 0.5 en 5/5 folds y en 10 particiones de folds, mínimo 0.647); S2min, S3 y S5 no. Los AUC OOF por debajo de 0.5 de las variantes sin señal son un **artefacto de la validación cruzada con pocos datos** (el nulo de permutación tiene media ≈ 0.49 con cola inferior en ≈ 0.38) y no una relación inversa.
- **Calidad STARLIGHT, observacionales y física no aportan señal detectable.** S3–S5 no superan a S2min, y añadir esos rasgos *sobre* S2 (variantes post-hoc S3c–S5c, AUC 0.63–0.67) no mejora a S2.
- **Los dos puntos de operación importan.** El modo `balanced` sube el recall del espectro pero cambia con probabilidad real cercana a la tasa base (≈ 1/3) y **pierde aciertos** (S2: 54 corregidos, 65 perdidos). El modo `prior` equilibra rescates y pérdidas, pero la ganancia neta tiene un IC que incluye 0.
- **El margen es grande pero no alcanzado.** El oráculo corrige 85 errores de RGB; los selectores, como máximo ≈ +5.

### 10.3 Diagnóstico no lineal: árboles y XGBoost

Pregunta: ¿hay dependencias no lineales (umbrales, interacciones entre confianza, calidad y propiedades físicas) que un modelo de árboles aproveche mejor que la regresión logística? Se compararon, sobre los mismos folds y seis conjuntos de rasgos fijados a priori (confianza; + observacionales; + calidad; + física compacta; combinado; + física completa), siete modelos con hiperparámetros fijos: LR, LR con términos cuadráticos, árbol de profundidad 3, Random Forest y XGBoost de profundidad 1, 2 (primario) y 3. Validaciones: diferencia pareada con IC bootstrap en AUC y log loss, 10 particiones de folds, **prueba de permutación de ΔAUC (XGBoost − LR) con Holm**, brecha entre AUC de entrenamiento y OOF, curvas univariadas LR vs XGBoost, importancia por permutación fuera de fold con una variable de ruido de control y simulación de ganancia neta.

Resultados:

- **Ningún conjunto de rasgos muestra ventaja de los árboles**: con XGBoost (profundidad 2) el ΔAUC frente a la LR con los mismos rasgos tiene IC95 % > 0 y p ajustado < 0.05 en **0 de 6** conjuntos, y no supera a la LR en ≥ 8 de 10 particiones en ninguno. Con solo confianza: AUC 0.618 (XGBoost) vs 0.671 (LR).
- **Sobreajuste**: brecha media AUC entrenamiento − OOF 0.30 en XGBoost frente a 0.07 en la LR. El mejor flexible (XGBoost profundidad 1 con calidad, AUC 0.665) no supera a la mejor LR (0.671).
- **Sin no linealidad univariada detectable**: 0 de 12 rasgos con ΔAUC (XGBoost profundidad 1 − LR) con IC > 0 tras Bonferroni; añadir contexto a T1 tampoco mejora a XGBoost.
- **Importancia fuera de fold**: solo `confidence_rgb` supera claramente al ruido de control.
- **Ganancia neta**: con XGBoost ninguna de las seis combinaciones tiene IC > 0 (en confianza: +4 [−9, 18] frente a +5 [−8, 19] de la LR).

Conclusión: la dependencia con la confianza ya la captura la regresión logística si se le entrega `2·|p−0.5|`; con **257 desacuerdos** los modelos flexibles no extraen estructura adicional. Es un resultado nulo con **potencia limitada**: no descarta no linealidad pequeña ni interacciones complejas.

### 10.4 Veredicto y decisiones

- Criterios para recomendar avanzar a un gate adaptativo: 3 de 5 cumplidos. Se cumplen **señal fuera de fold**, **estabilidad entre folds y semillas** y **rasgos disponibles en inferencia**; **no** se cumplen **ganancia neta positiva con IC > 0** ni **robustez a la pérdida de unas pocas galaxias**.
- **Veredicto: señal sin ganancia neta demostrada.** La confianza de los modelos contiene información modesta sobre cuál acierta, pero no mejora de forma demostrable a RGB sola; **no se justifica todavía entrenar un gate adaptativo** (MLP/atención de gating). Esto es coherente con §8: hay información distinta en las modalidades secundarias, pero cuándo usarla no es predecible con los rasgos actuales.
- Las propiedades STARLIGHT (calidad y físicas) no sirven como regla simple para elegir modalidad, consistente con §8.4.
- **Próximos pasos justificados:**
  1. Obtener **predicciones OOF de los propios encoders** (RGB y espectro, validación cruzada sobre todo el conjunto) para multiplicar los desacuerdos disponibles y reservar una evaluación final independiente. Es el único camino para convertir esta evidencia exploratoria en confirmatoria.
  2. Si hay más datos, repetir S1/S2 (confianza) y, solo entonces, probar un gate pequeño con los mismos rasgos mínimos; incluir la fusión fija α=0.85 como referencia.
  3. Estudiar calibración (p. ej. recalibrar las probabilidades dentro de validación cruzada) antes de combinar probabilidades, dado el exceso de confianza del espectro en los desacuerdos.
  4. Si alguna variable STARLIGHT mejorara un selector en el futuro, compararla contra un clasificador tabular directo de la morfología con esas variables, para separar *fiabilidad* de *información morfológica adicional*.

**Limitaciones de §10.** Pocos desacuerdos (257) y un solo split; sesgo favorable de `validation` por la selección de checkpoints; selectores evaluados con OOF solo del selector; la variante primaria S2 se eligió por AUC OOF sobre los mismos datos (algo optimista); las variantes S3c–S5c y la sección de árboles se añadieron después de ver que la señal estaba en la confianza (exploratorias); los IC bootstrap de la simulación no incluyen la incertidumbre de reentrenar el selector; no hay evidencia confirmatoria en `test`.

---

## 11. Fusión por tokens con attention pooling (M0 / M1 / M2) — complemento de §8

Notebook: [experimento_multimodal_token_fusion_attention.ipynb](experimento_multimodal_token_fusion_attention.ipynb); modelos en `src/models/multimodal_token_fusion.py` (guía: [../markdowns/MULTIMODAL_TOKEN_FUSION.md](../markdowns/MULTIMODAL_TOKEN_FUSION.md)). Artefactos: `models/experiments/multimodal_token_fusion_attention_{independent_mean,independent_attention,joint_attention}/` (`results.json` y `test_metrics.json` son idénticos en las tres). Pregunta de §8 que retoma: ¿la complementariedad *potencial* se vuelve *aprovechable* si se fusiona **antes** del pooling global, en vez de concatenar embeddings o promediar probabilidades?

**Diseño.** ResNet-18 (`imagenes_resnet18_fov`) → 64 tokens y patch-BiGRU FoV (`espectros_fov_patch_bigru`) → 128 tokens, ambos proyectados a 128, con LayerNorm y embedding de modalidad. M0: media por modalidad + concat; M1: attention pooling enmascarado por modalidad + concat; M2: un attention pooling sobre los 192 tokens. Mismo MLP final y misma optimización (AdamW, LR 1e-4, 50 épocas máx., paciencia 8, mismo split `_FoV`). **Ambos encoders congelados**: solo se entrenan proyecciones, embeddings, pooling y clasificador (≈ 0.10–0.13 M de parámetros entrenables de ≈ 11.5 M). Una semilla.

### 11.1 Resultados en test (n = 1403, umbral 0.5)

| Modelo | AUC [IC95 % bootstrap] | Acc. | Prec. T / Prec. L | Rec. T / Rec. L | F1 L | Test loss | Mejor ép. / total |
|---|---|---|---|---|---|---|---|
| ResNet-18 (imagen sola) | 0.9633 [0.9552, 0.9715] | 0.8881 | 0.8454 / 0.9145 | 0.8598 / 0.9051 | 0.9098 | 0.2480 | — |
| patch-BiGRU FoV (espectro solo) | 0.9170 [0.9009, 0.9307] | 0.8525 | 0.7841 / 0.8986 | 0.8390 / 0.8606 | 0.8792 | 0.3593 | — |
| Concat entrenada imagen+espectro (§2) | 0.9635 [0.9552, 0.9717] | 0.8874 | 0.8599 / 0.9033 | 0.8371 / 0.9177 | 0.9104 | 0.2372 | — |
| **M0** `independent_mean` | 0.9660 [0.9581, 0.9736] | 0.8988 | 0.8726 / 0.9141 | 0.8561 / 0.9246 | 0.9193 | 0.2462 | 27 / 35 |
| **M1** `independent_attention` | 0.9661 [0.9581, 0.9734] | 0.8995 | 0.8728 / 0.9152 | 0.8580 / 0.9246 | 0.9198 | 0.2445 | 14 / 22 |
| **M2** `joint_attention` | **0.9666** [0.9587, 0.9740] | **0.9016** | 0.8839 / 0.9117 | 0.8504 / 0.9326 | 0.9220 | 0.2479 | 23 / 31 |

Matrices de confusión (filas = real T, L; columnas = pred. T, L): M0 [[452, 76], [66, 809]]; M1 [[453, 75], [66, 809]]; M2 [[449, 79], [59, 816]]. AUC de validación (mejor época): 0.9665 / 0.9669 / 0.9659. Tiempo total de entrenamiento: 515 / 310 / 428 s. Las referencias salen de sus `results.json`; los IC y las comparaciones de abajo se calcularon con los `predictions.csv` (test) mediante bootstrap pareado (1000 remuestreos, semilla 0) y prueba binomial exacta de McNemar, de forma ad hoc, fuera del notebook de §8.

### 11.2 Comparaciones pareadas (test)

| Comparación | ΔAUC [IC95 % pareado] | Galaxias corregidas / rotas respecto a la referencia | McNemar exacto |
|---|---|---|---|
| M0 − ResNet-18 | +0.0027 [+0.0005, +0.0048] | +25 / −10 | p = 0.017 |
| M1 − ResNet-18 | +0.0028 [−0.0000, +0.0057] | +30 / −14 | p = 0.023 |
| M2 − ResNet-18 | +0.0033 [+0.0007, +0.0058] | +35 / −16 | p = 0.011 |
| M0 − concat previa | +0.0024 [−0.0026, +0.0080] | — | — |
| M1 − concat previa | +0.0025 [−0.0029, +0.0085] | — | — |
| M2 − concat previa | +0.0030 [−0.0024, +0.0090] | +54 / −34 | p = 0.042 |
| M2 − M0 | +0.0006 [−0.0009, +0.0022] | — | — |

### 11.3 Lectura

1. **Mejora pequeña y consistente sobre RGB, no concluyente.** Las tres variantes quedan ≈ +0.003 AUC y +0.011 a +0.014 de accuracy por encima de ResNet-18, y tienen más aciertos nuevos que fallos nuevos (≈ 2:1). Es la primera fusión de la serie que se separa de RGB en la dirección esperada (en §8 la fusión tardía daba +0.0005 y la concat era neutra). Pero el efecto equivale a ~0.5 errores estándar del AUC; dos de los tres IC de AUC apenas excluyen 0, el de M1 llega a −0.0000, y con tres variantes comparadas contra la misma referencia (Bonferroni ×3: p ≈ 0.03–0.07) y una sola semilla/split **no se puede afirmar una ganancia robusta**. Sigue siendo consistente con §1.2: añadir espectro no mejora de forma demostrable.
2. **El attention pooling no es lo que aporta.** M0, que solo promedia, ya obtiene +0.0027; M2−M0 = +0.0006 [−0.0009, +0.0022]. Además el orden en test (M2 > M1 > M0) no coincide con validación (M1 > M0 > M2), lo que sugiere ruido. No hay evidencia a favor de la hipótesis de que conservar tokens hasta el pooling supere al pooling independiente. M1 y M2 convergen antes (14 y 23 épocas) pero eso no se traduce en mejor métrica.
3. **Confundido con la calidad de los encoders.** Frente a la concat previa (que entrenaba una CNN 1D débil junto con ResNet) cambian tres cosas a la vez: encoders **preentrenados y más fuertes** (el espectro pasa de 0.857 a 0.917 AUC), **congelados** y cabeza nueva. M0 es en la práctica el control "embeddings preentrenados congelados + proyección + media", y ya logra casi todo. Con estos datos **no se puede atribuir** la mejora a la fusión a nivel de token. Esto encaja con §9.3: la rama espectral fuerte perjudicó la fusión cuando se entrenaba conjuntamente a LR 3e-4; congelarla evita esa interferencia.
4. **Complementariedad aprovechada (retoma §8.1).** Con los `predictions.csv`: ResNet-18 comete 157 errores; el espectro patch-BiGRU acierta en 82 de ellos y ambos fallan en 75 (oráculo RGB∪espectro: accuracy 0.9465, +82 galaxias sobre RGB). M0/M1/M2 corrigen 25/30/35 de los 157 errores RGB (15.9/19.1/22.3 %), pero rompen 10/14/16 aciertos; neto +15/+16/+19 galaxias, es decir, **≈ 18–23 % del margen del oráculo**. De las 82 galaxias donde "RGB falla y espectro acierta", las fusiones aciertan 20/27/28 (24/33/34 %); de las 75 donde ambos fallan, solo 5/3/7. Esto confirma el diagnóstico del Escenario B: la información distinta existe, pero la fusión aprendida recupera una fracción pequeña. La dificultad común (frontera Ttype 0–1, §8.4) sigue siendo el límite: ninguna variante la resuelve.
5. **Las fusiones son casi una copia de RGB.** Correlación de probabilidades con ResNet-18: M0 0.989, M1 0.983, M2 0.982 (concat previa 0.950); con el espectro solo: 0.82–0.84. Ni siquiera M2, que puede asignar toda la atención a un modalidad, se aleja de RGB; el modelo aprende a usar el espectro como un ajuste pequeño. (No se analizó la masa de atención por modalidad: `get_attention_weights` existe pero no se ejecutó en estas corridas, así que no se puede afirmar cuánta atención recibe cada modalidad ni interpretarla causalmente.)
6. **Punto de operación.** La mejora de accuracy viene de dos lados: la clase tardía gana recall (0.925–0.933 vs 0.905 de ResNet-18) con precisión similar (0.912–0.915 vs 0.9145), y la clase temprana gana precisión (0.873–0.884 vs 0.845) con recall casi igual o algo menor (0.850–0.858 vs 0.860). M2 es la que más desplaza el umbral hacia tardías (recall T 0.850, recall L 0.933, precisión L 0.912). La test loss (0.245–0.248) **no mejora** respecto a ResNet-18 (0.248) ni a la concat previa (0.237): el beneficio es de ranking/umbral, no de calibración.
7. **Sobreajuste de la cabeza.** Pérdida final de train 0.153–0.161 frente a val mínima 0.233–0.244, patrón ya visto en ResNet-18 (§1.5). Los encoders se entrenaron con el mismo conjunto de train, de modo que sus embeddings en train ya son "demasiado buenos" y la cabeza nueva tiene poco margen para aprender a combinar; es una hipótesis (no verificada aquí) coherente con el sobreajuste y con la correlación ≈ 0.98 con RGB. Además, los encoders y sus checkpoints se eligieron con `validation`, y este también selecciona la época de la cabeza: sesgo favorable leve en validación, no en test.

### 11.4 Conclusiones y próximos pasos

- **Conclusión para §8.** Fusionar a nivel de token con encoders fuertes y congelados da la mejor cifra de test del proyecto (AUC 0.9666) pero **no cambia el veredicto**: mejora no demostrable respecto a RGB ni a la concat previa, y aprovecha una fracción pequeña de la complementariedad potencial. No hay evidencia de que M2 > M1 > M0.
- **Decisión sugerida.** No priorizar más variantes de pooling. Sí vale la pena (a) repetir M0 y M2 con 3–5 semillas e IC pareados para decidir si el +0.003 es real; (b) añadir el control **concat de embeddings globales preentrenados y congelados** (misma cabeza, sin tokens) para aislar el efecto de la tokenización; (c) probar **descongelar** parcial o total con LR bajo; (d) ejecutar `get_attention_weights` en validación para medir la masa visual/espectral y revisar si M2 realmente usa el espectro; (e) evaluar estas fusiones dentro de §8/§10 (rescate, fusión tardía con α, confianza) para ver si corrigen los desacuerdos de alta confianza.
- **Limitaciones.** Una semilla y un solo split; los IC reflejan solo el muestreo de test, no la variabilidad de entrenamiento; las comparaciones con IC pareado se calcularon ad hoc y sin corrección por múltiples comparaciones; `test` ya se había inspeccionado en §8 y no se usó para elegir hiperparámetros; no se evaluó calibración (ECE/Brier) ni error por Ttype para estas variantes.

---

## 12. Preprocesamiento de imagen sobre la CNN pequeña (grises, gamma, percentil, NLM)

Pregunta: ¿alguna técnica de procesamiento aplicada *antes* del entrenamiento mejora la CNN visual? Se reentrenó la **misma CNN** de `experimento_binario_imagenes_cnn_gpu` sobre cutouts transformados (notebook [experimento_binario_imagenes_gpu_acc_image_processed.ipynb](experimento_binario_imagenes_gpu_acc_image_processed.ipynb); artefactos en `models/experiments/experimento_binario_imagenes_cnn_gpu_{grayscale,gamma,percentile,nlm}/`).

### 12.1 Diseño

- **Controlado:** mismo split (`splits_seed42_obj-label.csv`, test n = 1403), FoV 0, tamaño 256, `normalization: none`, batch 32, LR 3e-4, weight decay 1e-4, *clipping* 1.0, 50 épocas máx., *early stopping* 8, selección por AUC de validación, semilla 42. Los `config.json` de las cuatro corridas son idénticos salvo el nombre. El control es `cnn_gpu` (RGB sin transformar; 46 épocas, mejor época 38).
- **Transformaciones** (`src/data/image_transforms.py`, globales y deterministas): **gamma** γ = 0.85 (aclara tonos medios/bajos); **grayscale** luminancia Rec. 709 replicada en 3 canales (elimina el color); **percentile** reescala la luminancia entre los percentiles 1 y 99.5 conservando proporciones RGB (satura ≈ 1 % inferior y 0.5 % superior). **NLM** (*non-local means*, reducción de ruido) no está en `image_transforms.py` ni sus parámetros constan en `config.json`: **sus hiperparámetros no quedan documentados** en el repositorio.

### 12.2 Resultados en test (umbral 0.5)

| Variante | Accuracy | AUC | Test loss | Recall temprana | Recall tardía | Bal. acc. | Errores (FP+FN) |
|---|---|---|---|---|---|---|---|
| Base (RGB) | 0.9009 | 0.9623 | 0.2416 | 0.892 | 0.906 | 0.8992 | 57+82 = 139 |
| **Percentile** | **0.9059** | 0.9651 | **0.2328** | 0.869 | 0.928 | 0.8987 | 69+63 = 132 |
| **Grayscale** | 0.8967 | **0.9657** | 0.2404 | 0.911 | 0.888 | 0.8995 | 47+98 = 145 |
| Gamma 0.85 | 0.8874 | 0.9627 | 0.2550 | 0.919 | 0.869 | 0.8936 | 43+115 = 158 |
| NLM | 0.8952 | 0.9553 | 0.2735 | 0.831 | 0.934 | 0.8826 | 89+58 = 147 |

(FP = temprana predicha tardía; FN = tardía predicha temprana.)

Comparación pareada con el control sobre `predictions.csv` (bootstrap de 2000 remuestreos de test y McNemar exacto):

| Variante | ΔAUC [IC95 %] | Aciertos perdidos / ganados vs base | McNemar p | Corr. de probabilidades con base |
|---|---|---|---|---|
| Percentile | +0.0028 [−0.0023, +0.0083] | 36 / 43 | 0.50 | 0.959 |
| Grayscale | +0.0035 [−0.0030, +0.0102] | 63 / 57 | 0.65 | 0.929 |
| Gamma | +0.0005 [−0.0028, +0.0042] | 35 / 16 | 0.011 | 0.986 |
| NLM | **−0.0071 [−0.0123, −0.0019]** | 44 / 36 | 0.43 | 0.958 |

### 12.3 Entrenamiento (validación)

| Variante | Épocas (mejor) | Mejor AUC val. | Train / val. loss en la mejor época | Pérdida val. mínima |
|---|---|---|---|---|
| Base | 46 (38) | 0.9587 | 0.246 / 0.254 | 0.254 |
| Percentile | 50 (**50**) | 0.9655 | 0.239 / 0.233 | 0.233 |
| Grayscale | 50 (43) | 0.9614 | 0.260 / 0.251 | 0.251 |
| Gamma | 40 (32) | 0.9582 | 0.257 / 0.265 | 0.258 |
| NLM | 41 (33) | 0.9566 | 0.265 / 0.276 | 0.262 |

Tiempo por época ≈ 27 s en todas (el preprocesamiento es offline).

### 12.4 Lecturas

1. **Ninguna técnica mejora de forma demostrable.** Los ΔAUC son ≤ 0.0035 y sus IC incluyen 0, salvo NLM, con efecto **negativo**. Es coherente con el ruido de §1 (±0.005 de AUC) y con que el techo visual (≈ 0.962–0.965) se repite con ResNet-18, FoV aleatorio y las fusiones: el cuello de botella no parece ser el contraste o el color de entrada.
2. **Percentile es la candidata más sólida, aunque modesta.** Única con mejora simultánea de accuracy (+0.005), AUC (+0.003), test loss (0.233, la menor) y AUC de validación (0.9655 vs 0.9587). Su **mejor época es la 50 (el tope)**: seguía mejorando cuando se cortó, así que el límite de épocas pudo truncar el resultado. Tiene menos FN (63 vs 82) a costa de más FP (69 vs 57). La coincidencia validación/test le da cierta consistencia, pero con una semilla es un indicio, no una conclusión.
3. **Grayscale conserva toda la información útil sin color.** Obtiene el AUC más alto (0.9657) sin perder frente al RGB: la señal temprana/tardía parece estar en morfología y brillo más que en el color de los cutouts (hipótesis; coherente con el resultado). Permite entradas de 1 canal. Su perfil de errores favorece la clase temprana (recall 0.911, FP 47) y pierde tardías (FN 98); su correlación con el control es la menor (0.929), o sea que es el modelo **más distinto** del RGB, interesante para ensamblar.
4. **Gamma solo desplaza el umbral.** AUC igual al control (+0.0005, correlación 0.986) pero accuracy −0.0135 y McNemar p = 0.011: sesga las probabilidades hacia temprana (recall temprana 0.919, tardía 0.869; 115 FN). Es un problema de calibración/umbral (en principio corregible con umbral elegido en validación), no de capacidad; sin corrección por múltiples comparaciones la señal es débil y no hay motivo para preferirla.
5. **NLM perjudica.** Único con AUC significativamente menor (−0.007), mayor test loss (0.2735) y menor accuracy balanceada (0.883), con el desplazamiento opuesto (recall temprana 0.831, 89 FP). Hipótesis (no verificada): el suavizado elimina textura fina (brazos, grumos, polvo) útil para separar tipos tardíos. Al no estar registrados sus parámetros no se puede ajustar ni reproducir; se sugiere descartarla salvo con parámetros más suaves.
6. **Sin diferencias de sobreajuste.** Brecha train/val pequeña (≤ 0.03 en la mejor época) en todas, igual que la CNN base (§4); el preprocesamiento no cambia la dinámica de entrenamiento.
7. **Ensamble exploratorio.** El promedio de probabilidades de las cuatro variantes da AUC 0.9691 / accuracy 0.9116 en test (con el control: 0.9690 / 0.9138), frente a 0.9623 / 0.9009. Es descriptivo (sin IC ni pesos elegidos en validación) y sobre test, así que **no sirve para elegir modelo**, pero sugiere que las variantes cometen errores parcialmente distintos y que ensamblar vistas preprocesadas podría aportar más que cualquier transformación individual.

### 12.5 Conclusiones y próximos pasos

- **Conclusión:** el preprocesamiento global **no es una palanca fuerte** aquí. Percentile y grayscale son neutras o ligeramente positivas, gamma es neutra en AUC y NLM es perjudicial. No hay evidencia para cambiar el pipeline de imagen de las demás ramas; el resultado refuerza que el límite está en la frontera morfológica (Ttype 0–1, §8.4) más que en la presentación de la imagen.
- **Prioridad 1:** confirmar percentile/grayscale con 3–5 semillas (control incluido) y **ampliar el tope de épocas de percentile** o usar *scheduler*.
- **Prioridad 2:** ensamble de vistas (RGB + percentile + grayscale) con pesos elegidos en validación y evaluación pareada con IC; si se usa gamma, calibrar el umbral en validación.
- **Prioridad 3:** otros cuantiles y combinación percentile + grayscale; registrar en `config.json` los parámetros de cada transformación (incluido NLM); verificar si el efecto se mantiene con ResNet-18.
- **Limitaciones:** una semilla y un solo split; los IC reflejan solo el muestreo de test; cuatro comparaciones sin corrección; el test ya se había inspeccionado y aquí solo se usa para describir; el control (46 épocas) y las variantes (hasta 50) no tuvieron el mismo número de épocas por *early stopping*; estadísticos pareados ad hoc sobre `predictions.csv` (split `test`).

---

## Apéndice: limitaciones de este análisis

- Basado en una corrida por configuración (semilla 42) y en `results.json`/`history.csv` existentes; §1–§7 no usan predicciones por galaxia (§8 sí).
- Los errores estándar de AUC en §1–§7 y §9 son aproximaciones analíticas (Hanley–McNeil); los IC bootstrap pareados solo existen en §8.
- Los resultados de `test` se habían inspeccionado antes del estudio de §8; allí se usan de forma descriptiva y ningún α o umbral se eligió con test.
- §10 usa **solo `validation`** (nunca `test`) y 257 desacuerdos; sus cifras no son comparables con las de test de §3–§9, y los selectores se evalúan con OOF solo del selector (las redes base no se reentrenaron con validación cruzada).
- §11 se calculó ad hoc sobre `predictions.csv`/`results.json` (IC bootstrap pareados, McNemar exacto, correlaciones) sin notebook de análisis; las fusiones por tokens usan encoders ya entrenados en el mismo split de train y congelados (distinto de las fusiones de §2, que se entrenan de forma conjunta).
- Según los notebooks, las fusiones y el control usan un ResNet-18 sin preentrenar; la CNN visual es la definida en `experimento_binario_imagenes_gpu_acc.ipynb`.

---

## Anexo A. Conclusiones de las libretas de Alexander (`notebook/*.ipynb`)

Este anexo **se agrega al final sin modificar §1–§12 ni el apéndice anterior**. Reúne todas las conclusiones que se pueden sacar de las 12 libretas sueltas de la carpeta `notebook/` (autor: Alexander Carreño), escritas para que **personas no técnicas** puedan entenderlas. En cada conclusión se indica:

- 📍 **dónde está**: la libreta (en el título de cada subsección), la sección con su título tal como aparece en la libreta y el número de celda (contando desde 0, en el orden en que aparecen en el archivo `.ipynb`). Los números de celda corresponden a la versión de las libretas del commit `b70e18e` (2026-11-09); si una libreta cambió, busque por el título de la sección (ver [Cómo están hechas las referencias](#guía-de-lectura-por-dónde-empiezo));
- **Cálculo** (cuando hace falta explicar): qué funciones se usan y en qué archivo de `src/` o de la propia libreta se hacen las cuentas.

Las libretas `analisis_complementariedad_multimodal.ipynb` y `diagnostico_gate_multimodal.ipynb` ya están resumidas de forma técnica en §8 y §10; aquí se repiten sus conclusiones en lenguaje sencillo y con su ubicación exacta. Para una visión de conjunto de todo el documento, empiece por el [Resumen para lectores no técnicos](#resumen-para-lectores-no-técnicos).

Orden: primero la Fase 1 (datos de Kaggle / Galaxy Zoo), luego la exploración de la Fase 2 (catálogo MaNGA, archivos `.fits`, atajos, limpieza y elección de imágenes) y al final los análisis multimodales.

| # | Libreta | Tema |
|---|---|---|
| A.1 | `exploration.ipynb` | Exploración de Galaxy Zoo (Fase 1) |
| A.2 | `training_baselines.ipynb` | Modelos clásicos de referencia (Fase 1) |
| A.3 | `cnn_baseline.ipynb` | Primera red neuronal (Fase 1) |
| A.4 | `exploration_morfologia.ipynb` | Catálogo MaNGA y cortes de calidad (Fase 2) |
| A.5 | `exploration_fits.ipynb` | Archivos científicos `.fits` |
| A.6 | `normalization_fits.ipynb` | Normalización de los `.fits` |
| A.7 | `shortcuts.ipynb` | Atajos que el modelo podría aprender |
| A.8 | `image-curation.ipynb` | Máscara de la galaxia, variantes y tratamientos |
| A.9 | `variant-classification-for-images-only.ipynb` | Mejor variante de imagen |
| A.10 | `variant-classification-images-and-fits.ipynb` | ¿Aporta el `.fits`? |
| A.11 | `analisis_complementariedad_multimodal.ipynb` | Complementariedad entre imagen, espectro y poblaciones |
| A.12 | `diagnostico_gate_multimodal.ipynb` | ¿Cuándo confiar en cada fuente? |

### Glosarios

Hay dos glosarios porque los lectores tienen huecos distintos. Quien viene de la astronomía suele necesitar el primero; quien viene de la computación o del aprendizaje automático, el segundo. Los términos aparecen en orden lógico, no alfabético.

#### Glosario 1: términos de aprendizaje automático (para lectores de astronomía)

**Lo básico del entrenamiento**
- **Modelo / clasificador**: programa que, a partir de los datos de una galaxia, da la probabilidad de que sea tardía. Si la probabilidad es ≥ 0.5 se dice "tardía" (el **umbral 0.5**); si no, "temprana".
- **Entrenar**: ajustar el modelo mostrándole galaxias con su respuesta correcta para que aprenda a acertar.
- **Conjuntos de entrenamiento, validación y prueba** (*train / validation / test*): las galaxias se reparten en tres grupos. Con el primero se aprende; con el segundo se toman decisiones (cuándo parar, qué versión guardar); con el tercero se califica **una sola vez** al final, con galaxias que el modelo nunca vio. Es como estudiar con unos ejercicios, practicar con otros y presentar un examen nuevo.
- **Época**: una pasada completa por todas las galaxias de entrenamiento.
- **Semilla**: número que fija el azar (cómo se reparten las galaxias, por dónde empieza el modelo). "Una sola semilla" significa que el experimento se hizo una vez; repetirlo con otra semilla podría mover un poco los resultados.
- **Sobreajuste** (*overfitting*): el modelo memoriza las galaxias de entrenamiento en lugar de aprender reglas generales; le va muy bien con ellas y peor con galaxias nuevas. **Subajuste**: el modelo es demasiado simple o aprendió poco y falla incluso con las de entrenamiento.
- **Parada temprana** (*early stopping*): dejar de entrenar cuando el resultado en validación deja de mejorar. **Checkpoint**: la versión del modelo que se guarda (la de la mejor época).
- **Aumento de datos** (*data augmentation*): girar, voltear o cambiar el brillo de las imágenes al azar durante el entrenamiento, para que el modelo aprenda la forma y no la orientación.
- **Ablación**: quitar o poner una fuente de datos a la vez (imagen, espectro, poblaciones) para medir cuánto aporta cada una.

**Cómo se califica un modelo**
- **Exactitud** (*accuracy*): porcentaje de galaxias bien clasificadas. Puede engañar si una clase es mucho más común: con 62 % de tardías, decir siempre "tardía" ya da 62 %.
- **Exactitud balanceada** (*balanced accuracy*): promedio del acierto en cada clase por separado; no se deja engañar por la clase más común.
- **Sensibilidad** (*recall*) y **precisión** (*precision*): de las galaxias que son de una clase, qué porcentaje encuentra el modelo (sensibilidad); de las que el modelo asigna a una clase, qué porcentaje lo es de verdad (precisión). **F1** combina ambas; **F1 macro** promedia el F1 de las dos clases con el mismo peso.
- **Matriz de confusión**: tabla que cruza la clase real con la predicha; muestra cuántas tempranas se llamaron tardías y viceversa.
- **AUC** (*Area Under the ROC Curve*, área bajo la curva ROC — *Receiver Operating Characteristic*, característica operativa del receptor): si se toma al azar una tardía y una temprana, es la probabilidad de que el modelo le dé más probabilidad de "tardía" a la que sí es tardía. **0.5 = lanzar una moneda; 1.0 = nunca se equivoca al ordenarlas.** Guía: 0.7 débil, 0.9 bueno, 0.95 o más muy bueno. No depende del umbral 0.5.
- **Punto de operación**: dónde se pone el umbral. Dos modelos pueden ordenar igual de bien (mismo AUC) y aun así tener exactitudes distintas solo porque cortan en sitios diferentes.
- **Calibración**: si un modelo dice "80 % tardía", ¿de verdad son tardías ≈ 80 de cada 100 galaxias así? **ECE** (*Expected Calibration Error*, error de calibración esperado) y **Brier** (error cuadrático medio de la probabilidad) lo miden; menor es mejor. **Sobreconfiado**: da probabilidades más extremas de lo que justifica su acierto real.

**Cómo saber si una diferencia es real**
- **IC** (intervalo de confianza) al 95 %: rango de valores plausibles si se repitiera el estudio con otra muestra parecida. Si el IC de una diferencia incluye el 0, **es un empate**.
- **Valor p**: probabilidad de obtener un resultado así de fuerte por pura casualidad; p < 0.05 se considera "no explicable por azar". Un p alto no prueba que dos cosas sean iguales, solo que los datos no alcanzan para distinguirlas.
- **Bootstrap**: repetir el cálculo miles de veces sobre remuestreos aleatorios de las galaxias para medir cuánto podría variar un resultado. **Bootstrap pareado**: usa las mismas galaxias para los dos modelos que se comparan.
- **Prueba de McNemar**: compara dos modelos contando las galaxias que solo acierta uno o solo acierta el otro.
- **Prueba de permutación**: repetir el análisis con las respuestas barajadas al azar para ver qué resultado daría la pura casualidad.
- **Corrección por comparaciones múltiples** (Bonferroni, Holm): al probar muchas cosas a la vez, alguna sale "significativa" por azar; estas correcciones lo compensan.
- **Validación cruzada**: dividir las galaxias en grupos (*folds*), entrenar con todos menos uno y evaluar con el que quedó fuera, rotando. Así cada galaxia se evalúa con un modelo que no la vio. **OOF** (*out-of-fold*, fuera del pliegue) son las predicciones obtenidas de esta forma.

**Tipos de modelos**
- **Red neuronal**: modelo formado por muchas capas de operaciones simples que se ajustan durante el entrenamiento.
- **CNN** (*Convolutional Neural Network*, red neuronal convolucional): red que "mira" la imagen por zonas y aprende formas (bordes, brazos, núcleos). **CNN1D**: versión de una dimensión, para espectros.
- **ResNet** (*Residual Network*, red residual): familia de CNN muy usada; ResNet-18 y ResNet-34 tienen 18 y 34 capas.
- **ImageNet**: base de millones de fotos cotidianas. Un modelo **preentrenado** ya aprendió con ellas; **congelado** significa que no se vuelve a ajustar; ***fine-tuning*** (ajuste fino) es reentrenarlo con galaxias.
- ***Linear probe*** (sonda lineal): usar una red preentrenada y congelada como "instrumento" que resume cada imagen en números, y una fórmula sencilla para decidir la clase.
- **Regresión logística**: fórmula sencilla que convierte una suma ponderada de datos en una probabilidad.
- **MLP** (*Multilayer Perceptron*, perceptrón multicapa): red neuronal simple para datos en tabla (aquí, los 90 números de poblaciones).
- **RNN / BiGRU** (*Recurrent Neural Network*, red recurrente; *Bidirectional Gated Recurrent Unit*, unidad recurrente con compuertas bidireccional): redes que leen el espectro como una secuencia. **Atención** (*attention*): mecanismo que aprende a qué partes de la entrada darles más peso.
- **Árbol de decisión, Random Forest (bosque aleatorio), XGBoost** (*eXtreme Gradient Boosting*) **y HistGradientBoosting**: modelos que deciden mediante muchas preguntas del tipo "¿este valor es mayor que X?".

**Combinar fuentes de información**
- **Modalidad**: fuente de datos (imagen, espectro, poblaciones).
- **Fusión por concatenación** (*concat*): juntar lo que extrae cada fuente y entrenar un solo modelo. **Fusión tardía**: promediar las probabilidades de modelos separados, con un peso fijo α. **Fusión por tokens**: combinar las fuentes en pedazos antes de resumirlas.
- **Oráculo**: sistema imaginario que siempre elige la fuente correcta; sirve para medir el máximo posible, pero no es real porque necesita saber la respuesta.
- **Compuerta** (*gate*) **o selector**: modelo que decide, galaxia por galaxia, a qué fuente hacerle caso. **Ganancia neta**: errores corregidos menos aciertos perdidos al usarla.
- **Atajo** (*shortcut*): pista ajena a la forma de la galaxia que permite acertar en estos datos pero falla con datos nuevos (por ejemplo, el color que el visor le da a cada telescopio).
- **Fuga de información** (*data leakage*): que el modelo vea, sin querer, algo de las galaxias con que se le califica; infla los resultados.

#### Glosario 2: términos de astronomía (para lectores de aprendizaje automático)

**Tipos de galaxias**
- **Elíptica** (E): galaxia redondeada u ovalada, sin disco ni brazos, dominada por estrellas viejas y rojas.
- **Lenticular** (S0): tiene disco pero no brazos; es intermedia entre elípticas y espirales. **S0a / Sa**: transición entre lenticular y espiral, con un núcleo (bulbo) dominante.
- **Espiral** (Sa → Sd): galaxia con disco y brazos; de Sa a Sd los brazos son cada vez más abiertos y el bulbo más pequeño. **Barrada** (SB, SAB): con una barra de estrellas que cruza el centro. **Irregular** (Irr): sin forma definida.
- **Temprana / tardía**: nombres históricos de la secuencia de Hubble. Aquí, temprana = elípticas, lenticulares y S0a; tardía = espirales e irregulares. No indican edad.
- **`Ttype`** (escala de de Vaucouleurs): número que ordena la secuencia de Hubble: −5 = elíptica, −2 = lenticular, 0 = S0a, 1 = Sa … 7 = Sd, 9 = Sm, 10 = irregular; en este catálogo, 11 = fusiones o espirales sin subtipo. Temprana = `Ttype ≤ 0`; tardía = `1 ≤ Ttype ≤ 10`.
- **Bulbo**: componente central, esferoidal y de estrellas viejas. **Disco**: componente aplanada donde están los brazos y se forman estrellas nuevas.
- **Galaxias en interacción / fusión** (*merger*): dos galaxias que chocan o se unen; deforman su morfología.

**Propiedades medidas**
- **Corrimiento al rojo** (*redshift*, `z`, `nsa_z`): cuánto se estira la luz de la galaxia por la expansión del universo; a estas distancias es proporcional a la distancia. Aquí todas están cerca (z ≤ 0.15).
- **Masa estelar** (`LogMass`): masa total de las estrellas, en logaritmo base 10 de masas solares (10 = diez mil millones de veces la masa del Sol).
- **Inclinación** (`incl`): ángulo con que vemos el disco; 0° = de frente, 90° = de canto. Una espiral de canto se ve alargada y puede parecer otra cosa.
- **Radio de Petrosian** (`PETRO_TH90`): radio que encierra el 90 % de la luz de la galaxia, en segundos de arco; mide su tamaño aparente.
- **Segundo de arco** (″): 1/3600 de grado; unidad del tamaño aparente en el cielo.
- **Color** (g − r, r − z): diferencia de brillo entre dos filtros; un valor mayor indica una galaxia más roja (más estrellas viejas).
- **Concentración**: qué tan concentrada en el centro está la luz; las elípticas son más concentradas que los discos.
- **Brillo superficial**: brillo por unidad de área del cielo; describe qué tan "intensa" se ve cada zona de la galaxia.
- **Metalicidad**: proporción de elementos más pesados que el helio en las estrellas.
- **Sesgo de Malmquist**: a mayor distancia solo se alcanzan a ver las galaxias más brillantes y masivas; crea relaciones falsas entre distancia y masa.

**Imágenes y observaciones**
- **Legacy Survey**: gran programa que fotografió buena parte del cielo; de ahí salen las imágenes. **DR** (*Data Release*): cada versión publicada de sus datos (DR9, DR10, DR11). **Variante**: combinación de versión y filtros (por ejemplo `ls-dr9_grz`).
- **Filtros g, r, i, z**: dejan pasar solo un rango de colores: g = verde-azul, r = rojo, i = rojo lejano, z = infrarrojo cercano. La imagen a color (**RGB**, *Red-Green-Blue*) es de "falso color": z → rojo, r → verde, g → azul.
- **Norte / sur**: las imágenes del norte vienen de los telescopios de Kitt Peak (BASS — *Beijing–Arizona Sky Survey* — y MzLS — *Mayall z-band Legacy Survey*) y las del sur de la cámara DECam (*Dark Energy Camera*) en Chile. Tienen filtros ligeramente distintos.
- **Recorte** (*cutout*): imagen pequeña centrada en una galaxia, extraída de las imágenes completas del cielo.
- **FITS** (*Flexible Image Transport System*): formato científico que guarda la cantidad real de luz medida en cada filtro. El **`.jpg`** es la versión "bonita" que genera el visor web.
- **Nanomaggie**: unidad de cantidad de luz usada por estos catálogos.
- **PSF** (*Point Spread Function*, función de dispersión de punto): cuánto se "emborrona" una estrella en la imagen por la atmósfera y el telescopio; fija el detalle más fino visible.
- **Extinción galáctica**: el polvo de nuestra propia galaxia atenúa y enrojece la luz de las galaxias lejanas.
- **Ruido del cielo**: fluctuación aleatoria del fondo de la imagen contra la que se compara la luz de la galaxia.

**Espectros y poblaciones**
- **MaNGA** (*Mapping Nearby Galaxies at Apache Point Observatory*): proyecto que obtuvo espectros de miles de galaxias cercanas; de él sale el catálogo. **MPL-11**: versión de sus datos.
- **Espectro**: cómo se reparte la luz de la galaxia por longitud de onda (color fino); revela la edad de sus estrellas, su gas y su movimiento. **Espectro FoV** (*Field of View*, campo de visión): el observado, integrado en todo el campo de MaNGA. **Espectro STARLIGHT (SL)**: el modelo que ajusta el programa STARLIGHT a ese espectro.
- **STARLIGHT**: programa que descompone el espectro en una mezcla de poblaciones de estrellas de distintas edades y metalicidades. **Poblaciones estelares**: los 90 números resultantes (cuánta luz y masa aporta cada población). No son independientes del espectro: salen de él.
- **Quiebre de 4000 Å, Hα, Dn4000, Hδ_A, Mg b**: rasgos del espectro que indican edad estelar y formación de estrellas.

### A.1 `exploration.ipynb` — Exploración del dataset de Kaggle (Galaxy Zoo, Fase 1)

**Contexto en palabras simples:** en la primera fase del proyecto se usaron fotos de galaxias del concurso *Galaxy Zoo* de Kaggle. Miles de voluntarios votaron si cada galaxia se veía "suave y redonda" (elíptica) o "con disco o brazos" (espiral). Esta libreta revisa esos datos antes de entrenar nada.

1. **Los datos vienen limpios: no hay valores faltantes ni galaxias repetidas.** Las 61,578 galaxias del archivo de etiquetas tienen todas sus 37 columnas completas y ningún identificador duplicado.
   - 📍 Sección *"Carga de `training_solutions_rev1.csv` e inspección general"* (celdas 4–6 y nota de la celda 8).
   - Cálculo: `df.isnull().sum().sum()` (cuenta vacíos) y `df['GalaxyID'].duplicated().sum()` (cuenta repetidos) dentro de la propia libreta.

2. **Cada galaxia con etiqueta tiene su foto, y viceversa.** Hay exactamente 61,578 filas y 61,578 imágenes de entrenamiento. Las 79,975 imágenes de la carpeta de "test" de Kaggle **no tienen respuesta conocida**, por lo que no sirven para medir al modelo y no se usan.
   - 📍 Misma sección, celda 7 ("Consistencia con las imágenes en disco") y nota de la celda 8.
   - Cálculo: se comparan los conjuntos de identificadores del CSV contra los nombres de archivo `*.jpg` en disco (`TRAIN_IMAGES_DIR.glob`).

3. **Casi todos los objetos son galaxias de verdad (no estrellas ni defectos de imagen).** La columna que mide "es una estrella o un artefacto" está casi siempre en cero. Las otras dos (suave vs. disco) tienen muchas galaxias con votos muy claros, pero también un grupo importante con votos repartidos (los voluntarios no se pusieron de acuerdo).
   - 📍 Sección *"Distribución de vote fractions — Task 1"* (celda 10, histogramas, y lectura en la celda 11).
   - Cálculo: histogramas con `sns.histplot` de `Class1.1`, `Class1.2`, `Class1.3`; además se verifica que las tres sumen 1 (fracciones de votos).

4. **Al exigir que al menos el 80 % de los voluntarios coincida, se descarta el 60.6 % de las galaxias.** Quedan 8,132 elípticas (13.2 %) y 16,141 espirales (26.2 %), y se desechan 37,305 galaxias "ambiguas". Es una decisión consciente: menos datos, pero etiquetas mucho más confiables (mismo criterio usado en la literatura científica: Walmsley et al. 2022).
   - 📍 Sección *"Etiquetado binario (elíptica vs. espiral) con umbral de vote fraction confiable"* (celdas 12–14).
   - Cálculo: función `derive_binary_label` de `src/data/labeling.py`, con el umbral `CLEAN_LABEL_THRESHOLD = 0.8` definido en `src/config.py`.

5. **Hay el doble de espirales que de elípticas (desbalance moderado, 1.98 a 1).** De 24,273 galaxias limpias, el 66.5 % son espirales. No es un desbalance extremo, pero sí suficiente para que un modelo "flojo" que diga siempre "espiral" aparente un 66 % de aciertos. Por eso se decidió dar más peso a la clase minoritaria durante el entrenamiento (`class_weight`, peso por clase) y repartir los datos manteniendo las proporciones (división estratificada).
   - 📍 Sección *"Balance de clases resultante (subconjunto limpio)"* (celda 16 y conclusión en la celda 17).
   - Cálculo: `value_counts()` sobre la columna `label` y cociente mayoritaria/minoritaria.

6. **Las galaxias ambiguas tienen sentido visual.** Al mirar ejemplos, las elípticas se ven como manchas suaves y redondas; las espirales muestran brazos o disco; las ambiguas suelen ser galaxias con disco tenue sobre un núcleo brillante (posibles lenticulares, una forma intermedia), objetos pequeños o poco brillantes, o votos repartidos. Esto respalda el umbral de 80 %.
   - 📍 Sección *"Muestras de imágenes por clase (inspección visual cualitativa)"* (celda 19 y nota de la celda 20).
   - Cálculo: función local `show_class_grid`, que dibuja 5 galaxias al azar (semilla fija) de cada clase.

7. **Distinguir elíptica de espiral es el problema "fácil" del área.** La literatura reporta ≈ 99 % de exactitud en esta tarea de dos clases, frente a ≈ 82 % cuando se usan tres clases (Barchi et al. 2020). Esto fija expectativas altas para los modelos de la Fase 1.
   - 📍 Nota cualitativa de la celda 20 (misma sección que el punto anterior).

8. **Las imágenes son homogéneas y no están dañadas.** En una muestra aleatoria de 500 fotos, todas miden 424×424 píxeles, son a color (RGB, *Red-Green-Blue*: rojo-verde-azul) y tienen valores de brillo entre 0 y 255; ninguna está corrupta. El brillo promedio es bajo (≈ 9.7 de 255): la mayor parte de cada foto es cielo negro.
   - 📍 Sección *"Estadísticas de las imágenes (tamaño, canales, rango de píxeles, calidad)"* (celdas 22–24).
   - Cálculo: bucle que abre cada imagen con `PIL.Image` y guarda ancho, alto, modo de color y mínimo/máximo/promedio de píxeles.

9. **La división en entrenamiento / validación / prueba (70 % / 15 % / 15 %) conserva exactamente la proporción de clases** (33.5 % elípticas y 66.5 % espirales en los tres grupos: 16,991 / 3,641 / 3,641 galaxias). Así, las tres partes son comparables y la evaluación final es justa.
   - 📍 Sección *"Split train/val/test estratificado (subconjunto limpio)"* (celdas 26–27) y *"Conclusiones"* (celda 28).
   - Cálculo: `train_test_split` de scikit-learn, aplicado dos veces con `stratify=label` y la semilla `RANDOM_SEED` de `src/config.py`.

### A.2 `training_baselines.ipynb` — Modelos clásicos (no profundos) como punto de partida

**Contexto en palabras simples:** antes de usar redes neuronales, se probó qué tan bien funcionan métodos de aprendizaje automático "clásicos" usando directamente los píxeles de una versión reducida de cada imagen. Sirven de "piso": cualquier modelo más sofisticado debe superarlos.

1. **El mejor modelo clásico es HistGradientBoosting (un conjunto de árboles de decisión que se corrigen unos a otros).** En validación logró 87.0 % de exactitud y AUC (*Area Under the ROC Curve*, área bajo la curva ROC — *Receiver Operating Characteristic*; mide de 0.5 = azar a 1.0 = perfecto qué tan bien el modelo ordena a las galaxias de una clase por encima de las de la otra) de 0.932. Le siguen Random Forest (bosque aleatorio, AUC 0.903), KNN (*K-Nearest Neighbors*, vecinos más cercanos, 0.879), regresión logística (0.865) y SVM (*Support Vector Machine*, máquina de vectores de soporte) lineal (0.864).
   - 📍 Sección *"Entrenamiento y comparación de modelos"* (celda 8, tabla resumen) y gráficas en *"Gráficas de desempeño"* (celdas 10–13).
   - Cálculo: cada imagen se reduce a 64×64 píxeles y se aplana (`extract_pixel_features` en `src/data/features.py`); los cinco modelos vienen de `get_baseline_models` en `src/models/sklearn_baselines.py`, todos con el mismo preprocesamiento (`StandardScaler`, que estandariza, + PCA — *Principal Component Analysis*, análisis de componentes principales — que resume los 12,288 píxeles en 150 números). Las métricas se calculan con `evaluate_binary_classifier` de `src/evaluate.py`.

2. **En la prueba final (datos nunca vistos), el ganador mantiene su nivel: 86.95 % de exactitud y AUC 0.937.** No hubo "sorpresa" entre validación y prueba, lo que indica que la elección del modelo no fue engañosa.
   - 📍 Sección *"Evaluación final del mejor modelo sobre test"* (celdas 15–16).

3. **Los modelos clásicos se equivocan más con las elípticas (la clase minoritaria).** El ganador reconoce 83.6 % de las elípticas y 88.6 % de las espirales; Random Forest y SVM solo reconocen 61–66 % de las elípticas en validación. Es decir, con pocos datos de una clase, los modelos simples tienden a favorecer la clase más común.
   - 📍 Celdas 8 (columnas `recall_elliptical` / `recall_spiral`) y 15 (reporte por clase en test).
   - Cálculo: "recall" (sensibilidad) = proporción de galaxias de una clase que el modelo identifica correctamente; viene del reporte de `evaluate_binary_classifier`.

4. **Estos resultados quedan por debajo de lo que logra la literatura con redes neuronales (≈ 94.5–99 %), como se esperaba**: confirman que la "tubería" de datos funciona y fijan el piso que debe superar la red neuronal.
   - 📍 Sección *"Conclusiones"* (celda 17).

### A.3 `cnn_baseline.ipynb` — Primera red neuronal convolucional (Fase 1)

**Contexto en palabras simples:** una CNN (*Convolutional Neural Network*, red neuronal convolucional) es un tipo de modelo que "mira" la imagen por zonas y aprende formas (bordes, brazos, núcleos). Aquí se entrenó una pequeña desde cero.

1. **Una CNN pequeña supera ampliamente a todos los modelos clásicos.** En la prueba final logra 96.8 % de exactitud y AUC 0.996, frente a 86.95 % y 0.937 del mejor modelo clásico (HistGradientBoosting), usando exactamente las mismas galaxias de prueba.
   - 📍 Secciones *"Evaluación final sobre test"* (celda 17) y *"Comparación con el mejor baseline de sklearn"* (celda 20).
   - Cálculo: la arquitectura `BaselineCNN` (`src/models/baseline_cnn.py`, 4 bloques convolucionales, 422,530 parámetros) se entrena con `fit` de `src/train.py` y se evalúa con `evaluate` (`src/train.py`) + `evaluate_binary_classifier` (`src/evaluate.py`).

2. **Casi nunca confunde una espiral con una elíptica, pero sí algunas elípticas con espirales.** Reconoce el 99.4 % de las espirales (solo 14 errores de 2,421) y el 91.6 % de las elípticas (103 errores de 1,220). Las elípticas siguen siendo la clase más difícil.
   - 📍 Sección *"Evaluación final sobre test"* (celda 17, matriz de confusión `[[1117, 103], [14, 2407]]` y reporte por clase; gráfico en la celda 18).
   - Cálculo: matriz de confusión (tabla que cruza la clase real con la predicha) de `evaluate_binary_classifier`.

3. **Bastan pocas épocas (vueltas completas sobre los datos de entrenamiento) para aprender la tarea.** Con solo 8 épocas (~2 minutos en GPU — *Graphics Processing Unit*, tarjeta gráfica) el AUC de validación pasa de 0.970 en la primera época a ≈ 0.995. La pérdida de validación (medida del error) oscila bastante entre épocas (p. ej. sube a 0.46 en la época 4 y vuelve a 0.10), por eso se guarda la época con mejor AUC de validación en lugar de la última.
   - 📍 Secciones *"Entrenamiento"* (celda 12, registro por época) y *"Curvas de entrenamiento"* (celda 14).

4. **Se aplicó aumento de datos agresivo (girar, voltear, desplazar y cambiar brillo/contraste las fotos al azar),** porque una galaxia no tiene una orientación "correcta" en el cielo; así el modelo aprende la forma y no la posición.
   - 📍 Sección *"Efecto del data augmentation"* (celda 8) y *"Datasets y DataLoaders"* (celdas 5–6).
   - Cálculo: `get_train_transforms` / `get_eval_transforms` de `src/data/augmentation.py`; imágenes cacheadas en memoria con `InMemoryGalaxyDataset` (`src/data/torch_dataset.py`).

5. **Se compensó el desbalance de clases dando más peso a las elípticas** (peso 1.49 para elípticas vs. 0.75 para espirales) en la función de error.
   - 📍 Sección *"Entrenamiento"* (celdas 11–12).
   - Cálculo: `compute_class_weights` de `src/train.py`.

6. **Este resultado es un primer punto de referencia, no el final de la Fase 1.** Está dentro del rango de la literatura (94.5–99 %) y el siguiente paso propuesto era aprovechar redes ya preentrenadas (*transfer learning*, aprendizaje por transferencia, p. ej. ResNet50).
   - 📍 Sección *"Conclusiones"* (celda 21).

### A.4 `exploration_morfologia.ipynb` — Exploración del catálogo MaNGA (Fase 2)

**Contexto en palabras simples:** en la Fase 2 se trabaja con un catálogo de 10,126 galaxias del proyecto MaNGA (*Mapping Nearby Galaxies at Apache Point Observatory*, cartografía de galaxias cercanas), compartido por el Instituto de Astronomía de la UNAM (Universidad Nacional Autónoma de México). Ahora la pregunta es si una galaxia es **temprana** (elípticas y lenticulares: redondas, viejas, sin brazos) o **tardía** (espirales e irregulares). La etiqueta sale de la columna `Ttype`, un número que ordena las formas de galaxia de −5 (elíptica) a 10 (irregular).

1. **El catálogo no tiene galaxias repetidas,** ni por nombre ni por posición en el cielo.
   - 📍 Sección *"Duplicados"* (celda 17) y *"Conclusiones"* (celda 63, punto 1).
   - Cálculo: `df["name"].duplicated()` y `df.duplicated(subset=["objra", "objdec"])`.

2. **Faltan muy pocos datos (< 0.4 % en el peor caso) y nunca en la etiqueta.** Faltan 33 radios de galaxia (`PETRO_TH90`), 28 masas (`LogMass`), 28 inclinaciones (`incl`) y 1 distancia (`nsa_z`). El nombre, el tipo y el `Ttype` están completos, así que no hace falta "rellenar" datos de forma agresiva.
   - 📍 Sección *"Valores faltantes"* (celdas 10–12).
   - Cálculo: `df.isna().sum()` y `df.isna().mean()`.

3. **Hay un error oculto: 27 galaxias tienen la distancia marcada como −9999.** Ese número es un "código de dato no disponible" del catálogo, no una medida real. Si no se limpia, distorsiona por completo los promedios (el promedio de `nsa_z` sale −26.6 en vez de 0.046). Se creó una versión limpia (`nsa_z_clean`) que trata ese código como dato faltante.
   - 📍 Sección *"Valores faltantes"*, celdas 13–15 (*"Hallazgo de calidad de datos"*).
   - Cálculo: búsqueda de valores menores a −100 en las columnas numéricas y `replace(-9999.0, np.nan)`.

4. **El `Ttype` toma valores enteros con "picos", no es una medida continua.** Cada valor corresponde a un tipo concreto de galaxia (E, S0, Sa, …, Irr). Por eso la frontera temprana/tardía cae entre categorías bien definidas del catálogo.
   - 📍 Sección *"Distribución de `Ttype` (tipo numérico de Hubble)"* (celdas 19–20).

5. **La clasificación detallada (`Type`) tiene 50 subtipos con muchos casi vacíos** (S0: 1,038; Sc: 619; Sbc: 566; pero decenas con 1–5 galaxias). Es demasiado detallada para entrenar un modelo directamente; por eso el proyecto la reduce a dos clases.
   - 📍 Sección *"Distribución de `Type` (clase morfológica detallada)"* (celdas 22–23).

6. **Hay un desbalance moderado: ≈ 63 % tardías y ≈ 37 % tempranas.** Es más suave que en otros catálogos y se maneja con pesos por clase y división estratificada; no hace falta generar datos sintéticos (como SMOTE — *Synthetic Minority Over-sampling Technique*, técnica de sobremuestreo sintético de la minoría).
   - 📍 Sección *"Etiquetado binario temprana/tardía"* (celdas 25–27).
   - Cálculo: `np.select` con los umbrales `TTYPE_EARLY_MAX = 0` y `TTYPE_LATE_MIN = 1` de `src/config.py`.

7. **67 galaxias tienen `Ttype = 11`, un valor que el correo de la UNAM no contempla** (declara tardías solo de 1 a 10). Son fusiones/galaxias en interacción (`S-merger`, `interacting`) o espirales sin subtipo (`S`, `SAB`). Quedaron pendientes de confirmar con J. A. Vázquez y, después, se decidió excluirlas.
   - 📍 Celda 29 (nota tras el etiquetado), celda 31 (tabla `Type` vs `Ttype`, últimas filas) y *"Conclusiones"* (celda 63, punto 3).
   - ⚠️ **Observación al revisar la libreta:** en la salida guardada de la celda 25, las 10,126 galaxias quedan como temprana (3,741) o tardía (6,385) y la celda 28 no encuentra ninguna "fuera de rango". Es decir, **con el código tal como se ejecutó, las 67 galaxias con `Ttype = 11` se contaron como tardías**, porque la regla solo revisa `Ttype ≥ 1` y no el tope de 10. El texto de la libreta dice lo contrario. Esto no afecta al conjunto curado final (que sí las elimina en la sección de cortes de calidad), pero los porcentajes 63.1 % / 36.9 % de esa celda incluyen esas 67 galaxias.

8. **El `Ttype` es coherente con la clasificación detallada.** Todas las elípticas tienen −5, las lenticulares −2, y las espirales van subiendo en orden (Sa = 1 … Sd = 7) hasta las irregulares (10). Es una buena señal de calidad del catálogo.
   - 📍 Sección *"Consistencia `Ttype` vs. `Type` (chequeo del catálogo)"* (celdas 31–32).
   - Cálculo: `groupby("Type")["Ttype"]` con media, mediana, mínimo y máximo.

9. **Todas las galaxias están relativamente cerca** (corrimiento al rojo — *redshift*, medida de distancia — mediana 0.038 y máximo 0.15). A esas distancias las imágenes tienen buena resolución para ver la forma; para las más lejanas puede ser más difícil.
   - 📍 Sección *"Redshift (`nsa_z`, centinela limpiado)"* (celdas 34–35).

10. **Las galaxias tempranas son, en promedio, más masivas que las tardías** (mediana de masa en escala logarítmica 10.67 vs 9.97, es decir ≈ 5 veces más masa), como predice la astronomía. Pero ambas distribuciones se traslapan mucho: **la masa sola no basta para separarlas**; sirve como comprobación, no como sustituto de la imagen.
   - 📍 Sección *"Masa estelar (`LogMass`) por clase temprana/tardía"* (celdas 37–39).
   - Cálculo: `groupby("morph_label")["LogMass"].describe()` e histogramas por clase.

11. **Las galaxias se ven desde todos los ángulos (inclinación de 0° a ≈ 81°).** Una espiral vista "de canto" puede parecer una elíptica alargada en la foto, así que la inclinación es una posible fuente de errores del modelo de imágenes (no un problema de calidad de los datos).
   - 📍 Sección *"Inclinación del disco (`incl`)"* (celdas 41–42).

12. **El tamaño aparente de las galaxias varía muchísimo** (radio mediano 13 segundos de arco, máximo 148). Por eso **no se puede descargar la imagen con un zoom fijo**: las galaxias grandes saldrían recortadas y las pequeñas como un punto. El zoom debe calcularse por galaxia a partir de `PETRO_TH90`, como advirtió el correo de la UNAM.
   - 📍 Sección *"Radio de Petrosian (`PETRO_TH90`) — define el zoom del cutout de imagen"* (celdas 44–45) y *"Conclusiones"* (punto 6).

13. **Las galaxias cubren solo ciertas regiones del cielo** (las franjas que observó el telescopio de SDSS — *Sloan Digital Sky Survey*). No introduce sesgo de forma, y todas caen dentro de la zona con imágenes de Legacy Survey.
   - 📍 Sección *"Cobertura en el cielo (`objra` / `objdec`)"* (celdas 47–48).

14. **Masa y distancia están fuertemente relacionadas (correlación ≈ 0.70), pero por un efecto de observación, no por la forma**: a mayor distancia solo se alcanzan a ver las galaxias más grandes y brillantes (sesgo de Malmquist). El `Ttype` se relaciona de forma moderada y negativa con la masa, y positiva con la inclinación.
   - 📍 Sección *"Correlación entre variables numéricas"* (celdas 50–51).
   - Cálculo: matriz de correlación de Pearson con `df[...].corr()` dibujada como mapa de calor.

15. **Los cortes de calidad acordados con la UNAM eliminan solo ≈ 1.1 % de las galaxias (10,126 → 10,014) sin desbalancear las clases.** Los cortes son: quitar `Ttype > 10` (67 galaxias, el más grande), distancias fuera de 0–0.15 (29, que ya caen también por masa) y masas fuera de 8–12 en escala logarítmica (45 en total, 16 nuevas). Quitar galaxias con inclinación 0 solo quitaría 7 más, así que esa decisión puede tomarse revisando las imágenes, sin preocuparse por perder datos.
   - 📍 Sección *"Cortes de calidad — impacto antes de aplicarlos"* (celdas 52–56) y *"Aplicación de los cortes acordados → `data/morfologia_curada.csv`"* (celdas 60–62).
   - Cálculo: `quality_cut_masks`, `quality_cut_report` y `build_curated_morfologia` en `src/data/morfologia.py`; umbrales `NSA_Z_MIN/MAX` y `LOGMASS_MIN/MAX` en `src/config.py`. (Nota: las celdas 53–62 aparecen sin salida guardada en la libreta; las cifras provienen del texto de la celda 56.)

### A.5 `exploration_fits.ipynb` — Exploración de los archivos científicos `.fits`

**Contexto en palabras simples:** las fotos de las galaxias se pueden bajar de Legacy Survey en dos formatos: un `.jpg` (la foto "bonita" a color que ve la red neuronal) y un `.fits` (FITS, *Flexible Image Transport System*: el formato científico que guarda la cantidad de luz real medida en tres filtros, g = verde-azul, r = rojo y z = infrarrojo cercano). Esta libreta compara ambos con una muestra de 50 galaxias (25 tempranas y 25 tardías).

1. **Los 50 archivos `.fits` están completos y sin errores.** Todos tienen 3 capas (una por filtro) de 256×256 píxeles, sin zonas vacías, y su escala coincide exactamente con la registrada al descargarlos. Cada uno pesa ≈ 773 KB, unas 35 veces más que el `.jpg` (~22 KB).
   - 📍 Sección *"1. Inventario: qué hay en disco y qué trae cada archivo"* (celdas 5–11) y *"10. Conclusiones"* (punto 1).
   - Cálculo: `cutout_stats`, `load_fits_cutout` y `fits_pixscale` de `src/data/cutout_io.py`.

2. **El zoom de cada galaxia es distinto** (de 0.12 a 0.6 segundos de arco por píxel) porque se ajustó al tamaño de cada galaxia (`PETRO_TH90`). Además, los valores **no vienen normalizados**: el punto más brillante varía hasta 100 veces entre galaxias.
   - 📍 Sección 1, celdas 9–11.

3. **Los `.fits` vienen "de cabeza" respecto a los `.jpg`.** En el formato científico la primera fila de la imagen es la de abajo (sur); en un `.jpg` es la de arriba. Al voltearlos, la similitud píxel a píxel con el `.jpg` sube de 0.73 a 0.98 (en promedio), y mejora en las 50 galaxias. Este error no se había detectado porque nada comparaba ambos formatos; ya se corrigió en el código.
   - 📍 Sección *"2. Orientación: el `.fits` llega volteado respecto al `.jpg`"* (celdas 14–16).
   - Cálculo: función local `render_correlation` (correlación entre el brillo del `.jpg` y el del `.fits`), y la corrección en `fits_cube_to_rgb` / `fits_to_image_orientation` de `src/data/cutout_io.py`.

4. **La forma de la galaxia es la misma en ambos formatos**: brazos, barras, discos de canto y estrellas vecinas aparecen tanto en el `.jpg` como en el `.fits`. Por tanto, lo que la red necesita para reconocer la forma ya está en el `.jpg`.
   - 📍 Sección *"3. `.fits` junto al `.jpg` de la misma galaxia"* (celdas 19–21, lectura en la celda 20).

5. **Cada filtro muestra cosas distintas.** En el filtro azul (g) resaltan los brazos y las zonas donde nacen estrellas (típicas de tardías); en el infrarrojo (z) domina el núcleo y las galaxias se ven más lisas. Una galaxia temprana se ve casi igual en los tres filtros. El filtro z es el más ruidoso y a veces trae defectos del instrumento (rayas) que el `.jpg` disimula.
   - 📍 Sección 3, celda 20.

6. **Unidades: cada píxel guarda brillo "por píxel original del telescopio", sin importar el zoom pedido.** Se comprobó pidiendo la misma galaxia a dos zooms: la suma de luz bajó exactamente 4 veces y el ruido no cambió. Sin corregir esto, las galaxias pequeñas parecerían hasta 1.5 magnitudes (≈ 4 veces) más débiles de lo que son.
   - 📍 Sección *"4. Unidades: nanomaggies por píxel nativo"* (celdas 22–26).
   - Cálculo: corrección `suma × (pixscale / 0.262)²` en `native_flux_factor` de `src/data/fits_photometry.py`; la tabla de la prueba está en la celda 22 (la celda 24 que repite la descarga está desactivada).

7. **La mayoría de los píxeles son cielo con ruido alrededor de cero; la galaxia ocupa una "cola" de valores que abarca de 1,000 a 10,000 veces más brillo.** Por eso un `.fits` no se puede normalizar simplemente con mínimo y máximo: hay que "estirarlo" (comprimir lo brillante) y fijar la escala con el nivel de ruido.
   - 📍 Sección *"5. Cielo y ruido por banda"* (celdas 29–30).

8. **El ruido crece del filtro azul al infrarrojo, y las regiones norte y sur del cielo tienen calidad parecida** (diferencias de profundidad de solo 0.1–0.3 magnitudes). El cielo residual en el filtro infrarrojo del sur es ≈ 6 veces el del norte, pero sigue siendo pequeño.
   - 📍 Sección 5, celdas 31–32.
   - Cálculo: `sky_background` y `surface_brightness_mag` de `src/data/fits_photometry.py`, agrupando por región (`legacy_region` de `src/data/image_quality.py`).

9. **El tono morado de las galaxias del sur en las imágenes `ls-dr10_grz` lo pone el visor web, no está en los datos.** Para la misma cantidad de luz, el visor "topa" el canal verde en ≈ 127 (en vez de 255) y el rojo en ≈ 190 en las imágenes del sur. Rojo y azul altos con verde a media escala = morado.
   - 📍 Sección *"6. Cómo convierte el visor el flujo en el `.jpg` (y el morado del sur)"* (celdas 35–37).
   - Cálculo: función local `transfer_curves` (valor mediano del `.jpg` por cada nivel de luz del `.fits`) y `color_features` de `src/data/image_quality.py` con la máscara `galaxy_color_mask` (`src/data/galaxy_mask.py`).

10. **Generar la imagen a color directamente desde el `.fits` reduce 3 veces la diferencia de color entre norte y sur** (la fracción de verde pasa de una brecha de 0.134 en el `.jpg` del visor a 0.044 en el render propio).
    - 📍 Sección 6 (tabla de la celda 37) y explicación en la celda 52.

11. **Con solo medir colores y concentración de luz ya se distinguen muy bien tempranas de tardías, incluso con 50 galaxias y sin entrenar nada.** El color g − r (cuánto más roja que azul es la galaxia) separa las clases con AUC (*Area Under the ROC Curve*, área bajo la curva ROC; 0.5 = azar, 1 = perfecto) de 0.95, y la concentración de la luz en el centro con 0.89; ambas superan a la masa del catálogo (0.76). Las tempranas son más rojas (g − r mediano 0.905 vs 0.684) y más concentradas (4.10 vs 3.14). Son buenos candidatos de "modelo de referencia físico" contra el cual comparar la red neuronal.
    - 📍 Sección *"7. Fotometría: colores y concentración por clase"* (celdas 40–43) y *"10. Conclusiones"* (punto 6).
    - Cálculo: `build_fits_photometry_table` en `src/data/fits_photometry.py` (suma la luz en un círculo de 1.5 × `PETRO_TH90`, resta el cielo, calcula colores y la concentración C = 5·log10(r80/r20), es decir, cuántas veces más grande es el radio que encierra el 80 % de la luz que el que encierra el 20 %); el AUC con su IC (intervalo de confianza) al 95 % se calcula con la función local `auc_with_ci` (bootstrap, es decir, remuestreo aleatorio de 2,000 repeticiones).

12. **El perfil de brillo confirma la diferencia física:** las tempranas tienen un núcleo ≈ 1 magnitud más brillante que cae rápido y luego "alas" extendidas (perfil típico de elíptica); las tardías tienen un centro más plano y caen más rápido por fuera (perfil de disco). Más allá de 1.5 veces el radio de la galaxia, la medición se vuelve ruidosa.
    - 📍 Sección *"8. Perfiles de brillo superficial"* (celdas 46–47).
    - Cálculo: `radial_profile`, `profile_edges` y `radius_grid` de `src/data/fits_photometry.py`.

13. **Se confirmó con datos qué significa la columna `PETRO_TH90`**: el radio que encierra el 90 % de la luz medido en el `.fits` es ≈ 0.94 veces `PETRO_TH90` (mediana). Antes solo se suponía por convención; ahora está respaldado. Los casos raros (1.3–1.48) son galaxias con una vecina brillante o recortadas por el borde.
    - 📍 Sección 8, celdas 48–49.

14. **Aun en los datos científicos hay una pequeña diferencia de color entre norte y sur**, porque se usaron telescopios y filtros distintos: en el sur el color r − z es 0.1–0.15 mayor. Ese color por sí solo adivina la región con AUC 0.71. Es decir, **el `.fits` reduce pero no elimina la "pista" norte/sur**, y hay que vigilarla si un modelo usa colores.
    - 📍 Sección *"9. Norte y sur en el dato físico"* (celdas 53–55) y *"10. Conclusiones"* (punto 5).

### A.6 `normalization_fits.ipynb` — Cómo preparar los `.fits` para la red neuronal

**Contexto en palabras simples:** una red neuronal necesita recibir números en una escala parecida para todas las imágenes. Los `.fits` vienen en unidades físicas con rangos enormes, así que esta libreta decide, paso a paso, cómo "normalizarlos" (llevarlos a una escala común) **sin perder la forma ni el color de la galaxia y sin darle a la red pistas engañosas**. Se usan las mismas 50 galaxias de la libreta anterior.

1. **El brillo crudo es inmanejable tal cual:** el punto más brillante varía 67 veces entre galaxias, y dentro de una misma imagen el núcleo es ≈ 100 veces más brillante que el disco donde están los brazos. Con una escala lineal común solo se ve el núcleo y el disco queda en negro. Además, las tempranas tienen el núcleo ≈ 2 veces más brillante (mediana 4.17 vs 2.20).
   - 📍 Sección *"2. El problema: el rango crudo"* (celdas 6–7).

2. **Hay que restar siempre el brillo residual del cielo.** Es pequeño (0.2–0.4 veces el ruido en los filtros g y r), pero en el filtro infrarrojo (z) cambia según la región: 0.04 en el norte vs 0.22 en el sur. Si no se resta, el fondo de las imágenes del sur sería más alto y **la red podría usar eso como pista de la región** en lugar de mirar la galaxia.
   - 📍 Sección *"3. Paso 1: restar el cielo"* (celdas 9–10).
   - Cálculo: `subtract_sky` de `src/data/fits_normalization.py` (mide el cielo fuera de 1.5 × `PETRO_TH90` descartando estrellas y galaxias).

3. **Se eligió el "estiramiento asinh tipo Lupton" porque es el único que muestra el disco sin saturar el núcleo y conserva el color.** La escala lineal pierde el disco; la logarítmica hace visible el ruido; el asinh aplicado por separado a cada filtro "lava" los colores hacia el gris. Con Lupton, el color medido en la imagen estirada sigue casi perfectamente al color físico (correlación de Spearman — correlación por rangos — de 0.98) y conserva su capacidad de separar clases (AUC — *Area Under the ROC Curve*, área bajo la curva ROC — de 0.93), mientras que por filtro baja a 0.85 y 0.89.
   - 📍 Sección *"4. Paso 2: cómo estirar el rango dinámico"* (celdas 12–15).
   - Cálculo: `lupton_asinh` de `src/data/fits_normalization.py` (aplica el mismo factor a los tres filtros de cada píxel); la comparación usa la función local `stretched_color` y `spearmanr` / `roc_auc_score`.

4. **La "suavidad" del estiramiento se fijó en β = 3 veces el ruido del cielo.** Con ese valor el cielo ocupa solo ≈ 6 % de la escala, el disco se ve y el núcleo no se satura. Valores más bajos llenan la imagen de "granos" de ruido; valores más altos apagan el disco externo. Es un valor razonable, no necesariamente el óptimo.
   - 📍 Sección *"5. Paso 3: la suavidad β"* (celdas 17–18).
   - Cálculo: `intensity_noise` y `reference_flux` de `src/data/fits_normalization.py`; parámetro `FITS_NORM_SOFTENING_SIGMA` en `src/config.py`.

5. **Se adoptó una escala global (igual para todo el conjunto) en lugar de una por imagen,** porque la escala por imagen "esconde" el brillo de la galaxia en el ruido: en una galaxia tenue, el cielo ocupa más espacio en la escala, y el nivel de ruido de la imagen termina separando las clases (AUC 0.667). Eso sería una pista falsa, porque el ruido depende de la observación, no de la galaxia. Con escala global, lo que separa las clases es el brillo del núcleo (AUC 0.746), que sí es una propiedad física (aunque también se relaciona con la masa y la distancia).
   - 📍 Sección *"6. Paso 4: escala global o por imagen"* (celdas 20–23).
   - Cálculo: `fit_global_reference` y `normalize_fits_cube` de `src/data/fits_normalization.py`; la tabla de "fugas" usa la función local `image_summary` (núcleo, mediana de la galaxia y ruido del cielo de cada imagen) con AUC y correlación de Spearman frente a distancia y masa. Las constantes globales deben calcularse **solo con el conjunto de entrenamiento**.

6. **Como la escala global conserva el brillo, hay que comprobar que el modelo no dependa de él**: evaluar dando el mismo peso a galaxias de igual masa y distancia (`matching_weights` en `src/data/shortcuts.py`) y, al entrenar, multiplicar el brillo de cada imagen por un factor aleatorio.
   - 📍 Sección 6, celda 23 (último párrafo).

7. **Recortar los valores al rango [−0.5, 3.0] casi no afecta a la imagen** (como mucho 0.03 % de los píxeles del filtro z). Con un tope de 1.5 se cortaban núcleos muy rojos, por eso se subió a 3.0.
   - 📍 Sección *"7. Paso 5: recorte, estandarización y almacenamiento"* (celdas 25–26).

8. **Guardar las imágenes normalizadas a media precisión (`float16`) no pierde nada útil y ahorra la mitad del espacio.** El error de redondeo máximo es ≈ 1.5 % del ruido del cielo; cada imagen pasa de 768 KB a 384 KB y el conjunto completo (~10 mil galaxias) de ~7.6 GB a ~3.8 GB.
   - 📍 Sección 7, celdas 27–28.
   - Cálculo: `channel_stats` y `standardize` de `src/data/fits_normalization.py` (llevan cada canal a media 0 y desviación 1 con estadísticas de entrenamiento).

9. **La normalización conserva la pequeña diferencia de color entre norte y sur** (en las tempranas del sur el filtro z aporta 55 % de la luz vs 49 % en el norte), porque conserva el color. Si el modelo resultara sensible a la región, se podría corregir con transformaciones de color entre telescopios, estandarización por región o pequeñas variaciones aleatorias de color; primero hay que medir si hace falta.
   - 📍 Sección *"8. ¿Y la diferencia norte/sur?"* (celdas 30–31).

10. **Proceso final adoptado:** (1) restar cielo, (2) asinh tipo Lupton con β = 3σ, (3) escala global calculada con entrenamiento, (4) recorte a [−0.5, 3.0] y misma orientación que el `.jpg`, (5) estandarización por canal y guardado en `float16`. Quedan pendientes: corregir el polvo de nuestra galaxia (extinción), la diferencia de nitidez entre imágenes (PSF — *Point Spread Function*, función de dispersión de punto), la diferencia norte/sur y probar otros valores de β con el modelo ya entrenado.
    - 📍 Sección *"9. Conclusiones y pipeline adoptado"* (celda 32).
    - Cálculo: todo el proceso está en `normalize_fits_cube` de `src/data/fits_normalization.py`.

### A.7 `shortcuts.ipynb` — "Atajos": qué podría aprender el modelo en lugar de la forma

**Contexto en palabras simples:** un "atajo" es una pista que permite adivinar la respuesta sin entender el problema. Por ejemplo, si todas las galaxias tempranas estuvieran más lejos, un modelo podría aprender "lejos = temprana" en lugar de mirar la forma, y fallaría con galaxias nuevas. Un atributo es un atajo peligroso solo si (1) predice la etiqueta y (2) se puede ver en la imagen. Esta libreta revisa cada candidato con el catálogo curado completo.

> Nota: en la libreta guardada, las celdas de cálculo de esta libreta no tienen salidas guardadas; las cifras citadas provienen de los textos de conclusión escritos en la propia libreta.

1. **Solo tres atributos predicen la etiqueta por sí solos: la inclinación, la masa y la distancia (redshift).** Los atributos propios de la imagen (región del cielo, zoom, interpolación, latitud galáctica, tamaño aparente) quedan prácticamente en azar (AUC — *Area Under the ROC Curve*, área bajo la curva ROC — entre 0.50 y 0.55).
   - 📍 Sección *"2. Panorama: ¿qué atributo predice la etiqueta por sí solo?"* (celdas 6–8).
   - Cálculo: `auc_table` de `src/data/shortcuts.py` (AUC de cada columna usada como puntaje, con IC — intervalo de confianza — al 95 % por bootstrap, es decir, remuestreo aleatorio); los atributos se agregan con `observational_attributes` del mismo archivo.

2. **La distancia NO es un atajo propio: solo "parece" predecir la clase porque las galaxias masivas se eligieron más lejos.** El catálogo MaNGA se diseñó de forma que las galaxias más masivas se observan a mayor distancia, y las tempranas son más masivas. Al comparar galaxias de **igual masa**, la distancia ya no separa nada (AUC 0.51). En cambio, a **igual distancia**, la masa sigue separando (0.59): la masa es la relación física real.
   - 📍 Sección *"3. Redshift: ¿atajo propio o reflejo de la masa?"* (celdas 10–13).
   - Cálculo: `matching_weights` (repondera las galaxias para que ambas clases tengan la misma distribución de masa o distancia) y `weighted_auc` de `src/data/shortcuts.py`.

3. **Lo que la imagen muestra de la distancia casi no predice la clase**, porque cada foto se encuadra a 4 veces el tamaño de la galaxia y todas llegan con el mismo tamaño aparente. El tamaño angular (`PETRO_TH90`) solo da AUC 0.52. Riesgo: **bajo a moderado**; conviene verificarlo con el modelo entrenado (si su AUC cae al comparar galaxias de igual distancia, el modelo estaría usando la distancia).
   - 📍 Sección 3, celda 13 (párrafo *"¿Es visible el redshift en la imagen?"* y *"Conclusión"*).

4. **La región del cielo (norte/sur) no es un atajo, pero sí un "cambio de dominio".** La proporción de tardías es prácticamente igual en ambas regiones (AUC 0.51), así que aprender "esto es del sur" no ayuda a acertar. El riesgo real es otro: como las imágenes del sur se ven distintas (tono morado en `ls-dr10_grz`), el modelo podría **funcionar peor en una región que en otra**. Esto corrige una idea previa del README (el peligro no es que la red aprenda la región, sino que generalice peor entre regiones).
   - 📍 Sección *"4. Región del survey (norte / sur)"* (celdas 15–16).
   - Cálculo: tabla cruzada `pd.crosstab` región × etiqueta; la evaluación cruzada entre regiones se propone con `cross_group_auc` de `src/evaluate.py`.

5. **≈ 70 % de las imágenes están "ampliadas" (interpoladas),** porque el zoom pedido es más fino que el píxel original del telescopio. Ocurre por igual en ambas clases, así que no es un atajo, pero significa que en esas imágenes los "detalles finos" son relleno, no información real. Las pocas galaxias en el límite del zoom (`zoom_topado`, AUC 0.55) conviene marcarlas en el control de calidad.
   - 📍 Sección *"5. Encuadre y zoom del cutout"* (celdas 18–19).

6. **La inclinación es la pista individual más fuerte (AUC 0.72) e independiente de masa y distancia, pero es engañosa.** Se calcula a partir de qué tan alargada se ve la galaxia, suponiendo que es un disco; por eso una elíptica redonda "sale" con inclinación baja. El caso peligroso son las **lenticulares (S0) vistas de canto**: son tempranas pero se ven alargadas como espirales, y un modelo que aprenda "alargada = tardía" fallará ahí. Recomendación: no quitar la elongación (es parte de la forma), sino **reportar el error por rangos de inclinación** y vigilar las S0 de canto y las espirales de frente sin brazos visibles.
   - 📍 Sección *"6. Inclinación"* (celdas 20–23).
   - Cálculo: familias E / S0-S0a / S-Irr a partir de `Ttype` con `pd.cut`, histogramas por familia y fracción de galaxias con inclinación ≥ 70°.

7. **Hay una "fuga de identidad": algunas galaxias aparecen dos veces (dos observaciones de la misma galaxia) o caen dentro de la foto de otra.** Si una queda en entrenamiento y la otra en prueba, el modelo "reconoce" la imagen en vez de clasificarla. Son pocas (< 1 % del catálogo), pero evitarlo no cuesta nada: **la división de datos debe hacerse por grupo**, no por galaxia individual.
   - 📍 Sección *"7. Galaxias repetidas y vecinas dentro del encuadre"* (celdas 25–28).
   - Cálculo: `galaxy_groups` de `src/data/shortcuts.py` (une galaxias separadas por < 3 segundos de arco o que caen en el encuadre de otra); se compara `StratifiedKFold` (división normal) con `StratifiedGroupKFold` (división que respeta grupos) de scikit-learn mediante la función local `broken_groups`.

8. **Se dejó montada una prueba para detectar atajos en las propias imágenes** (ruido del cielo, nivel del cielo y zoom), usando los colores como "control positivo" que sí deben separar. Con solo 50 imágenes únicamente detectaría señales fuertes; la propuesta es repetirla con el conjunto completo.
   - 📍 Sección *"8. Primera mirada en la imagen: ¿delatan la clase las propiedades del `.fits`?"* (celdas 30–31).
   - Cálculo: `predictability_test` de `src/evaluate.py` (regresión logística con validación cruzada repetida y valor p por permutación, es decir, comparando contra resultados obtenidos con etiquetas barajadas al azar).

9. **Protocolo de evaluación recomendado:** (1) dividir los datos por grupo y estratificado por etiqueta; (2) reportar, además del AUC global, el AUC a igual distancia y a igual masa, el AUC por región y entrenando en una región para evaluar en la otra, y el error por rangos de inclinación con las S0 de canto aparte; (3) repetir la prueba de atajos en imagen con el conjunto completo. La masa (AUC 0.72) se considera una relación física legítima, no un atajo: se compara contra ella, no se elimina.
   - 📍 Sección *"9. Conclusiones y protocolo propuesto"* (celda 32, tabla resumen).

### A.8 `image-curation.ipynb` — Limpieza de imágenes (máscara de la galaxia) y elección de variantes

**Contexto en palabras simples:** las fotos de Legacy Survey traen, además de la galaxia del centro, estrellas y otras galaxias alrededor. Se probó una "máscara" (un recorte automático) que deja solo la galaxia central y pinta el resto de negro. También se compararon 8 "variantes" de las fotos (distintas versiones del catálogo — DR9, DR10, DR11; DR = *Data Release*, versión publicada de los datos — y distintas combinaciones de filtros), con una muestra de 50 galaxias.

1. **La máscara se construye por color: la galaxia es brillante y blanco-amarillenta en el centro, el cielo es oscuro y rojizo.** El tono de la galaxia se toma del centro de la foto y el del cielo del borde, **por cada imagen** (porque el visor cambia los tonos de foto a foto). En la imagen de referencia, la galaxia cubre el 30.8 % de la foto.
   - 📍 Secciones *"1. Imagen de referencia: regiones de muestreo de color"* (celdas 4–6) y *"2. Paso a paso sobre la referencia"* (celdas 7–8).
   - Cálculo: `galaxy_color_mask` y `apply_mask` en `src/data/galaxy_mask.py` (distancia de Mahalanobis — una distancia de color que considera la variación típica de cada región — y luego quedarse con la zona conectada al centro); parámetros `GALAXY_MASK_*` en `src/config.py`.

2. **El color por sí solo no basta para quitar estrellas**, porque las estrellas azules se parecen al disco de las espirales. Lo que las elimina es quedarse solo con la zona "pegada" al centro.
   - 📍 Sección 1, subsección *"Colores de galaxia vs. cielo"* (celda 5).

3. **La máscara elimina bien las estrellas separadas, pero tiene dos limitaciones:** (a) si otra galaxia se superpone a la central (pares en interacción), queda dentro de la máscara; (b) en espirales con núcleo muy dominante, el disco externo débil se confunde con el cielo y queda fuera. En las 50 galaxias la máscara cubre entre 11 % y 43 % de la imagen (mediana 26 %), ninguna quedó vacía y 4 tocan el borde.
   - 📍 Secciones *"3. Validación visual sobre toda la muestra"* (celdas 10–12, *"Observaciones"*) y *"4. Guardar los cutouts enmascarados"* (celda 14).
   - Cálculo: `save_masked_cutout` de `src/data/galaxy_mask.py`.

4. **La máscara cubre algo más en las tempranas que en las tardías** (p. ej. 0.282 vs 0.248 en `ls-dr10_grz`; 0.304 vs 0.257 en `ls-dr9_grz`). Esto importa porque el contorno de la máscara por sí solo podría dar una pista de la clase.
   - 📍 Sección *"5. Comparación entre variantes de capa y bandas"* (celda 17, segunda tabla).
   - Cálculo: `summarize_variants` de `src/data/download_cutouts.py` y una tabla dinámica (`pivot_table`) del índice de descargas.

5. **Ninguna de las 8 variantes tiene huecos de cobertura ni máscaras vacías.** Las máscaras que tocan el borde son casi siempre las mismas 3–4 galaxias en todas las variantes: es un problema de encuadre (galaxia más grande que la foto), no de la variante; se resuelve ampliando el zoom (`CUTOUT_FOV_FACTOR`).
   - 📍 Sección *"6. Filtro 1: calidad y atajos por variante"*, subsección *"6.1 Calidad"* (celdas 20–21).
   - Cálculo: `quality_flags` de `src/data/image_quality.py`.

6. **Las variantes `ls-dr10_grz` y `ls-dr10_gri` quedan descartadas porque pintan de distinto color el cielo del norte y del sur** (morado en `grz`, verde en `gri`). El modelo vería dos "mundos" de color distintos. En las demás, las dos regiones se superponen con una diferencia pequeña (≈ 0.04–0.06). `ls-dr11_gri` pasa por muy poco (0.121 con límite 0.122) y además recorta disco externo, así que hay que tratarla con cautela.
   - 📍 Subsección *"6.2 Homogeneidad de color entre regiones"* (celdas 22–23) y *"Observaciones (muestra de 50, 2026-10-07)"* (celda 31).
   - Cálculo: `region_color_gap` de `src/data/image_quality.py` (distancia entre el tono mediano del norte y del sur; se marca como heterogénea la que supera 2 veces la mediana entre variantes, `COLOR_GAP_MAX_FACTOR`).

7. **La región del cielo no está asociada a la clase:** en el catálogo curado completo hay 62.2 % de tardías en el norte y 63.3 % en el sur. Por eso "adivinar la región" no ayuda al modelo a adivinar la clase.
   - 📍 Subsección *"6.3 Fuga norte/sur"*, paso 1 (celda 25).

8. **En todas las variantes se puede adivinar la región desde la imagen (AUC — *Area Under the ROC Curve*, área bajo la curva ROC — entre 0.8 y 1.0), incluso con la imagen enmascarada.** Es una diferencia física entre telescopios (profundidad, nitidez, filtros), no un defecto del visor, y la máscara no la quita. **No descalifica a la variante**, porque la región no predice la clase; lo que sí hay que vigilar es que el modelo funcione igual de bien en ambas regiones.
   - 📍 Subsección 6.3, paso 2 (celdas 27–28) y *"6.4 Veredicto del filtro 1"* (celdas 29–31).
   - Cálculo: `predictability_test` de `src/evaluate.py` (validación cruzada 5×10 y valor p por permutación) sobre colores (`COLOR_FEATURES`) o sobre las 512 características de una ResNet-34 (*Residual Network* de 34 capas) preentrenada en ImageNet y congelada (`build_backbone`, `embed_cutouts`, `make_logistic_probe` de `src/models/linear_probe.py`).

9. **Pasan el primer filtro:** `ls-dr9_grz`, `ls-dr10_griz`, `ls-dr10-grz_grz`, `ls-dr11_grz`, `ls-dr11_griz` y, al límite, `ls-dr11_gri`.
   - 📍 Celda 30 (tabla de veredicto, columna `pasa_filtro_1`) y celda 31.

10. **Ningún tratamiento de imagen mejora al clasificador; la mayoría lo empeora.** Se probaron 8 tratamientos sobre las 4 variantes finalistas (32 comparaciones) y ninguno salió mejor que la imagen original; ninguno la supera en más de 6 de 40 repeticiones.
    - 📍 Sección *"7. Tratamientos de imagen: ¿alguno mejora al clasificador?"* (celdas 35–39).
    - Cálculo: tratamientos en `IMAGE_TREATMENTS` de `src/data/image_treatments.py`; comparación con `multi_seed_paired_comparison` de `src/evaluate.py` (AUC con validación cruzada en 10 particiones distintas y diferencia contra la original con bootstrap pareado — remuestreo de las mismas galaxias — de 2,000 repeticiones); clasificador = ResNet-34 congelada + regresión logística (`src/models/linear_probe.py`).

11. **La máscara con fondo negro es el peor tratamiento:** pierde entre 0.10 y 0.15 de AUC en las 4 variantes, de forma significativa. Suavizar la máscara reduce la pérdida, pero no la elimina: fondo de cielo −0.07 a −0.10, borde suave −0.08 a −0.09, margen ancho −0.05 a −0.07, círculo del tamaño de la galaxia −0.04 a −0.06. **Cuanta más imagen se conserva, menos se pierde.**
    - 📍 Sección 7, tablas de las celdas 35–36, figura de la celda 37 y *"Observaciones (muestra de 50, 2026-10-27)"* (celda 39).

12. **Quitar solo las estrellas pierde poco (−0.01 a −0.05)** y empata en las variantes DR11; **aclarar la imagen con gamma (γ = 0.6) no aporta** (−0.01 a −0.02, empate).
    - 📍 Celda 39.

13. **La explicación de la pérdida: fuera de la galaxia también hay información útil.** Con la galaxia tapada en negro y solo el resto visible, el modelo todavía acierta con AUC 0.69–0.84. Lo que queda afuera es la luz débil del disco y el halo (que la máscara corta) y el entorno (galaxias vecinas). Por la relación entre forma y densidad de entorno, el entorno sí ayuda a predecir la clase.
    - 📍 Celda 39 (punto *"El control `control_solo_fondo` explica la pérdida"*).

14. **Decisión: entrenar con la imagen original, sin tratamiento.** La máscara solo se usará para **medir** (tamaño, forma, color), no como entrada del modelo. Si la UNAM exige que la entrada contenga solo la galaxia central, la opción menos costosa es `sin_estrellas`. Pregunta abierta para la UNAM: ¿el modelo debe clasificar solo la galaxia o puede apoyarse en su entorno? Además, con 50 galaxias y una red congelada, conviene repetirlo con ~2,000 galaxias y con el modelo final.
    - 📍 Celda 39 (*"Decisión"* y *"Alcance y preguntas abiertas"*) y resumen en la celda 0 (*"Resultado clave"*).

### A.9 `variant-classification-for-images-only.ipynb` — ¿Con qué versión de la imagen se clasifica mejor?

**Contexto en palabras simples:** se tomaron las mismas 50 galaxias (25 tempranas / 25 tardías) en sus 8 variantes de imagen, en versión original y enmascarada (solo la galaxia, fondo negro). Para comparar de forma justa se usó una red ResNet-34 (*Residual Network* de 34 capas) **ya entrenada con fotos cotidianas (ImageNet) y sin volver a entrenarla**: funciona como un "instrumento fijo" que resume cada imagen en 512 números; después, una fórmula sencilla (regresión logística) aprende a separar las clases con esos números. A esto se le llama *linear probe* (sonda lineal). Como el instrumento es siempre el mismo, si una variante sale mejor es por la imagen.

1. **6 de las 8 variantes pasan el control de calidad previo;** `ls-dr10_grz` (sur morado) y `ls-dr10_gri` (sur verdoso) no, porque colorean distinto según el telescopio.
   - 📍 Celdas 3–4 (*"Lectura del resultado"* tras la configuración del proyecto).
   - Cálculo: se recalcula el filtro de `image-curation.ipynb` con `quality_flags` y `region_color_gap` de `src/data/image_quality.py`.

2. **El contorno de la máscara difiere un poco entre clases, por razones físicas:** las máscaras de las tempranas son algo más grandes (ocupan 28 % vs 25 % de la imagen) y las de las tardías más alargadas (elongación 0.34 vs 0.22), como se espera de discos inclinados.
   - 📍 Sección *"1. Atajo de la máscara"*, subsección *"1.1 Silueta media por clase"* (celdas 6–7).
   - Cálculo: función local `variant_masks` + `mask_cutout` de `src/data/galaxy_mask.py`; descriptores `SHAPE_FEATURES` (área, elongación, compacidad, solidez) de `src/data/image_quality.py`.

3. **El contorno de la máscara por sí solo da, como mucho, una pista débil de la clase — no es un atajo preocupante.** Resumido en 4 números, predice con AUC (*Area Under the ROC Curve*, área bajo la curva ROC; 0.5 = azar, 1 = perfecto) de 0.58–0.71 (significativo solo en 3 variantes); como imagen de 32×32 píxeles no predice nada (0.45–0.57). La pista viene sobre todo de la elongación, que es información real de la forma. Como el clasificador con imagen enmascarada logra 0.82–0.90, usa mucho más que el contorno.
   - 📍 Subsección *"1.2 ¿La silueta sola predice la clase?"* (celdas 9–10) y *"Observaciones (muestra de 50, 2026-10-07)"* (celda 29).
   - Cálculo: `mask_silhouette` de `src/data/galaxy_mask.py` y `predictability_test` de `src/evaluate.py` (validación cruzada 5×10 y valor p por 200 permutaciones; p < 0.05 = no explicable por azar).

4. **Todas las variantes contienen información clara de la forma, pero la imagen original siempre funciona mejor que la enmascarada.** Con imagen original: AUC 0.84–0.98 y 82–92 % de aciertos; las mejores (`ls-dr9_grz`, `ls-dr10_grz`, `ls-dr10-grz_grz`) aciertan 46 de 50. Con imagen enmascarada: AUC 0.82–0.90, 74–82 % de aciertos y resultados más inestables. Todos los resultados son significativos (p = 0.005, el mínimo posible).
   - 📍 Sección *"2. Clasificador temprana/tardía (linear probe)"* (celdas 13–16).
   - Cálculo: `build_backbone`, `embed_cutouts` y `make_logistic_probe` de `src/models/linear_probe.py`; predicciones "fuera de pliegue" (*out-of-fold*: cada galaxia la predice un modelo que no la vio) con `out_of_fold_proba` e intervalo de confianza con `bootstrap_auc_ci` de `src/evaluate.py`; F1 macro (promedio, con el mismo peso por clase, de la media armónica entre precisión y sensibilidad) con `f1_score` de scikit-learn.

5. **Con solo 50 galaxias no se usó una división fija entrenamiento/validación/prueba**, sino validación cruzada: 5 grupos de 10, se aprende con 40 y se califica con 10, rotando, de modo que cada galaxia se evalúa una vez con un modelo que no la vio. No hace falta conjunto de validación porque no se ajusta nada (la red está congelada y la regularización se fijó de antemano). Con ~2,000 galaxias sí convendrá una división 70/15/15.
   - 📍 Sección 2, celda 12 (*"¿Qué porcentaje se usa para entrenamiento, validación y prueba?"*).

6. **Cambiar de variante no mejora el rendimiento, pero cinco variantes rinden igual que la actual y además sí pasan el control de calidad** (`ls-dr9_grz`, `ls-dr10-grz_grz`, `ls-dr11_grz`, `ls-dr11_griz`, `ls-dr11_gri`; diferencias de −0.002 a −0.024, todas compatibles con cero). Son claramente peores `ls-dr10_griz` (−0.09) y `ls-dr10_gri` (−0.14), y todas las enmascaradas (−0.08 a −0.16). Ninguna supera a la referencia.
   - 📍 Subsección *"2.1 Comparación pareada contra la variante de referencia"* (celdas 18–19).
   - Cálculo: `paired_bootstrap_auc_diff` de `src/evaluate.py` (bootstrap pareado: se remuestrean las mismas galaxias para ambas variantes 2,000 veces, así se cancela el efecto de "qué galaxias tocaron").

7. **El mejor conjunto que pasa el control (`ls-dr9_grz`, original) acierta 46 de 50 (92 %) y casi siempre se equivoca en la misma dirección:** 1 temprana llamada tardía y 3 tardías llamadas tempranas. Hipótesis no verificada: podrían ser tardías "de transición" (tipo Sa, con núcleo dominante) o espirales con brazos débiles; se sugiere que un astrónomo revise esas 3 galaxias.
   - 📍 Subsección *"2.2 Matriz de confusión del mejor dataset"* (celdas 21–22).
   - Cálculo: `confusion_matrix` de scikit-learn sobre las predicciones *out-of-fold* con umbral 0.5.

8. **Las buenas variantes funcionan igual con los telescopios del norte y del sur:** AUC 0.93–1.00 en cada región y 0.94–1.00 entrenando en una región y evaluando en la otra. En cambio, las variantes DR10 con filtro *i* caen en el sur (0.82 y 0.74) porque **en DR10 el filtro *i* solo existe para el telescopio del sur**, así que las imágenes se construyen distinto según la región. Esta prueba detectó en `ls-dr10_griz` un problema que el control de calidad había dejado pasar. (Con ~25 galaxias por región, solo las caídas grandes son señal clara.)
   - 📍 Sección *"3. Rendimiento por región y generalización norte ↔ sur"* (celdas 24–25) y celda 29.
   - Cálculo: `cross_group_auc` de `src/evaluate.py` (entrena en un grupo y evalúa en el otro) y `bootstrap_auc_ci` por región.

9. **La configuración que se usaba (`ls-dr10_grz`) rinde igual que las mejores (AUC 0.979) pese al tono morado del sur**, porque la región no está asociada a la clase. Se excluye por precaución (dos "estilos" de color), no porque se haya demostrado que rinde peor.
   - 📍 Sección *"4. Resumen"* (celdas 27–28) y celda 29.

10. **Finalistas: `ls-dr9_grz`, `ls-dr10-grz_grz`, `ls-dr11_grz` y `ls-dr11_griz` (imagen original).** Con 50 galaxias no hay ganador único; para separarlas hay que repetir con ~2,000 galaxias. El desacuerdo entre colaboradores sobre DR9 vs DR11 no se resuelve con rendimiento (empatan) y requiere un criterio astronómico.
    - 📍 Celda 29 (*"Siguiente paso"*) y celda 30 (*"Conclusiones en términos simples"*, puntos 4 y 6).

11. **Resultado general: las imágenes de Legacy Survey bastan para separar tempranas de tardías.** Una red que nunca vio una galaxia, más una fórmula simple, logra AUC ≈ 0.96–0.98 y 88–92 % de aciertos, cerca del rango de la literatura (94.5–99 %), sin haber entrenado aún la red con galaxias.
    - 📍 Celda 30, punto 1.

12. **Preguntas abiertas para el equipo de astronomía:** razones astronómicas para preferir DR9 o DR11; si las 3 tardías mal clasificadas son casos frontera; y qué alternativa a la máscara dura sería aceptable (rellenar el fondo con ruido de cielo, suavizar el borde o borrar solo estrellas).
    - 📍 Celda 30 (*"Preguntas abiertas para el equipo de astronomía"*).

### A.10 `variant-classification-images-and-fits.ipynb` — ¿Aporta algo el `.fits` además de la imagen?

**Contexto en palabras simples:** continuación de la libreta anterior, con el mismo método (red ResNet-34 congelada + regresión logística) y las mismas 50 galaxias, ahora en las 4 variantes finalistas más `ls-dr10_grz` como referencia. En cada variante se compararon 6 "entradas": la imagen `.jpg`, el `.fits` normalizado, 3 números de fotometría (colores g − r, r − z y concentración de la luz) y sus combinaciones (cada fuente con el mismo peso en la decisión final).

1. **El `.fits` y el `.jpg` muestran la misma galaxia; el `.fits` no tiene el tono morado del sur y conserva el brillo real entre galaxias.** Los `.fits` con filtros g, r, z de DR9, DR10 y DR11 son casi idénticos (el de `ls-dr10-grz_grz` es literalmente el mismo archivo que el de `ls-dr10_grz`).
   - 📍 Sección *"1. Qué recibe cada rama"*, subsección *"1.1 `.jpg` contra `.fits` normalizado"* (celdas 7–8) y *"Conclusión de la sección 1"* (celda 12).
   - Cálculo: `normalize_fits_cube` (`src/data/fits_normalization.py`) y `fits_rgb_channels` (`src/models/fusion_probe.py`).

2. **Defecto propio de `ls-dr11_griz`: el filtro *i* no existe en el norte.** El archivo trae ceros en ese filtro en las 23 galaxias del norte (y en 2 del sur, con otras 2 cubiertas a medias). Como *i* y *z* se promedian en el canal rojo, donde falta *i* el rojo baja a la mitad y la galaxia se ve verdosa.
   - 📍 Celda 8 (último punto) y celda 12.

3. **La fotometría da casi los mismos números en todas las variantes** (diferencias ≤ 0.006) y reproduce las diferencias físicas clásicas: las tempranas son más rojas (g − r ≈ 0.91 vs 0.67) y con la luz más concentrada en el centro (≈ 4.0 vs 3.1).
   - 📍 Subsección *"1.2 Fotometría por clase"* (celdas 10–11).
   - Cálculo: `build_fits_photometry_table` de `src/data/fits_photometry.py`.

4. **En todas las variantes, la imagen sola es la mejor entrada:** AUC (*Area Under the ROC Curve*, área bajo la curva ROC; 0.5 = azar, 1 = perfecto) entre 0.963 y 0.979 y 44–46 aciertos de 50. Estos números reproducen exactamente los de la libreta anterior, lo que confirma que ambas miden igual.
   - 📍 Sección *"2. Clasificador temprana/tardía por variante y representación"* (celdas 16–21).
   - Cálculo: `make_fusion_probe` (`src/models/fusion_probe.py`, regresión logística que iguala el peso de cada bloque con `block_weights`), `out_of_fold_proba`, `predictability_test` y `bootstrap_auc_ci` de `src/evaluate.py`.

5. **Añadir el `.fits`, la fotometría o ambos nunca mejora a la imagen:** de 25 comparaciones, 0 a favor, 19 empates y 6 significativamente peores. El `.fits` solo es peor en las 5 variantes (AUC 0.83–0.88, diferencias −0.085 a −0.131), y la combinación imagen + `.fits` llega a empeorar a la imagen en `ls-dr11_griz` (−0.053).
   - 📍 Subsección *"2.1 Comparación pareada contra `imagen` sola, dentro de cada variante"* (celdas 24–27).
   - Cálculo: `paired_bootstrap_auc_diff` de `src/evaluate.py` (diferencia galaxia por galaxia con 2,000 remuestreos; p < 0.05 = diferencia real).

6. **Tres números de fotometría clásica casi igualan a la imagen** (AUC 0.94, empate estadístico en todas las variantes). Es una validación física del problema: color y concentración son los indicadores morfológicos conocidos en astronomía.
   - 📍 Celdas 17, 25 y *"Observaciones y conclusiones"* (celda 50, punto 5).

7. **Los errores cambian de galaxia al combinar, pero su número no baja.** La imagen sola se equivoca casi siempre llamando "tempranas" a tardías (3–5 por variante). La combinación completa reparte mejor los errores, pero no comete menos en total.
   - 📍 Subsección *"2.2 Aciertos y errores"* (celdas 30–31).
   - Cálculo: `confusion_matrix` de scikit-learn por variante y entrada.

8. **Hay galaxias difíciles para cualquier versión, lo que apunta a la galaxia y no al formato:** `manga-9037-6102` y `manga-12495-1901` (tardías) y `manga-9863-6102` (temprana) fallan en (casi) todas las variantes y entradas, probablemente casos frontera. En cambio, `manga-8728-9101`, `manga-8728-6104` y `manga-12075-3704` (tardías) las confunde la imagen pero las corrigen el color y la concentración; y `manga-11979-6101` y `manga-8481-6104` las "estropea" la combinación.
   - 📍 Celda 31 (*"Tabla de galaxias"*) y celda 50 (*"Preguntas para el equipo de astronomía"*).

9. **Interpretación: la imagen del visor ya contiene lo que aportan el `.fits` y la fotometría** (forma, estructura y color). El `.fits` rinde peor por cómo se usó, no porque tenga menos información: la red se entrenó con fotos cotidianas y el `.fits` normalizado tiene otra "estadística" a la que una red congelada no puede adaptarse.
   - 📍 *"Conclusión de la sección 2"* (celda 32).

10. **El `.fits` solo no funciona bien con las galaxias del sur en ninguna variante** (AUC 0.75–0.79 en el sur y 0.79–0.81 entrenando en el norte y evaluando en el sur), porque conserva la diferencia real de filtros entre telescopios y la red no "sabe" que es la misma galaxia vista con otro filtro. La imagen, en cambio, generaliza entre telescopios (0.94–1.00).
    - 📍 Sección *"3. Rendimiento por región y generalización norte ↔ sur"* (celdas 35–36) y *"Conclusión de la sección 3"* (celda 41).
    - Cálculo: `cross_group_auc` de `src/evaluate.py`.

11. **Ninguna entrada está libre de "pistas del telescopio".** Prediciendo norte/sur: la imagen de las finalistas lo delata con AUC 0.81–0.84 (vs 1.00 en `ls-dr10_grz` por el morado); el `.fits` grz 0.77–0.79; el `.fits` de `ls-dr11_griz` 0.97 (por el filtro *i* vacío); la fotometría 0.82–0.87; y imagen + fotometría 0.95–0.98 (cada una lleva una pista débil y la fórmula las suma). Como la región no está asociada a la clase, esto no infla los resultados, pero **añadir `.fits` o fotometría reintroduce pistas del telescopio sin mejorar la clasificación**. Si se usara el `.fits`, habría que corregir antes la diferencia de filtros y no usar el filtro *i* de DR11.
    - 📍 Subsección *"3.1 ¿La representación delata la región?"* (celdas 39–41).
    - Cálculo: `predictability_test` con la región como objetivo.

12. **Prueba de honestidad: medir las constantes de normalización del `.fits` solo con las galaxias de entrenamiento de cada ronda cambia el AUC en 0.003 o menos.** El procedimiento rápido no infló los resultados; con el conjunto completo se recomienda usar siempre la versión estricta.
    - 📍 Sección *"4. Control: normalización del `.fits` medida dentro de cada fold"* (celdas 44–46).
    - Cálculo: `fits_out_of_fold_proba_foldwise` de `src/models/fusion_probe.py`.

13. **Recomendación: entrenar con el `.jpg` de una variante finalista y guardar el `.fits` solo para medir.** El `.fits` pesa ~770 KB por galaxia (~1 MB con 4 filtros) frente a pocos KB del `.jpg`, necesita su propio proceso de normalización y no da ninguna ganancia medible. Lo que este resultado **no** descarta: que con una red reentrenada (*fine-tuning*, ajuste fino) y ~2,000 galaxias el `.fits` rinda distinto, y mejoras pequeñas (~0.01 de AUC) que 50 galaxias no pueden detectar. Tampoco resuelve la elección entre DR9 y DR11 (siguen empatadas).
    - 📍 Secciones *"5. Resumen"* (celdas 48–49) y *"Observaciones y conclusiones (muestra de 50, 2026-10-19)"* (celda 50, puntos 1–6 y *"Qué no descarta este resultado"*).

### A.11 `analisis_complementariedad_multimodal.ipynb` — ¿Las distintas fuentes de datos "saben" cosas diferentes?

**Contexto en palabras simples:** en los experimentos de ablación (§1–§7 de este documento) se entrenaron modelos con distintas fuentes ("modalidades"): la **imagen a color (RGB)**, el **espectro** (cómo se reparte la luz de la galaxia por color/longitud de onda) y las **poblaciones estelares STARLIGHT** (90 números que resumen la edad y composición de las estrellas, calculados a partir del espectro). Esta libreta **no entrena nada**: compara, galaxia por galaxia, las predicciones ya guardadas de 11 modelos para saber si una fuente acierta donde otra falla y si eso se puede aprovechar. Esta libreta es la fuente de §8; aquí se explica en lenguaje no técnico y con su ubicación exacta.

1. **Las predicciones de los 11 modelos son comparables galaxia por galaxia.** Todos evaluaron exactamente las mismas galaxias (6,545 de entrenamiento, 1,403 de validación y 1,403 de prueba), sin duplicados, sin datos faltantes, sin galaxias repetidas entre grupos, y todos usan la misma convención (probabilidad de "tardía", umbral 0.5).
   - 📍 Sección *"2. Auditoría de predicciones y alineación de galaxias"* (celdas 4–6).
   - Cálculo: tabla `audit` y comparación de cohortes por `name` + `split` dentro de la libreta.

2. **La imagen es, por mucho, la fuente más informativa.** Orden por AUC (*Area Under the ROC Curve*, área bajo la curva ROC; 0.5 = azar, 1 = perfecto) en prueba: ResNet-18 con imagen 0.963 ≈ CNN (*Convolutional Neural Network*, red neuronal convolucional) compacta con imagen 0.962 > espectro STARLIGHT con atención 0.907 > poblaciones 0.895 > espectro STARLIGHT con CNN1D (CNN de una dimensión) 0.864 > espectro observado (FoV, *Field of View*, campo de visión) con CNN1D 0.857.
   - 📍 Sección *"3. Rendimiento individual de las modalidades"* (celdas 8–10).
   - Cálculo: función local `metrics_row` (AUC, precisión promedio, exactitud, F1 macro, exactitud balanceada, matriz de confusión y Brier — error cuadrático medio de la probabilidad —) con funciones de scikit-learn.

3. **Todos los modelos reconocen mejor las tardías que las tempranas** (favorecen la clase más común). El reconocimiento de tempranas va de 63.6 % (espectro STARLIGHT con CNN1D) a 89.2 % (CNN de imagen).
   - 📍 Sección 3, celda 10.

4. **Las dos redes de imagen empatan en su capacidad de ordenar galaxias (AUC −0.0009, compatible con cero), pero difieren en exactitud balanceada (+0.017 a favor de la CNN).** Es decir, la diferencia no está en qué tan bien distinguen, sino en dónde ponen el "punto de corte" (punto de operación).
   - 📍 Sección 3 (celda 10) y sección *"9. Incertidumbre estadística (bootstrap pareado)"* (celda 32).

5. **Hay complementariedad "potencial": las otras fuentes corrigen cerca de la mitad de los errores de la imagen.** La ResNet-18 se equivoca en 157 de 1,403 galaxias (11.2 %). Las poblaciones aciertan en 51.6 % de esos casos, el espectro FoV en 51.0 %, el espectro STARLIGHT en 51.6 % y el espectro con atención en 49.7 %. Al menos una de las cuatro corrige el 64.3 %; las cuatro a la vez, el 38.2 %.
   - 📍 Sección *"4. Matrices de complementariedad y tasas de rescate"* (celdas 12–14).
   - Cálculo: función local `pair_stats` (tasa de rescate = errores de A que B acierta / errores de A; índice de Jaccard = errores en común / errores de cualquiera de los dos).

6. **Un "oráculo" que siempre eligiera la fuente correcta subiría la exactitud de 88.8 % a 94.6 % (+5.8 puntos), pero es un límite teórico, no un sistema real**, porque para elegir necesita saber la respuesta.
   - 📍 Sección 4 (advertencia de la celda 11 y resultados de la celda 14); intervalo de confianza en la sección 9 (celda 33: +0.058 [+0.046, +0.068]).

7. **Además de complementariedad hay "dificultad común": 76 galaxias fallan con la imagen y con las poblaciones a la vez,** cuando por puro azar se esperarían ≈ 26. Hay galaxias intrínsecamente difíciles para todos. El rescate es consistente entre validación (56.2 %) y prueba (51.6 %).
   - 📍 Sección 4, celda 14.

8. **Los modelos no solo discrepan en la clase, también en su seguridad.** La diferencia típica de probabilidad entre imagen y poblaciones es 0.12 en general, 0.57 cuando discrepan en la clase y 0.09 cuando ambos aciertan.
   - 📍 Sección *"5. Desacuerdos, confianza y calibración"* (celdas 16 y 19).

9. **La CNN compacta de imagen es la mejor "calibrada"** (sus probabilidades se parecen más a las frecuencias reales): ECE (*Expected Calibration Error*, error de calibración esperado) de 0.014 frente a 0.038 de la ResNet-18, con la misma capacidad de distinguir. El ECE cambia según cuántos grupos (bins) se usen para calcularlo, así que hay que leerlo con cautela.
   - 📍 Sección 5, celda 17 (diagrama de confiabilidad y tabla).
   - Cálculo: función local `ece` y `brier_score_loss` de scikit-learn.

10. **Que un modelo esté muy seguro no garantiza que acierte:** 35 galaxias (22.3 % de los errores de la imagen) se clasificaron mal con mucha seguridad. Hay 29 galaxias donde la imagen duda y las poblaciones aciertan con seguridad: son las candidatas naturales para un sistema que "elija" fuente según la confianza (*gating*). Las 9 galaxias donde ambos modelos están muy seguros pero dicen lo contrario son mayoritariamente tempranas jóvenes (edad estelar ≈ 1.9 mil millones de años, masa baja): posibles formas peculiares o etiquetas dudosas.
    - 📍 Sección 5, celdas 18–19, y sección *"7. Galería de galaxias discordantes"* (celda 26).

11. **Los errores se concentran en la frontera entre tipos: las galaxias con `Ttype` 0 y 1** (lenticulares-espirales de transición) son el 16.1 % de la muestra pero acumulan el 47.8 % de los errores de la imagen. Ahí la imagen se equivoca 33.2 % de las veces vs 7.0 % en el resto; las poblaciones, 46.5 % vs 10.5 %. Las elípticas puras casi nunca fallan con imagen (3.0 %), pero las lenticulares S0 sí (21.5 %).
    - 📍 Sección *"6. Caracterización física y morfológica de los errores"* (celdas 22–23).
    - Cálculo: función local `ttype_bin` y tasas de error por `Ttype` con intervalo de Wilson (función local `wilson`).

12. **Las poblaciones casi no distinguen las espirales tempranas (`Ttype` 1) de las lenticulares:** fallan en el 51.7 % de las `Ttype` 1. Físicamente la transición es continua; la etiqueta pone una frontera arbitraria sobre ella. El mayor desacuerdo imagen–poblaciones también está en `Ttype` 1 (44.1 %).
    - 📍 Sección 6, celdas 22–23.

13. **No hay una regla física simple para decidir qué fuente usar:** ninguna propiedad física o de STARLIGHT (masa, distancia, inclinación, edad, metalicidad, polvo, etc.) distingue a las galaxias "solo las poblaciones aciertan" de "ambos fallan" después de corregir por comparaciones múltiples (corrección de Bonferroni).
    - 📍 Sección 6, celdas 21 y 23.
    - Cálculo: prueba de Kruskal-Wallis y de Mann-Whitney (MWU, *Mann-Whitney U*) con corrección de Bonferroni, de `scipy.stats`.

14. **La galería sugiere hipótesis (no pruebas):** formas peculiares, alta inclinación y galaxias compañeras. Las galaxias donde "la imagen falla y las poblaciones aciertan" son 60 % tardías con `Ttype` mediano 1; donde "la imagen acierta y las poblaciones fallan", 37 % tardías con `Ttype` mediano −1: son grupos complementarios.
    - 📍 Sección 7, celda 26. ⚠️ La celda 25 (la que dibuja la galería) termina con un error de sintaxis (`SyntaxError: incomplete input`) en la libreta guardada; el resumen numérico de la celda 26 sí se ejecutó.

15. **Mezclar probabilidades con un peso fijo no aprovecha la complementariedad.** Combinando `α·imagen + (1−α)·otra` con α elegido solo en validación: con poblaciones, el mejor peso es 0.85 hacia la imagen (rango estable 0.70–1.00), y en prueba el AUC pasa de 0.9633 a 0.9637 (+0.0005, compatible con cero). Con el espectro FoV, el mejor peso es 1.0: **la mejor mezcla es no mezclar**. El promedio simple 50/50 **empeora** (p. ej. 0.9581 con poblaciones).
    - 📍 Sección *"8. Fusión tardía de probabilidades (exploración solo en validation)"* (celdas 28–30).
    - Cálculo: función local `fusion_curve` (rejilla de α de 0 a 1 en pasos de 0.05, curva suavizada con media móvil de 3 puntos).

16. **Ninguna fusión supera de forma demostrable a la imagen sola,** ni las entrenadas (concatenación) ni la mezcla fija: todos los intervalos de confianza de la diferencia de AUC incluyen el cero y la prueba de McNemar (que compara en cuántas galaxias acierta solo uno u otro) da p = 0.14–1.0. En cambio, sí son diferencias reales: imagen vs poblaciones (+0.068), poblaciones vs espectro CNN1D (+0.031), espectro STARLIGHT vs FoV con CNN1D (+0.007), espectro con atención vs CNN1D (+0.043) e imagen vs espectro con atención (+0.057).
    - 📍 Sección 9, celdas 32–33.
    - Cálculo: funciones locales `strat_indices`, `boot` y `run_bootstrap` (bootstrap pareado y estratificado por clase, 1,000 remuestreos) y prueba exacta de McNemar.

17. **Escenario resultante: "B — hay información distinta, pero falta saber *cuándo* confiar en cada fuente".** Según la regla de la libreta (oráculo ≥ 4 puntos, errores compartidos < 70 % y sin mejora demostrable de la fusión), las fuentes secundarias sí aportan algo distinto en ~la mitad de los fallos de la imagen, pero ni un promedio fijo ni una concatenación sabe cuándo usarlas. Siguiente paso recomendado: fusión con "compuerta" (*gated fusion*), basada en incertidumbre o calibración, o *stacking* (modelo que aprende a combinar) con validación cruzada.
    - 📍 Sección *"10. Síntesis de hallazgos y próximos experimentos"* (celda 35).

18. **Limitaciones:** una sola semilla y una sola división de datos; los intervalos solo reflejan qué galaxias tocaron en prueba, no la variabilidad del entrenamiento; los resultados de prueba ya se habían visto antes, por eso aquí solo se usan para describir, y ningún peso o umbral se eligió con ellos.
    - 📍 Nota de la celda 0 (*"Limitación metodológica global"*) y texto de la sección 9 (celda 31, *"Qué NO hace"*).

### A.12 `diagnostico_gate_multimodal.ipynb` — ¿Se puede saber de antemano cuándo confiar en la imagen y cuándo en el espectro?

**Contexto en palabras simples:** la libreta anterior mostró que el espectro acierta en muchos casos donde la imagen falla, pero que mezclar sus respuestas con un peso fijo no ayuda. La idea siguiente es una "compuerta" (*gate*): un sistema que, para cada galaxia, decida a cuál de los dos modelos hacerle caso. Antes de construirla, esta libreta revisa **si existe evidencia suficiente para justificarla**. Usa solo las predicciones ya guardadas de la ResNet-18 con imagen y de la CNN con atención sobre el espectro STARLIGHT, y trabaja **solo con el conjunto de validación** (el de prueba no se toca). Es la fuente de §10 de este documento; aquí se explica en lenguaje no técnico y con su ubicación exacta.

1. **Los datos están completos y alineados:** 1,403 galaxias de validación (62.3 % tardías) con predicciones de todos los modelos, propiedades del catálogo y de STARLIGHT, sin faltantes ni pérdidas al unir tablas. Los valores extremos no se eliminan porque son normales en astronomía.
   - 📍 Sección *"1. Auditoría y preparación"* (celdas 4–8).

2. **La imagen y el espectro solo discrepan en el 18.3 % de las galaxias (257 de 1,403), y ahí la imagen tiene razón dos de cada tres veces** (172 vs 85; el espectro acierta 33.1 %). Solo en esas 257 galaxias una compuerta podría cambiar algo, lo que limita desde el inicio cuánto se puede ganar. No importa la "dirección": el espectro gana ≈ 33 % tanto cuando la imagen dice tardía como cuando dice temprana.
   - 📍 Celdas 10–11 (*"Variables derivadas"*) y sección *"5. Experimento 4: análisis restringido a desacuerdos"* (celda 30).
   - Cálculo: función local `add_derived` (calcula `confidence = 2·|p − 0.5|`, que va de 0 = indeciso a 1 = totalmente seguro, el desacuerdo y quién gana).

3. **No existe ninguna "zona" de probabilidades donde el espectro sea claramente más confiable que la imagen.** En una cuadrícula de 5×5 combinaciones de probabilidad, ninguna celda con suficientes galaxias muestra ventaja del espectro; 5 celdas muestran ventaja de la imagen, todas cuando la imagen está muy segura.
   - 📍 Sección *"2. Experimento 1: mapa bidimensional de probabilidades"* (celdas 13–17).

4. **La seguridad de cada modelo sí sirve para detectar sus propios errores:** la confianza de la imagen distingue aciertos de errores con AUC (*Area Under the ROC Curve*, área bajo la curva ROC; 0.5 = azar, 1 = perfecto) de 0.878; la del espectro, 0.754. Cuando la imagen está muy indecisa (confianza < 0.4) el espectro gana ≈ 51–55 % de los desacuerdos; cuando la imagen está muy segura (> 0.9), solo 16 %.
   - 📍 Sección *"3. Experimento 2: incertidumbre y desacuerdo"* (celdas 19–23).
   - Cálculo: función local `binned_table` (agrupa por niveles de confianza con intervalos de Wilson) y AUC de la confianza como detector de aciertos.

5. **Pero el espectro corrige casi igual los errores "seguros" y los "dudosos" de la imagen:** de 146 errores de la imagen, 34 se cometieron con mucha seguridad; el espectro rescata 52.9 % de esos y 59.8 % de los demás. Cuando ambos están muy seguros y discrepan (solo 8 casos), el espectro gana 25 % (muestra insuficiente para concluir).
   - 📍 Sección 3, celda 21.

6. **El espectro está "sobreconfiado" justo donde discrepa con la imagen.** En los desacuerdos, la imagen asigna en promedio 0.79 de probabilidad a su respuesta (y acierta 67 %), mientras el espectro asigna 0.72 (y acierta solo 33 %). Además, ambos modelos usan "escalas" de probabilidad distintas (la imagen da probabilidades mucho más extremas). Esto ayuda a explicar por qué promediar sus probabilidades empeora el resultado. En general el espectro tiene más error de probabilidad (Brier 0.123 vs 0.074).
   - 📍 Sección *"4. Experimento 3: calibración"* (celdas 25–28).
   - Cálculo: función local `ece` (ECE, *Expected Calibration Error*, error de calibración esperado, con grupos uniformes o por cuantiles), Brier y *log loss* (pérdida logarítmica).

7. **Ninguna de 14 propiedades de la galaxia o de la calidad del espectro distingue los casos donde gana el espectro** (distancia, tamaño, masa, señal/ruido del espectro, calidad del ajuste STARLIGHT, edad, metalicidad, polvo, etc.) después de corregir por comparaciones múltiples (corrección de Holm). La más cercana fue la señal/ruido del espectro (`sn_window`), con un efecto pequeño.
   - 📍 Sección *"6. Experimento 5: propiedades asociadas con el rescate"* (celdas 34–36).
   - Cálculo: función local `compare_groups` (δ de Cliff y d de Cohen como tamaño del efecto; prueba de Mann-Whitney con corrección de Holm).

8. **El espectro gana más en la frontera morfológica:** en las S0a (`Ttype` 0) gana el 48.8 % de los desacuerdos y en las `Ttype` 1 el 41.7 %, frente a 15.6 % en las elípticas. El `Ttype` solo se usa para interpretar, nunca para decidir, porque contiene la respuesta.
   - 📍 Sección 6, celdas 37–38.

9. **Un selector aprendido encuentra una señal modesta, y está en la seguridad de los modelos, no en el valor de sus probabilidades.** Se probaron variantes de regresión logística entrenadas solo con los 257 desacuerdos (validación cruzada de 5 grupos). La que usa probabilidades + confianzas (S2) logra AUC 0.676, claramente mejor que el azar (p de permutación 0.005) y estable en los 5 grupos y en 10 particiones distintas. La que solo ve las probabilidades "en línea recta" (S2min) no encuentra nada (AUC 0.398), igual que las que agregan datos del catálogo (S3), de calidad (S4) o físicos (S5). Los AUC por debajo de 0.5 no significan una relación inversa: son un efecto conocido de la validación cruzada con pocos datos.
   - 📍 Sección *"7. Experimento 6: ¿podemos aprender a elegir la modalidad?"* (celdas 40–46), en particular la prueba de permutación (celdas 41–42).
   - Cálculo: funciones locales `make_pipeline` (imputación por mediana + estandarización + regresión logística, todo ajustado dentro de cada grupo), `cv_logistic` y `permutation_null`.

10. **Esa señal no se traduce en más aciertos que usar solo la imagen.** Simulando el sistema (si coinciden, usar la imagen; si discrepan, que decida el selector), el mejor caso (S2 sin reponderar clases) cambia a espectro en 49 galaxias: corrige 27 y estropea 22, ganancia neta +5 galaxias (intervalo −9 a +18, es decir, compatible con cero). El modo que "equilibra clases" pierde más de lo que gana (−11). Como referencia: mezclar con peso fijo 0.85 da +3, usar siempre el espectro −87, y el oráculo (que conoce la respuesta) +85.
    - 📍 Sección *"8. Experimento 7: simulación de un sistema de selección"* (celdas 48–50).
    - Cálculo: función local `system_eval` (simulación del sistema; ganancia neta = errores corregidos − aciertos perdidos, con bootstrap).

11. **Los errores del selector también se concentran en la frontera morfológica** (`Ttype` 0–1), donde la etiqueta corta una propiedad física continua y parte del "error" puede ser ambigüedad de la etiqueta.
    - 📍 Sección *"9. Interpretación de las decisiones del selector"* (celdas 52–55) e hipótesis de la celda 59.

12. **Veredicto: "señal sin ganancia neta demostrada"; todavía no se justifica construir una compuerta adaptativa.** Se cumplen 3 de 5 criterios fijados de antemano (hay señal fuera de muestra, es estable y usa datos disponibles al predecir), pero no los otros 2 (ganancia neta con intervalo positivo y que no dependa de unas pocas galaxias).
    - 📍 Sección *"10. Conclusiones y recomendación"* (celdas 57–58).

13. **Los modelos de árboles (XGBoost — *eXtreme Gradient Boosting* —, Random Forest, árbol de decisión) no encuentran más estructura que la regresión logística; al contrario, empeoran por sobreajuste.** En 0 de 6 conjuntos de variables superan a la regresión logística; con solo la confianza, XGBoost logra AUC 0.618 vs 0.671 de la logística. Los árboles "memorizan" mucho más (diferencia entrenamiento − prueba de 0.30 vs 0.07). Solo la confianza de la imagen supera claramente a una variable de ruido de control en importancia. Ninguna combinación logra ganancia neta positiva demostrable.
    - 📍 Sección *"12. Diagnóstico final: ¿hay dependencias no lineales que los árboles (XGBoost) aprovechen mejor que la regresión logística?"* (celdas 63–75; veredicto en la celda 74).
    - Cálculo: modelos de scikit-learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `PolynomialFeatures`) y `xgboost`; funciones locales `paired_boot` (diferencia pareada con bootstrap) e importancia por permutación fuera de grupo con una variable `_ruido_control`.

14. **Resultado nulo con poca potencia:** con 257 desacuerdos no se puede descartar que exista algo de no linealidad pequeña o interacciones complejas; solo que estos datos no la respaldan.
    - 📍 Celda 62 (*"Advertencia de potencia"*) y celda 75.

15. **Siguientes pasos justificados:** (1) obtener predicciones "fuera de muestra" de los propios modelos de imagen y espectro con validación cruzada sobre todo el conjunto, para tener muchos más desacuerdos y una evaluación final independiente; (2) recalibrar las probabilidades antes de combinarlas, dado el exceso de confianza del espectro; (3) si alguna variable STARLIGHT llegara a mejorar al selector, compararla contra un clasificador directo de la morfología con esas variables, para separar "fiabilidad" de "información de forma adicional".
    - 📍 Celda 59 (*"Experimentos posteriores justificados"*).

16. **Limitaciones:** pocos desacuerdos y una sola división; los modelos base se eligieron con validación (sus resultados ahí son algo favorables); el selector se evaluó fuera de muestra, pero los modelos base no; la variante principal (S2) se eligió con los mismos datos (algo optimista); no hay evidencia confirmatoria en prueba.
    - 📍 Nota de la celda 0 (*"Sesgo conocido de `validation`"*), advertencia de la celda 39 y *"Limitaciones de esta fase"* (celda 59).


### A.13 Lo que se intentó y no funcionó

**Para qué sirve esta sección:** reúne en un solo lugar todo lo que se probó en las libretas y salió mal, empeoró los resultados o se descartó. Así nadie del equipo repite un intento ya probado. Cada fila indica dónde está la evidencia (libreta, sección y celda) y en qué conclusión de este anexo se explica con más detalle. "Descartado" significa que no se usará; "falló" significa que el resultado fue peor que la opción de referencia.

#### Preparación de datos e imágenes

| Qué se intentó | Qué pasó | Dónde está la evidencia | Más detalle |
|---|---|---|---|
| Etiquetar temprana/tardía con la regla `Ttype ≤ 0` / `Ttype ≥ 1` | **Falló en silencio:** las 67 galaxias con `Ttype = 11` (fuera del rango 1–10 del correo de la UNAM) se contaron como tardías, aunque el texto de la libreta dice que se excluían. El conjunto curado final sí las elimina. | `exploration_morfologia.ipynb`, *"Etiquetado binario temprana/tardía"*, celdas 25 y 28 | A.4, punto 7 |
| Tomar la columna de distancia (`nsa_z`) tal como viene | **Falló:** 27 galaxias traen el código −9999 ("sin dato") en vez de un valor real, y eso arruina los promedios (−26.6 en lugar de 0.046). Se creó una versión limpia. | `exploration_morfologia.ipynb`, *"Valores faltantes"*, celdas 13–15 | A.4, punto 3 |
| Descargar todas las imágenes con un zoom fijo | **Descartado:** el tamaño de las galaxias varía demasiado (mediana ≈ 13 y máximo 148 segundos de arco); las grandes saldrían cortadas y las pequeñas como un punto. | `exploration_morfologia.ipynb`, *"Radio de Petrosian (`PETRO_TH90`)…"*, celdas 44–45 | A.4, punto 12 |
| Dibujar los `.fits` directamente junto a los `.jpg` | **Falló:** los `.fits` salían de cabeza (similitud con el `.jpg` de 0.73 en vez de 0.98). Ya está corregido en `src/data/cutout_io.py`. | `exploration_fits.ipynb`, *"2. Orientación…"*, celdas 14–16 | A.5, punto 3 |
| Sumar la luz de los `.fits` sin corregir por el zoom | **Falló:** las galaxias pequeñas salían hasta ≈ 4 veces más débiles de lo real. Se corrigió con `native_flux_factor` (`src/data/fits_photometry.py`). | `exploration_fits.ipynb`, *"4. Unidades…"*, celdas 22–26 | A.5, punto 6 |
| Normalizar los `.fits` con mínimo y máximo | **Descartado:** el núcleo es ≈ 100 veces más brillante que el disco, así que solo se ve el centro y los brazos quedan en negro. | `normalization_fits.ipynb`, *"2. El problema: el rango crudo"*, celdas 6–7 | A.6, punto 1 |
| Estirar el brillo con escala lineal, logarítmica o asinh aplicado por separado a cada filtro | **Descartado:** la lineal pierde el disco, la logarítmica hace visible el ruido y el asinh por filtro "lava" los colores (su relación con el color real baja de 0.98 a 0.85). | `normalization_fits.ipynb`, *"4. Paso 2…"*, celdas 12–15 | A.6, punto 3 |
| Suavidad del estiramiento β ≤ 1σ o β ≥ 10σ (σ = ruido del cielo) | **Descartado:** con valores bajos la imagen se llena de "granos" de ruido; con valores altos se apaga el disco externo. | `normalization_fits.ipynb`, *"5. Paso 3: la suavidad β"*, celdas 17–18 | A.6, punto 4 |
| Escalar cada imagen por separado | **Descartado:** el nivel de ruido termina delatando la clase (AUC 0.667), una pista falsa que depende de la observación y no de la galaxia. | `normalization_fits.ipynb`, *"6. Paso 4…"*, celdas 20–23 | A.6, punto 5 |
| Recortar valores normalizados con tope 1.5 | **Falló:** cortaba los núcleos muy rojos en el filtro z; se subió el tope a 3.0. | `normalization_fits.ipynb`, *"7. Paso 5…"*, celda 26 | A.6, punto 7 |
| Usar la máscara de color para quitar todo menos la galaxia central | **Falló parcialmente:** deja dentro a las galaxias vecinas superpuestas y, en espirales con núcleo muy dominante, corta el disco externo débil. | `image-curation.ipynb`, *"3. Validación visual…"*, celda 12 | A.8, punto 3 |
| Variantes de imagen `ls-dr10_grz` y `ls-dr10_gri` | **Descartadas:** pintan de distinto color el cielo del norte y del sur (morado o verde). | `image-curation.ipynb`, *"6.2 Homogeneidad de color…"*, celdas 22–23 y 31 | A.8, punto 6 |
| Variantes `ls-dr10_griz` y `ls-dr10_gri` para clasificar | **Fallaron:** son significativamente peores (AUC −0.09 y −0.14) y caen en el sur, porque en DR10 el filtro *i* solo existe para el telescopio del sur. | `variant-classification-for-images-only.ipynb`, *"2.1…"* y *"3. Rendimiento por región…"*, celdas 18–19 y 24–25 | A.9, puntos 6 y 8 |

#### Entradas y tratamientos para el clasificador

| Qué se intentó | Qué pasó | Dónde está la evidencia | Más detalle |
|---|---|---|---|
| Clasificar con la imagen enmascarada (fondo negro) | **Falló:** empeora en todas las variantes (−0.08 a −0.16 de AUC) y da resultados más inestables. | `variant-classification-for-images-only.ipynb`, *"2."* y *"2.1"*, celdas 13–19 | A.9, puntos 4 y 6 |
| Arreglar la máscara: fondo con ruido de cielo, borde suave, margen ancho o círculo del tamaño de la galaxia | **Falló:** todas siguen siendo peores que la imagen original (−0.04 a −0.10 de AUC). | `image-curation.ipynb`, *"7. Tratamientos de imagen…"*, celdas 35–39 | A.8, punto 11 |
| Quitar solo las estrellas, aclarar con gamma (γ = 0.6) o ambas cosas | **No mejoró:** empate o pérdida leve (−0.003 a −0.053); ninguna supera a la original. | `image-curation.ipynb`, sección 7, celda 39 | A.8, punto 12 |
| Clasificar solo con el contorno de la máscara | **No sirvió como atajo:** como mucho da una pista débil (AUC ≤ 0.71), que viene de la elongación, una propiedad real de la forma. | `variant-classification-for-images-only.ipynb`, *"1.2…"*, celdas 9–10 | A.9, punto 3 |
| Clasificar con el `.fits` en lugar del `.jpg` | **Falló:** es peor en las 5 variantes (AUC 0.83–0.88) y sobre todo en el sur (0.75–0.79). | `variant-classification-images-and-fits.ipynb`, *"2.1…"* y *"3."*, celdas 24–27 y 35–36 | A.10, puntos 5 y 10 |
| Combinar imagen + `.fits` y/o fotometría | **No mejoró:** 0 de 25 comparaciones a favor; con `ls-dr11_griz` empeora a la imagen (−0.053). Además vuelve a meter pistas del telescopio. | `variant-classification-images-and-fits.ipynb`, celdas 24–27 y 39–41 | A.10, puntos 5 y 11 |
| Usar el filtro *i* de DR11 en el `.fits` | **Falló:** el filtro viene vacío en todo el norte; el `.fits` delata el telescopio casi sin error (AUC 0.97) y es el peor de todos. | `variant-classification-images-and-fits.ipynb`, celdas 8 y 39–40 | A.10, puntos 2 y 11 |

#### Combinar imagen, espectro y poblaciones

| Qué se intentó | Qué pasó | Dónde está la evidencia | Más detalle |
|---|---|---|---|
| Fusiones entrenadas por concatenación (imagen + espectro y/o poblaciones) | **No mejoraron:** ninguna supera a la imagen sola de forma demostrable (todas las diferencias son compatibles con cero). | `analisis_complementariedad_multimodal.ipynb`, *"9. Incertidumbre estadística…"*, celdas 32–33 | A.11, punto 16 |
| Mezclar probabilidades con un peso fijo (`α·imagen + (1−α)·otra`) | **No mejoró:** con poblaciones +0.0005 de AUC (empate); con el espectro FoV, la mejor mezcla es no mezclar. | `analisis_complementariedad_multimodal.ipynb`, *"8. Fusión tardía…"*, celdas 28–30 | A.11, punto 15 |
| Promedio simple 50/50 de probabilidades | **Falló:** empeora a la imagen sola (p. ej. AUC 0.9581 vs 0.9633 con poblaciones), en parte porque el espectro está sobreconfiado justo donde discrepa. | Misma sección, celda 29; `diagnostico_gate_multimodal.ipynb`, *"4. Experimento 3: calibración"*, celda 27 | A.11, punto 15; A.12, punto 6 |
| Buscar una regla física simple (masa, distancia, edad, calidad, etc.) para elegir fuente | **Falló:** ninguna propiedad distingue los casos en que conviene cada fuente. | `analisis_complementariedad_multimodal.ipynb`, *"6."*, celdas 21–23; `diagnostico_gate_multimodal.ipynb`, *"6. Experimento 5"*, celdas 34–36 | A.11, punto 13; A.12, punto 7 |
| Buscar "zonas" de probabilidad donde el espectro sea más confiable | **Falló:** ninguna celda del mapa muestra ventaja del espectro. | `diagnostico_gate_multimodal.ipynb`, *"2. Experimento 1"*, celdas 13–17 | A.12, punto 3 |
| Selectores con solo probabilidades (S2min) o con datos de catálogo, calidad o físicos (S3, S4, S5) | **Fallaron:** no encuentran señal (AUC 0.31–0.53, no distinguible del azar). | `diagnostico_gate_multimodal.ipynb`, *"7. Experimento 6"*, celdas 40–46 | A.12, punto 9 |
| Selector que "equilibra clases" (modo `balanced`) | **Falló:** cambia demasiado a espectro y pierde más aciertos de los que gana (ganancia neta −11). | `diagnostico_gate_multimodal.ipynb`, *"8. Experimento 7"*, celdas 48–50 | A.12, punto 10 |
| El mejor selector (S2, basado en la seguridad de los modelos) | **Señal sin ganancia:** detecta algo (AUC 0.676), pero solo suma +5 galaxias de 1,403 (intervalo −9 a +18, empate). | Mismas celdas 48–50 y veredicto en celdas 57–58 | A.12, puntos 10 y 12 |
| Árboles, Random Forest y XGBoost como selectores | **Fallaron:** son peores que la regresión logística por sobreajuste (memorizan) y ninguna combinación da ganancia neta. | `diagnostico_gate_multimodal.ipynb`, *"12. Diagnóstico final…"*, celdas 63–75 | A.12, punto 13 |
| Construir ya una "compuerta" (*gate*) adaptativa | **Descartado por ahora:** solo se cumplen 3 de 5 criterios; falta demostrar ganancia neta. | `diagnostico_gate_multimodal.ipynb`, *"10. Conclusiones y recomendación"*, celda 58 | A.12, punto 12 |

#### Fallas técnicas en las propias libretas

| Qué pasó | Dónde | Consecuencia |
|---|---|---|
| La celda que dibuja la galería de galaxias discordantes termina con `SyntaxError: incomplete input` | `analisis_complementariedad_multimodal.ipynb`, *"7. Galería…"*, celda 25 | La galería no se ve en la libreta guardada; el resumen numérico (celda 26) sí corrió. |
| Celdas de cálculo sin salidas guardadas | `shortcuts.ipynb` (todas las de cálculo) y `exploration_morfologia.ipynb`, celdas 53–62 | Las cifras solo constan en los textos escritos en la libreta; conviene volver a ejecutarlas para respaldarlas. |
| Se cita un documento que no existe en el repositorio (`RESUMEN_INVESTIGACION_COMPLEMENTARIEDAD_MULTIMODAL.md`) | `diagnostico_gate_multimodal.ipynb`, celda 0 | Ninguna en la práctica: la libreta lo sustituye con §8 de este documento y la libreta de complementariedad. |

**Lección general:** casi todo lo que "limpia", recorta o agrega información a la imagen original salió peor o igual que usarla tal cual. La información que aportan el espectro y las poblaciones existe (corrigen ≈ la mitad de los errores de la imagen), pero ningún método probado ha logrado saber **cuándo** usarla.
