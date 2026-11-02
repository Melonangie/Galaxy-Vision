# Análisis de los experimentos de ablación (imágenes, espectros, poblaciones STARLIGHT)

Clasificación binaria `temprana (0)` vs `tardia (1)` en galaxias MaNGA. Este documento resume qué se ha ejecutado, qué muestran las métricas y las curvas de entrenamiento, cuánto overfitting hay y qué decisiones sugiere para la siguiente fase.

Fuentes: los notebooks de esta carpeta y los artefactos en `models/experiments/<experimento>/` (`history.csv`, `results.json`, `config.json`). Las curvas están en [figuras/curvas_entrenamiento.png](figuras/curvas_entrenamiento.png).

---

## 1. Resumen ejecutivo

1. **Las imágenes RGB dominan.** Cualquier modelo que incluya imagen llega a **AUC test ≈ 0.962–0.965** y accuracy ≈ 0.89–0.90. Sin imagen, el techo es **AUC ≈ 0.86 (espectro) y ≈ 0.90 (poblaciones)**.
2. **Añadir espectros o poblaciones a la imagen no mejora de forma demostrable.** Las diferencias entre las 6 configuraciones visuales (0.962–0.965) están dentro del ruido (error estándar del AUC en test ≈ ±0.005).
3. **Dos arquitecturas visuales sencillas ya son líneas base fuertes.** La CNN pequeña (AUC test 0.9623) y el control ResNet-18 (0.9633) empatan prácticamente. Extender la CNN a 46 épocas no mejoró su AUC de test ni su mejor AUC de validación.
4. **Las poblaciones STARLIGHT (90 números) superan al espectro con la CNN 1D** (0.895 vs 0.857–0.864) aunque derivan del mismo espectro. Esto indica que la CNN 1D actual **no está extrayendo bien la información espectral**; no que el espectro carezca de ella.
5. **ResNet-18 sobreajusta incluso con imagen sola** (loss final train/val: 0.159/0.317). Es el mismo patrón de sus fusiones, por lo que no se debe atribuir el sobreajuste a añadir modalidades. La CNN compacta solo muestra sobreajuste leve al final (0.237/0.287). Los modelos sin imagen **subajustan** (train ≈ val, plateau).
6. El control **ResNet-18 solo imagen ya cierra la ablación arquitectónica**: usa el mismo split FoV, tamaño, regularización, LR y selección que las fusiones. Queda la limitación central: **una sola semilla y un solo split**.

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

### Decisión sugerida
**Seguir con imagen como insumo central y poblaciones STARLIGHT como complemento candidato; poner en pausa el espectro en bruto hasta mejorar su rama.** El control ResNet-18 confirma que las fusiones aún no ganan a la imagen. Los esfuerzos inmediatos: (1) semillas/intervalos de confianza de ResNet-18 y las dos fusiones candidatas, (2) baseline de metadatos y análisis de complementariedad de errores, (3) augmentations/scheduler para imagen y comparación de FoV. Solo después tiene sentido invertir en arquitecturas más complejas.

---

## Apéndice: limitaciones de este análisis

- Basado en una corrida por configuración (semilla 42) y en `results.json`/`history.csv` existentes; no se reentrenó ni se calcularon predicciones por galaxia.
- Los errores estándar de AUC son aproximaciones analíticas (Hanley–McNeil), no bootstrap sobre predicciones.
- Según los notebooks, las fusiones y el control usan un ResNet-18 sin preentrenar; la CNN visual es la definida en `experimento_binario_imagenes_gpu_acc.ipynb`.
