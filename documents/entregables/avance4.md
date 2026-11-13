# Análisis de los experimentos de ablación (imágenes, espectros, poblaciones STARLIGHT)

Clasificación binaria `temprana (0)` vs `tardia (1)` en galaxias MaNGA. Este documento resume qué se ha ejecutado, qué muestran las métricas y las curvas de entrenamiento, cuánto overfitting hay y qué decisiones sugiere para la siguiente fase.

Fuentes: los notebooks de esta carpeta y los artefactos en `models/experiments/<experimento>/` (`history.csv`, `results.json`, `config.json`). Las curvas están en [figuras/curvas_entrenamiento.png](figuras/curvas_entrenamiento.png).

Estructura: §1–§7 cubren la **ablación** (imagen / espectro / poblaciones y sus fusiones); §8 resume el **estudio de complementariedad** ([analisis_complementariedad_multimodal.ipynb](../../analisis_complementariedad_multimodal.ipynb), solo analiza predicciones, no entrena); §9 documenta la **optimización del experto espectral** (RNN/atención), que *no* forma parte de la ablación pero se acumula aquí; §10 resume el **diagnóstico de confiabilidad y selección de modalidad** (primera fase del *gate*, [diagnostico_gate_multimodal.ipynb](../../diagnostico_gate_multimodal.ipynb), solo `validation`, entrena únicamente selectores tabulares pequeños y árboles).

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

## Apéndice: limitaciones de este análisis

- Basado en una corrida por configuración (semilla 42) y en `results.json`/`history.csv` existentes; §1–§7 no usan predicciones por galaxia (§8 sí).
- Los errores estándar de AUC en §1–§7 y §9 son aproximaciones analíticas (Hanley–McNeil); los IC bootstrap pareados solo existen en §8.
- Los resultados de `test` se habían inspeccionado antes del estudio de §8; allí se usan de forma descriptiva y ningún α o umbral se eligió con test.
- §10 usa **solo `validation`** (nunca `test`) y 257 desacuerdos; sus cifras no son comparables con las de test de §3–§9, y los selectores se evalúan con OOF solo del selector (las redes base no se reentrenaron con validación cruzada).
- Según los notebooks, las fusiones y el control usan un ResNet-18 sin preentrenar; la CNN visual es la definida en `experimento_binario_imagenes_gpu_acc.ipynb`.
