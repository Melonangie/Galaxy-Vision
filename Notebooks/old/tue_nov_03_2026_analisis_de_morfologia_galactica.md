# Análisis de morfología galáctica

**User:** @maria-rabelero-mx_nestle

## @maria-rabelero-mx_nestle

based on the EDA https://github.com/Melonangie/Galaxy-Vision/blob/main/EDA_Starlight.ipynb  do data preprocessing, feature selection, balancing and feature engineering. 

I want to find which galaxy properties and spectral characteristics are most useful for classifying galaxies by morphological type ttype.

The primary objective is not simply to maximize classification accuracy. It is to discover which features best distinguish morphological classes and understand their astrophysical significance.

In addition also 
 inspect the datasets and report: Number of features and their data types, Outliers and unusual values, correlations between numerical features,Potential data-quality and measurement issues. Identify which datasets can be joined and validate the matching using galaxy identifiers or coordinates.
Compare the advantages and disadvantages of classify
 Detailed morphological classes
early-type, intermediate-type, and late-type
T-type intervals

The EDA was made using .fits.gz files and .out files. A complementary .fits  file was used to

Each .fits.gz file is of a galaxy. The galaxies are of ttypes 4, 7, -2,  -5, 3, 2, 0, 6, 5, 1, 11, 8, 9, 10. The .fits.gz files have the same name pattern and columns as the csv files below, these csv have Gaps in the spectra but the .fits.gz files do not.  Use wave and flux_fov, ignore ERROR_FoV is incorrect. Wave needs to be converted to rest-frame wavelength. Wavelength is in vacuum
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-12701-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-1901-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10214-12705-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10216-1902-1D.csv
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10217-6103-1D.csv

The .out files that were obtained using http://www.starlight.ufsc.br/downloads/, and the wave and flux_fov from the .fits.gz files.
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-12701-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10001-1901-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10214-12705-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10216-1902-1D-out.txt
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/manga-10217-6103-1D-out.txt

The complementary .fits file has the same columns as this csv file.
https://github.com/Melonangie/Galaxy-Vision/blob/main/data/sample-data/morfologia.csv


Use astronomy libraries such as astropy, astroML, etc

Make the code as simple as possible and human readable as possible, using inline comments and codeblocks explain what the code is doing and why in spanish

Document as much as you can about galaxy morphology with references, and add a conclusion to each analysis  in spanish

Show me the cells to run on google colab, dont run it, dont test it


## Copilot

I'll start by inspecting the EDA notebook and sample data to understand the structure.

The EDA is thorough. Now let me inspect the sample CSV, .out, and morfologia files.

Notable: `Type` encodes bars (SB/SAB) and "Edc"; some catalog NaNs. Now the .out format and the FITS headers (for coordinates).

Valuable issues found: `nsa_z=-9999` sentinels (27), a negative z, Ttype 3 containing "Sa", Ttype 10 mixing dSph/BCD with Irr, inclination capped at 81.37°. Continuing checks.

DRPall isn't reachable from this sandbox (I'll make that cell optional with a fallback). Let me check what the FITS contains where the CSV has gaps.

Inspection is done. Now I'm writing the notebook. I'll build it in parts and then convert them to a single `.ipynb`.

Understood — no files. I'll remove the partial files I created and the temporary clone, then print every cell in the chat.

Sigo desde la **Celda 80**. La vuelvo a poner completa porque quedó cortada.

## Celda 80 · Código (completa)
```python
def importancia_familias(modelo, X, y, n_rep=10):
    """Caída de balanced accuracy al permutar juntas todas las features de una familia."""
    rng = np.random.default_rng(SEMILLA)
    base = balanced_accuracy_score(y, modelo.predict(X))
    res = {}
    for fam, cols in FAMILIAS.items():
        caidas = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[cols] = X[cols].values[rng.permutation(len(X))]   # misma permutación para toda la familia
            caidas.append(base - balanced_accuracy_score(y, modelo.predict(Xp)))
        res[fam] = (np.mean(caidas), np.std(caidas))
    return pd.DataFrame(res, index=['caida_media', 'std']).T.sort_values('caida_media', ascending=False)

IMP_FAMILIA = {}
for e in ESQUEMAS:
    modelo = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_ajuste, FEATURES], df.loc[idx_ajuste, f'y_{e}'])
    IMP_FAMILIA[e] = importancia_familias(modelo, df.loc[idx_valid, FEATURES], df.loc[idx_valid, f'y_{e}'])

fig, axs = plt.subplots(1, 3, figsize=(22, 4))
for ax, e in zip(axs, ESQUEMAS):
    t = IMP_FAMILIA[e][::-1]
    ax.barh(t.index, t['caida_media'], xerr=t['std'], color=[colores_familia[f] for f in t.index])
    ax.set_title(f'Importancia por familia — {e}'); ax.set_xlabel('Caída de balanced accuracy')
plt.tight_layout(); plt.show()
```

## Celda 81 · Markdown
```markdown
**Conclusión 10.4–10.5:**
* La familia **"Edad / historia"** debería tener la mayor caída individual.
* **"Masa, cinemática y estructura"** debería ser la segunda, y la **más importante para separar E de S0**: las dos son viejas,
  pero difieren en σ y Σ* (Cappellari et al. 2011).
* Las caídas por familia son **pequeñas** si otra familia contiene la misma información. Por ejemplo, la edad y la formación
  estelar actual se sustituyen mutuamente (la sSFR actual determina buena parte de D4000). Una caída pequeña no significa
  "irrelevante", sino "redundante dado el resto".
* Si **b_a** saliera muy importante, el modelo estaría usando la forma aparente. Es útil para separar E de S0 (las E son más
  redondas), pero no es una propiedad intrínseca.

### 10.6 ¿Qué feature separa cada par de tipos vecinos?
Calculamos el poder discriminante **univariado** = 2·|AUC − 0.5| (0 = nada, 1 = separación perfecta) para cada par de clases
adyacentes. Así vemos **dónde** de la secuencia actúa cada propiedad física.
```

## Celda 82 · Código
```python
def poder_auc(col_clase, pares, columnas):
    datos = df[es_train]
    tabla = {}
    for a, b in pares:
        s = datos[datos[col_clase].isin([a, b])]
        fila = {}
        for f in columnas:
            ok = s[f].notna()
            if ok.sum() < 10 or s.loc[ok, col_clase].nunique() < 2:
                fila[f] = np.nan
                continue
            auc = roc_auc_score(s.loc[ok, col_clase] == b, s.loc[ok, f])
            fila[f] = 2 * abs(auc - 0.5)
        tabla[f'{a} vs {b}'] = fila
    return pd.DataFrame(tabla)

orden_det = ORDEN_CLASES['detallado']
AUC_VECINOS = poder_auc('y_detallado', list(zip(orden_det[:-1], orden_det[1:])), FEATURES)
AUC_TRES = poder_auc('y_tres_clases', [('Temprana', 'Intermedia'), ('Intermedia', 'Tardía'), ('Temprana', 'Tardía')], FEATURES)

fig, axs = plt.subplots(1, 2, figsize=(22, 14), gridspec_kw={'width_ratios': [4, 1.3]})
sns.heatmap(AUC_VECINOS, cmap='magma', vmin=0, vmax=1, annot=True, fmt='.1f', annot_kws={'size': 6}, ax=axs[0])
axs[0].set_title('Poder 2|AUC − 0.5| entre T-types vecinos')
sns.heatmap(AUC_TRES, cmap='magma', vmin=0, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 7}, ax=axs[1])
axs[1].set_title('Tres clases')
plt.tight_layout(); plt.show()

mejores = pd.concat([AUC_VECINOS, AUC_TRES], axis=1).apply(
    lambda col: ', '.join(f'{f} ({v:.2f})' for f, v in col.nlargest(3).items()))
display(mejores.to_frame('3 mejores features'))
```

## Celda 83 · Markdown
```markdown
**Conclusión 10.6** (es la tabla más reveladora del análisis):
* **E vs S0:** el poder espectral es **bajo** (≲ 0.3). Los mejores separadores deberían ser `b_a`, log σ, log Σ* o la masa,
  no la edad. Es consistente con que las S0 sean discos que dejaron de formar estrellas (van den Bergh 1976; Kormendy & Bender
  2012).
* **S0a vs Sa y Sa vs Sab:** aquí suelen estar los saltos de **sSFR, EW(Hα) y D4000**. Es la transición de la secuencia roja a
  la nube azul.
* **Sb … Sd:** el poder es pequeño en cada par (~0.1–0.2). La secuencia continúa de forma **gradual** en edad, sSFR y Σ*, sin
  una frontera espectral clara.
* **Sdm vs Sm vs Im:** el poder es casi nulo. Estas clases se distinguen por la estructura visual (brazos frente a irregularidad),
  que el espectro integrado no ve.
* Con **tres clases**, Temprana vs Tardía debería separarse casi perfectamente (≳ 0.9). La dificultad se concentra en la
  Intermedia.

### 10.7 Ranking de consenso
Para cada esquema se promedia el **rango** de cada representante en cuatro criterios: ε², MI, estabilidad y permutación.
Promediar rangos evita que un solo método domine (es la idea de los *ensembles* de selección).
```

## Celda 84 · Código
```python
CONSENSO, TOP10 = {}, {}
for e in ESQUEMAS:
    r = pd.DataFrame({'epsilon2': FILTROS[e].loc[REPS, 'epsilon2'], 'MI': FILTROS[e].loc[REPS, 'MI'],
                      'estabilidad': ESTAB[e], 'permutacion': PERM[e]['media']})
    t = r.assign(rango_medio=r.rank(ascending=False).mean(axis=1)).sort_values('rango_medio')
    t['rho_ordinal'] = FILTROS[e].loc[t.index, 'rho_ordinal']
    t['familia'] = t.index.map(FAMILIA_DE)
    t['significado'] = t.index.map(lambda f: DESCRIPCION.get(f, ''))
    t['representa_a'] = t.index.map(lambda f: ', '.join(representa[f]))
    CONSENSO[e] = t
    TOP10[e] = t.index[:10].tolist()
    print(f'\n=== {e} ===')
    display(t.head(15).round(3))
    t.to_csv(os.path.join(RUTA_RES, f'consenso_{e}.csv'))
```

## Celda 85 · Markdown
```markdown
### 10.8 ¿Cuántas features hacen falta?
Se mide la balanced accuracy en validación al usar las k primeras features del consenso. Es ligeramente optimista, porque la
permutación también usó validación, pero sirve para ver la **forma** de la curva.
```

## Celda 86 · Código
```python
ks = sorted({k for k in [1, 2, 3, 4, 5, 7, 10, 15, len(REPS)] if k <= len(REPS)})
curva = []
for e in ESQUEMAS:
    for k in ks:
        cols = CONSENSO[e].index[:k].tolist()
        m = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_ajuste, cols], df.loc[idx_ajuste, f'y_{e}'])
        curva.append({'esquema': e, 'k': k, 'balanced_acc':
                      balanced_accuracy_score(df.loc[idx_valid, f'y_{e}'], m.predict(df.loc[idx_valid, cols]))})
curva = pd.DataFrame(curva)

plt.figure(figsize=(8, 5))
for e in ESQUEMAS:
    c = curva[curva['esquema'] == e]
    plt.plot(c['k'], c['balanced_acc'], 'o-', label=e)
    plt.axhline(1 / len(ORDEN_CLASES[e]), ls=':', color=plt.gca().lines[-1].get_color())   # nivel de azar
plt.xscale('log'); plt.xlabel('número de features (k)'); plt.ylabel('balanced accuracy (validación)')
plt.legend(); plt.title('Rendimiento vs número de features (punteado = azar)'); plt.show()
```

## Celda 87 · Markdown
```markdown
**Conclusión 10.8:** la curva debería **saturarse en k ≈ 3–5**. Con una feature de edad (D4000n o HdA), una de formación estelar
actual (sSFR o EW(Hα)) y una de estructura (Σ* o σ) se obtiene casi todo el rendimiento. El resto aporta poco. Esto refuerza la
idea física: **la morfología espectral es esencialmente de baja dimensión** (edad + estructura).

### 10.9 Análisis discriminante lineal (LDA)
LDA (Fisher 1936) encuentra las combinaciones lineales que mejor separan las clases. Sus coeficientes estandarizados son
directamente interpretables. Lo aplicamos al esquema de tres clases con su top-10.
```

## Celda 88 · Código
```python
cols = TOP10['tres_clases']
prep = Pipeline([('imputar', SimpleImputer(strategy='median')), ('escalar', StandardScaler())])
X_lda = prep.fit_transform(df.loc[idx_train, cols])
y_lda = df.loc[idx_train, 'y_tres_clases']
lda = LinearDiscriminantAnalysis(n_components=2).fit(X_lda, y_lda)
print('Varianza discriminante explicada:', lda.explained_variance_ratio_.round(3))
display(pd.DataFrame(lda.scalings_[:, :2], index=cols, columns=['LD1', 'LD2']).sort_values('LD1').round(2))

proy_lda = lda.transform(X_lda)
plt.figure(figsize=(7, 6))
for cl, col in zip(ORDEN_CLASES['tres_clases'], ['tab:red', 'tab:green', 'tab:blue']):
    m = (y_lda == cl).values
    plt.scatter(proy_lda[m, 0], proy_lda[m, 1], s=4, alpha=0.4, color=col, label=cl)
plt.xlabel('LD1'); plt.ylabel('LD2'); plt.legend(); plt.title('Proyección LDA (tres clases)'); plt.show()
```

## Celda 89 · Markdown
```markdown
**Conclusión 10.9:**
* **LD1** debería explicar > 90 % de la separación y combinar edad (D4000n), formación estelar (sSFR) y estructura (Σ*, σ): es un
  **"eje de Hubble espectral"**.
* **LD2**, si es relevante, suele separar la Intermedia, por ejemplo mediante Δedad (un bulbo viejo con un disco joven).
* Que una sola dimensión domine confirma que las tres clases forman una **secuencia continua**, no tres grupos aislados.
```

## Celda 90 · Markdown
```markdown
# 11. Comparación de los tres esquemas de clasificación (en TEST)
Cada esquema se entrena con su top-10 de consenso y su mejor balanceo, sobre **todo train**, y se evalúa **una sola vez** en test.
* **Mejora sobre el azar** = (BA − 1/K)/(1 − 1/K). Permite comparar esquemas con distinto número de clases K.
* También proyectamos las predicciones detalladas y por intervalos a tres clases, para ver si **más detalle ayuda o perjudica**
  a la clasificación gruesa.
```

## Celda 91 · Código
```python
COMPARACION, PRED_TEST = [], {}
fig, axs = plt.subplots(1, 3, figsize=(26, 8), gridspec_kw={'width_ratios': [2, 1, 1.4]})
for ax, e in zip(axs, ESQUEMAS):
    cols = TOP10[e]
    m = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_train, cols], df.loc[idx_train, f'y_{e}'])
    y_t = df.loc[idx_test, f'y_{e}']
    pred = m.predict(df.loc[idx_test, cols])
    PRED_TEST[e] = pred
    K = len(ORDEN_CLASES[e])
    met = metricas(y_t, pred, e, df.loc[idx_test, 'Ttype'])
    met['azar'] = 1 / K
    met['mejora_sobre_azar'] = (met['balanced_acc'] - 1 / K) / (1 - 1 / K)
    COMPARACION.append({'esquema': e, 'n_clases': K, 'balanceo': MEJOR_BALANCEO[e], **met})
    ConfusionMatrixDisplay.from_predictions(y_t, pred, labels=ORDEN_CLASES[e], normalize='true', values_format='.2f',
                                            ax=ax, colorbar=False, xticks_rotation=60)
    ax.set_title(f'{e} (normalizada por fila)')
plt.tight_layout(); plt.show()
COMPARACION = pd.DataFrame(COMPARACION).set_index('esquema')
display(COMPARACION.round(3))
```

## Celda 92 · Código
```python
# ¿Las mismas features importan en todos los esquemas? (Spearman entre importancias por permutación)
imp = pd.DataFrame({e: PERM[e]['media'] for e in ESQUEMAS})
display(imp.corr(method='spearman').round(2))
print('Features en el top-10 de los tres esquemas:', sorted(set.intersection(*[set(TOP10[e]) for e in ESQUEMAS])))

# Proyectar todas las predicciones a tres clases
A_TRES = {'detallado': {NOMBRE_TTYPE[t]: esquema_tres(t) for t in ORDEN_TTYPE if t != 11},
          'intervalos': {'E': 'Temprana', 'S0–S0a': 'Temprana', 'Sa–Sab': 'Intermedia',
                         'Sb–Sbc': 'Intermedia', 'Sc–Sd': 'Tardía', 'Sdm–Im': 'Tardía'},
          'tres_clases': {cl: cl for cl in ORDEN_CLASES['tres_clases']}}
y3 = df.loc[idx_test, 'y_tres_clases']
proyeccion = pd.DataFrame({e: {'balanced_acc_3c': balanced_accuracy_score(y3, pd.Series(PRED_TEST[e]).map(A_TRES[e])),
                               'F1_macro_3c': f1_score(y3, pd.Series(PRED_TEST[e]).map(A_TRES[e]), average='macro')}
                           for e in ESQUEMAS}).T
display(proyeccion.round(3))
```

## Celda 93 · Markdown
```markdown
### 11.1 Ventajas y desventajas de cada esquema

| | **T-type detallado (13 clases)** | **Temprana / Intermedia / Tardía** | **Intervalos de T (6 clases)** |
|---|---|---|---|
| **Ventajas** | Máxima resolución física; conserva el orden completo de de Vaucouleurs; permite métricas ordinales finas (MAE en T) | Clases grandes y balanceadas (~2:1); fronteras con significado físico (secuencia roja, discos con bulbo, discos puros); resultados estables e interpretables | Compromiso: absorbe la incertidumbre de ±1–2 T de los clasificadores humanos; mantiene la E separada de las S0; desbalance moderado |
| **Desventajas** | Desbalance ~100:1; la diferencia entre vecinos (~1 T) es menor que el error de la etiqueta (σ ≈ 1.8 T; Naim et al. 1995); recall ≈ 0 en Sdm–Im; las métricas nominales castigan igual errores de 1 y 8 T | Pierde información: mezcla E con S0 (dos poblaciones distintas) y Sa con Sbc; la clase Intermedia es heterogénea y es la que más se confunde; los resultados dependen del corte elegido (T ≤ 0 y T ≥ 5) | Los cortes siguen siendo arbitrarios; Sdm–Im queda pequeña (~180); las métricas siguen siendo nominales aunque la etiqueta es ordinal |
| **Mejor uso** | Regresión de T o clasificación **ordinal**; estudiar tendencias continuas | Preguntas de evolución (quenching, bimodalidad) | Catálogos de clasificación automática "práctica" |

**Conclusión 11** (verifícala con tus resultados):
* La **mejora sobre el azar** debería ser máxima en tres clases y mínima en el esquema detallado. El κ cuadrático y el acierto ±1
  muestran que los errores del esquema detallado son casi siempre **entre vecinos**: el modelo aprende el orden, pero no la
  frontera fina.
* La correlación alta entre importancias (ρ ≳ 0.7) indica que **la misma física** (edad, formación estelar y estructura)
  sostiene los tres esquemas. Cambiar de esquema cambia la **resolución**, no la física.
* Si el detallado proyectado a tres clases rinde igual que el modelo directo, entrenar con más detalle no perjudica. Si rinde
  peor, el ruido de la etiqueta fina se propaga.
* **Recomendación:** tratar T como variable **ordinal**, por ejemplo con la descomposición en K − 1 clasificadores binarios
  "T > k" (Frank & Hall 2001) o con regresión ordinal. Así se aprovecha el orden sin pagar el desbalance del esquema detallado.
```

## Celda 94 · Markdown
```markdown
# 12. Conclusiones generales
1. **Calidad de datos.**
   * `ERROR_FoV` es incorrecto: se sustituyó por DER_SNR, y `adev` por χ².
   * El v0 de STARLIGHT contiene el offset aire→vacío (≈ 83 km/s), que se corrigió (el EDA lo trataba como velocidad).
   * El catálogo tiene centinelas (z = −9999), topes (incl = 81.37°), valores de Petrosian repetidos, etiquetas mezcladas
     (dSph en T = 10 y T = −5, mergers en T = 11, `Sa` con T = 3) y **9 galaxias observadas dos veces**.
   * Todo esto se trató antes del modelado, y el cruce se validó con `arq_obs`, v0, coordenadas y DRPall.
2. **Qué distingue la morfología.** Tres ejes físicos casi independientes:
   * **Edad estelar / historia de formación** (D4000n, HδA, edad STARLIGHT, PC1): es la más informativa en general.
   * **Formación estelar actual** (sSFR, EW(Hα)): marca el salto entre tempranas y espirales.
   * **Estructura y cinemática** (Σ*, σ, M*): es la única que separa bien **E de S0** y aporta información que el espectro de
     poblaciones no contiene.
   * La química ([MgFe]′, Mg b/⟨Fe⟩) y el polvo aportan matices (formación rápida en las E; discos polvorientos en Sb–Sc), pero
     son en gran parte redundantes con la masa y la edad.
3. **Significado astrofísico.** El espectro integrado ve la **historia de formación estelar**, que está acoplada a la morfología
   a través de la masa del bulbo (quenching morfológico; Martig et al. 2009) y del entorno (Dressler 1980). Por eso:
   * Clasifica bien la dicotomía **temprana/tardía**.
   * Pierde resolución donde la morfología depende de detalles **geométricos** (brazos, barras, anillos), como Sb–Sd y Sdm–Im.
   * Falla cuando morfología y color se desacoplan, como en las espirales rojas (Masters et al. 2010) y las elípticas azules
     (Schawinski et al. 2009).
4. **Techo intrínseco.** Con σ ≈ 1.8 T entre clasificadores humanos, una precisión baja en el esquema detallado **no es un
   fracaso del modelo**, sino un límite de la etiqueta.
5. **Próximos pasos:**
   * Añadir parámetros estructurales fotométricos (índice de Sérsic, concentración, B/T de NSA o de Fischer et al. 2019).
   * Usar el perfil radial de MaNGA, comparando el espectro dentro de 1 R_e con el de todo el campo (`FLUX_1RE` frente a
     `FLUX_FoV`), como medida espectroscópica del gradiente bulbo/disco.
   * Probar clasificación ordinal.

## Referencias
* Abdurro'uf et al. 2022, ApJS 259, 35 — SDSS DR17
* Asari N. V. et al. 2007, MNRAS 381, 263
* Astropy Collaboration 2022, ApJ 935, 167
* Baldry I. K. et al. 2004, ApJ 600, 681
* Baldwin J. A., Phillips M. M., Terlevich R. 1981, PASP 93, 5
* Balogh M. L. et al. 1999, ApJ 527, 54
* Bamford S. P. et al. 2009, MNRAS 393, 1324
* Blanton M. R., Moustakas J. 2009, ARA&A 47, 159
* Breiman L. 2001, Machine Learning 45, 5
* Bruzual G., Charlot S. 2003, MNRAS 344, 1000
* Bundy K. et al. 2015, ApJ 798, 7
* Buta R. J. 2013, "Galaxy Morphology", en *Planets, Stars and Stellar Systems* Vol. 6, Springer
* Cappellari M. et al. 2011, MNRAS 416, 1680
* Cardelli J. A., Clayton G. C., Mathis J. S. 1989, ApJ 345, 245
* Chawla N. V. et al. 2002, JAIR 16, 321
* Cid Fernandes R. et al. 2005, MNRAS 358, 363
* Cid Fernandes R. et al. 2011, MNRAS 413, 1687
* Ciddor P. E. 1996, Applied Optics 35, 1566
* Cohen J. 1968, Psychological Bulletin 70, 213
* Concas A. et al. 2019, A&A 622, A188
* Connolly A. J. et al. 1995, AJ 110, 1071
* de Vaucouleurs G. 1959, *Handbuch der Physik* 53, 275
* de Vaucouleurs G. et al. 1991, *Third Reference Catalogue of Bright Galaxies* (RC3), Springer
* Díaz A. I., Terlevich E., Terlevich R. 1989, MNRAS 239, 325
* Domínguez Sánchez H. et al. 2022, MNRAS 509, 4024
* Dressler A. 1980, ApJ 236, 351
* Drory N. et al. 2015, AJ 149, 77
* Emsellem E. et al. 2011, MNRAS 414, 888
* Faber S. M., Jackson R. E. 1976, ApJ 204, 668
* Fischer J.-L., Domínguez Sánchez H., Bernardi M. 2019, MNRAS 483, 2057
* Fisher R. A. 1936, Annals of Eugenics 7, 179
* Frank E., Hall M. 2001, ECML, LNCS 2167, 145
* Gallazzi A. et al. 2005, MNRAS 362, 41
* Hubble E. P. 1926, ApJ 64, 321
* Hubble E. P. 1936, *The Realm of the Nebulae*, Yale UP
* Ivezić Ž., Connolly A. J., VanderPlas J. T., Gray A. 2014, *Statistics, Data Mining, and Machine Learning in Astronomy*,
  Princeton UP (astroML)
* Kauffmann G. et al. 2003a, MNRAS 341, 33
* Kauffmann G. et al. 2003b, MNRAS 341, 54
* Kauffmann G. et al. 2003c, MNRAS 346, 1055
* Kennicutt R. C. 1992, ApJ 388, 310
* Kennicutt R. C. 1998, ARA&A 36, 189
* Kennicutt R. C., Evans N. J. 2012, ARA&A 50, 531
* Kewley L. J. et al. 2001, ApJ 556, 121
* Kewley L. J. et al. 2006, MNRAS 372, 961
* Kormendy J. 1985, ApJ 295, 73
* Kormendy J. et al. 2009, ApJS 182, 216
* Kormendy J., Bender R. 2012, ApJS 198, 2
* Lahav O. et al. 1995, Science 267, 859
* Law D. R. et al. 2016, AJ 152, 83
* Le Borgne J.-F. et al. 2003, A&A 402, 433
* Lemaître G., Nogueira F., Aridas C. K. 2017, JMLR 18, 1
* Martig M. et al. 2009, ApJ 707, 250
* Masters K. L. et al. 2010, MNRAS 405, 783
* Meinshausen N., Bühlmann P. 2010, JRSS B 72, 417
* Nair P. B., Abraham R. G. 2010, ApJS 186, 427
* Naim A. et al. 1995, MNRAS 274, 1107
* Osterbrock D. E., Ferland G. J. 2006, *Astrophysics of Gaseous Nebulae and AGN*, University Science Books
* Pedregosa F. et al. 2011, JMLR 12, 2825
* Pettini M., Pagel B. E. J. 2004, MNRAS 348, L59
* Planck Collaboration 2020, A&A 641, A6
* Roberts M. S., Haynes M. P. 1994, ARA&A 32, 115
* Sandage A. 1961, *The Hubble Atlas of Galaxies*, Carnegie Institution
* Sandage A. 2005, ARA&A 43, 581
* Schawinski K. et al. 2009, MNRAS 396, 818
* Schawinski K. et al. 2014, MNRAS 440, 889
* Simien F., de Vaucouleurs G. 1986, ApJ 302, 564
* Stoehr F. et al. 2008, ASP Conf. Ser. 394, 505
* Strateva I. et al. 2001, AJ 122, 1861
* Thomas D., Maraston C., Bender R. 2003, MNRAS 339, 897
* Thomas D. et al. 2005, ApJ 621, 673
* Tibshirani R. 1996, JRSS B 58, 267
* Tomczak M., Tomczak E. 2014, Trends in Sport Sciences 21, 19
* Trager S. C. et al. 1998, ApJS 116, 1
* van den Bergh S. 1976, ApJ 206, 883
* Vázquez-Mata J. A. et al. 2022, MNRAS 512, 2222
* Wake D. A. et al. 2017, AJ 154, 86
* Worthey G. et al. 1994, ApJS 94, 687
* Worthey G., Ottaviani D. L. 1997, ApJS 111, 377
* Yip C. W. et al. 2004, AJ 128, 585
```

## @maria-rabelero-mx_nestle

show me the cells

## Copilot

Sigo desde la **Celda 54**. La vuelvo a poner completa porque quedó cortada.

## Celda 54 · Código
```python
SN_MIN = 3
def detectada(*lineas):
    """True si TODAS las líneas tienen S/N ≥ SN_MIN."""
    ok = pd.Series(True, index=df.index)
    for l in lineas:
        ok &= df[f'SN_{l}'] >= SN_MIN
    return ok

def log_cociente(num, den):
    """log10(F_num/F_den), solo si ambas líneas están detectadas."""
    return np.log10(df[f'F_{num}'] / df[f'F_{den}']).where(detectada(num, den))

z_ok = df['nsa_z'] >= 0.005                                   # distancia fiable

# --- 6.1 Edad e historia de formación estelar ---
df['Delta_edad'] = df['logEdad_M'] - df['logEdad_L']

# --- 6.2 Formación estelar actual ---
for linea in ['Ha_6563', 'Hb_4861', 'OII_3727', 'NII_6584', 'OIII_5007', 'SII_6717']:
    df[f'log_EW_{linea.split("_")[0]}'] = np.log10(df[f'EW_{linea}'].clip(lower=0.1))   # piso de 0.1 Å
df['Ha_detectada'] = detectada('Ha_6563').astype(int)

# --- 6.3 Ionización ---
df['log_NII_Ha'] = log_cociente('NII_6584', 'Ha_6563')
df['log_OIII_Hb'] = log_cociente('OIII_5007', 'Hb_4861')
df['log_OI_Ha'] = log_cociente('OI_6300', 'Ha_6563')
ok_sii = detectada('SII_6717', 'SII_6731', 'Ha_6563')
df['log_SII_Ha'] = np.log10((df['F_SII_6717'] + df['F_SII_6731']) / df['F_Ha_6563']).where(ok_sii)

# BPT (Baldwin et al. 1981; Kauffmann et al. 2003c; Kewley et al. 2001, 2006)
ka03     = lambda x: 0.61 / (x - 0.05) + 1.30
ke01_nii = lambda x: 0.61 / (x - 0.47) + 1.19
def clase_bpt(x, y, x_sii):
    if np.isnan(x) or np.isnan(y):    return 'Sin líneas'
    if x < 0.05 and y < ka03(x):      return 'SF'
    if x < 0.47 and y < ke01_nii(x):  return 'Compuesta'
    if not np.isnan(x_sii):           return 'Seyfert' if y > 1.89 * x_sii + 0.76 else 'LINER'
    return 'Seyfert' if y > 1.05 * x + 0.45 else 'LINER'
df['BPT'] = [clase_bpt(x, y, s) for x, y, s in zip(df['log_NII_Ha'], df['log_OIII_Hb'], df['log_SII_Ha'])]

# WHAN (Cid Fernandes et al. 2011): también clasifica a las galaxias de líneas débiles
def clase_whan(ew_ha, ew_nii, x):
    if np.isnan(ew_ha):               return 'Sin datos'
    if ew_ha < 0.5 and ew_nii < 0.5:  return 'Pasiva'
    if ew_ha < 3:                     return 'Retirada'
    if np.isnan(x):                   return 'Indeterminada'
    if x < -0.4:                      return 'SF'
    return 'sAGN' if ew_ha > 6 else 'wAGN'
x_whan = np.log10(df['F_NII_6584'] / df['F_Ha_6563']).where(detectada('Ha_6563') & (df['SN_NII_6584'] >= 2))
df['WHAN'] = [clase_whan(a, b, x) for a, b, x in zip(df['EW_Ha_6563'], df['EW_NII_6584'], x_whan)]
```

## Celda 55 · Código
```python
# --- 6.4 Polvo ---
ok_balmer = detectada('Ha_6563', 'Hb_4861')
cociente_balmer = (df['F_Ha_6563'] / df['F_Hb_4861']).where(ok_balmer)
df['log_HaHb'] = np.log10(cociente_balmer)
# Decremento de Balmer: caso B, Hα/Hβ = 2.86 (Osterbrock & Ferland 2006); ley de Cardelli et al. (1989),
# k(Hβ) = 3.61 y k(Hα) = 2.53
df['AV_gas'] = (3.1 * 2.5 / (3.61 - 2.53) * np.log10(cociente_balmer / 2.86)).clip(lower=0)
df['NaD_ISM'] = df['NaD'] - df['NaD_syn']                     # exceso de Na D sobre el modelo estelar = gas frío

# --- 6.5 Química ---
fe_medio = (df['Fe5270'] + df['Fe5335']) / 2
fe_ponderado = 0.72 * df['Fe5270'] + 0.28 * df['Fe5335']      # Thomas et al. 2003
df['MgFe'] = np.sqrt((df['Mgb'] * fe_ponderado).where((df['Mgb'] > 0) & (fe_ponderado > 0)))
df['Mgb_Fe'] = (df['Mgb'] / fe_medio).where(fe_medio > 0.5)   # evita dividir por ~0
df['CaT'] = df['CaT1'] + df['CaT2'] + df['CaT3']
df['OH_N2'] = (8.90 + 0.57 * df['log_NII_Ha']).where(df['BPT'] == 'SF')   # Pettini & Pagel 2004, solo regiones H II

# --- 6.6 SFR específica a partir de Hα corregido por polvo (Kennicutt & Evans 2012) ---
A_Ha = (2.53 / 3.1) * df['AV_gas'].fillna(0)
d_L = cosmo.luminosity_distance(df['nsa_z'].values).to(u.cm).value
L_Ha = 4 * np.pi * d_L ** 2 * df['F_Ha_6563'] * 1e-17 * 10 ** (0.4 * A_Ha)   # supone flujo en 1e-17 erg/s/cm²
log_ssfr = np.log10(L_Ha) - 41.27 - df['LogMass']
log_ssfr = log_ssfr.where(detectada('Ha_6563'), -14)          # sin Hα detectado: límite inferior (pasiva)
df['log_sSFR_Ha'] = log_ssfr.clip(lower=-14).where(z_ok & df['LogMass'].notna())

# --- 6.7 Masa, cinemática y estructura ---
kpc_por_arcsec = cosmo.kpc_proper_per_arcmin(df['nsa_z'].values).to(u.kpc / u.arcsec).value
R90_kpc = df['PETRO_TH90'] * kpc_por_arcsec
df['log_R90_kpc'] = np.log10(R90_kpc).where(z_ok)
df['log_Sigma90'] = (df['LogMass'] - np.log10(np.pi * R90_kpc ** 2)).where(z_ok)   # densidad superficial de masa
df['log_sigma'] = np.log10(df['sigma_estelar'].clip(lower=50))   # piso: por debajo de la resolución instrumental
df['b_a'] = np.cos(np.radians(df['incl']))                    # cociente de ejes aparente
```

## Celda 56 · Markdown
```markdown
### 6.8 Familias físicas
Cada feature pertenece a **una** familia. Más adelante mediremos la importancia de cada familia completa (10.5).

> **Nota sobre los NaN:** los cocientes de líneas no existen en galaxias sin emisión (casi todas las E y S0). Imputarlos con la
> mediana les daría valores "de espiral". Por eso existe `Ha_detectada`, y en la sección 7 se descartan las features con > 50 %
> de NaN en train.

> **`b_a` es una variable de control:** la orientación no es morfología. Si resultara muy importante, el modelo estaría usando
> la forma aparente (las E son más redondas).
```

## Celda 57 · Código
```python
FAMILIAS = {
    'Edad / historia': ['D4000n', 'HdA', 'HgA', 'Hb_Lick', 'G4300', 'CaII_K', 'HdA_syn', 'HgA_syn', 'D4000n_syn',
                        'color_esp', 'logEdad_L', 'logEdad_M', 'Delta_edad', 'xY', 'xI', 'xO', 'mY_1Gyr'],
    'Forma espectral (PCA)': ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
    'Formación estelar actual': ['log_EW_Ha', 'log_EW_Hb', 'log_EW_OII', 'log_sSFR_Ha', 'Ha_detectada'],
    'Química': ['Mgb', 'Fe5270', 'Fe5335', 'MgFe', 'Mgb_Fe', 'logZ_L', 'logZ_M', 'CaT', 'OH_N2'],
    'Polvo': ['AV_estelar', 'AV_gas', 'log_HaHb', 'NaD', 'NaD_ISM'],
    'Ionización': ['log_NII_Ha', 'log_OIII_Hb', 'log_SII_Ha', 'log_OI_Ha', 'log_EW_NII', 'log_EW_OIII', 'log_EW_SII'],
    'Masa, cinemática y estructura': ['LogMass', 'log_sigma', 'log_R90_kpc', 'log_Sigma90', 'b_a'],
}
FEATURES = [f for cols in FAMILIAS.values() for f in cols]
FAMILIA_DE = {f: fam for fam, cols in FAMILIAS.items() for f in cols}

DESCRIPCION = {
    'D4000n': 'Quiebre en 4000 Å: edad media (↑ = vieja)', 'HdA': 'Absorción Hδ: estrellas A, brote hace 0.1–1 Gyr',
    'HgA': 'Absorción Hγ: como HdA', 'Hb_Lick': 'Absorción Hβ: edad (afectada por emisión)',
    'G4300': 'Banda G (CH): estrellas frías y viejas', 'CaII_K': 'Ca II K: estrellas viejas / metalicidad',
    'HdA_syn': 'Hδ estelar sin emisión', 'HgA_syn': 'Hγ estelar sin emisión', 'D4000n_syn': 'D4000n del modelo estelar',
    'color_esp': 'Color del continuo (≈ g − r)', 'logEdad_L': 'Edad media pesada en luz',
    'logEdad_M': 'Edad media pesada en masa', 'Delta_edad': 'Edad masa − luz: estrellas jóvenes sobre una base vieja',
    'xY': 'Luz de estrellas < 100 Myr', 'xI': 'Luz de 0.1–1 Gyr', 'xO': 'Luz de > 1 Gyr', 'mY_1Gyr': 'Masa formada en el último Gyr',
    'log_EW_Ha': 'EW(Hα): formación estelar relativa al continuo', 'log_EW_Hb': 'EW(Hβ) en emisión',
    'log_EW_OII': 'EW([OII]): formación estelar / ionización', 'log_sSFR_Ha': 'SFR(Hα)/M*: tasa específica',
    'Ha_detectada': 'Hay gas ionizado (Hα S/N ≥ 3)', 'Mgb': 'Mg b: metalicidad / α', 'Fe5270': 'Hierro',
    'Fe5335': 'Hierro', 'MgFe': "[MgFe]': metalicidad total", 'Mgb_Fe': 'Mg b/<Fe>: [α/Fe], rapidez de formación',
    'logZ_L': 'Metalicidad estelar (luz)', 'logZ_M': 'Metalicidad estelar (masa)', 'CaT': 'Triplete Ca II: gigantes rojas',
    'OH_N2': 'Abundancia de O del gas', 'AV_estelar': 'Extinción de las estrellas', 'AV_gas': 'Extinción del gas (Balmer)',
    'log_HaHb': 'Decremento de Balmer', 'NaD': 'Na D total', 'NaD_ISM': 'Na D interestelar (gas frío y polvo)',
    'log_NII_Ha': 'Ionización / metalicidad del gas', 'log_OIII_Hb': 'Grado de ionización', 'log_SII_Ha': 'Choques / gas difuso',
    'log_OI_Ha': 'Choques / AGN', 'log_EW_NII': 'EW([NII])', 'log_EW_OIII': 'EW([OIII])', 'log_EW_SII': 'EW([SII])',
    'LogMass': 'Masa estelar', 'log_sigma': 'Dispersión de velocidades: soporte por presión',
    'log_R90_kpc': 'Tamaño físico', 'log_Sigma90': 'Densidad superficial de masa: concentración / bulbo',
    'b_a': 'Cociente de ejes aparente (control)',
}
DESCRIPCION.update({f'PC{k}': f'Componente principal {k} del espectro' for k in range(1, 6)})
print(len(FEATURES), 'features en', len(FAMILIAS), 'familias')
```

## Celda 58 · Código
```python
def caja_por_ttype(columnas, ncol=4):
    """Diagrama de cajas de cada feature por T-type (solo train)."""
    nfil = int(np.ceil(len(columnas) / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(5 * ncol, 3.3 * nfil))
    orden = [t for t in ORDEN_TTYPE if t in df['Ttype'].unique()]
    for ax, col in zip(np.ravel(axs), columnas):
        sns.boxplot(data=df[es_train], x='Ttype', y=col, order=orden,
                    palette=[COLOR_TTYPE[t] for t in orden], fliersize=0, ax=ax)
        ax.set_xticklabels([NOMBRE_TTYPE[t] for t in orden], rotation=60, fontsize=7); ax.set_xlabel('')
        lo, hi = df[col].quantile([0.01, 0.99]); ax.set_ylim(lo, hi)
    for ax in np.ravel(axs)[len(columnas):]:
        ax.axis('off')
    plt.tight_layout(); plt.show()

caja_por_ttype(['Delta_edad', 'log_sSFR_Ha', 'MgFe', 'Mgb_Fe', 'AV_gas', 'NaD_ISM',
                'log_NII_Ha', 'OH_N2', 'log_sigma', 'log_R90_kpc', 'log_Sigma90', 'b_a'])

# ¿Qué tipo de ionización domina en cada T-type? (porcentaje por columna)
display((pd.crosstab(df['BPT'], df['Ttype'], normalize='columns') * 100).round(0))
display((pd.crosstab(df['WHAN'], df['Ttype'], normalize='columns') * 100).round(0))
```

## Celda 59 · Markdown
```markdown
**Conclusión 6** (verifícala con tus resultados):
* **log Σ*₉₀ y log σ** deberían bajar de forma monótona de E a Im. Trazan la importancia del bulbo, el criterio principal de
  Hubble (Kauffmann et al. 2003b; Fischer et al. 2019).
* **log sSFR(Hα)** debería mostrar un **salto** entre S0a y Sa–Sb: el paso de la secuencia roja a la nube azul.
* **Mg b/⟨Fe⟩** debería ser mayor en las E masivas, que formaron sus estrellas rápido, antes del hierro de las SN Ia
  (Thomas et al. 2005).
* **Δedad** debería ser máximo en las Sa–Sbc (bulbo viejo + disco joven).
* **WHAN:** las E y S0 deberían ser mayoritariamente "Pasiva/Retirada" (ionización por estrellas viejas; Cid Fernandes et al.
  2011), y las Sc–Im "SF".
```

## Celda 60 · Markdown
```markdown
# 7. Outliers, NaN y redundancia de las features finales
1. Se descartan las features con > 50 % de NaN en train.
2. Se revisan los outliers en train.
3. Se **winsorizan** los percentiles 0.5 y 99.5 de train: se recortan colas sin eliminar galaxias.
4. Se agrupan las features redundantes por Spearman, con la distancia 1 − |ρ|.
```

## Celda 61 · Código
```python
frac_nan = df.loc[es_train, FEATURES].isna().mean()
display(frac_nan[frac_nan > 0].sort_values(ascending=False).to_frame('frac_NaN_train').round(2))
descartadas = frac_nan[frac_nan > 0.5].index.tolist()
FEATURES = [f for f in FEATURES if f not in descartadas]
FAMILIAS = {fam: [f for f in cols if f in FEATURES] for fam, cols in FAMILIAS.items()}
print('Descartadas por NaN:', descartadas)

display(tabla_outliers(df[es_train], FEATURES))

limites = df.loc[es_train, FEATURES].quantile([0.005, 0.995])   # límites aprendidos SOLO con train
for f in FEATURES:
    if df[f].nunique() > 2:                                    # no recortamos binarias
        df[f] = df[f].clip(limites.loc[0.005, f], limites.loc[0.995, f])
```

## Celda 62 · Código
```python
rho = df.loc[es_train, FEATURES].corr(method='spearman').fillna(0)
distancia = 1 - np.abs(rho.values)
np.fill_diagonal(distancia, 0)
distancia = (distancia + distancia.T) / 2                     # simetría numérica exacta
enlace = hierarchy.linkage(squareform(distancia, checks=False), method='average')
grupos_corr = pd.Series(hierarchy.fcluster(enlace, t=0.15, criterion='distance'), index=FEATURES)   # |ρ| ≳ 0.85

plt.figure(figsize=(18, 5))
hierarchy.dendrogram(enlace, labels=FEATURES, color_threshold=0.15, leaf_rotation=90, leaf_font_size=8)
plt.axhline(0.15, color='r', ls='--'); plt.ylabel('1 − |ρ|'); plt.title('Agrupamiento de features redundantes')
plt.show()

for g, miembros in grupos_corr.groupby(grupos_corr):
    if len(miembros) > 1:
        print(f'Grupo {g}:', ', '.join(miembros.index))
```

## Celda 63 · Markdown
```markdown
**Conclusión 7:**
* Debería formarse un **gran grupo de "edad"** (D4000n, D4000n_syn, HdA, HgA, PC1, color, logEdad_L, xO, EW(Hα)…): gran parte del
  espectro mide **una sola variable física**, la proporción de estrellas jóvenes y viejas.
* Sin tener en cuenta esta redundancia, el "top-10" se llenaría con diez versiones de la edad y ocultaría la estructura y la
  química. Por eso en la sección 10 se usa **un representante por grupo**.
```

## Celda 64 · Markdown
```markdown
# 8. Esquemas de clasificación
| Esquema | Clases | Idea física |
|---|---|---|
| **Detallado** | 13 T-types (E … Im) | Resolución máxima de de Vaucouleurs |
| **Tres clases** | Temprana (T ≤ 0), Intermedia (1–4), Tardía (5–10) | Esferoides, espirales con bulbo, discos puros e irregulares |
| **Intervalos** | E, S0–S0a, Sa–Sab, Sb–Sbc, Sc–Sd, Sdm–Im | Pares de vecinos: absorben la incertidumbre de ±1–2 T |

T ≤ 0 como "temprana" es la convención habitual (Nair & Abraham 2010). El corte en T = 5 separa los discos con bulbo
(B/T ≳ 0.15) de los discos casi puros.
```

## Celda 65 · Código
```python
def esquema_tres(t):
    return 'Temprana' if t <= 0 else ('Intermedia' if t <= 4 else 'Tardía')

def esquema_intervalos(t):
    if t == -5: return 'E'
    if t <= 0:  return 'S0–S0a'
    if t <= 2:  return 'Sa–Sab'
    if t <= 4:  return 'Sb–Sbc'
    if t <= 7:  return 'Sc–Sd'
    return 'Sdm–Im'

ESQUEMAS = ['detallado', 'tres_clases', 'intervalos']
df['y_detallado'] = df['Ttype'].map(NOMBRE_TTYPE)
df['y_tres_clases'] = df['Ttype'].apply(esquema_tres)
df['y_intervalos'] = df['Ttype'].apply(esquema_intervalos)

ORDEN_CLASES = {'detallado': [NOMBRE_TTYPE[t] for t in ORDEN_TTYPE if t in set(df['Ttype'])],
                'tres_clases': ['Temprana', 'Intermedia', 'Tardía'],
                'intervalos': ['E', 'S0–S0a', 'Sa–Sab', 'Sb–Sbc', 'Sc–Sd', 'Sdm–Im']}
# T representativo de cada clase (mediana en train): permite medir el error en unidades de T
T_REP = {e: df[es_train].groupby(f'y_{e}')['Ttype'].median().to_dict() for e in ESQUEMAS}

for e in ESQUEMAS:
    n = df.loc[es_train, f'y_{e}'].value_counts().reindex(ORDEN_CLASES[e])
    print(f'{e}: {len(n)} clases | desbalance máx/mín = {n.max() / n.min():.0f}')
    display(n.to_frame('n_train').T)
```

## Celda 66 · Markdown
```markdown
**Conclusión 8:** el esquema detallado tiene un desbalance de ~100:1 (E ≈ 1800 frente a Im ≈ 15), así que Sdm, Sm e Im son casi
imposibles de aprender sin balanceo. Tres clases y los intervalos reducen el desbalance a ~2:1 y ~10:1.
```

## Celda 67 · Markdown
```markdown
# 9. Balanceo de clases
Comparamos cuatro estrategias **dentro de un pipeline**. El remuestreo se aplica solo a los pliegues de entrenamiento: nunca se
crean galaxias sintéticas en los datos de evaluación.

| Estrategia | Cómo funciona | Riesgo |
|---|---|---|
| ninguno | — | ignora las clases raras |
| class_weight | pesa más los errores en las clases raras | casi ninguno |
| submuestreo | reduce todas las clases al tamaño de la menor | tira datos |
| SMOTE (Chawla et al. 2002) | interpola galaxias sintéticas entre vecinas | espectros "imposibles", sobreajuste |

**Métricas** (no usamos *accuracy*, que favorece a la clase mayoritaria):
* **Balanced accuracy:** recall medio por clase.
* **F1 macro.**
* **κ ponderado cuadrático** (Cohen 1968): adecuado para una etiqueta **ordinal**.
* **Acierto ±1 clase.**
* **MAE en unidades de T.**
```

## Celda 68 · Código
```python
def hacer_modelo(estrategia, n_arboles=300):
    """Pipeline: imputar (mediana) -> estandarizar -> [balancear] -> Random Forest (Breiman 2001)."""
    pasos = [('imputar', SimpleImputer(strategy='median')), ('escalar', StandardScaler())]
    if estrategia == 'submuestreo':
        pasos.append(('balancear', RandomUnderSampler(random_state=SEMILLA)))
    if estrategia == 'smote':
        pasos.append(('balancear', SMOTE(k_neighbors=3, random_state=SEMILLA)))
    peso = 'balanced' if estrategia == 'class_weight' else None
    pasos.append(('modelo', RandomForestClassifier(n_estimators=n_arboles, min_samples_leaf=2, class_weight=peso,
                                                   n_jobs=-1, random_state=SEMILLA)))
    return Pipeline(pasos)

def metricas(y_true, y_pred, esquema, t_verdadero):
    """Métricas nominales y ordinales. t_verdadero = T-type real (para el MAE en unidades de T)."""
    pos = {cl: i for i, cl in enumerate(ORDEN_CLASES[esquema])}
    i_t = np.array([pos[cl] for cl in y_true]); i_p = np.array([pos[cl] for cl in y_pred])
    t_pred = np.array([T_REP[esquema][cl] for cl in y_pred])
    return {'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            'F1_macro': f1_score(y_true, y_pred, average='macro'),
            'kappa_cuadratico': cohen_kappa_score(i_t, i_p, weights='quadratic'),
            'acierto_±1_clase': np.mean(np.abs(i_t - i_p) <= 1),
            'MAE_T': np.mean(np.abs(np.asarray(t_verdadero) - t_pred))}
```

## Celda 69 · Código
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
X_train = df.loc[es_train, FEATURES]
resultados, recalls = [], {}
for e in ESQUEMAS:
    y = df.loc[es_train, f'y_{e}']
    for estrategia in ['ninguno', 'class_weight', 'submuestreo', 'smote']:
        pred = cross_val_predict(hacer_modelo(estrategia), X_train, y, cv=cv)
        resultados.append({'esquema': e, 'balanceo': estrategia, **metricas(y, pred, e, df.loc[es_train, 'Ttype'])})
        recalls[(e, estrategia)] = pd.Series(recall_score(y, pred, labels=ORDEN_CLASES[e], average=None),
                                             index=ORDEN_CLASES[e])
tabla_balanceo = pd.DataFrame(resultados)
display(tabla_balanceo.round(3))

fig, axs = plt.subplots(1, 3, figsize=(22, 4))
for ax, e in zip(axs, ESQUEMAS):
    m = pd.DataFrame({estr: recalls[(e, estr)] for estr in ['ninguno', 'class_weight', 'submuestreo', 'smote']}).T
    sns.heatmap(m, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1, ax=ax, cbar=False)
    ax.set_title(f'Recall por clase — {e}')
plt.tight_layout(); plt.show()

MEJOR_BALANCEO = (tabla_balanceo.loc[tabla_balanceo.groupby('esquema')['F1_macro'].idxmax()]
                  .set_index('esquema')['balanceo'].to_dict())
print('Mejor balanceo por esquema:', MEJOR_BALANCEO)
```

## Celda 70 · Markdown
```markdown
**Conclusión 9:**
* Sin balanceo, el recall de Sdm, Sm e Im debería ser ≈ 0.
* `class_weight` y SMOTE suben el recall de las clases raras a costa de un poco de E, S0 y Sb.
* El submuestreo suele ser el peor en el esquema detallado, pero es competitivo con tres clases.
* El balanceo **no crea información**: si Sm e Im siguen confundiéndose con Sd, es porque sus espectros integrados son realmente
  parecidos (discos de baja masa con formación estelar).
```

## Celda 71 · Markdown
```markdown
# 10. Selección de features
Combinamos métodos con supuestos **distintos**; una feature robusta debe destacar en varios:

| Método | Qué mide | Supuesto |
|---|---|---|
| ε² de Kruskal–Wallis (Tomczak & Tomczak 2014) | diferencia de distribuciones entre clases | univariado, no paramétrico |
| ρ de Spearman con el orden de clases | tendencia monótona en la secuencia | univariado, ordinal |
| Información mutua | dependencia no lineal | univariado |
| Estabilidad L1 (Meinshausen & Bühlmann 2010; Tibshirani 1996) | ¿sobrevive en un modelo disperso al cambiar la muestra? | multivariado, lineal |
| Permutación (Breiman 2001) | ¿cuánto empeora el RF si se "rompe" la feature? | multivariado, no lineal |

Todo se calcula con **train**. El test se reserva para la sección 11.

### 10.1 Filtros univariados
```

## Celda 72 · Código
```python
def ranking_filtros(esquema):
    datos = df[es_train]
    y = datos[f'y_{esquema}'].values
    pos = {cl: i for i, cl in enumerate(ORDEN_CLASES[esquema])}
    y_ord = np.array([pos[cl] for cl in y])
    X_imp = SimpleImputer(strategy='median').fit_transform(datos[FEATURES])
    mi = mutual_info_classif(X_imp, y, random_state=SEMILLA)
    filas = []
    for f, mi_f in zip(FEATURES, mi):
        x = datos[f].values.astype(float)
        ok = np.isfinite(x)
        grupos = [x[ok & (y == cl)] for cl in ORDEN_CLASES[esquema]]
        grupos = [g for g in grupos if len(g) > 1]
        try:
            eps2 = kruskal(*grupos).statistic / (ok.sum() - 1)   # ε² = H/(n−1)
        except ValueError:                                         # todos los valores iguales
            eps2 = 0.0
        filas.append({'feature': f, 'epsilon2': eps2, 'rho_ordinal': spearmanr(x[ok], y_ord[ok]).correlation,
                      'MI': mi_f, 'familia': FAMILIA_DE[f]})
    return pd.DataFrame(filas).set_index('feature').sort_values('epsilon2', ascending=False)

FILTROS = {e: ranking_filtros(e) for e in ESQUEMAS}
colores_familia = dict(zip(FAMILIAS, sns.color_palette('tab10', len(FAMILIAS))))
fig, axs = plt.subplots(1, 3, figsize=(22, 7))
for ax, e in zip(axs, ESQUEMAS):
    top = FILTROS[e].head(20)
    ax.barh(top.index[::-1], top['epsilon2'][::-1], color=[colores_familia[f] for f in top['familia'][::-1]])
    ax.set_title(f'ε² — {e}'); ax.set_xlabel('ε² (efecto de Kruskal–Wallis)')
handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in colores_familia.values()]
fig.legend(handles, colores_familia.keys(), loc='lower center', ncol=len(FAMILIAS), fontsize=8)
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()
display(FILTROS['tres_clases'].head(20).round(3))
```

## Celda 73 · Markdown
```markdown
**Conclusión 10.1:**
* El EDA ya encontró arriba del todo HdA (ε² = 0.585), HgA, G4300, D4000n y EW(Hα) (≈ 0.56). Aquí deberían aparecer también
  **log_sSFR_Ha, log_Sigma90 y log_sigma**. ε² ≥ 0.26 es un efecto "grande".
* ρ indica la **dirección**: D4000n, σ y Σ* tienen ρ < 0 con el orden de clases (bajan hacia las tardías), y EW(Hα) y xY tienen
  ρ > 0.
* Si la MI pone una feature más alta que ε², su relación no es monótona (p. ej. Δedad, con máximo en las intermedias).

### 10.2 Un representante por grupo de redundancia
En cada grupo de la sección 7 elegimos la feature con mayor ε² medio entre esquemas (`REPS`). Los modelos multivariados usan solo
estos representantes.
```

## Celda 74 · Código
```python
eps_medio = pd.concat([FILTROS[e]['epsilon2'] for e in ESQUEMAS], axis=1).mean(axis=1)
REPS = [eps_medio[grupos_corr.index[grupos_corr == g]].idxmax() for g in sorted(grupos_corr.unique())]
representa = {r: [f for f in FEATURES if grupos_corr[f] == grupos_corr[r] and f != r] for r in REPS}
print(len(REPS), 'representantes:')
for r in REPS:
    print(f'  {r:14s} <- {", ".join(representa[r]) or "—"}')
```

## Celda 75 · Markdown
```markdown
### 10.3 Selección de estabilidad con regresión logística L1
1. Se ajusta una regresión logística multinomial L1 sobre 50 submuestras del 50 % de train.
2. Una feature es **estable** si su coeficiente es ≠ 0 en ≥ 80 % de ellas (Meinshausen & Bühlmann 2010).
3. Los coeficientes por clase muestran **hacia qué clase empuja** cada feature.

Tarda unos 5–15 min.
```

## Celda 76 · Código
```python
def estabilidad(esquema, n_sub=50, C=0.05):
    datos = df[es_train]
    X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(datos[REPS]))
    y = datos[f'y_{esquema}'].values
    rng = np.random.default_rng(SEMILLA)
    nueva_lr = lambda: LogisticRegression(penalty='l1', solver='saga', C=C, class_weight='balanced', max_iter=3000)
    conteo = np.zeros(len(REPS))
    for _ in range(n_sub):
        i = rng.choice(len(y), size=len(y) // 2, replace=False)
        conteo += (np.abs(nueva_lr().fit(X[i], y[i]).coef_) > 1e-6).any(axis=0)
    lr = nueva_lr().fit(X, y)
    coefs = pd.DataFrame(lr.coef_, index=lr.classes_, columns=REPS).loc[ORDEN_CLASES[esquema]]
    return pd.Series(conteo / n_sub, index=REPS), coefs

ESTAB, COEFS = {}, {}
for e in ESQUEMAS:
    ESTAB[e], COEFS[e] = estabilidad(e)
display(pd.DataFrame(ESTAB).sort_values('tres_clases', ascending=False).round(2))

fig, axs = plt.subplots(3, 1, figsize=(16, 16))
for ax, e in zip(axs, ESQUEMAS):
    lim = np.abs(COEFS[e].values).max()
    sns.heatmap(COEFS[e], cmap='coolwarm', center=0, vmin=-lim, vmax=lim, ax=ax)
    ax.set_title(f'Coeficientes L1 por clase — {e} (rojo = favorece a la clase)')
plt.tight_layout(); plt.show()
```

## Celda 77 · Markdown
```markdown
**Conclusión 10.3:** para tres clases deberías ver:
* **Temprana:** D4000n/edad (+), σ y Σ* (+), sSFR (−).
* **Tardía:** lo contrario, con EW(Hα) (+) y Σ* muy baja.
* **Intermedia:** pocos coeficientes grandes; es una **clase "del medio"** definida por exclusión. Es la señal de que la etiqueta
  es **ordinal**.

Que una feature tenga estabilidad ~1 en todos los esquemas indica información independiente y robusta.

### 10.4 Importancia por permutación (Random Forest)
Separamos train en **ajuste** (75 %) y **validación** (25 %). El test no se toca.
```

## Celda 78 · Código
```python
idx_ajuste, idx_valid = train_test_split(idx_train, test_size=0.25, stratify=df.loc[idx_train, 'Ttype'],
                                         random_state=SEMILLA)
PERM = {}
for e in ESQUEMAS:
    modelo = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_ajuste, REPS], df.loc[idx_ajuste, f'y_{e}'])
    r = permutation_importance(modelo, df.loc[idx_valid, REPS], df.loc[idx_valid, f'y_{e}'],
                               scoring='balanced_accuracy', n_repeats=10, random_state=SEMILLA)
    PERM[e] = pd.DataFrame({'media': r.importances_mean, 'std': r.importances_std}, index=REPS) \
                .sort_values('media', ascending=False)

fig, axs = plt.subplots(1, 3, figsize=(22, 7))
for ax, e in zip(axs, ESQUEMAS):
    p = PERM[e].head(15)[::-1]
    ax.barh(p.index, p['media'], xerr=p['std'], color=[colores_familia[FAMILIA_DE[f]] for f in p.index])
    ax.set_title(f'Permutación — {e}'); ax.set_xlabel('Caída de balanced accuracy')
plt.tight_layout(); plt.show()
```

## Celda 79 · Markdown
```markdown
### 10.5 Importancia por familia física
Se permutan **todas las features de una familia a la vez**, con la misma permutación de filas. Así se mide cuánto aporta ese
**tipo de física**, aunque esté repartido en muchas columnas. Aquí el modelo usa todas las `FEATURES`.
```

## Celda 80 · Código
```python
def importancia_familias(modelo, X, y, n_rep=10):
    """Caída de balanced accuracy al permutar juntas todas las features de una familia."""
    rng = np.random.default_rng(SEMILLA)
    base = balanced_accuracy_score(y, modelo.predict(X))
    res = {}
    for fam, cols in FAMILIAS.items():
        caidas = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[cols] = X[cols].values[rng.permutation(len(X))]   # misma permutación para toda la familia
            caidas.append(base - balanced_accuracy_score(y, modelo.predict(Xp)))
        res[fam] = (np.mean(caidas), np.std(caidas))
    return pd.DataFrame(res, index=['caida_media', 'std']).T.sort_values('caida_media', ascending=False)

IMP_FAMILIA = {}
for e in ESQUEMAS:
    modelo = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_ajuste, FEATURES], df.loc[idx_ajuste, f'y_{e}'])
    IMP_FAMILIA[e] = importancia_familias(modelo, df.loc[idx_valid, FEATURES], df.loc[idx_valid, f'y_{e}'])

fig, axs = plt.subplots(1, 3, figsize=(22, 4))
for ax, e in zip(axs, ESQUEMAS):
    t = IMP_FAMILIA[e][::-1]
    ax.barh(t.index, t['caida_media'], xerr=t['std'], color=[colores_familia[f] for f in t.index])
    ax.set_title(f'Importancia por familia — {e}'); ax.set_xlabel('Caída de balanced accuracy')
plt.tight_layout(); plt.show()
```

## Celda 81 · Markdown
```markdown
**Conclusión 10.4–10.5:**
* **"Edad / historia"** debería tener la mayor caída.
* **"Masa, cinemática y estructura"** debería ser la segunda, y la **clave para separar E de S0**: ambas son viejas, pero difieren
  en σ y Σ* (Cappellari et al. 2011).
* Una caída pequeña no significa "irrelevante", sino "redundante dado el resto". Por ejemplo, la edad y la formación estelar
  actual se sustituyen mutuamente.
* Si **b_a** saliera muy importante, el modelo estaría usando la forma aparente, no una propiedad intrínseca.

### 10.6 ¿Qué feature separa cada par de tipos vecinos?
Poder discriminante **univariado** = 2·|AUC − 0.5| (0 = nada, 1 = separación perfecta) para cada par de clases adyacentes. Así
vemos **dónde** de la secuencia actúa cada propiedad física.
```

## Celda 82 · Código
```python
def poder_auc(col_clase, pares, columnas):
    datos = df[es_train]
    tabla = {}
    for a, b in pares:
        s = datos[datos[col_clase].isin([a, b])]
        fila = {}
        for f in columnas:
            ok = s[f].notna()
            if ok.sum() < 10 or s.loc[ok, col_clase].nunique() < 2:
                fila[f] = np.nan
                continue
            auc = roc_auc_score(s.loc[ok, col_clase] == b, s.loc[ok, f])
            fila[f] = 2 * abs(auc - 0.5)
        tabla[f'{a} vs {b}'] = fila
    return pd.DataFrame(tabla)

orden_det = ORDEN_CLASES['detallado']
AUC_VECINOS = poder_auc('y_detallado', list(zip(orden_det[:-1], orden_det[1:])), FEATURES)
AUC_TRES = poder_auc('y_tres_clases', [('Temprana', 'Intermedia'), ('Intermedia', 'Tardía'), ('Temprana', 'Tardía')], FEATURES)

fig, axs = plt.subplots(1, 2, figsize=(22, 14), gridspec_kw={'width_ratios': [4, 1.3]})
sns.heatmap(AUC_VECINOS, cmap='magma', vmin=0, vmax=1, annot=True, fmt='.1f', annot_kws={'size': 6}, ax=axs[0])
axs[0].set_title('Poder 2|AUC − 0.5| entre T-types vecinos')
sns.heatmap(AUC_TRES, cmap='magma', vmin=0, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 7}, ax=axs[1])
axs[1].set_title('Tres clases')
plt.tight_layout(); plt.show()

mejores = pd.concat([AUC_VECINOS, AUC_TRES], axis=1).apply(
    lambda col: ', '.join(f'{f} ({v:.2f})' for f, v in col.nlargest(3).items()))
display(mejores.to_frame('3 mejores features'))
```

## Celda 83 · Markdown
```markdown
**Conclusión 10.6** (la tabla más reveladora):
* **E vs S0:** el poder espectral es **bajo** (≲ 0.3). Los mejores separadores deberían ser `b_a`, log σ, log Σ* o la masa, no la
  edad. Es consistente con que las S0 sean discos que dejaron de formar estrellas (van den Bergh 1976; Kormendy & Bender 2012).
* **S0a vs Sa y Sa vs Sab:** aquí suelen estar los saltos de **sSFR, EW(Hα) y D4000**: el paso de la secuencia roja a la nube
  azul.
* **Sb … Sd:** el poder es pequeño en cada par (~0.1–0.2). La secuencia continúa de forma **gradual**.
* **Sdm vs Sm vs Im:** casi nulo. Se distinguen por la estructura visual, que el espectro integrado no ve.
* Con **tres clases**, Temprana vs Tardía debería separarse casi perfectamente (≳ 0.9). La dificultad está en la Intermedia.

### 10.7 Ranking de consenso
Para cada esquema se promedia el **rango** de cada representante en ε², MI, estabilidad y permutación, para que ningún método
domine.
```

## Celda 84 · Código
```python
CONSENSO, TOP10 = {}, {}
for e in ESQUEMAS:
    r = pd.DataFrame({'epsilon2': FILTROS[e].loc[REPS, 'epsilon2'], 'MI': FILTROS[e].loc[REPS, 'MI'],
                      'estabilidad': ESTAB[e], 'permutacion': PERM[e]['media']})
    t = r.assign(rango_medio=r.rank(ascending=False).mean(axis=1)).sort_values('rango_medio')
    t['rho_ordinal'] = FILTROS[e].loc[t.index, 'rho_ordinal']
    t['familia'] = t.index.map(FAMILIA_DE)
    t['significado'] = t.index.map(lambda f: DESCRIPCION.get(f, ''))
    t['representa_a'] = t.index.map(lambda f: ', '.join(representa[f]))
    CONSENSO[e] = t
    TOP10[e] = t.index[:10].tolist()
    print(f'\n=== {e} ===')
    display(t.head(15).round(3))
    t.to_csv(os.path.join(RUTA_RES, f'consenso_{e}.csv'))
```

## Celda 85 · Markdown
```markdown
### 10.8 ¿Cuántas features hacen falta?
Se mide la balanced accuracy en validación con las k primeras features del consenso. Es algo optimista (la permutación usó
validación), pero muestra la **forma** de la curva.
```

## Celda 86 · Código
```python
ks = sorted({k for k in [1, 2, 3, 4, 5, 7, 10, 15, len(REPS)] if k <= len(REPS)})
curva = []
for e in ESQUEMAS:
    for k in ks:
        cols = CONSENSO[e].index[:k].tolist()
        m = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_ajuste, cols], df.loc[idx_ajuste, f'y_{e}'])
        curva.append({'esquema': e, 'k': k, 'balanced_acc':
                      balanced_accuracy_score(df.loc[idx_valid, f'y_{e}'], m.predict(df.loc[idx_valid, cols]))})
curva = pd.DataFrame(curva)

plt.figure(figsize=(8, 5))
for e in ESQUEMAS:
    c = curva[curva['esquema'] == e]
    linea, = plt.plot(c['k'], c['balanced_acc'], 'o-', label=e)
    plt.axhline(1 / len(ORDEN_CLASES[e]), ls=':', color=linea.get_color())   # nivel de azar
plt.xscale('log'); plt.xlabel('número de features (k)'); plt.ylabel('balanced accuracy (validación)')
plt.legend(); plt.title('Rendimiento vs número de features (punteado = azar)'); plt.show()
```

## Celda 87 · Markdown
```markdown
**Conclusión 10.8:** la curva debería **saturarse en k ≈ 3–5**. Una feature de edad (D4000n/HdA), una de formación estelar
actual (sSFR/EW(Hα)) y una de estructura (Σ*/σ) dan casi todo el rendimiento: **la morfología espectral es de baja
dimensión**.

### 10.9 Análisis discriminante lineal (LDA)
LDA (Fisher 1936) encuentra las combinaciones lineales que mejor separan las clases, con coeficientes interpretables. Se aplica a
tres clases con su top-10.
```

## Celda 88 · Código
```python
cols = TOP10['tres_clases']
prep = Pipeline([('imputar', SimpleImputer(strategy='median')), ('escalar', StandardScaler())])
X_lda = prep.fit_transform(df.loc[idx_train, cols])
y_lda = df.loc[idx_train, 'y_tres_clases']
lda = LinearDiscriminantAnalysis(n_components=2).fit(X_lda, y_lda)
print('Varianza discriminante explicada:', lda.explained_variance_ratio_.round(3))
display(pd.DataFrame(lda.scalings_[:, :2], index=cols, columns=['LD1', 'LD2']).sort_values('LD1').round(2))

proy_lda = lda.transform(X_lda)
plt.figure(figsize=(7, 6))
for cl, col in zip(ORDEN_CLASES['tres_clases'], ['tab:red', 'tab:green', 'tab:blue']):
    m = (y_lda == cl).values
    plt.scatter(proy_lda[m, 0], proy_lda[m, 1], s=4, alpha=0.4, color=col, label=cl)
plt.xlabel('LD1'); plt.ylabel('LD2'); plt.legend(); plt.title('Proyección LDA (tres clases)'); plt.show()
```

## Celda 89 · Markdown
```markdown
**Conclusión 10.9:**
* **LD1** debería explicar > 90 % y combinar edad, formación estelar y estructura: un **"eje de Hubble espectral"**.
* **LD2** suele separar la Intermedia (p. ej. vía Δedad: bulbo viejo + disco joven).
* Que domine una sola dimensión confirma que las clases forman una **secuencia continua**.
```

## Celda 90 · Markdown
```markdown
# 11. Comparación de los tres esquemas (en TEST)
Cada esquema se entrena con su top-10 y su mejor balanceo, sobre **todo train**, y se evalúa **una sola vez** en test.
* **Mejora sobre el azar** = (BA − 1/K)/(1 − 1/K). Permite comparar esquemas con distinto número de clases K.
* También proyectamos las predicciones detalladas y por intervalos a tres clases, para ver si **más detalle ayuda o perjudica**.
```

## Celda 91 · Código
```python
COMPARACION, PRED_TEST = [], {}
fig, axs = plt.subplots(1, 3, figsize=(26, 8), gridspec_kw={'width_ratios': [2, 1, 1.4]})
for ax, e in zip(axs, ESQUEMAS):
    cols = TOP10[e]
    m = hacer_modelo(MEJOR_BALANCEO[e]).fit(df.loc[idx_train, cols], df.loc[idx_train, f'y_{e}'])
    y_t = df.loc[idx_test, f'y_{e}']
    pred = m.predict(df.loc[idx_test, cols])
    PRED_TEST[e] = pred
    K = len(ORDEN_CLASES[e])
    met = metricas(y_t, pred, e, df.loc[idx_test, 'Ttype'])
    met['azar'] = 1 / K
    met['mejora_sobre_azar'] = (met['balanced_acc'] - 1 / K) / (1 - 1 / K)
    COMPARACION.append({'esquema': e, 'n_clases': K, 'balanceo': MEJOR_BALANCEO[e], **met})
    ConfusionMatrixDisplay.from_predictions(y_t, pred, labels=ORDEN_CLASES[e], normalize='true', values_format='.2f',
                                            ax=ax, colorbar=False, xticks_rotation=60)
    ax.set_title(f'{e} (normalizada por fila)')
plt.tight_layout(); plt.show()
COMPARACION = pd.DataFrame(COMPARACION).set_index('esquema')
display(COMPARACION.round(3))
```

## Celda 92 · Código
```python
# ¿Las mismas features importan en todos los esquemas? (Spearman entre importancias por permutación)
imp = pd.DataFrame({e: PERM[e]['media'] for e in ESQUEMAS})
display(imp.corr(method='spearman').round(2))
print('Features en el top-10 de los tres esquemas:', sorted(set.intersection(*[set(TOP10[e]) for e in ESQUEMAS])))

# Proyectar todas las predicciones a tres clases
A_TRES = {'detallado': {NOMBRE_TTYPE[t]: esquema_tres(t) for t in ORDEN_TTYPE if t != 11},
          'intervalos': {'E': 'Temprana', 'S0–S0a': 'Temprana', 'Sa–Sab': 'Intermedia',
                         'Sb–Sbc': 'Intermedia', 'Sc–Sd': 'Tardía', 'Sdm–Im': 'Tardía'},
          'tres_clases': {cl: cl for cl in ORDEN_CLASES['tres_clases']}}
y3 = df.loc[idx_test, 'y_tres_clases'].values
proyeccion = pd.DataFrame({e: {'balanced_acc_3c': balanced_accuracy_score(y3, pd.Series(PRED_TEST[e]).map(A_TRES[e])),
                               'F1_macro_3c': f1_score(y3, pd.Series(PRED_TEST[e]).map(A_TRES[e]), average='macro')}
                           for e in ESQUEMAS}).T
display(proyeccion.round(3))
```

## Celda 93 · Markdown
```markdown
### 11.1 Ventajas y desventajas de cada esquema

| | **T-type detallado (13 clases)** | **Temprana / Intermedia / Tardía** | **Intervalos de T (6 clases)** |
|---|---|---|---|
| **Ventajas** | Máxima resolución; conserva el orden completo; permite métricas ordinales finas (MAE en T) | Clases grandes y balanceadas (~2:1); fronteras con significado físico (secuencia roja, discos con bulbo, discos puros); resultados estables | Absorbe la incertidumbre de ±1–2 T humana; mantiene E separada de S0; desbalance moderado |
| **Desventajas** | Desbalance ~100:1; la diferencia entre vecinos (~1 T) es menor que el error de la etiqueta (σ ≈ 1.8 T; Naim et al. 1995); recall ≈ 0 en Sdm–Im; las métricas nominales castigan igual un error de 1 T que uno de 8 T | Mezcla E con S0 y Sa con Sbc; la Intermedia es heterogénea y la más confundida; depende de los cortes (T ≤ 0, T ≥ 5) | Cortes arbitrarios; Sdm–Im pequeña (~180); métricas nominales para una etiqueta ordinal |
| **Mejor uso** | Regresión de T o clasificación **ordinal**; tendencias continuas | Evolución (quenching, bimodalidad) | Catálogos automáticos "prácticos" |

**Conclusión 11** (verifícala con tus resultados):
* La **mejora sobre el azar** debería ser máxima en tres clases y mínima en el detallado. El κ cuadrático y el acierto ±1
  muestran que los errores del detallado son casi siempre **entre vecinos**: el modelo aprende el orden, no la frontera fina.
* Una correlación alta entre importancias (ρ ≳ 0.7) indica que **la misma física** (edad, formación estelar y estructura)
  sostiene los tres esquemas. Cambiar de esquema cambia la **resolución**, no la física.
* Si el detallado proyectado a tres clases rinde igual que el modelo directo, más detalle no perjudica. Si rinde peor, el ruido de
  la etiqueta fina se propaga.
* **Recomendación:** tratar T como **ordinal**, con K − 1 clasificadores "T > k" (Frank & Hall 2001) o regresión ordinal.
```

## Celda 94 · Markdown
```markdown
# 12. Conclusiones generales
1. **Calidad de datos.**
   * `ERROR_FoV` es incorrecto: se sustituyó por DER_SNR, y `adev` por χ².
   * El v0 de STARLIGHT contiene el offset aire→vacío (≈ 83 km/s), que se corrigió.
   * El catálogo tiene centinelas (z = −9999), topes (incl = 81.37°), valores de Petrosian repetidos, etiquetas mezcladas (dSph
     en T = 10 y T = −5, mergers en T = 11, `Sa` con T = 3) y **9 galaxias observadas dos veces**.
   * El cruce se validó con `arq_obs`, v0, coordenadas y DRPall.
2. **Qué distingue la morfología.** Tres ejes físicos casi independientes:
   * **Edad estelar / historia de formación** (D4000n, HδA, edad STARLIGHT, PC1): la más informativa.
   * **Formación estelar actual** (sSFR, EW(Hα)): el salto entre tempranas y espirales.
   * **Estructura y cinemática** (Σ*, σ, M*): la única que separa bien **E de S0**.
   * La química ([MgFe]′, Mg b/⟨Fe⟩) y el polvo aportan matices, pero son en gran parte redundantes con la masa y la edad.
3. **Significado astrofísico.** El espectro integrado ve la **historia de formación estelar**, acoplada a la morfología a través
   de la masa del bulbo (quenching morfológico; Martig et al. 2009) y del entorno (Dressler 1980). Por eso:
   * Clasifica bien la dicotomía **temprana/tardía**.
   * Pierde resolución donde la morfología depende de detalles **geométricos** (brazos, barras, anillos), como Sb–Sd y Sdm–Im.
   * Falla cuando morfología y color se desacoplan: espirales rojas (Masters et al. 2010) y elípticas azules (Schawinski et al.
     2009).
4. **Techo intrínseco.** Con σ ≈ 1.8 T entre clasificadores humanos, una precisión baja en el detallado **no es un fracaso del
   modelo**, sino un límite de la etiqueta.
5. **Próximos pasos:**
   * Añadir parámetros estructurales (índice de Sérsic, concentración, B/T de Fischer et al. 2019).
   * Comparar `FLUX_1RE` con `FLUX_FoV` como gradiente espectroscópico bulbo/disco.
   * Probar clasificación ordinal.

## Referencias
* Abdurro'uf et al. 2022, ApJS 259, 35 — SDSS DR17
* Asari N. V. et al. 2007, MNRAS 381, 263
* Astropy Collaboration 2022, ApJ 935, 167
* Baldry I. K. et al. 2004, ApJ 600, 681
* Baldwin J. A., Phillips M. M., Terlevich R. 1981, PASP 93, 5
* Balogh M. L. et al. 1999, ApJ 527, 54
* Bamford S. P. et al. 2009, MNRAS 393, 1324
* Blanton M. R., Moustakas J. 2009, ARA&A 47, 159
* Breiman L. 2001, Machine Learning 45, 5
* Bruzual G., Charlot S. 2003, MNRAS 344, 1000
* Bundy K. et al. 2015, ApJ 798, 7
* Buta R. J. 2013, "Galaxy Morphology", en *Planets, Stars and Stellar Systems* Vol. 6, Springer
* Cappellari M. et al. 2011, MNRAS 416, 1680
* Cardelli J. A., Clayton G. C., Mathis J. S. 1989, ApJ 345, 245
* Chawla N. V. et al. 2002, JAIR 16, 321
* Cid Fernandes R. et al. 2005, MNRAS 358, 363
* Cid Fernandes R. et al. 2011, MNRAS 413, 1687
* Ciddor P. E. 1996, Applied Optics 35, 1566
* Cohen J. 1968, Psychological Bulletin 70, 213
* Concas A. et al. 2019, A&A 622, A188
* Connolly A. J. et al. 1995, AJ 110, 1071
* de Vaucouleurs G. 1959, *Handbuch der Physik* 53, 275
* de Vaucouleurs G. et al. 1991, *Third Reference Catalogue of Bright Galaxies* (RC3), Springer
* Díaz A. I., Terlevich E., Terlevich R. 1989, MNRAS 239, 325
* Domínguez Sánchez H. et al. 2022, MNRAS 509, 4024
* Dressler A. 1980, ApJ 236, 351
* Drory N. et al. 2015, AJ 149, 77
* Emsellem E. et al. 2011, MNRAS 414, 888
* Faber S. M., Jackson R. E. 1976, ApJ 204, 668
* Fischer J.-L., Domínguez Sánchez H., Bernardi M. 2019, MNRAS 483, 2057
* Fisher R. A. 1936, Annals of Eugenics 7, 179
* Frank E., Hall M. 2001, ECML, LNCS 2167, 145
* Gallazzi A. et al. 2005, MNRAS 362, 41
* Hubble E. P. 1926, ApJ 64, 321
* Hubble E. P. 1936, *The Realm of the Nebulae*, Yale UP
* Ivezić Ž., Connolly A. J., VanderPlas J. T., Gray A. 2014, *Statistics, Data Mining, and Machine Learning in Astronomy*,
  Princeton UP (astroML)
* Kauffmann G. et al. 2003a, MNRAS 341, 33
* Kauffmann G. et al. 2003b, MNRAS 341, 54
* Kauffmann G. et al. 2003c, MNRAS 346, 1055
* Kennicutt R. C. 1998, ARA&A 36, 189
* Kennicutt R. C., Evans N. J. 2012, ARA&A 50, 531
* Kewley L. J. et al. 2001, ApJ 556, 121
* Kewley L. J. et al. 2006, MNRAS 372, 961
* Kormendy J. 1985, ApJ 295, 73
* Kormendy J. et al. 2009, ApJS 182, 216
* Kormendy J., Bender R. 2012, ApJS 198, 2
* Lahav O. et al. 1995, Science 267, 859
* Law D. R. et al. 2016, AJ 152, 83
* Le Borgne J.-F. et al. 2003, A&A 402, 433
* Lemaître G., Nogueira F., Aridas C. K. 2017, JMLR 18, 1
* Martig M. et al. 2009, ApJ 707, 250
* Masters K. L. et al. 2010, MNRAS 405, 783
* Meinshausen N., Bühlmann P. 2010, JRSS B 72, 417
* Nair P. B., Abraham R. G. 2010, ApJS 186, 427
* Naim A. et al. 1995, MNRAS 274, 1107
* Osterbrock D. E., Ferland G. J. 2006, *Astrophysics of Gaseous Nebulae and AGN*, University Science Books
* Pedregosa F. et al. 2011, JMLR 12, 2825
* Pettini M., Pagel B. E. J. 2004, MNRAS 348, L59
* Planck Collaboration 2020, A&A 641, A6
* Roberts M. S., Haynes M. P. 1994, ARA&A 32, 115
* Sandage A. 1961, *The Hubble Atlas of Galaxies*, Carnegie Institution
* Sandage A. 2005, ARA&A 43, 581
* Schawinski K. et al. 2009, MNRAS 396, 818
* Schawinski K. et al. 2014, MNRAS 440, 889
* Simien F., de Vaucouleurs G. 1986, ApJ 302, 564
* Stoehr F. et al. 2008, ASP Conf. Ser. 394, 505
* Strateva I. et al. 2001, AJ 122, 1861
* Thomas D., Maraston C., Bender R. 2003, MNRAS 339, 897
* Thomas D. et al. 2005, ApJ 621, 673
* Tibshirani R. 1996, JRSS B 58, 267
* Tomczak M., Tomczak E. 2014, Trends in Sport Sciences 21, 19
* Trager S. C. et al. 1998, ApJS 116, 1
* van den Bergh S. 1976, ApJ 206, 883
* Vázquez-Mata J. A. et al. 2022, MNRAS 512, 2222
* Wake D. A. et al. 2017, AJ 154, 86
* Worthey G. et al. 1994, ApJS 94, 687
* Worthey G., Ottaviani D. L. 1997, ApJS 111, 377
* Yip C. W. et al. 2004, AJ 128, 585
```

## @maria-rabelero-mx_nestle

where arethe cells?

## Copilot

Las celdas no te están llegando en el chat. Te las mando en partes dentro del mensaje final. Primero reviso los nombres exactos de las columnas para que el código coincida.

Las celdas no se ven porque tu cliente solo muestra mi mensaje final. Por eso ahora van **dentro del mensaje final**, en 3 partes. Esta es la **Parte 1/3 (Celdas 1–20)**.
