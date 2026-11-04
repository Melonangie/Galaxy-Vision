from sklearn.decomposition import PCA

# PCA needs complete rows -> fill the few NaNs with the mean spectrum value at that wavelength
S = SPEC[spec_index.isin(d.index)]
S = np.where(np.isnan(S), np.nanmean(S, axis=0), S)
lab = d.loc[spec_index[spec_index.isin(d.index)], "morph_class"].values

pca = PCA(n_components=10).fit(S)
P = pca.transform(S)
print("explained variance:", pca.explained_variance_ratio_.round(3))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for c in CLASS_ORDER:
    m = lab == c
    ax[0].scatter(P[m, 0], P[m, 1], s=6, alpha=0.5, color=CLASS_COLORS[c], label=c)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(markerscale=3)
for k in range(3):
    ax[1].plot(GRID, pca.components_[k] + 0.05 * k, lw=0.7, label=f"PC{k+1}")
ax[1].legend(); ax[1].set_xlabel("Rest wavelength [Å]"); ax[1].set_title("Eigenspectra")
plt.show()
# PC1 usually follows the young-vs-old continuum shape plus emission-line strength (Yip et al. 2004, AJ 128, 585)