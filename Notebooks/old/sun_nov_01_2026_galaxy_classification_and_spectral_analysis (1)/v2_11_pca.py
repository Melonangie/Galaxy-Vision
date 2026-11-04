from sklearn.decomposition import PCA

sel = spec_index.isin(d.index)
S = SPEC[sel]
S = np.where(np.isnan(S), np.nanmean(S, axis=0), S)     # fill rare NaNs with the mean spectrum
tt = d.loc[spec_index[sel], "Ttype"].values

pca = PCA(n_components=10).fit(S); P = pca.transform(S)
print("explained variance:", pca.explained_variance_ratio_.round(3))

fig, ax = plt.subplots(1, 2, figsize=(17, 6))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=tt, cmap="RdYlBu", s=6, alpha=0.6)
plt.colorbar(sc, ax=ax[0], label="T-type"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
for k in range(3):
    ax[1].plot(GRID, pca.components_[k] + 0.05 * k, lw=0.7, label=f"PC{k+1}")
ax[1].legend(); ax[1].set_xlabel("Rest wavelength [Å]"); ax[1].set_title("Eigenspectra")
plt.show()
# PC1 normally tracks the old-vs-young continuum shape plus line strength (Connolly et al. 1995; Yip et al. 2004)