import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from PIL import Image

OLIVE = "#9AA542"; OLIVEDARK = "#1B2208"; OLIVELINK = "#6B7A1F"; OLIVESOFT = "#E4E8C7"
INK = "#1A1A10"; MUTED = "#6B6B5E"
RED = "#B5462B"; BLUE = "#3E6E8E"; LBLUE = "#8DB3C9"; GOLD = "#D9A520"
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 250,
})
OUT = "figs/"


def save(fig, name):
    fig.savefig(OUT + name, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def hanley_se(auc, npos=875, nneg=528):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg)
    return np.sqrt(v)


# 1. Hubble sequence + T-type scale --------------------------------------
def fig_hubble():
    img = Image.open(OUT + "hubble_crop.png")
    w, h = img.size
    img = img.resize((w * 3, h * 3), Image.LANCZOS)
    fig = plt.figure(figsize=(24 * CM, 7.8 * CM))
    ax = fig.add_axes([0, 0.40, 1, 0.60]); ax.imshow(img, aspect="auto"); ax.axis("off")
    ax2 = fig.add_axes([0, 0, 1, 0.38]); ax2.set_xlim(-5.6, 10.6); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.add_patch(plt.Rectangle((-5.6, 0.50), 6.1, 0.40, color=RED))
    ax2.add_patch(plt.Rectangle((0.5, 0.50), 10.1, 0.40, color=BLUE))
    ax2.text(-2.55, 0.70, "Temprana (0)  ·  T-Type ≤ 0", color="white", ha="center", va="center", fontsize=17, weight="bold")
    ax2.text(5.55, 0.70, "Tardía (1)  ·  1 ≤ T-Type ≤ 10", color="white", ha="center", va="center", fontsize=17, weight="bold")
    ax2.plot([0.5, 0.5], [0.42, 0.98], ls="--", color=INK, lw=2.5)
    for t, lab in [(-5, "E"), (-2, "S0"), (0, "S0a"), (1, "Sa"), (3, "Sb"), (5, "Sc"), (7, "Sd"), (10, "Irr")]:
        ax2.text(t, 0.30, f"{t}\n{lab}", ha="center", va="top", fontsize=15, color=INK)
    fig.text(0.5, 0.99, "", ha="center")
    save(fig, "f1_hubble_ttype.png")


# 2. Cohort funnel + split --------------------------------------------
def fig_cohorte():
    fig, ax = plt.subplots(figsize=(24 * CM, 4.6 * CM))
    parts = ["Prueba", "Validación", "Entrenamiento"]
    early = [528, 529, 2467]; late = [875, 874, 4078]
    y = np.arange(3)
    ax.barh(y, early, color=RED, height=0.62, label="Tempranas")
    ax.barh(y, late, left=early, color=BLUE, height=0.62, label="Tardías")
    for i in range(3):
        ax.text(early[i] + late[i] + 80, i, f"{early[i] + late[i]:,}", va="center", fontsize=18, weight="bold", color=INK)
    ax.set_yticks(y, parts, fontsize=18); ax.set_xlim(0, 7700)
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    ax.legend(loc="center right", fontsize=16, frameon=False, ncol=1, bbox_to_anchor=(1.0, 0.25))
    ax.set_title("Cohorte alineada 9,351 · split estratificado (semilla 42) · 62.3 % tardías",
                 fontsize=17, color=OLIVEDARK, loc="left")
    save(fig, "f2_cohorte.png")


# 3. CRISP-ML(Q) phases mapped to project activities -------------------
def fig_crisp():
    fig, ax = plt.subplots(figsize=(24 * CM, 9.0 * CM)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    rows = [
        ("1", "Business & Data\nUnderstanding", "Etiqueta binaria por T-Type; EDA de\ncatálogo, FITS, atajos y norte/sur", OLIVESOFT),
        ("2", "Data\nEngineering", "Cortes de calidad, FoV ∝ PETRO_TH90,\nespectros 4,096 pts + máscara, split", OLIVESOFT),
        ("3", "Model\nEngineering", "Ablación: CNN, ResNet-18, CNN1D,\nMLP, atención/BiGRU, fusiones", OLIVESOFT),
        ("4", "Evaluation\n(QA)", "AUC test, IC bootstrap pareado,\nMcNemar, oráculo, error por T-Type", OLIVESOFT),
        ("5–6", "Deployment &\nMonitoring", "Propuesta: CNN RGB + monitoreo de\ncalibración y frontera T-Type 0–1", "#F1F1EA"),
    ]
    hgt = 17.5; gap = 2.6
    for i, (n, ph, act, bg) in enumerate(rows):
        y0 = 100 - (i + 1) * hgt - i * gap
        ax.add_patch(FancyBboxPatch((0, y0), 35, hgt, boxstyle="round,pad=0,rounding_size=2", fc=OLIVE if i < 4 else MUTED, ec="none"))
        ax.text(4.0, y0 + hgt / 2, n, color="white", fontsize=20 if len(n)>1 else 24, weight="bold", ha="center", va="center")
        ax.text(21.5, y0 + hgt / 2, ph, color="white", fontsize=13.5, weight="bold", ha="center", va="center", linespacing=1.05)
        ax.add_patch(FancyBboxPatch((38, y0), 62, hgt, boxstyle="round,pad=0,rounding_size=2", fc=bg, ec="none"))
        ax.text(40, y0 + hgt / 2, act, color=INK, fontsize=14, ha="left", va="center", linespacing=1.15)
        ax.annotate("", xy=(38, y0 + hgt / 2), xytext=(35, y0 + hgt / 2),
                    arrowprops=dict(arrowstyle="-|>", color=OLIVEDARK, lw=2.5, mutation_scale=22))
    save(fig, "f3_crisp.png")


# 4. Hero: AUC test of all experiments ---------------------------------
def fig_auc():
    data = [
        ("CNN compacta · LR 3e-4 (seleccionada)", 0.9667, "img"),
        ("Fusión tokens M2 (encoders congelados)", 0.9666, "fus"),
        ("Img + Esp SL + Pob · concat", 0.9650, "fus"),
        ("Img + Pob · concat", 0.9639, "fus"),
        ("Img + Esp FoV · concat", 0.9635, "fus"),
        ("ResNet-18 (baseline)", 0.9633, "img"),
        ("Img + Esp SL · concat", 0.9632, "fus"),
        ("CNN pequeña (GPU)", 0.9623, "img"),
        ("Img + Esp SL atención · concat", 0.9334, "fus"),
        ("Espectro FoV · patch-BiGRU", 0.9170, "esp2"),
        ("Espectro FoV · CNN+BiGRU", 0.9122, "esp2"),
        ("Espectro SL · CNN+atención", 0.9066, "esp2"),
        ("Esp SL + Pob · concat", 0.8960, "pob"),
        ("Poblaciones STARLIGHT · MLP", 0.8949, "pob"),
        ("Espectro SL · CNN1D", 0.8642, "esp1"),
        ("Espectro FoV · CNN1D", 0.8572, "esp1"),
    ]
    cols = {"img": OLIVELINK, "fus": OLIVE, "esp2": BLUE, "esp1": LBLUE, "pob": GOLD}
    labs = {"img": "Solo imagen", "fus": "Fusión con imagen", "esp2": "Espectro (encoder mejorado)",
            "esp1": "Espectro (CNN1D base)", "pob": "Poblaciones STARLIGHT"}
    fig, ax = plt.subplots(figsize=(30 * CM, 12.6 * CM))
    y = np.arange(len(data))[::-1]
    for yi, (n, a, k) in zip(y, data):
        se = hanley_se(a)
        ax.barh(yi, a - 0.80, left=0.80, color=cols[k], height=0.74, label=labs[k])
        ax.errorbar(a, yi, xerr=1.96 * se, color=INK, capsize=4, lw=1.6)
        ax.text(a + 1.96 * se + 0.003, yi, f"{a:.4f}", va="center", fontsize=16,
                weight="bold" if "seleccionada" in n else "normal", color=INK)
    ax.set_yticks(y, [d[0] for d in data], fontsize=17)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.set_xlim(0.80, 1.0); ax.set_xlabel("ROC-AUC en prueba (n = 1,403) · IC95 % aprox. Hanley–McNeil", fontsize=16)
    ax.tick_params(axis="x", labelsize=15)
    ax.axvspan(0.958, 0.972, color=OLIVESOFT, zorder=0)
    ax.text(0.965, len(data) - 0.35, "techo visual ≈ 0.96", ha="center", va="bottom", fontsize=14, color=OLIVEDARK, style="italic")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), loc="upper center", bbox_to_anchor=(0.36, -0.12), ncol=5, fontsize=13.5, frameon=False, columnspacing=1.0, handletextpad=0.4)
    save(fig, "f4_auc.png")


# 5. Spectral expert evolution -----------------------------------------
def fig_espectro():
    m = ["CNN1D\n(LR 1e-4)", "Atención\n(sin RNN)", "CNN +\nBiGRU", "Patch +\nBiGRU"]
    fov = [0.8572, 0.9011, 0.9122, 0.9170]; sl = [0.8642, 0.9066, 0.9064, 0.8997]
    fig, ax = plt.subplots(figsize=(30 * CM, 7.0 * CM))
    x = np.arange(4); w = 0.36
    b1 = ax.bar(x - w / 2, fov, w, color=BLUE, label="FoV observado")
    b2 = ax.bar(x + w / 2, sl, w, color=LBLUE, label="STARLIGHT modelado")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}", ha="center", fontsize=14)
    ax.axhline(0.8949, color=GOLD, ls="--", lw=2.2); ax.text(-0.55, 0.8935, "poblaciones 0.895", color="#8A6A10", fontsize=14, ha="left", va="top")
    ax.set_ylim(0.84, 0.952); ax.set_xticks(x, m, fontsize=15); ax.set_ylabel("AUC test", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.text(3 - w / 2, 0.933, "+0.060 vs CNN1D", fontsize=17, weight="bold", color=OLIVEDARK, ha="center", va="bottom")
    ax.legend(loc="upper left", ncol=2, fontsize=14, frameon=False, borderaxespad=0.2)
    save(fig, "f5_espectro.png")


# 6. Complementarity vs fusion -----------------------------------------
def fig_complementariedad():
    fig = plt.figure(figsize=(30 * CM, 9.0 * CM))
    ax1 = fig.add_axes([0.06, 0.02, 0.34, 0.92])
    vals = [81, 76]
    ax1.pie(vals, colors=[GOLD, RED], startangle=90, counterclock=False,
            wedgeprops=dict(width=0.36, edgecolor="white", linewidth=3))
    ax1.text(0, 0.08, "157", ha="center", va="center", fontsize=34, weight="bold", color=INK)
    ax1.text(0, -0.25, "errores RGB", ha="center", va="center", fontsize=14, color=MUTED)
    ax1.text(0, -1.28, "81 (51.6 %) corregidos por poblaciones", fontsize=14, color="#8A6A10", weight="bold", va="center", ha="center")
    ax1.text(0, -1.58, "76 fallan en ambos (dificultad común)", fontsize=14, color=RED, weight="bold", ha="center", va="center"); ax1.set_ylim(-1.75, 1.05); ax1.set_xlim(-1.6, 1.6)
    ax1.set_title("Complementariedad potencial", fontsize=18, color=OLIVEDARK)
    ax2 = fig.add_axes([0.66, 0.18, 0.34, 0.72])
    labs = ["Fusión tardía α=0.85*", "Selector S2 (gate)*", "Fusión tokens M2", "Oráculo RGB ∪ espectro"]
    v = [3, 5, 19, 82]
    c = [MUTED, MUTED, OLIVE, OLIVESOFT]
    bars = ax2.barh(np.arange(4)[::-1], v, color=c, edgecolor=[MUTED, MUTED, OLIVE, OLIVE], linewidth=2, height=0.68)
    for yi, val in zip(np.arange(4)[::-1], v):
        ax2.text(val + 1.5, yi, f"+{val}", va="center", fontsize=16, weight="bold")
    ax2.set_yticks(np.arange(4)[::-1], labs, fontsize=16); ax2.set_xlim(0, 100)
    ax2.set_xlabel("Galaxias netas ganadas sobre RGB", fontsize=14); ax2.tick_params(axis="x", labelsize=13)
    ax2.set_title("Lo que la fusión recupera", fontsize=18, color=OLIVEDARK)
    save(fig, "f6_complementariedad.png")


# 7. Error rate by T-Type -----------------------------------------------
def fig_ttype():
    t = ["E\n(−5)", "S0\n(−2)", "S0a\n(0)", "Sa\n(1)", "Sab\n(2)", "≥ Sb\n(≥3)"]
    rgb = [3.0, 21.5, 34.6, 32.4, 17.0, 4.5]; pob = [12.2, 24.3, 37.0, 51.7, 19.2, 10.0]
    fig, ax = plt.subplots(figsize=(24 * CM, 6.8 * CM))
    x = np.arange(len(t)); w = 0.38
    ax.axvspan(1.5, 3.5, color=OLIVESOFT, zorder=0)
    ax.text(2.5, 62, "frontera T-Type 0–1", ha="center", fontsize=15, color=OLIVEDARK, weight="bold")
    b1 = ax.bar(x - w / 2, rgb, w, color=OLIVELINK, label="ResNet-18 (imagen)")
    b2 = ax.bar(x + w / 2, pob, w, color=GOLD, label="Poblaciones (MLP)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f}", ha="center", fontsize=13)
    ax.set_xticks(x, t, fontsize=14); ax.set_ylabel("Tasa de error (%)", fontsize=15); ax.set_ylim(0, 72)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    save(fig, "f7_ttype.png")


# 8. Future work: adaptive gate diagram --------------------------------
def fig_futuro():
    fig, ax = plt.subplots(figsize=(24 * CM, 7.0 * CM)); ax.set_xlim(0, 100); ax.set_ylim(9, 31); ax.axis("off"); ax.set_aspect("equal")
    items = [(15, "3–5 semillas\n+ IC pareados"), (50, "Calibración\nECE / Brier"), (85, "Gate α(x)\ncon OOF")]
    for i, (cx, txt) in enumerate(items):
        ax.add_patch(Circle((cx, 20), 10.5, fc=OLIVESOFT if i < 2 else OLIVE, ec="none"))
        ax.text(cx, 20, txt, ha="center", va="center", fontsize=13.5, weight="bold", color=OLIVEDARK if i < 2 else "white", linespacing=1.15)
    for a, b in [(15, 50), (50, 85)]:
        ax.add_patch(FancyArrowPatch((a + 10.8, 20), (b - 10.8, 20), arrowstyle="-|>", mutation_scale=26, color=OLIVE, lw=3))
    save(fig, "f8_futuro.png")


if __name__ == "__main__":
    fig_hubble(); fig_cohorte(); fig_crisp(); fig_auc(); fig_espectro(); fig_complementariedad(); fig_ttype(); fig_futuro()
    print("ok")
