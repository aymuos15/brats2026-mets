"""Slide figure: one training case shown as T1-Gd, primary labels, compound labels
and the regions the challenge assesses."""
import numpy as np, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion

D = "/home/localssk23/CAI4CAI/SegData/raw/Dataset006_BraTSMets2025"
CID = "sub-01321-002"
OUT = "/home/localssk23/CAI4CAI/brats2026-mets/slides/figures/labels_case.png"

seg = np.asarray(nib.as_closest_canonical(nib.load(f"{D}/labelsTr/{CID}.nii.gz")).dataobj).astype(np.uint8)
cnt = np.stack([(seg == k).sum((0, 1)) for k in range(1, 5)], 1)
z = int(cnt.min(1).argmax())


def img(ch):
    a = np.asarray(nib.as_closest_canonical(nib.load(f"{D}/imagesTr/{CID}_{ch}.nii.gz")).dataobj,
                   dtype=np.float32)[:, :, z]
    lo, hi = np.percentile(a[a > 0], [1, 99.5])
    return np.clip((a - lo) / (hi - lo), 0, 1)


t1c, lab = img("0000"), seg[:, :, z]
ys, xs = np.where(t1c > 0)
crop = lambda x: np.rot90(x[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
t1c, lab = crop(t1c), crop(lab)
# keep the tumour-bearing hemisphere so the labels are legible at slide size
half = lambda x: x[:, : int(x.shape[1] * 0.56)]
t1c, lab = half(t1c), half(lab)

PRIMARY = [("NETC", lab == 1, "#e31a1c"), ("SNFH", lab == 2, "#33a02c"),
           ("ET", lab == 3, "#ffd320"), ("RC", lab == 4, "#1f78b4")]
WT, TC, ET, RC = np.isin(lab, [1, 2, 3]), np.isin(lab, [1, 3]), lab == 3, lab == 4
COMPOUND = [("WT = NETC+SNFH+ET", WT, "#ff7f00"), ("TC = NETC+ET", TC, "#c05bff")]
ASSESSED = [("WT", WT, "#ff7f00"), ("TC", TC, "#c05bff"), ("ET", ET, "#ffd320"), ("RC", RC, "#1f78b4")]


def fill(ax, parts, alpha=0.6):
    ov = np.zeros(lab.shape + (4,))
    for _, m, c in parts:
        ov[m] = (*to_rgb(c), alpha)
    ax.imshow(ov)


def outline(ax, parts):
    ov = np.zeros(lab.shape + (4,))
    for n, m, c in parts:
        # TC is drawn thicker so its edge stays visible where ET is drawn over it
        edge = m & ~binary_erosion(m, iterations=4 if n == "TC" else 2)
        ov[edge] = (*to_rgb(c), 1)
    ax.imshow(ov)


def legend(ax, parts):
    ax.legend(handles=[Patch(color=c, label=n) for n, _, c in parts], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2 if len(parts) > 2 else 1, frameon=False,
              fontsize=13, handlelength=1, columnspacing=1)


fig, ax = plt.subplots(1, 4, figsize=(11.5, 5.6), facecolor="white")
titles = ["T1-Gd", "Primary labels", "Compound labels", "Assessed regions"]
for a, t in zip(ax, titles):
    a.imshow(t1c, cmap="gray")
    a.set_title(t, fontsize=17)
    a.axis("off")
fill(ax[1], PRIMARY); legend(ax[1], PRIMARY)
fill(ax[2], COMPOUND[:1], 0.45); fill(ax[2], COMPOUND[1:], 0.7); legend(ax[2], COMPOUND)
outline(ax[3], ASSESSED); legend(ax[3], ASSESSED)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
