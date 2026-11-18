
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
GRAPH_DIR = BASE / "outputs" / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load experimental results
df = pd.read_csv(RESULTS / "quantitative_results.csv")
ocr = pd.read_csv(RESULTS / "ocr_results.csv")

# 1. PSNR comparison
metrics = [
    ("PSNR", "PSNR (dB)", "psnr_comparison.png"),
    ("SSIM", "SSIM", "ssim_comparison.png"),
    ("Time_ms", "Processing Time (ms)", "processing_time.png")
]

for column, ylabel, filename in metrics:
    pivot = df.pivot(
        index="Degradation",
        columns="Filter",
        values=column
    )

    ax = pivot.plot(kind="bar", figsize=(13, 7))
    ax.set_title(f"{ylabel} Comparison Across Degradation Types")
    ax.set_xlabel("Degradation Type")
    ax.set_ylabel(ylabel)
    ax.legend(title="Filter", bbox_to_anchor=(1.02, 1))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300)
    plt.close()

# 2. OCR score comparison
image_col = ocr.columns[0]
score_col = next(
    col for col in ocr.columns
    if "score" in col.lower()
)

degradations = [
    "gaussian_noise",
    "salt_pepper_noise",
    "gaussian_blur",
    "periodic_noise"
]

filter_names = [
    "No Restoration",
    "Mean Filter",
    "Gaussian Filter",
    "Median Filter",
    "Ideal LPF",
    "Butterworth LPF",
    "Gaussian LPF"
]

def identify_degradation(filename):
    name = str(filename).lower()

    for degradation in degradations:
        if name == degradation + ".png":
            return degradation
        if name.startswith(degradation + "_"):
            return degradation

    return None

def identify_filter(filename, degradation):
    if pd.isna(filename) or pd.isna(degradation):
        return None

    name = str(filename).lower()
    degradation = str(degradation).lower()

    if name == degradation + ".png":
        return "No Restoration"

    for filter_name in filter_names[1:]:
        suffix = filter_name.lower().replace(" ", "_")
        if name == f"{degradation}_{suffix}.png":
            return filter_name

    return None
    

ocr["Degradation"] = ocr[image_col].apply(identify_degradation)
ocr["Filter"] = ocr.apply(
    lambda row: identify_filter(
        row[image_col], row["Degradation"]
    ) if row["Degradation"] else None,
    axis=1
)

ocr = ocr.dropna(subset=["Degradation", "Filter"])

ocr_pivot = ocr.pivot_table(
    index="Degradation",
    columns="Filter",
    values=score_col,
    aggfunc="mean"
)

ax = ocr_pivot.plot(kind="bar", figsize=(13, 7))
ax.set_title("OCR Similarity Score Comparison")
ax.set_xlabel("Degradation Type")
ax.set_ylabel("OCR Score (%)")
ax.legend(title="Filter", bbox_to_anchor=(1.02, 1))
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "ocr_comparison.png", dpi=300)
plt.close()

print("All graphs generated successfully!")
print("Saved in:", GRAPH_DIR)
print("OCR comparison rows:", len(ocr))
