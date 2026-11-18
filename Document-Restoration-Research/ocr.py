
import subprocess
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

OCR_ENGINE = BASE_DIR / "ocr_runner"
GROUND_TRUTH = BASE_DIR / "dataset" / "original" / "document1_ground_truth.txt"

ORIGINAL_DIR = BASE_DIR / "dataset" / "original"
DEGRADED_DIR = BASE_DIR / "dataset" / "degraded"
RESTORED_DIR = BASE_DIR / "dataset" / "restored"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_text(image_path):

    result = subprocess.run(
        [str(OCR_ENGINE), str(image_path)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or "OCR failed")

    return result.stdout.strip()


def normalize_text(text):

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def calculate_edit_distance(reference, predicted):

    previous = list(range(len(predicted) + 1))

    for i, ref_char in enumerate(reference, start=1):

        current = [i]

        for j, pred_char in enumerate(predicted, start=1):

            cost = 0 if ref_char == pred_char else 1

            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost
            ))

        previous = current

    return previous[-1]


def calculate_cer(reference, predicted):

    if not reference:
        return 0.0

    distance = calculate_edit_distance(reference, predicted)

    return distance / len(reference)


def main():

    if not OCR_ENGINE.exists():
        raise FileNotFoundError("Compile ocr_engine.swift first.")

    reference = normalize_text(
        GROUND_TRUTH.read_text(encoding="utf-8")
    )

    image_paths = []

    image_paths.extend(ORIGINAL_DIR.glob("*.png"))
    image_paths.extend(DEGRADED_DIR.glob("*.png"))
    image_paths.extend(RESTORED_DIR.glob("*.png"))

    rows = []

    print("\nStarting OCR evaluation...\n")

    for image_path in image_paths:

        predicted_text = extract_text(image_path)
        normalized_prediction = normalize_text(predicted_text)

        cer = calculate_cer(reference, normalized_prediction)

        ocr_score = max(0.0, 1.0 - cer) * 100

        rows.append({
            "Image": image_path.name,
            "OCR_Score_Percent": ocr_score,
            "CER": cer
        })

        print(
            f"{image_path.name} | "
            f"OCR Score: {ocr_score:.2f}% | "
            f"CER: {cer:.4f}"
        )

    dataframe = pd.DataFrame(rows)

    output_path = RESULTS_DIR / "ocr_results.csv"

    dataframe.to_csv(output_path, index=False)

    print("\nOCR evaluation completed!")
    print("Results saved:", output_path)


if __name__ == "__main__":
    main()
