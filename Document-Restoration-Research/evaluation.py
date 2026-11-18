
import cv2
import pandas as pd
import time

from pathlib import Path
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

from filters import (
    mean_filter,
    gaussian_filter,
    median_filter,
    ideal_low_pass_filter,
    butterworth_low_pass_filter,
    gaussian_low_pass_filter
)

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_PATH = BASE_DIR / "dataset" / "original" / "document1.png"
DEGRADED_DIR = BASE_DIR / "dataset" / "degraded"
RESTORED_DIR = BASE_DIR / "dataset" / "restored"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILTERS = {
    "Mean Filter": mean_filter,
    "Gaussian Filter": gaussian_filter,
    "Median Filter": median_filter,
    "Ideal LPF": ideal_low_pass_filter,
    "Butterworth LPF": butterworth_low_pass_filter,
    "Gaussian LPF": gaussian_low_pass_filter
}


def calculate_metrics(reference, restored):

    mse = mean_squared_error(reference, restored)

    psnr = peak_signal_noise_ratio(
        reference, restored, data_range=255
    )

    ssim = structural_similarity(
        reference, restored, data_range=255
    )

    return mse, psnr, ssim


def measure_runtime(filter_function, image, repetitions=5):

    # Warm-up
    filter_function(image)

    start = time.perf_counter()

    for _ in range(repetitions):
        filter_function(image)

    elapsed = time.perf_counter() - start

    return (elapsed / repetitions) * 1000


def main():

    original = cv2.imread(
        str(ORIGINAL_PATH), cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise FileNotFoundError("Original image not found.")

    degradation_types = [
        "gaussian_noise",
        "salt_pepper_noise",
        "gaussian_blur",
        "periodic_noise"
    ]

    rows = []

    print("\nStarting corrected evaluation...\n")

    for degradation in degradation_types:

        degraded_path = DEGRADED_DIR / f"{degradation}.png"

        degraded = cv2.imread(
            str(degraded_path), cv2.IMREAD_GRAYSCALE
        )

        if degraded is None:
            print("Missing:", degraded_path)
            continue

        # Baseline metrics before restoration
        mse, psnr, ssim = calculate_metrics(original, degraded)

        rows.append({
            "Degradation": degradation,
            "Filter": "No Restoration",
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim,
            "Time_ms": 0.0
        })

        for filter_name, filter_function in FILTERS.items():

            filename = (
                f"{degradation}_"
                f"{filter_name.lower().replace(' ', '_')}.png"
            )

            restored_path = RESTORED_DIR / filename

            if not restored_path.exists():
                print("Missing:", filename)
                continue

            restored = cv2.imread(
                str(restored_path), cv2.IMREAD_GRAYSCALE
            )

            if restored is None:
                print("Could not read:", filename)
                continue

            mse, psnr, ssim = calculate_metrics(original, restored)

            # Measure only this individual filter
            runtime = measure_runtime(filter_function, degraded)

            rows.append({
                "Degradation": degradation,
                "Filter": filter_name,
                "MSE": mse,
                "PSNR": psnr,
                "SSIM": ssim,
                "Time_ms": runtime
            })

            print(
                f"{degradation} | {filter_name} | "
                f"PSNR={psnr:.2f} | SSIM={ssim:.4f} | "
                f"Time={runtime:.2f} ms"
            )

    dataframe = pd.DataFrame(rows)

    output_file = RESULTS_DIR / "quantitative_results.csv"

    dataframe.to_csv(output_file, index=False)

    print("\nCorrected evaluation completed!")
    print("Results saved:", output_file)


if __name__ == "__main__":
    main()
