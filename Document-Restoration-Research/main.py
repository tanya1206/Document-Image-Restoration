
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "dataset" / "original" / "document1.png"
OUTPUT_DIR = BASE_DIR / "dataset" / "degraded"
RESULT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible random generator
rng = np.random.default_rng(42)


def add_gaussian_noise(image):
    """Add controlled Gaussian noise."""
    noise = rng.normal(0, 15, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.03):
    """Add salt-and-pepper noise."""
    noisy = image.copy()

    random_values = rng.random(image.shape)

    noisy[random_values < amount / 2] = 0
    noisy[random_values > 1 - amount / 2] = 255

    return noisy


def add_gaussian_blur(image):
    """Simulate scanning or camera blur."""
    return cv2.GaussianBlur(image, (5, 5), 1.2)


def add_periodic_noise(image):
    """Add sinusoidal periodic interference."""
    height, width = image.shape

    x = np.arange(width)
    y = np.arange(height)

    X, Y = np.meshgrid(x, y)

    noise = 15 * np.sin(2 * np.pi * X / 30)

    noisy = image.astype(np.float32) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():

    print("Starting document degradation experiment...")

    # Load clean reference image
    original = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise FileNotFoundError(
            f"Could not find image: {INPUT_PATH}"
        )

    print("Original image loaded successfully.")
    print("Image dimensions:", original.shape)

    # Generate four degraded images
    degraded_images = {
        "gaussian_noise": add_gaussian_noise(original),
        "salt_pepper_noise": add_salt_pepper_noise(original),
        "gaussian_blur": add_gaussian_blur(original),
        "periodic_noise": add_periodic_noise(original)
    }

    # Save degraded images
    for name, image in degraded_images.items():

        output_path = OUTPUT_DIR / f"{name}.png"

        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path.name}")

    # Create visual comparison
    titles = [
        "Original",
        "Gaussian Noise",
        "Salt & Pepper",
        "Gaussian Blur",
        "Periodic Noise"
    ]

    images = [original] + list(degraded_images.values())

    fig, axes = plt.subplots(1, 5, figsize=(20, 7))

    for ax, image, title in zip(axes, images, titles):

        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    comparison_path = RESULT_DIR / "degradation_comparison.png"

    plt.savefig(
        str(comparison_path),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print("\nExperiment completed successfully!")
    print("Degraded images saved in:", OUTPUT_DIR)
    print("Comparison saved in:", comparison_path)


def run_restoration():

    from filters import apply_all_filters

    print("\nStarting restoration experiment...")

    degradation_types = [
        "gaussian_noise",
        "salt_pepper_noise",
        "gaussian_blur",
        "periodic_noise"
    ]

    for degradation in degradation_types:

        input_path = OUTPUT_DIR.parent / "degraded" / f"{degradation}.png"

        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Missing image: {input_path}")
            continue

        print(f"\nProcessing: {degradation}")

        restored_images = apply_all_filters(image)

        for filter_name, restored in restored_images.items():

            filename = f"{degradation}_{filter_name.lower().replace(' ', '_')}.png"

            save_path = BASE_DIR / "dataset" / "restored" / filename

            cv2.imwrite(str(save_path), restored)

            print(f"Saved: {filename}")

    print("\nAll restoration experiments completed!")


if __name__ == "__main__":
    main()
    run_restoration()
