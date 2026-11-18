
import cv2
import numpy as np


# ---------------------------------
# 1. SPATIAL DOMAIN FILTERS
# ---------------------------------

def mean_filter(image):
    return cv2.blur(image, (5, 5))


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 1.0)


def median_filter(image):
    return cv2.medianBlur(image, 5)


# ---------------------------------
# 2. FREQUENCY DOMAIN FILTERS
# ---------------------------------

def frequency_filter(image, filter_type, cutoff=100, order=2):

    # Convert image to floating point
    image_float = np.float32(image)

    # Apply 2D Fourier Transform
    fft = np.fft.fft2(image_float)

    # Shift zero frequency to centre
    fft_shift = np.fft.fftshift(fft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create frequency distance matrix
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol

    U, V = np.meshgrid(v, u)

    D = np.sqrt(U**2 + V**2)

    # Avoid division by zero
    D = np.maximum(D, 1e-10)

    # Create selected low-pass filter
    if filter_type == "ideal":

        H = np.zeros_like(D)
        H[D <= cutoff] = 1

    elif filter_type == "butterworth":

        H = 1 / (1 + (D / cutoff) ** (2 * order))

    elif filter_type == "gaussian":

        H = np.exp(-(D**2) / (2 * cutoff**2))

    else:
        raise ValueError("Unknown frequency filter type")

    # Apply frequency response
    filtered_fft = fft_shift * H

    # Inverse Fourier Transform
    inverse_shift = np.fft.ifftshift(filtered_fft)

    restored = np.fft.ifft2(inverse_shift)

    restored = np.real(restored)

    return np.clip(restored, 0, 255).astype(np.uint8)


def ideal_low_pass_filter(image):
    return frequency_filter(image, "ideal", cutoff=200)


def butterworth_low_pass_filter(image):
    return frequency_filter(
        image, "butterworth", cutoff=200, order=2
    )


def gaussian_low_pass_filter(image):
    return frequency_filter(image, "gaussian", cutoff=200)


# ---------------------------------
# 3. ALL SIX FILTERS
# ---------------------------------

def apply_all_filters(image):

    return {
        "Mean Filter": mean_filter(image),
        "Gaussian Filter": gaussian_filter(image),
        "Median Filter": median_filter(image),
        "Ideal LPF": ideal_low_pass_filter(image),
        "Butterworth LPF": butterworth_low_pass_filter(image),
        "Gaussian LPF": gaussian_low_pass_filter(image)
    }
