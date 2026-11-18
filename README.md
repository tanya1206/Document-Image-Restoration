# Document Image Restoration: Spatial vs FFT Low-Pass Filters

## Project Overview
This research project presents an empirical evaluation of spatial-domain and frequency-domain (FFT-based) low-pass filters for document image restoration.

The study compares six filtering techniques under four types of image degradation using quantitative image quality metrics and OCR-based evaluation.

## Objectives
- Compare spatial and frequency-domain filtering methods.
- Evaluate restoration quality under different degradation conditions.
- Measure MSE, PSNR, SSIM, processing time, and OCR similarity.
- Identify the strengths and limitations of each filtering approach.

## Methods Evaluated

### Spatial Filters
- Mean Filter
- Gaussian Filter
- Median Filter

### Frequency-Domain Filters
- Ideal Low-Pass Filter
- Butterworth Low-Pass Filter
- Gaussian Low-Pass Filter

## Degradation Types
- Gaussian Noise
- Salt-and-Pepper Noise
- Gaussian Blur
- Periodic Noise

## Evaluation Metrics
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Processing Time
- Character Error Rate (CER) and OCR Similarity

## Technologies Used
- Python
- NumPy
- OpenCV
- Matplotlib
- Apple Vision Framework (Swift) for OCR

## Project Structure
```text
Document-Restoration-Research/
├── dataset/
│   ├── original/
│   ├── degraded/
│   └── restored/
├── results/
├── outputs/
│   └── graphs/
├── main.py
├── filters.py
├── evaluation.py
├── ocr.py
├── plot_results.py
└── README.md
```

## Key Findings
- The Gaussian spatial filter achieved the highest PSNR among the restoration filters across the four tested degradation types.
- The Median filter achieved the highest SSIM for salt-and-pepper noise.
- Spatial filters had substantially lower processing times than the FFT-based filters in this implementation.
- The results demonstrate that filter performance depends on the degradation type and evaluation metric.

## Limitations
The experiment uses one synthetic document and one severity level for each degradation type. Further testing with diverse real-world documents is required.

## Authors
- Tanya Singh
- Vaibhav
- Mohammad Ali

## Research Focus
An Empirical Evaluation of Spatial vs FFT Low-Pass Filters for Document Image Restoration.
## Research Paper

[Read the Full Research Paper (PDF)](Updated_IEEE_Document_Image_Restoration_Paper.pdf)
