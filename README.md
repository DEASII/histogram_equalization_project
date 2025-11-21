# CLAHE Brain Tumor MRI - CPU vs GPU

A parallel computing project applying **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to Brain Tumor MRI images, comparing **CPU sequential** and **GPU parallel** implementations for large-scale batch processing.

---

## 📝 Project Overview

**Problem:** Enhance contrast in MRI images for different tumor types (glioma, pituitary, meningioma, no tumor) to improve visibility of features.

**Goal:** Demonstrate significant speedup using GPU parallelization over CPU sequential processing.

**Dataset:** Brain Tumor MRI (BraTS or custom) with 4 classes:

**Dataset Link:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

* `notumor`
* `glioma`
* `pituitary`
* `meningioma`

* Images vary in resolution; processing uses grayscale for CLAHE.

---

## 🏗️ Project Structure

```
clahe_mri/
├── src/
│   ├── cpu_clahe.cpp            # CPU sequential CLAHE
│   ├── gpu_clahe.cu             # GPU CUDA CLAHE
│   └── utils.cpp                 # Helper functions
├── include/
│   ├── stb_image.h               # Image I/O
│   └── stb_image_write.h
├── data/
│   ├── input/                    # Original MRI images
│   └── output/
│       ├── cpu/                  # CPU processed
│       └── gpu/                  # GPU processed
├── results/
│   ├── benchmark_results.txt     # Performance comparison
│   └── comparison_samples/       # Before/after images
├── Run with google GPU/.         #ผมไม่มี GPU ใช้ colab แทน
│   ├── CLAHE Notebook.ipynb      #Preview result
│   └── Colab-Link.txt
├── Makefile
├── README.md
└── presentation.pdf
```

---


## 📊 Performance Results

**Test Setup:**

* GPU: Google Colab Tesla T4 / A100 (depends on runtime)
* CPU: Google Colab CPU (Intel Xeon @ ~2.3GHz)
* Dataset: 5,712 MRI images

| Metric        | CPU       | GPU       | Speedup   |
| ------------- | --------- | --------- | --------- |
| Total Time    | 779.69 s  | 29.46 s   | **26.5x** |
| Avg per Image | 0.136 s   | 0.0051 s  | **26.7x** |

---
### GPU Parallel

1. **Tile Histogram Kernel**

   * One block per tile
   * Shared memory + atomic operations for histogram

2. **Pixel Mapping Kernel**

   * Bilinear interpolation between 4 neighboring tile LUTs

---

## 👨‍💻 Author

Thapat Jirametharat

Term Project – Parallel Computing Course

Kasetsart University

---

