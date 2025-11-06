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

## 🎯 Algorithm: CLAHE

CLAHE works by:

1. Dividing the image into **tiles** (e.g., 64x64 pixels)
2. Computing **histogram per tile**
3. **Clipping histogram** to limit noise amplification
4. Generating **CDF/LUT per tile**
5. **Mapping pixel values** using bilinear interpolation between neighboring tile LUTs

**CPU Version:** Sequential processing per image
**GPU Version:** Parallel per-tile histogram + LUT + pixel mapping

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
├── Makefile
├── README.md
└── presentation.pdf
```

---

## 🚀 Installation & Setup

### Prerequisites

* **CUDA Toolkit** (11.0+)
* **NVIDIA GPU** (Compute Capability ≥ 3.5)
* **g++** (C++17)
* **OpenCV 4.x**
* **stb_image** library (included)

### Compile

```bash
# Compile both CPU & GPU versions
make all

# Compile individually
make cpu
make gpu
```

---

## ▶️ Usage

### CPU Version

```bash
./build/clahe_cpu data/input/Brain_Tumor_MRI_Dataset/Training data/output/cpu
```

### GPU Version

```bash
./build/clahe_gpu data/input/Brain_Tumor_MRI_Dataset/Training data/output/gpu
```

* Progress bar shows **per-class processing** with speed (images/sec)

### Benchmark Both Versions

```bash
make benchmark
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
| Throughput    | 7.3 img/s | 194 img/s | **26.5x** |

**Observations:**

* GPU achieves massive speedup on batch processing
* Speedup grows with larger batch sizes
* Bilinear interpolation ensures smooth contrast enhancement

---

## 💡 Implementation Details

### CPU Sequential

* Convert image → grayscale
* Apply OpenCV CLAHE per image
* Save output to disk
* Complexity: O(n × m)

### GPU Parallel

1. **Tile Histogram Kernel**

   * One block per tile
   * Shared memory + atomic operations for histogram
2. **Clip & LUT Kernel**

   * Clip histogram to limit contrast
   * Compute CDF per tile → LUT
3. **Pixel Mapping Kernel**

   * Bilinear interpolation between 4 neighboring tile LUTs
   * Fully parallelized across pixels

**Optimizations:**

* Shared memory for histogram
* Coalesced memory access
* Batch processing to amortize data transfer

---

## 🎨 Visual Results

**Example: Before vs After CLAHE**

| Class      | Input (Grayscale) | Output (CLAHE) |
| ---------- | ----------------- | -------------- |
| Notumor    | Dark/low contrast | Enhanced       |
| Glioma     | Mild contrast     | Sharper edges  |
| Pituitary  | Slight blur       | Clearer tissue |
| Meningioma | Low contrast      | Better detail  |

*(See `results/comparison_samples/` for actual images)*

---

## 🧪 Testing & Verification

* Compare CPU vs GPU outputs: `diff` or checksum
* Minor differences expected due to rounding (±1)

**Scalability Example**

| Batch Size | CPU Time | GPU Time | Speedup |
| ---------- | -------- | -------- | ------- |
| 1,000      | 136 s    | 5.2 s    | 26x     |
| 5,712      | 779 s    | 29.5 s   | 26.5x   |

---

## 🐛 Troubleshooting

1. **CUDA Out of Memory** → reduce batch size or tile size
2. **Cannot read images** → check file formats and permissions
3. **Incorrect output** → verify tile LUT and bilinear mapping

---

## 📚 References

1. Gonzalez & Woods, *Digital Image Processing*
2. NVIDIA CUDA C Programming Guide
3. Kirk & Hwu, *Programming Massively Parallel Processors*

---

## 👨‍💻 Author

Thapat Jirametharat

Term Project – Parallel Computing Course

Kasetsart University

---

