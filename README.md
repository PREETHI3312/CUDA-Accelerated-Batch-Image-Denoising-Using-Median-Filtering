# CUDA-Accelerated-Batch-Image-Denoising-Using-Median-Filtering

## 📌 Project Overview

This project implements a **GPU-accelerated image denoising system using CUDA**.

The program applies a **3×3 Median Filter** to a large collection of images. Median filtering is commonly used to remove noise from images, especially **salt-and-pepper noise**, while preserving important edges.

The project compares the performance of:

* CPU-based median filtering
* GPU-based median filtering using a custom CUDA kernel

The GPU implementation uses **CUDA shared memory** to improve memory access efficiency during neighborhood-based image processing.

The program is designed to process **hundreds of images**, satisfying the requirement of performing image processing on a large amount of data.

---

## 🎯 Objectives

The main objectives of this project are:

1. To implement image denoising using a 3×3 median filter.
2. To process a large collection of images automatically.
3. To implement the median filter using a custom CUDA kernel.
4. To use CUDA shared memory for efficient neighborhood access.
5. To compare CPU and GPU processing performance.
6. To calculate the GPU speedup.
7. To verify the correctness of GPU results against CPU results.
8. To save the denoised images for visual comparison.

---

## 🧠 What is Median Filtering?

A median filter is a nonlinear image-processing technique used to remove noise from images.

For every pixel, a 3×3 neighborhood is considered:

```text
P1  P2  P3
P4  P5  P6
P7  P8  P9
```

The nine pixel values are sorted:

```text
P1 ≤ P2 ≤ P3 ≤ P4 ≤ P5 ≤ P6 ≤ P7 ≤ P8 ≤ P9
```

The middle value is selected as the new pixel value:

```text
Median = P5
```

For example:

```text
Input neighborhood:

10   12   11
13  255   14
15   16   12

Sorted values:

10 11 12 12 13 14 15 16 255

Median = 13
```

The noisy value `255` is removed while the surrounding image information is preserved.

---

## 🖼️ Why Median Filtering?

Median filtering is particularly effective for **salt-and-pepper noise**.

Salt-and-pepper noise appears as random:

* White pixels
* Black pixels

in an image.

A median filter replaces abnormal pixel values using the median of their surrounding neighborhood.

Compared with simple averaging, median filtering is better at preserving edges.

---

## 🚀 Why Use CUDA?

A conventional CPU implementation processes pixels sequentially.

For example:

```text
CPU
 │
 ├── Pixel 1
 ├── Pixel 2
 ├── Pixel 3
 ├── Pixel 4
 ├── ...
 └── Pixel N
```

CUDA allows many pixels to be processed simultaneously:

```text
GPU
 │
 ├── Thread 1 → Pixel 1
 ├── Thread 2 → Pixel 2
 ├── Thread 3 → Pixel 3
 ├── Thread 4 → Pixel 4
 ├── ...
 └── Thread N → Pixel N
```

Since the filtering operation for each pixel is largely independent, median filtering is suitable for GPU parallelization.

---

# ⚡ CUDA Implementation

The GPU implementation uses a custom CUDA kernel.

Each CUDA thread is responsible for one output pixel.

For each pixel, the thread:

1. Loads neighboring pixels.
2. Stores them in shared memory.
3. Collects the 3×3 neighborhood.
4. Finds the median value.
5. Writes the result to the output image.

---

## 🧩 Shared Memory

The project uses CUDA shared memory to reduce repeated global-memory accesses.

Instead of every thread repeatedly reading neighboring pixels from global memory, a block of threads cooperatively loads an image tile into shared memory.

The shared-memory layout contains a one-pixel border:

```text
+--------------------+
|      Halo          |
|  +--------------+  |
|  |              |  |
|  | Image Tile   |  |
|  |              |  |
|  +--------------+  |
|      Halo          |
+--------------------+
```

This is useful because every 3×3 median-filter operation needs neighboring pixels.

---

# 📂 Project Structure

```text
CUDA-Median-Filter/
│
├── median_filter.cu
├── README.md
│
├── input_images/
│   ├── image001.jpg
│   ├── image002.jpg
│   ├── image003.jpg
│   └── ...
│
└── output_images/
    ├── denoised_image001.jpg
    ├── denoised_image002.jpg
    ├── denoised_image003.jpg
    └── ...
```

---

# 📊 Dataset

The program accepts a large number of images from the `input_images` directory.

Supported image formats include:

* JPG
* JPEG
* PNG
* BMP
* TIFF
* TIF

For the assignment, the directory can contain:

```text
100+ small images
```

or

```text
10+ large images
```

A dataset containing several hundred images is recommended for demonstrating the advantage of GPU processing.

---

# 🛠️ Technologies Used

* C++
* CUDA
* OpenCV
* NVIDIA GPU
* CUDA Runtime API

---

# 💻 System Requirements

## Hardware

* NVIDIA CUDA-compatible GPU
* Sufficient GPU memory for processing the images

## Software

* NVIDIA CUDA Toolkit
* `nvcc`
* C++ compiler
* OpenCV

---

# 🔧 Installation

## 1. Check CUDA

Run:

```bash
nvcc --version
```

You should see the installed CUDA compiler version.

You can also check the NVIDIA GPU using:

```bash
nvidia-smi
```

---

## 2. Install OpenCV

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install libopencv-dev
```

Check the installation:

```bash
pkg-config --modversion opencv4
```

---

# 🔨 Compilation

Compile the program using:

```bash
nvcc median_filter.cu -o median_filter $(pkg-config --cflags --libs opencv4)
```

---

# ▶️ Running the Program

Create the required directories:

```bash
mkdir -p input_images
mkdir -p output_images
```

Place the dataset images inside:

```text
input_images/
```

Then run:

```bash
./median_filter input_images output_images
```

---

# 📈 Example Program Output

The actual execution time depends on the CPU, NVIDIA GPU, image resolution, and number of images.

Example:

```text
==============================================
CUDA Batch Image Denoising
3x3 Median Filter
==============================================

Input Directory  : input_images
Output Directory : output_images

Images found: 500

CUDA Device:
NVIDIA GeForce RTX ...

Processing images...

[1/500] image001.jpg
[2/500] image002.jpg
[3/500] image003.jpg
...
[500/500] image500.jpg

==============================================
Performance Results
==============================================

Images processed : 500

CPU Time         : 2145.73 ms
GPU Time         : 534.28 ms

GPU Speedup      : 4.02x

Maximum Error    : 0
Average Error    : 0

==============================================
Processing Complete
==============================================
```

**Note:** These are example values. Your actual timings will be different.

---

# 📊 Performance Measurement

The program measures the total processing time for all images.

The speedup is calculated as:

```text
GPU Speedup = CPU Time / GPU Time
```

For example:

```text
CPU Time = 2000 ms
GPU Time = 500 ms

Speedup = 2000 / 500

Speedup = 4x
```

This means that the measured GPU execution took approximately one-fourth of the CPU execution time for the benchmark.

---

# 🔍 Correctness Verification

To verify that the CUDA implementation produces the correct result, the program compares the CPU and GPU outputs.

Two values are calculated:

### Maximum Error

The largest difference between corresponding CPU and GPU pixels.

### Average Error

The average absolute difference between corresponding CPU and GPU pixels.

For an identical implementation:

```text
Maximum Error = 0
Average Error = 0
```

This indicates that the GPU result matches the CPU reference result.

---

# 🧵 CUDA Thread Organization

The image is divided into 2D blocks.

The implementation uses:

```text
16 × 16 threads per block
```

Conceptually:

```text
Grid
+-----------------------------+
| Block | Block | Block       |
|-------|-------|-------------|
| Block | Block | Block       |
|-------|-------|-------------|
| Block | Block | Block       |
+-----------------------------+
```

Each thread processes one output pixel.

---

# 🧠 CUDA Concepts Demonstrated

This project demonstrates:

* CUDA kernels
* CUDA threads
* Thread blocks
* 2D grid configuration
* `threadIdx.x`
* `threadIdx.y`
* `blockIdx.x`
* `blockIdx.y`
* Global memory
* Shared memory
* Host-to-device memory transfer
* Device-to-host memory transfer
* `cudaMalloc`
* `cudaMemcpy`
* `cudaFree`
* `cudaDeviceSynchronize`
* GPU parallel processing
* Performance measurement

---

# 🔄 Processing Workflow

```text
             Input Dataset
                   │
                   ▼
           Read Image Files
                   │
                   ▼
          Convert to Grayscale
                   │
             ┌─────┴─────┐
             │           │
             ▼           ▼
        CPU Median   GPU Median
          Filter       Filter
             │           │
             ▼           ▼
        CPU Result   GPU Result
             │           │
             └─────┬─────┘
                   ▼
           Compare Results
                   │
                   ▼
           Calculate Speedup
                   │
                   ▼
         Save GPU Output Images
```

---

# 📌 Advantages

* Processes a large number of images automatically.
* Uses GPU parallelism.
* Uses shared memory for neighborhood data.
* Effective for salt-and-pepper noise.
* Preserves image edges better than simple averaging.
* Provides CPU vs GPU performance comparison.
* Demonstrates practical CUDA programming.

---

# ⚠️ Limitations

* Requires an NVIDIA CUDA-compatible GPU.
* Requires CUDA Toolkit and OpenCV.
* GPU memory-transfer overhead can affect performance.
* Very small images may not provide significant GPU speedup.
* The current implementation processes images individually rather than processing an entire batch concurrently.

---

# 🔮 Future Enhancements

Possible improvements include:

1. Implementing a 5×5 median filter.
2. Processing multiple images concurrently.
3. Using CUDA streams.
4. Using pinned host memory.
5. Using asynchronous memory transfers.
6. Optimizing the median calculation.
7. Supporting RGB images directly.
8. Comparing 3×3 and 5×5 filters.
9. Comparing different CUDA block sizes.
10. Using CUDA NPP functions for image processing.

---

# 🧪 Experimental Evaluation

The project can be evaluated using different dataset sizes.

For example:

| Number of Images | Image Resolution | CPU Time | GPU Time |   Speedup |
| ---------------: | ---------------: | -------: | -------: | --------: |
|              100 |          256×256 |  Measure |  Measure | Calculate |
|              250 |          256×256 |  Measure |  Measure | Calculate |
|              500 |          256×256 |  Measure |  Measure | Calculate |
|              100 |          512×512 |  Measure |  Measure | Calculate |
|              100 |        1024×1024 |  Measure |  Measure | Calculate |

The values should be obtained by running the program on the target system.

---

# 🏁 Conclusion

This project demonstrates the use of **CUDA GPU parallelism for large-scale image processing**.

A 3×3 median filter was implemented using a custom CUDA kernel. The GPU implementation assigns individual image pixels to CUDA threads and uses shared memory to efficiently access neighboring pixels.

The project also provides a CPU implementation for comparison and measures the performance difference between CPU and GPU processing.

Processing a large collection of images demonstrates how GPU acceleration can be useful for computationally intensive image-processing workloads.

---

# 👩‍💻 Author

**Preethi A K**

CUDA / GPU Image Processing Project

---

# 📜 License

This project is intended for academic and educational purposes.
