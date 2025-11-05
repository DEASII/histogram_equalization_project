// gpu_histogram_equalization.cu
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace chrono;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

// Kernel 1: Compute Histogram (using Atomic Operations)
__global__ void compute_histogram_kernel(unsigned char* image, int* histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        atomicAdd(&histogram[image[idx]], 1);
    }
}

// Kernel 1b: Optimized Histogram with Shared Memory
__global__ void compute_histogram_shared_kernel(unsigned char* image, int* histogram, int size) {
    __shared__ int shared_hist[256];
    
    // Initialize shared histogram
    if (threadIdx.x < 256) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Compute local histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&shared_hist[image[idx]], 1);
    }
    __syncthreads();
    
    // Merge to global histogram
    if (threadIdx.x < 256) {
        atomicAdd(&histogram[threadIdx.x], shared_hist[threadIdx.x]);
    }
}

// Kernel 2: Compute CDF (Parallel Scan - Simple Version)
__global__ void compute_cdf_kernel(int* histogram, int* cdf) {
    // Simple sequential scan on GPU (1 thread)
    // For better performance, implement parallel scan (Blelloch algorithm)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i-1] + histogram[i];
        }
    }
}

// Kernel 3: Equalize Image (Mapping)
__global__ void equalize_kernel(unsigned char* input, unsigned char* output, 
                               int* cdf, int size, int cdf_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        int old_value = input[idx];
        float scale = 255.0f / (size - cdf_min);
        int new_value = (int)((cdf[old_value] - cdf_min) * scale + 0.5f);
        output[idx] = (unsigned char)new_value;
    }
}

// Convert RGB to Grayscale on GPU
__global__ void rgb_to_gray_kernel(unsigned char* rgb, unsigned char* gray, 
                                  int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx < total) {
        int r = rgb[idx * 3];
        int g = rgb[idx * 3 + 1];
        int b = rgb[idx * 3 + 2];
        gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Main GPU Histogram Equalization function
void histogram_equalization_gpu(unsigned char* h_input, unsigned char* h_output,
                               int width, int height) {
    int size = width * height;
    int histogram[256] = {0};
    int cdf[256] = {0};
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    int *d_histogram, *d_cdf;
    
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_histogram, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cdf, 256 * sizeof(int)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(unsigned char), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, 256 * sizeof(int)));
    
    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Step 1: Compute Histogram
    compute_histogram_shared_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_histogram, size);
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: Compute CDF
    compute_cdf_kernel<<<1, 1>>>(d_histogram, d_cdf);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy CDF back to find minimum
    CUDA_CHECK(cudaMemcpy(cdf, d_cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost));
    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // Step 3: Equalize Image
    equalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, d_cdf, size, cdf_min);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram);
    cudaFree(d_cdf);
}

// Process multiple images on GPU
void process_batch_gpu(vector<string>& image_paths, string output_dir) {
    auto start = high_resolution_clock::now();
    
    int total_images = image_paths.size();
    cout << "Processing " << total_images << " images on GPU..." << endl;
    
    for (int idx = 0; idx < total_images; idx++) {
        // Load image
        int width, height, channels;
        unsigned char* img = stbi_load(image_paths[idx].c_str(),
                                      &width, &height, &channels, 0);
        
        if (!img) {
            cerr << "Failed to load: " << image_paths[idx] << endl;
            continue;
        }
        
        int size = width * height;
        unsigned char* gray = new unsigned char[size];
        
        // Convert to grayscale
        if (channels == 3) {
            // Can also use GPU kernel for this
            for (int i = 0; i < size; i++) {
                int r = img[i * 3];
                int g = img[i * 3 + 1];
                int b = img[i * 3 + 2];
                gray[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            }
        } else {
            memcpy(gray, img, size);
        }
        
        // Equalize on GPU
        unsigned char* output = new unsigned char[size];
        histogram_equalization_gpu(gray, output, width, height);
        
        // Save output
        string output_path = output_dir + "/output_" + to_string(idx) + ".png";
        stbi_write_png(output_path.c_str(), width, height, 1, output, width);
        
        // Cleanup
        stbi_image_free(img);
        delete[] gray;
        delete[] output;
        
        if ((idx + 1) % 100 == 0) {
            cout << "Processed " << (idx + 1) << "/" << total_images << endl;
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    cout << "\n=== GPU Results ===" << endl;
    cout << "Total images: " << total_images << endl;
    cout << "Total time: " << duration.count() << " ms" << endl;
    cout << "Avg time per image: " << (float)duration.count() / total_images << " ms" << endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_dir> <output_dir>" << endl;
        return 1;
    }
    
    string input_dir = argv[1];
    string output_dir = argv[2];
    
    // TODO: Load all image paths
    vector<string> image_paths;
    
    cout << "Starting GPU Histogram Equalization..." << endl;
    process_batch_gpu(image_paths, output_dir);
    
    return 0;
}

/*
Compile:
nvcc -o gpu_histogram gpu_histogram_equalization.cu -O3

Run:
./gpu_histogram ./input_images ./output_images
*/