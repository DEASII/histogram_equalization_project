// utils.cpp - Helper functions for loading images and directories
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;
using namespace std;

// Load all image paths from directory
vector<string> load_image_paths(const string& directory) {
    vector<string> image_paths;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                string path = entry.path().string();
                string ext = entry.path().extension().string();
                
                // Convert extension to lowercase
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                // Check if it's an image file
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || 
                    ext == ".bmp" || ext == ".tga") {
                    image_paths.push_back(path);
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Error reading directory: " << e.what() << endl;
    }
    
    sort(image_paths.begin(), image_paths.end());
    return image_paths;
}

// Create output directory if it doesn't exist
void create_directory(const string& path) {
    try {
        if (!fs::exists(path)) {
            fs::create_directories(path);
            cout << "Created directory: " << path << endl;
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Error creating directory: " << e.what() << endl;
    }
}

// Compare two images (for verification)
double compare_images(unsigned char* img1, unsigned char* img2, int size) {
    long long diff_sum = 0;
    for (int i = 0; i < size; i++) {
        int diff = abs((int)img1[i] - (int)img2[i]);
        diff_sum += diff;
    }
    return (double)diff_sum / size;
}

// Print histogram statistics
void print_histogram_stats(int* histogram) {
    cout << "\nHistogram Statistics:" << endl;
    
    int min_val = 255, max_val = 0;
    int min_count = INT_MAX, max_count = 0;
    
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            if (i < min_val) min_val = i;
            if (i > max_val) max_val = i;
            if (histogram[i] < min_count) min_count = histogram[i];
            if (histogram[i] > max_count) max_count = histogram[i];
        }
    }
    
    cout << "Intensity range: [" << min_val << ", " << max_val << "]" << endl;
    cout << "Count range: [" << min_count << ", " << max_count << "]" << endl;
}

// Save benchmark results to file
void save_benchmark(const string& filename, 
                   int num_images,
                   double cpu_time_ms,
                   double gpu_time_ms) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Failed to open benchmark file" << endl;
        return;
    }
    
    double speedup = cpu_time_ms / gpu_time_ms;
    
    file << "=== Histogram Equalization Benchmark ===" << endl;
    file << "\nNumber of images: " << num_images << endl;
    file << "\nCPU Performance:" << endl;
    file << "  Total time: " << cpu_time_ms << " ms" << endl;
    file << "  Avg per image: " << cpu_time_ms / num_images << " ms" << endl;
    file << "\nGPU Performance:" << endl;
    file << "  Total time: " << gpu_time_ms << " ms" << endl;
    file << "  Avg per image: " << gpu_time_ms / num_images << " ms" << endl;
    file << "\nSpeedup: " << speedup << "x" << endl;
    file << "GPU is " << speedup << " times faster than CPU" << endl;
    
    file.close();
    cout << "\nBenchmark results saved to: " << filename << endl;
}

/*
Example usage in main():

vector<string> image_paths = load_image_paths("./input_images");
create_directory("./output_images");

// Process with CPU
auto cpu_start = chrono::high_resolution_clock::now();
process_batch_cpu(image_paths, "./output_images/cpu");
auto cpu_end = chrono::high_resolution_clock::now();
double cpu_time = chrono::duration_cast<chrono::milliseconds>(cpu_end - cpu_start).count();

// Process with GPU  
auto gpu_start = chrono::high_resolution_clock::now();
process_batch_gpu(image_paths, "./output_images/gpu");
auto gpu_end = chrono::high_resolution_clock::now();
double gpu_time = chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start).count();

// Save benchmark
save_benchmark("benchmark_results.txt", image_paths.size(), cpu_time, gpu_time);
*/