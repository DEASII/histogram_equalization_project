// cpu_histogram_equalization.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace chrono;

// Function to convert RGB to Grayscale
void rgb_to_grayscale(unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int r = rgb[i * 3];
        int g = rgb[i * 3 + 1];
        int b = rgb[i * 3 + 2];
        gray[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

// Step 1: Compute Histogram
void compute_histogram(unsigned char* image, int* histogram, int size) {
    // Initialize histogram to zero
    for (int i = 0; i < 256; i++) {
        histogram[i] = 0;
    }
    
    // Count frequency of each intensity
    for (int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}

// Step 2: Compute CDF (Cumulative Distribution Function)
void compute_cdf(int* histogram, int* cdf, int size) {
    cdf[0] = histogram[0];
    
    // Accumulate histogram values
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
}

// Step 3: Equalize Image using CDF
void equalize_image(unsigned char* input, unsigned char* output, 
                   int* cdf, int size, int cdf_min) {
    float scale = 255.0f / (size - cdf_min);
    
    for (int i = 0; i < size; i++) {
        int old_value = input[i];
        int new_value = round((cdf[old_value] - cdf_min) * scale);
        output[i] = (unsigned char)new_value;
    }
}

// Main Histogram Equalization function
void histogram_equalization_cpu(unsigned char* input, unsigned char* output, 
                               int width, int height) {
    int size = width * height;
    int histogram[256];
    int cdf[256];
    
    // Step 1: Compute histogram
    compute_histogram(input, histogram, size);
    
    // Step 2: Compute CDF
    compute_cdf(histogram, cdf, size);
    
    // Find CDF minimum (first non-zero value)
    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // Step 3: Equalize
    equalize_image(input, output, cdf, size, cdf_min);
}

// Process multiple images
void process_batch_cpu(vector<string>& image_paths, string output_dir) {
    auto start = high_resolution_clock::now();
    
    int total_images = image_paths.size();
    cout << "Processing " << total_images << " images on CPU..." << endl;
    
    for (int idx = 0; idx < total_images; idx++) {
        // Load image
        int width, height, channels;
        unsigned char* img = stbi_load(image_paths[idx].c_str(), 
                                      &width, &height, &channels, 0);
        
        if (!img) {
            cerr << "Failed to load: " << image_paths[idx] << endl;
            continue;
        }
        
        // Convert to grayscale if needed
        unsigned char* gray = new unsigned char[width * height];
        if (channels == 3) {
            rgb_to_grayscale(img, gray, width, height);
        } else {
            memcpy(gray, img, width * height);
        }
        
        // Equalize
        unsigned char* output = new unsigned char[width * height];
        histogram_equalization_cpu(gray, output, width, height);
        
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
    
    cout << "\n=== CPU Results ===" << endl;
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
    
    // TODO: Load all image paths from input_dir
    // For now, using placeholder
    vector<string> image_paths;
    // Add your images here:
    // image_paths.push_back(input_dir + "/image1.png");
    // image_paths.push_back(input_dir + "/image2.png");
    
    cout << "Starting CPU Histogram Equalization..." << endl;
    process_batch_cpu(image_paths, output_dir);
    
    return 0;
}

/* 
Compile:
g++ -o cpu_histogram cpu_histogram_equalization.cpp -std=c++11 -O3

Run:
./cpu_histogram ./input_images ./output_images
*/