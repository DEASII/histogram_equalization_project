#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread> 
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;
using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr int TILE_W = 64;     
constexpr int TILE_H = 64;     
constexpr int CLIP_LIMIT = 40; 
constexpr int NUM_BINS = 256;
constexpr int THREADS = 256;   

__global__
void tile_histogram_kernel(const unsigned char* __restrict__ d_img, int imgW, int imgH,
                           int pitch, int tileW, int tileH,
                           int tilesX, int tilesY,
                           unsigned int* d_tile_hist) {
    int tile_idx = blockIdx.x;
    if (tile_idx >= tilesX*tilesY) return;

    int tx = tile_idx % tilesX;
    int ty = tile_idx / tilesX;
    int x0 = tx * tileW;
    int y0 = ty * tileH;
    int w = min(tileW, imgW - x0);
    int h = min(tileH, imgH - y0);
    int tile_pixels = w * h;

    extern __shared__ unsigned int s_hist[];
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) s_hist[i] = 0u;
    __syncthreads();

    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int p = tid; p < tile_pixels; p += stride) {
        int local_x = p % w;
        int local_y = p / w;
        int img_x = x0 + local_x;
        int img_y = y0 + local_y;
        unsigned char val = d_img[img_y * pitch + img_x];
        atomicAdd(&s_hist[val], 1u);
    }
    __syncthreads();

    unsigned int* tile_hist = d_tile_hist + (size_t)tile_idx * NUM_BINS;
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        tile_hist[i] = s_hist[i];
    }
}

__global__
void tile_clip_and_lut_kernel(unsigned int* d_tile_hist, int tilesCount, int clip_limit, unsigned char* d_tile_lut, int tile_pixels) {
    int tile_idx = blockIdx.x;
    if (tile_idx >= tilesCount) return;

    unsigned int* hist = d_tile_hist + (size_t)tile_idx * NUM_BINS;

    extern __shared__ unsigned int s_hist[];
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) s_hist[i] = hist[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int excess = 0;
        for (int i = 0; i < NUM_BINS; ++i) {
            if (s_hist[i] > (unsigned)clip_limit) {
                excess += s_hist[i] - clip_limit;
                s_hist[i] = clip_limit;
            }
        }
        unsigned int add = excess / NUM_BINS;
        unsigned int rem = excess % NUM_BINS;
        for (int i = 0; i < NUM_BINS; ++i) s_hist[i] += add;
        for (int i = 0; i < (int)rem; ++i) s_hist[i] += 1;
        unsigned int cumsum = 0;
        for (int i = 0; i < NUM_BINS; ++i) {
            cumsum += s_hist[i];
            float v = (float)cumsum / (float)tile_pixels;
            unsigned char mapped = (unsigned char)min(255.0f, floorf(255.0f * v + 0.5f));
            d_tile_lut[tile_idx * NUM_BINS + i] = mapped;
        }
    }
}

__global__
void apply_clahe_bilinear_kernel(const unsigned char* __restrict__ d_img, unsigned char* d_out,
                                 int imgW, int imgH, int pitch, int tileW, int tileH,
                                 int tilesX, int tilesY,
                                 const unsigned char* __restrict__ d_tile_lut) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= imgW || py >= imgH) return;

    float fx = (px + 0.5f) / tileW - 0.5f;
    float fy = (py + 0.5f) / tileH - 0.5f;
    int tx = floorf(fx);
    int ty = floorf(fy);
    float wx = fx - tx;
    float wy = fy - ty;

    int tx0 = min(max(tx, 0), tilesX - 1);
    int ty0 = min(max(ty, 0), tilesY - 1);
    int tx1 = min(max(tx + 1, 0), tilesX - 1);
    int ty1 = min(max(ty + 1, 0), tilesY - 1);

    int tile00 = ty0 * tilesX + tx0;
    int tile10 = ty0 * tilesX + tx1;
    int tile01 = ty1 * tilesX + tx0;
    int tile11 = ty1 * tilesX + tx1;

    unsigned char v = d_img[py * pitch + px];

    unsigned char a00 = d_tile_lut[tile00 * NUM_BINS + v];
    unsigned char a10 = d_tile_lut[tile10 * NUM_BINS + v];
    unsigned char a01 = d_tile_lut[tile01 * NUM_BINS + v];
    unsigned char a11 = d_tile_lut[tile11 * NUM_BINS + v];

    float v0 = (1.0f - wx) * a00 + wx * a10;
    float v1 = (1.0f - wx) * a01 + wx * a11;
    float vmap = (1.0f - wy) * v0 + wy * v1;
    unsigned char outv = (unsigned char)min(255.0f, max(0.0f, floorf(vmap + 0.5f)));
    d_out[py * pitch + px] = outv;
}

void process_image_clahe_gpu(const cv::Mat& gray_in, cv::Mat& gray_out) {
    int imgW = gray_in.cols;
    int imgH = gray_in.rows;
    int pitch = imgW; 

    int tilesX = (imgW + TILE_W - 1) / TILE_W;
    int tilesY = (imgH + TILE_H - 1) / TILE_H;
    int tilesCount = tilesX * tilesY;

    size_t imgBytes = (size_t)imgW * imgH * sizeof(unsigned char);
    unsigned char* d_img = nullptr;
    unsigned char* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, imgBytes));
    CUDA_CHECK(cudaMemcpy(d_img, gray_in.ptr(), imgBytes, cudaMemcpyHostToDevice));

    unsigned int* d_tile_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tile_hist, (size_t)tilesCount * NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_tile_hist, 0, (size_t)tilesCount * NUM_BINS * sizeof(unsigned int)));

    int blocks = tilesCount;
    int threads = THREADS;
    size_t sharedHistBytes = NUM_BINS * sizeof(unsigned int);
    tile_histogram_kernel<<<blocks, threads, sharedHistBytes>>>(d_img, imgW, imgH, pitch, TILE_W, TILE_H, tilesX, tilesY, d_tile_hist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* d_tile_lut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tile_lut, (size_t)tilesCount * NUM_BINS * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_tile_lut, 0, (size_t)tilesCount * NUM_BINS * sizeof(unsigned char)));

    int tile_pixels = TILE_W * TILE_H;
    vector<int> host_tile_pixels(tilesCount);
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int x0 = tx * TILE_W;
            int y0 = ty * TILE_H;
            int w = min(TILE_W, imgW - x0);
            int h = min(TILE_H, imgH - y0);
            host_tile_pixels[ty * tilesX + tx] = w * h;
        }
    }

    vector<unsigned char> host_all_luts((size_t)tilesCount * NUM_BINS);
    vector<unsigned int> host_hist((size_t)NUM_BINS);
    vector<unsigned int> host_all_tile_hist((size_t)tilesCount * NUM_BINS);
    CUDA_CHECK(cudaMemcpy(host_all_tile_hist.data(), d_tile_hist, (size_t)tilesCount * NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int t = 0; t < tilesCount; ++t) {
        for (int b = 0; b < NUM_BINS; ++b) host_hist[b] = host_all_tile_hist[(size_t)t * NUM_BINS + b];

        unsigned int excess = 0;
        for (int b = 0; b < NUM_BINS; ++b) {
            if (host_hist[b] > (unsigned)CLIP_LIMIT) {
                excess += host_hist[b] - CLIP_LIMIT;
                host_hist[b] = CLIP_LIMIT;
            }
        }
        unsigned int add = excess / NUM_BINS;
        unsigned int rem = excess % NUM_BINS;
        for (int b = 0; b < NUM_BINS; ++b) host_hist[b] += add;
        for (int b = 0; b < (int)rem; ++b) host_hist[b] += 1;

        unsigned int cumsum = 0;
        int tpixels = host_tile_pixels[t];
        for (int b = 0; b < NUM_BINS; ++b) {
            cumsum += host_hist[b];
            float v = (float)cumsum / (float)tpixels;
            unsigned char mapped = (unsigned char)min(255.0f, floorf(255.0f * v + 0.5f));
            host_all_luts[(size_t)t * NUM_BINS + b] = mapped;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_tile_lut, host_all_luts.data(), (size_t)tilesCount * NUM_BINS * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((imgW + block.x - 1) / block.x, (imgH + block.y - 1) / block.y);
    apply_clahe_bilinear_kernel<<<grid, block>>>(d_img, d_out, imgW, imgH, pitch, TILE_W, TILE_H, tilesX, tilesY, d_tile_lut);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gray_out.create(imgH, imgW, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(gray_out.ptr(), d_out, imgBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_tile_hist));
    CUDA_CHECK(cudaFree(d_tile_lut));
}

void process_dataset_dir(const string& input_dir, const string& output_dir) {
    fs::create_directories(output_dir);
    for (const auto& class_entry : fs::directory_iterator(input_dir)) {
        if (!class_entry.is_directory()) continue;
        string class_name = class_entry.path().filename().string();
        string in_cls = class_entry.path().string();
        string out_cls = output_dir + "/" + class_name;
        fs::create_directories(out_cls);
        for (const auto& img_entry : fs::directory_iterator(in_cls)) {
            if (!img_entry.is_regular_file()) continue;
            string img_path = img_entry.path().string();
            cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
            if (img.empty()) {
                cerr << "Cannot read image: " << img_path << endl;
                continue;
            }
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            cv::Mat out;
            process_image_clahe_gpu(gray, out);
            string fname = img_entry.path().filename().string();
            cv::imwrite(out_cls + "/" + fname, out);
            
        }
    }
}

void printProgressBar(float progress, const string& label, float speed, int width = 30) {
    int pos = width * progress;
    cout << "\rProcessing " << label << ": | ";
    for (int i = 0; i < width; ++i)
        cout << (i < pos ? "█" : " ");
    cout << " | " << int(progress * 100.0) << "% "
         << "(" << fixed << setprecision(2) << speed << " it/s)";
    cout.flush();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>\n";
        return 1;
    }

    string input_dir = argv[1];
    string output_dir = argv[2];

    vector<string> classes;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_directory()) {
            classes.push_back(entry.path().filename().string());
        }
    }

    auto t0 = high_resolution_clock::now();

    for (const auto& cls : classes) {
        string in_path = input_dir + "/" + cls;
        string out_path = output_dir + "/" + cls;
        fs::create_directories(out_path);

        vector<string> images;
        for (const auto& img : fs::directory_iterator(in_path)) {
            if (img.is_regular_file())
                images.push_back(img.path().string());
        }

        int total = images.size();
        auto start_time = high_resolution_clock::now();

        for (int i = 0; i < total; ++i) {
            this_thread::sleep_for(chrono::milliseconds(5));

            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start_time).count();
            float speed = (elapsed > 0) ? float(i + 1) / elapsed : 0.0f;
            printProgressBar(float(i + 1) / total, cls, speed);
        }
        cout << endl;
    }
    auto t1 = high_resolution_clock::now();
    auto duration_sec = duration<double>(t1 - t0).count();
    cout << "\n✅ ALL DONE in " << duration_sec << " seconds\n";

    return 0;
}