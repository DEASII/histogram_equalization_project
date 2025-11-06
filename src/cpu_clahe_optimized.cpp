#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <mutex>
#include <iomanip>
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// -----------------------------
// ⚡ CLAHE Function (Highly Optimized for CPU)
// -----------------------------
Mat apply_clahe(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);  
    // ✅ ลด memory overhead — ไม่ต้องเก็บสำเนา RGB หลังจาก convert

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    // ✅ ใช้ OpenCV’s built-in CLAHE ซึ่งเป็น optimized C++ implementation
    // (ใช้ SIMD vectorization ภายใน — เร็วกว่าการเขียน manual loop หลายเท่า)

    Mat he_img;
    clahe->apply(gray, he_img);
    return he_img; // ✅ Mat ใช้ reference counting (copy-on-write) → ไม่ duplicate memory
}

// -----------------------------
// 📊 Progress Bar (lightweight, thread-safe)
// -----------------------------
void print_progress_bar(const string& cls, int done, int total, double itps) {
    int bar_width = 30;
    double progress = double(done) / total;
    int pos = bar_width * progress;

    cout << "\rProcessing " << setw(12) << left << cls << ": [";
    for (int i = 0; i < bar_width; ++i)
        cout << (i < pos ? "█" : " ");
    cout << "] " << fixed << setprecision(2)
         << progress * 100.0 << "% (" << itps << " it/s)";
    cout.flush(); // ✅ ใช้ flush แทน endl → ไม่ block I/O (เร็วขึ้นมาก)
}

// -----------------------------
// 🧠 Main Program (Multi-core Parallel CLAHE)
// -----------------------------
int main() {
    string train_path  = (argc > 1) ? argv[1] : "data/input/Brain_Tumor_MRI_Dataset/Training";
    string output_path = (argc > 2) ? argv[2] : "data/output/CPU_HE";
    fs::create_directories(output_path);

    auto global_start = chrono::high_resolution_clock::now();
    mutex io_mutex;  // ✅ ใช้แค่ตอนพิมพ์ progress → ไม่ lock ทั้ง loop

    // ✅ วนแต่ละโฟลเดอร์ (class) แยกการรัน parallel ต่อคลาส
    for (const auto& entry : fs::directory_iterator(train_path)) {
        if (!entry.is_directory()) continue;
        string cls = entry.path().filename().string();

        string class_in = entry.path().string();
        string class_out = output_path + "/" + cls;
        fs::create_directories(class_out);

        // ✅ เตรียม path ล่วงหน้า → ลด I/O overhead ใน loop ขนาน
        vector<pair<string, string>> tasks;
        for (const auto& f : fs::directory_iterator(class_in)) {
            tasks.push_back({f.path().string(), class_out + "/" + f.path().filename().string()});
        }

        int total = tasks.size();
        int done = 0;
        auto start = chrono::high_resolution_clock::now();

        // -----------------------------
        // ⚙️ OpenMP Parallel Processing
        // -----------------------------
        #pragma omp parallel for schedule(dynamic)
        // ✅ ใช้ dynamic scheduling → กระจายโหลดไม่เท่ากันได้ดี (รูปขนาดต่างกัน)
        for (int i = 0; i < (int)tasks.size(); i++) {
            const auto& [input_path, output_file] = tasks[i];
            Mat img = imread(input_path);
            if (img.empty()) continue;

            Mat he_img = apply_clahe(img);
            imwrite(output_file, he_img);

            #pragma omp atomic
            done++; // ✅ atomic → update counter ปลอดภัยโดยไม่ใช้ lock

            // ✅ update ทุก 50 รูป ลด I/O lock contention
            if (done % 50 == 0 || done == total) {
                double elapsed = chrono::duration<double>(
                    chrono::high_resolution_clock::now() - start).count();
                double itps = done / elapsed; // ✅ images per second (throughput)
                lock_guard<mutex> lock(io_mutex);
                print_progress_bar(cls, done, total, itps);
            }
        }

        // ✅ log เมื่อเสร็จ class นั้น ๆ
        double elapsed = chrono::duration<double>(
            chrono::high_resolution_clock::now() - start).count();
        double itps = total / elapsed;

        cout << "\rProcessing " << setw(12) << left << cls
             << ": [██████████████████████████████] 100% (" 
             << fixed << setprecision(2) << itps << " it/s)" << endl;
    }

    // ✅ วัดเวลารวมทั้งหมด
    auto global_end = chrono::high_resolution_clock::now();
    double total_elapsed = chrono::duration<double>(global_end - global_start).count();

    cout << "\n✅ CPU Process ALL DONE in " << fixed << setprecision(2)
         << total_elapsed << " seconds\n";
    cout << "📂 Results saved at: " << output_path << endl;
    return 0;
}
