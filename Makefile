NVCC = nvcc
GCC = g++
NVCC_FLAGS = -O3 -std=c++11
GCC_FLAGS = -O3 -std=c++17

# Directories
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data

# Targets
CPU_TARGET = $(BUILD_DIR)/histogram_cpu
GPU_TARGET = $(BUILD_DIR)/histogram_gpu

# Default target
all: directories $(CPU_TARGET) $(GPU_TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DATA_DIR)/input
	@mkdir -p $(DATA_DIR)/output/cpu
	@mkdir -p $(DATA_DIR)/output/gpu
	@mkdir -p results

# Compile CPU version
$(CPU_TARGET): $(SRC_DIR)/cpu_histogram_equalization.cpp
	$(GCC) $(GCC_FLAGS) $< -o $@
	@echo "CPU version compiled successfully!"

# Compile GPU version
$(GPU_TARGET): $(SRC_DIR)/gpu_histogram_equalization.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@echo "GPU version compiled successfully!"

# Run CPU version
run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) $(DATA_DIR)/input $(DATA_DIR)/output/cpu

# Run GPU version
run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) $(DATA_DIR)/input $(DATA_DIR)/output/gpu

# Run both and compare
benchmark: $(CPU_TARGET) $(GPU_TARGET)
	@echo "Running CPU version..."
	./$(CPU_TARGET) $(DATA_DIR)/input $(DATA_DIR)/output/cpu
	@echo "\nRunning GPU version..."
	./$(GPU_TARGET) $(DATA_DIR)/input $(DATA_DIR)/output/gpu
	@echo "\nBenchmark complete! Check results directory."

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(DATA_DIR)/output/*
	@echo "Cleaned build files and outputs"

# Clean everything including data
clean_all: clean
	rm -rf $(DATA_DIR)/input/*
	rm -rf results/*
	@echo "Cleaned all generated files"

# Help
help:
	@echo "Available targets:"
	@echo "  make all        - Compile both CPU and GPU versions"
	@echo "  make cpu        - Compile CPU version only"
	@echo "  make gpu        - Compile GPU version only"
	@echo "  make run_cpu    - Run CPU version"
	@echo "  make run_gpu    - Run GPU version"
	@echo "  make benchmark  - Run both and compare performance"
	@echo "  make clean      - Remove build files"
	@echo "  make clean_all  - Remove all generated files"

.PHONY: all directories run_cpu run_gpu benchmark clean clean_all help