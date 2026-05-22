#include <iostream>
#include <csignal>
#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>

#include "utils/helpers.hpp"
#include "core/gpu_props.hpp"
#include "monitoring/monitor.hpp"
#include "strategies/base_strategy.hpp"
#include "strategies/stress_strategy.hpp"
#include "strategies/benchmark_strategy.hpp"
#include "ui/tui.hpp"

// Global flag to signal shutdown from Ctrl+C
std::atomic<bool> g_shutdown_flag(false);

void signal_handler(int signum) {
    if (signum == SIGINT) {
        g_shutdown_flag = true;
    }
}

// Helper function to print usage
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " --mode [stress|benchmark] [--precision <p1> <p2> ...]" << std::endl;
    std::cerr << "Example: " << prog_name << " --mode benchmark --precision FP16 FP32" << std::endl;
}

int main(int argc, char** argv) {
    int gpu_id = -1;
    std::string mode;
    std::vector<std::string> selected_precisions;

    // --- Argument Parsing ---
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--gpu-id") {
                if (i + 1 < argc) {
                    gpu_id = std::stoi(argv[++i]);
                }
            } else if (arg == "--mode") {
                if (i + 1 < argc) {
                    mode = argv[++i];
                }
            } else if (arg == "--precision") {
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    selected_precisions.push_back(argv[++i]);
                }
            }
        }
    }

    // Initialize minimal CUDA to list devices
    cudaFree(0);
    auto devices = GpuProperties::list_devices();
    if (devices.empty()) {
        std::cerr << "Critical Error: No CUDA-capable GPU found." << std::endl;
        return 1;
    }

    // --- Validate --gpu-id ---
    if (gpu_id >= 0 && gpu_id >= static_cast<int>(devices.size())) {
        std::cerr << "Error: GPU ID " << gpu_id << " is out of range (0-"
                  << devices.size() - 1 << ")." << std::endl;
        return 1;
    }

    // --- Interactive Menu if no mode is specified ---
    if (mode.empty()) {
        std::cout << "=================================" << std::endl;
        std::cout << "  CUDABurner - Mode Selection    " << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "Available GPUs:" << std::endl;
        for (const auto& dev : devices) {
            std::cout << "  " << dev << std::endl;
        }
        std::cout << "---------------------------------" << std::endl;
        if (gpu_id < 0) {
            std::cout << "Enter GPU ID [0]: ";
            int tmp;
            std::cin >> tmp;
            gpu_id = tmp;
        }
        int choice = 0;
        std::cout << "  1. Stress Test (Max Power)     " << std::endl;
        std::cout << "  2. Benchmark Test (All Precisions)" << std::endl;
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Enter your choice (1 or 2): ";
        std::cin >> choice;
        switch (choice) {
            case 1: mode = "stress"; break;
            case 2: mode = "benchmark"; break;
            default: std::cerr << "Invalid choice. Exiting." << std::endl; return 1;
        }
    }

    if (gpu_id < 0) gpu_id = 0;

    if (mode != "stress" && mode != "benchmark") {
        print_usage(argv[0]);
        return 1;
    }
    if (mode == "stress" && !selected_precisions.empty()) {
        std::cerr << "Warning: --precision is only applicable to benchmark mode." << std::endl;
    }

    signal(SIGINT, signal_handler);

    try {
        // Set the target GPU before any CUDA operations
        CUDA_CHECK(cudaSetDevice(gpu_id));
        CUDA_CHECK(cudaFree(0));

        std::cout << "Initializing CUDABurner for GPU [" << gpu_id << "] in '" << mode << "' mode..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        GpuProperties gpu_props(gpu_id);
        
        // --- DIAGNOSTIC PRINT ---
        std::cout << "---------------------------------------------------------" << std::endl;
        std::cout << "DIAGNOSTIC INFO:" << std::endl;
        std::cout << "  GPU Name            : " << gpu_props.name << std::endl;
        GpuMonitor monitor(gpu_id);
        
        // Start the monitor early so the strategy constructor can use it.
        monitor.start();
        
        std::unique_ptr<BaseStrategy> strategy;
        if (mode == "benchmark") {
            strategy = std::make_unique<BenchmarkStrategy>(monitor, gpu_props, selected_precisions);
        } else { // "stress"
            strategy = std::make_unique<StressStrategy>(gpu_props, monitor);
        }

        TUI tui(monitor, *strategy, mode);

        strategy->start();
        tui.start();

        while (!g_shutdown_flag && !strategy->is_done()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if (strategy->is_done()) {
            std::cout << "\nTest completed. Waiting for exit signal..." << std::endl;
            while (!g_shutdown_flag) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        std::cout << "\nShutdown signal received. Stopping threads..." << std::endl;
        
        strategy->stop();
        monitor.stop();
        tui.stop();

        std::cout << "CUDABurner finished cleanly." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\nAn unrecoverable error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
