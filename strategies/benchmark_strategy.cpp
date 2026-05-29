#include "strategies/benchmark_strategy.hpp"
#include "monitoring/monitor.hpp"
#include "operators/operator_factory.hpp"
#include "utils/helpers.hpp"
#include <chrono>
#include <algorithm>
#include <map>

BenchmarkStrategy::BenchmarkStrategy(GpuMonitor& monitor, const GpuProperties& props, const std::vector<std::string>& selected_precisions)
    : monitor_(monitor), gpu_props_(props) {
    
    // Build the full list of all known precision/sparsity combinations.
    // Unsupported entries will show as "N/A" in the benchmark output,
    // but we still include them so the user sees the capability report.
    std::vector<OperatorDescriptor> all_tests = {
        {Precision::FP64, Sparsity::DENSE},
        {Precision::FP32, Sparsity::DENSE},
        {Precision::TF32, Sparsity::DENSE},
        {Precision::FP16, Sparsity::DENSE},
        {Precision::FP16, Sparsity::SPARSE},
        {Precision::BF16, Sparsity::DENSE},
        {Precision::INT8, Sparsity::DENSE},
        {Precision::INT8, Sparsity::SPARSE},
        {Precision::INT4, Sparsity::DENSE},
        {Precision::FP8,  Sparsity::DENSE},
        // FP8 Sparse and FP4/F4 Sparse not implemented on any GPU
        // {Precision::FP8,  Sparsity::SPARSE},
        {Precision::FP4,  Sparsity::DENSE},
    };

    if (selected_precisions.empty()) {
        tests_to_run_ = all_tests;
    } else {
        // Create a reverse map from string to Precision enum
        std::map<std::string, Precision> name_to_precision;
        for(const auto& pair : PRECISION_NAMES) {
            name_to_precision[pair.second] = pair.first;
        }

        for (const auto& selected : selected_precisions) {
            if (name_to_precision.count(selected)) {
                Precision p = name_to_precision.at(selected);
                // Find all tests (dense and sparse) matching this precision
                for(const auto& test : all_tests){
                    if(test.precision == p){
                        tests_to_run_.push_back(test);
                    }
                }
            } else {
                std::cerr << "Warning: Unknown precision '" << selected << "' ignored." << std::endl;
            }
        }
    }
}

const std::vector<BenchmarkResult>& BenchmarkStrategy::get_results() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return results_;
}

void BenchmarkStrategy::start() {
    stop_flag_ = false;
    worker_thread_ = std::thread(&BenchmarkStrategy::run_loop, this);
}

void BenchmarkStrategy::run_loop() {
    CUDA_CHECK(cudaSetDevice(gpu_props_.device_id));
    cudaFree(0);

    const double test_duration_seconds = 10.0;

    for (size_t idx = 0; idx < tests_to_run_.size(); ++idx) {
        if (stop_flag_) break;

        const auto& test_desc = tests_to_run_[idx];

        BenchmarkResult current_result;
        current_result.descriptor = test_desc;

        std::unique_ptr<BaseOperator> op;
        try {
            op = OperatorFactory::create_gemm_operator(test_desc, gpu_props_, current_result.notes);
        } catch (const std::exception& e) {
            current_result.is_supported = false;
            if (current_result.notes.empty()) {
                current_result.notes = "Runtime Error: " + std::string(e.what());
            }
        }

        if (op == nullptr) {
            current_result.is_supported = false;
        } else {
            current_result.is_supported = true;
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            for (int i = 0; i < 10; ++i) {
                op->execute(stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            long long iterations = 0;
            auto start_time = std::chrono::high_resolution_clock::now();

            while (!stop_flag_) {
                op->execute(stream);
                cudaError_t serr = cudaStreamSynchronize(stream);
                if (serr != cudaSuccess) break;
                iterations++;

                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(now - start_time).count() >= test_duration_seconds) break;

                auto state = monitor_.get_state();
                if (state.throttled) current_result.was_throttled = true;
                current_result.peak.max_temp = std::max(current_result.peak.max_temp, (int)state.temperature);
                current_result.peak.max_power = std::max(current_result.peak.max_power, (double)state.power_usage);
                current_result.peak.max_gpu_clock = std::max(current_result.peak.max_gpu_clock, (unsigned int)state.gpu_clock);
                current_result.peak.max_mem_clock = std::max(current_result.peak.max_mem_clock, (unsigned int)state.mem_clock);
            }

            auto final_end_time = std::chrono::high_resolution_clock::now();
            double precise_duration = std::chrono::duration<double>(final_end_time - start_time).count();

            CUDA_CHECK(cudaStreamDestroy(stream));

            if (iterations > 0 && precise_duration > 0) {
                double total_gops = op->get_gflops_or_gops() * iterations;
                current_result.performance = total_gops / precise_duration / 1000.0;
            }
            current_result.is_native = op->is_native();
            if (test_desc.precision == Precision::INT8 || test_desc.precision == Precision::INT4) {
                current_result.unit = "TOPS";
            } else {
                current_result.unit = "TFLOPS";
            }
        }

        cudaFree(0);

        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(current_result);
        }

        if (!stop_flag_) {
            for (int i = 5; i > 0; --i) {
                countdown_seconds_ = i;
                if(stop_flag_) break;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            countdown_seconds_ = 0;
        }
    }
    done_flag_ = true;
}
