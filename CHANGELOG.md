# Changelog

## [Unreleased] — YMMV

### Added
- `--gpu-id <N>` CLI argument to select a specific GPU on multi-GPU systems
- Interactive menu shows available GPU list and lets you pick a device ID
- `GpuProperties::list_devices()` static helper to enumerate CUDA-capable GPUs
- Adaptive matrix sizing (`operators/matrix_size.hpp` / `.cpp`): matrix dimensions now scale automatically to 60% of available VRAM instead of hardcoded `4096`/`8192`
- `Precision::INT4` support:
  - `operators/gemm_int4_tensorcore.hpp` / `.cu` — INT4 packed GEMM via cuBLASLt (sm_90 / Ada 8.9 + CUDA 12.3+)
  - Factory support and benchmark test entry for `INT4 Dense`
- `Precision::FP4` support:
  - Factory support (`operator_factory.cu`) with CUDA 12.3+ compile guard
  - Benchmark test entry for `FP4 Dense`
- `Precision::FP8` support re-enabled:
  - Benchmark test entries for `FP8 Dense` and `FP8 Sparse`
- CMake now auto-detects CUDA include directory (`/usr/local/cuda/include`) and supports manual override via `-DCMAKE_CXX_FLAGS="-I…/include"` / `-DDCMAKE_CUDA_FLAGS="-I…/include"`
- `Precision::INT4` and `Precision::FP4` added to `PRECISION_NAMES` enum map

### Changed
- Benchmark strategies now include `INT4`, `FP8`, `FP4` test cases (previously commented out)
- `stress_strategy.cpp` probe loop: replaced per-iteration `cudaDeviceSynchronize()` with a lightweight 200 ms `sleep_for` to avoid CPU hammering while still producing accurate power samples
- `GpuMonitor::stop()` now resets `monitor_thread_` after join for safe repeated calls
- `GpuMonitor` destructor cleans up `nvmlShutdown()` without redundant comments
- `cudaFree(0)` CUDA initialization call now appears only once in `main.cu`
- `operator_factory.cu` rewritten to use `adaptive_matrix_size()` throughout
- `benchmark_strategy.cpp` now reports `"TOPS"` units for `INT4` results

### Removed
- `main.cpp` — stale duplicate (project uses `main.cu`)
- `operators/vector_add.cu` / `.hpp` — unused operator
- `operators/power_virus_kernel.cu` / `.hpp` — unused operator / dead source

### Fixed
- `GpuMonitor` thread safety on repeated `stop()` / destructor calls
- Stress probe loop no longer blocks each iteration on a full device sync
- `--gpu-id` now validated against actual GPU count before `cudaSetDevice()`

---

## [1.0]

### Added
- `debug_mode` flag
- Performance-limiting-factor display (throttle, power cap, thermal, idle)
- Driver version display
- Per-test 5-second cool-down countdown between benchmark precisions
- Throttle warning indicator during benchmark runs
- Stress test auto-selects the most power-hungry kernel (FP16 / TF32 / FP32 probe)
- `cudaburner appimage` packaging support and `.desktop` / icon install targets

### Changed
- Performance limiter display logic refined (non-trivial reason filtering)
- Overheat / thermal display corrected
