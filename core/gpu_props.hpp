#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstring>

#include "operators/operator_traits.hpp"

struct GpuProperties {
    int device_id;
    std::string name;
    int cc_major;
    int cc_minor;
    size_t total_global_mem;
    char pci_bus_id[32];

    GpuProperties(int dev_id = 0) : device_id(dev_id), total_global_mem(0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);
        name = prop.name;
        cc_major = prop.major;
        cc_minor = prop.minor;
        total_global_mem = prop.totalGlobalMem;
        std::snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%x",
                      prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.pciDeviceID >> 3);
    }

    int sm_version() const { return cc_major * 10 + cc_minor; }

    bool supports_precision(Precision p, Sparsity s = Sparsity::DENSE) const {
        int sm = sm_version();
        switch (p) {
            case Precision::FP64: return sm >= 60;
            case Precision::FP32: return true;
            case Precision::TF32: return sm >= 80;
            case Precision::FP16: return sm >= 70;
            case Precision::BF16: return sm >= 80;
            case Precision::INT8:
                if (s == Sparsity::SPARSE) return sm >= 80; // Ampere+ sparse tensor core
                return sm >= 70; // Volta+ tensor core INT8
            case Precision::FP8:
#if CUDA_VERSION < 11080
                return false; // FP8 requires CUDA 11.8+
#else
                return sm >= 89; // Ada (89), Hopper (90), Blackwell (100+)
#endif
            case Precision::INT4:
#if CUDA_VERSION < 12030
                return false; // INT4 requires CUDA 12.3+
#else
                return sm >= 100; // Blackwell+ (100, 120)
#endif
            case Precision::FP4:
#if CUDA_VERSION < 12030
                return false; // FP4 requires CUDA 12.3+
#else
                return sm >= 100; // Blackwell+ (100, 120)
#endif
            default: return false;
        }
    }

    static std::vector<std::string> list_devices() {
        std::vector<std::string> devices;
        int count;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            devices.push_back(std::string{"[ "} + std::to_string(i) + std::string{" ] "}
                + std::string{prop.name});
        }
        return devices;
    }
};

size_t precision_element_size(Precision p);

int cuda_to_nvml_device_index(int cuda_device_id);
