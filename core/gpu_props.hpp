#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

struct GpuProperties {
    int device_id;
    std::string name;
    int cc_major; // Compute Capability Major
    int cc_minor; // Compute Capability Minor
    size_t total_global_mem; // VRAM in bytes

    GpuProperties(int dev_id = 0) : device_id(dev_id), total_global_mem(0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);
        name = prop.name;
        cc_major = prop.major;
        cc_minor = prop.minor;
        total_global_mem = prop.totalGlobalMem;
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
