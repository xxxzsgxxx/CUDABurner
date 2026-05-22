#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "operators/operator_traits.hpp"

struct GpuProperties {
    int device_id;
    std::string name;
    int cc_major;
    int cc_minor;
    size_t total_global_mem;
    char pci_bus_id[16];

    GpuProperties(int dev_id = 0) : device_id(dev_id), total_global_mem(0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev_id);
        name = prop.name;
        cc_major = prop.major;
        cc_minor = prop.minor;
        total_global_mem = prop.totalGlobalMem;
        std::memcpy(pci_bus_id, prop.pciBusID, 16);
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
