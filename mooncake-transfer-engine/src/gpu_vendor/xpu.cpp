// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gpu_vendor/xpu.h"

#include <glog/logging.h>
#include <level_zero/ze_api.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {

struct XpuDevice {
    ze_device_handle_t handle = nullptr;
    ze_context_handle_t context = nullptr;
    uint32_t copy_ordinal = 0;
    // Synchronous immediate command list, created on first copy. Immediate
    // lists are not thread-safe, so copies on one device are serialized.
    ze_command_list_handle_t copy_list = nullptr;
    std::unique_ptr<std::mutex> copy_mutex = std::make_unique<std::mutex>();
};

struct XpuRuntime {
    // One context per driver: allocations are looked up in every context, so
    // memory allocated by SYCL/PyTorch in their own context is still found.
    std::vector<ze_context_handle_t> contexts;
    std::vector<XpuDevice> devices;
};

thread_local int current_device = 0;

// Prefer a copy-only engine group (the blitter) and fall back to compute,
// which can also copy.
uint32_t findCopyOrdinal(ze_device_handle_t device) {
    uint32_t count = 0;
    zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr);
    std::vector<ze_command_queue_group_properties_t> groups(count);
    for (auto &group : groups) {
        group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        group.pNext = nullptr;
    }
    zeDeviceGetCommandQueueGroupProperties(device, &count, groups.data());
    uint32_t compute = 0;
    bool has_compute = false;
    for (uint32_t i = 0; i < count; ++i) {
        const auto flags = groups[i].flags;
        if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
            return i;
        }
        if (!has_compute &&
            (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
            compute = i;
            has_compute = true;
        }
    }
    return compute;
}

XpuRuntime &runtime() {
    static XpuRuntime rt;
    static std::once_flag once;
    std::call_once(once, [] {
        ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
        if (result != ZE_RESULT_SUCCESS) {
            LOG(WARNING) << "Level Zero initialization failed (0x" << std::hex
                         << result << "); no XPU devices are available";
            return;
        }
        uint32_t driver_count = 0;
        zeDriverGet(&driver_count, nullptr);
        std::vector<ze_driver_handle_t> drivers(driver_count);
        zeDriverGet(&driver_count, drivers.data());
        for (auto driver : drivers) {
            uint32_t device_count = 0;
            zeDeviceGet(driver, &device_count, nullptr);
            std::vector<ze_device_handle_t> handles(device_count);
            zeDeviceGet(driver, &device_count, handles.data());
            if (device_count == 0) continue;

            ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                                              nullptr, 0};
            ze_context_handle_t context = nullptr;
            if (zeContextCreate(driver, &context_desc, &context) !=
                ZE_RESULT_SUCCESS) {
                LOG(WARNING) << "zeContextCreate failed; skipping driver";
                continue;
            }
            rt.contexts.push_back(context);
            for (auto handle : handles) {
                ze_device_properties_t props = {
                    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
                zeDeviceGetProperties(handle, &props);
                if (props.type != ZE_DEVICE_TYPE_GPU) continue;
                XpuDevice device;
                device.handle = handle;
                device.context = context;
                device.copy_ordinal = findCopyOrdinal(handle);
                rt.devices.push_back(std::move(device));
            }
        }
    });
    return rt;
}

xpuError_t toXpuError(ze_result_t result) {
    switch (result) {
        case ZE_RESULT_SUCCESS:
            return xpuSuccess;
        case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        case ZE_RESULT_ERROR_INVALID_SIZE:
            return xpuErrorInvalidValue;
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
            return xpuErrorNotSupported;
        default:
            return xpuErrorUnknown;
    }
}

int deviceOrdinal(ze_device_handle_t handle) {
    auto &devices = runtime().devices;
    for (int pass = 0; pass < 2 && handle; ++pass) {
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].handle == handle) return static_cast<int>(i);
        }
        // Under ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE an allocation may report
        // a sub-device; attribute it to its root device.
        ze_device_handle_t root = nullptr;
        if (zeDeviceGetRootDevice(handle, &root) != ZE_RESULT_SUCCESS) break;
        handle = root;
    }
    return -1;
}

// Finds the context that knows ptr. Returns the context, or nullptr when no
// Level Zero context owns the pointer (e.g. ordinary malloc memory).
ze_context_handle_t findAllocation(const void *ptr,
                                   ze_memory_allocation_properties_t *props,
                                   ze_device_handle_t *device) {
    for (auto context : runtime().contexts) {
        *device = nullptr;
        if (zeMemGetAllocProperties(context, ptr, props, device) ==
                ZE_RESULT_SUCCESS &&
            props->type != ZE_MEMORY_TYPE_UNKNOWN) {
            return context;
        }
    }
    props->type = ZE_MEMORY_TYPE_UNKNOWN;
    return nullptr;
}

int deviceOfPointer(const void *ptr) {
    xpuPointerAttributes attributes;
    if (xpuPointerGetAttributes(&attributes, ptr) != xpuSuccess) return -1;
    return attributes.type == xpuMemoryTypeDevice ? attributes.device : -1;
}

}  // namespace

xpuError_t xpuGetDeviceCount(int *count) {
    if (!count) return xpuErrorInvalidValue;
    *count = static_cast<int>(runtime().devices.size());
    return *count > 0 ? xpuSuccess : xpuErrorNoDevice;
}

xpuError_t xpuSetDevice(int device) {
    if (device < 0 || device >= static_cast<int>(runtime().devices.size())) {
        return xpuErrorInvalidDevice;
    }
    current_device = device;
    return xpuSuccess;
}

xpuError_t xpuGetDevice(int *device) {
    if (!device) return xpuErrorInvalidValue;
    if (runtime().devices.empty()) return xpuErrorNoDevice;
    *device = current_device;
    return xpuSuccess;
}

xpuError_t xpuDeviceGetPCIBusId(char *pci_bus_id, int len, int device) {
    auto &devices = runtime().devices;
    if (!pci_bus_id || len <= 0) return xpuErrorInvalidValue;
    if (device < 0 || device >= static_cast<int>(devices.size())) {
        return xpuErrorInvalidDevice;
    }
    ze_pci_ext_properties_t pci = {ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
    ze_result_t result =
        zeDevicePciGetPropertiesExt(devices[device].handle, &pci);
    if (result != ZE_RESULT_SUCCESS) return toXpuError(result);
    int n = snprintf(pci_bus_id, len, "%04x:%02x:%02x.%x", pci.address.domain,
                     pci.address.bus, pci.address.device, pci.address.function);
    return (n > 0 && n < len) ? xpuSuccess : xpuErrorInvalidValue;
}

xpuError_t xpuPointerGetAttributes(xpuPointerAttributes *attributes,
                                   const void *ptr) {
    if (!attributes) return xpuErrorInvalidValue;
    *attributes = {xpuMemoryTypeUnregistered, -1, nullptr, nullptr};
    ze_memory_allocation_properties_t props = {
        ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    ze_device_handle_t device = nullptr;
    if (!ptr || !findAllocation(ptr, &props, &device)) return xpuSuccess;

    void *p = const_cast<void *>(ptr);
    switch (props.type) {
        case ZE_MEMORY_TYPE_HOST:
            attributes->type = xpuMemoryTypeHost;
            attributes->hostPointer = p;
            attributes->devicePointer = p;
            break;
        case ZE_MEMORY_TYPE_DEVICE:
            attributes->type = xpuMemoryTypeDevice;
            attributes->device = deviceOrdinal(device);
            attributes->devicePointer = p;
            break;
        case ZE_MEMORY_TYPE_SHARED:
            attributes->type = xpuMemoryTypeManaged;
            attributes->device = deviceOrdinal(device);
            attributes->hostPointer = p;
            attributes->devicePointer = p;
            break;
        default:
            break;
    }
    return xpuSuccess;
}

xpuError_t xpuMemcpy(void *dst, const void *src, size_t count,
                     xpuMemcpyKind kind) {
    (void)kind;  // direction is derived from the pointers, as for Default
    if (count == 0) return xpuSuccess;
    if (!dst || !src) return xpuErrorInvalidValue;

    int ordinal = deviceOfPointer(dst);
    if (ordinal < 0) ordinal = deviceOfPointer(src);
    if (ordinal < 0) {
        // Neither side is device memory; everything else is host-accessible.
        memcpy(dst, src, count);
        return xpuSuccess;
    }

    auto &device = runtime().devices[ordinal];
    std::lock_guard<std::mutex> guard(*device.copy_mutex);
    if (!device.copy_list) {
        ze_command_queue_desc_t desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
        desc.ordinal = device.copy_ordinal;
        desc.index = 0;
        desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        ze_result_t result = zeCommandListCreateImmediate(
            device.context, device.handle, &desc, &device.copy_list);
        if (result != ZE_RESULT_SUCCESS) {
            device.copy_list = nullptr;
            return toXpuError(result);
        }
    }
    ze_result_t result = zeCommandListAppendMemoryCopy(
        device.copy_list, dst, src, count, nullptr, 0, nullptr);
    if (result == ZE_RESULT_SUCCESS) {
        result = zeCommandListHostSynchronize(device.copy_list, UINT64_MAX);
    }
    return toXpuError(result);
}

const char *xpuGetErrorString(xpuError_t error) {
    switch (error) {
        case xpuSuccess:
            return "xpuSuccess";
        case xpuErrorInvalidValue:
            return "xpuErrorInvalidValue";
        case xpuErrorInvalidDevice:
            return "xpuErrorInvalidDevice";
        case xpuErrorNoDevice:
            return "xpuErrorNoDevice";
        case xpuErrorNotSupported:
            return "xpuErrorNotSupported";
        default:
            return "xpuErrorUnknown";
    }
}

xpuError_t xpuExportDmabuf(const void *ptr, int *fd, size_t *offset) {
    if (!ptr || !fd || !offset) return xpuErrorInvalidValue;
    ze_external_memory_export_fd_t export_fd = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD, nullptr,
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, -1};
    ze_memory_allocation_properties_t props = {
        ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = findAllocation(ptr, &props, &device);
    if (!context || props.type != ZE_MEMORY_TYPE_DEVICE) {
        return xpuErrorInvalidValue;
    }

    props.pNext = &export_fd;
    ze_result_t result = zeMemGetAllocProperties(context, ptr, &props, nullptr);
    if (result != ZE_RESULT_SUCCESS) return toXpuError(result);
    if (export_fd.fd < 0) return xpuErrorNotSupported;

    void *base = nullptr;
    size_t size = 0;
    result = zeMemGetAddressRange(context, ptr, &base, &size);
    if (result != ZE_RESULT_SUCCESS) return toXpuError(result);

    // The driver caches the exported fd on the allocation and closes it when
    // the allocation is freed, so hand the caller its own descriptor.
    int owned = dup(export_fd.fd);
    if (owned < 0) return xpuErrorUnknown;
    *fd = owned;
    *offset = static_cast<size_t>(static_cast<const char *>(ptr) -
                                  static_cast<const char *>(base));
    return xpuSuccess;
}
