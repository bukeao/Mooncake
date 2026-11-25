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

#pragma once

#include <cstddef>
#include <string>

// Intel GPUs (XPU) through Level Zero. Unlike the other vendors there is no
// CUDA-like runtime to alias, so this header declares the small subset of the
// CUDA runtime API the engine uses and src/gpu_vendor/xpu.cpp implements it
// on top of Level Zero. Device ordinals follow zeDeviceGet order, which is
// the order SYCL and PyTorch (torch.xpu) use for the same process.
const static std::string GPU_PREFIX = "xpu:";

enum xpuError_t {
    xpuSuccess = 0,
    xpuErrorInvalidValue = 1,
    xpuErrorInvalidDevice = 101,
    xpuErrorNoDevice = 100,
    xpuErrorNotSupported = 801,
    xpuErrorUnknown = 999,
};

enum xpuMemoryType {
    xpuMemoryTypeUnregistered = 0,
    xpuMemoryTypeHost = 1,
    xpuMemoryTypeDevice = 2,
    xpuMemoryTypeManaged = 3,
};

struct xpuPointerAttributes {
    xpuMemoryType type;
    int device;  // ordinal of the owning device, -1 for host memory
    void *devicePointer;
    void *hostPointer;
};

enum xpuMemcpyKind {
    xpuMemcpyHostToHost = 0,
    xpuMemcpyHostToDevice = 1,
    xpuMemcpyDeviceToHost = 2,
    xpuMemcpyDeviceToDevice = 3,
    xpuMemcpyDefault = 4,
};

xpuError_t xpuGetDeviceCount(int *count);
xpuError_t xpuSetDevice(int device);
xpuError_t xpuGetDevice(int *device);
xpuError_t xpuDeviceGetPCIBusId(char *pci_bus_id, int len, int device);
// Classifies any pointer. Like cudaPointerGetAttributes, memory that Level
// Zero does not know about is reported as xpuMemoryTypeUnregistered with
// xpuSuccess rather than as an error.
xpuError_t xpuPointerGetAttributes(xpuPointerAttributes *attributes,
                                   const void *ptr);
// Blocking copy. The device that executes the copy is taken from whichever
// side is device memory, so it works from threads that never called
// xpuSetDevice. Host pointers need not be Level Zero allocations.
xpuError_t xpuMemcpy(void *dst, const void *src, size_t count,
                     xpuMemcpyKind kind);
const char *xpuGetErrorString(xpuError_t error);

// Exports the device allocation containing ptr as a dma-buf for
// ibv_reg_dmabuf_mr. On success *fd is a new descriptor owned by the caller
// (close it once every NIC has imported it), and *offset is ptr's offset
// within the exported allocation.
xpuError_t xpuExportDmabuf(const void *ptr, int *fd, size_t *offset);

#define cudaError_t xpuError_t
#define cudaSuccess xpuSuccess
#define cudaGetErrorString xpuGetErrorString
#define cudaMemoryTypeUnregistered xpuMemoryTypeUnregistered
#define cudaMemoryTypeHost xpuMemoryTypeHost
#define cudaMemoryTypeDevice xpuMemoryTypeDevice
#define cudaMemoryTypeManaged xpuMemoryTypeManaged
#define cudaPointerAttributes xpuPointerAttributes
#define cudaMemcpyHostToHost xpuMemcpyHostToHost
#define cudaMemcpyHostToDevice xpuMemcpyHostToDevice
#define cudaMemcpyDeviceToHost xpuMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice xpuMemcpyDeviceToDevice
#define cudaMemcpyDefault xpuMemcpyDefault

#define cudaGetDeviceCount xpuGetDeviceCount
#define cudaSetDevice xpuSetDevice
#define cudaGetDevice xpuGetDevice
#define cudaDeviceGetPCIBusId xpuDeviceGetPCIBusId
#define cudaPointerGetAttributes xpuPointerGetAttributes
#define cudaMemcpy xpuMemcpy
