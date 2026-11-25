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

// Tests for the Level Zero XPU backend (USE_XPU). Device memory is allocated
// in a Level Zero context owned by the test, separate from the engine's, as
// SYCL/PyTorch allocations are in practice. Tests skip on hosts without an
// Intel GPU.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <level_zero/ze_api.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "cuda_alike.h"
#include "memory_location.h"
#include "topology.h"
#include "transport/rdma_transport/rdma_context.h"

using mooncake::DmabufExport;
using mooncake::RdmaContext;

namespace {

class XpuTest : public ::testing::Test {
   protected:
    void SetUp() override {
        int count = 0;
        if (xpuGetDeviceCount(&count) != xpuSuccess || count == 0) {
            GTEST_SKIP() << "no Intel GPU available";
        }
        device_count_ = count;
        ASSERT_EQ(zeInit(ZE_INIT_FLAG_GPU_ONLY), ZE_RESULT_SUCCESS);
        uint32_t n = 1;
        ASSERT_EQ(zeDriverGet(&n, &driver_), ZE_RESULT_SUCCESS);
        uint32_t ndev = 0;
        ASSERT_EQ(zeDeviceGet(driver_, &ndev, nullptr), ZE_RESULT_SUCCESS);
        devices_.resize(ndev);
        ASSERT_EQ(zeDeviceGet(driver_, &ndev, devices_.data()),
                  ZE_RESULT_SUCCESS);
        ze_context_desc_t desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        ASSERT_EQ(zeContextCreate(driver_, &desc, &context_),
                  ZE_RESULT_SUCCESS);
    }

    void TearDown() override {
        for (void *p : allocations_) zeMemFree(context_, p);
        if (context_) zeContextDestroy(context_);
    }

    void *allocDevice(size_t size, int device) {
        ze_device_mem_alloc_desc_t desc = {
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
        void *ptr = nullptr;
        EXPECT_EQ(zeMemAllocDevice(context_, &desc, size, 4096,
                                   devices_[device], &ptr),
                  ZE_RESULT_SUCCESS);
        allocations_.push_back(ptr);
        return ptr;
    }

    void *allocHost(size_t size) {
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
        void *ptr = nullptr;
        EXPECT_EQ(zeMemAllocHost(context_, &desc, size, 4096, &ptr),
                  ZE_RESULT_SUCCESS);
        allocations_.push_back(ptr);
        return ptr;
    }

    static std::vector<char> pattern(size_t size, unsigned seed) {
        std::vector<char> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<char>((i * 131 + seed * 7) & 0xff);
        }
        return data;
    }

    int device_count_ = 0;
    ze_driver_handle_t driver_ = nullptr;
    ze_context_handle_t context_ = nullptr;
    std::vector<ze_device_handle_t> devices_;
    std::vector<void *> allocations_;
};

TEST_F(XpuTest, PciBusIdsResolveToSysfs) {
    for (int i = 0; i < device_count_; ++i) {
        char bus_id[32];
        ASSERT_EQ(xpuDeviceGetPCIBusId(bus_id, sizeof(bus_id), i), xpuSuccess);
        std::string path = std::string("/sys/bus/pci/devices/") + bus_id;
        EXPECT_EQ(access(path.c_str(), F_OK), 0) << path;
    }
    char bus_id[32];
    EXPECT_EQ(xpuDeviceGetPCIBusId(bus_id, sizeof(bus_id), device_count_),
              xpuErrorInvalidDevice);
}

TEST_F(XpuTest, ClassifiesPointersFromAnotherContext) {
    for (int i = 0; i < device_count_; ++i) {
        char *ptr = static_cast<char *>(allocDevice(1 << 20, i));
        for (char *p : {ptr, ptr + 4097}) {
            xpuPointerAttributes attributes;
            ASSERT_EQ(xpuPointerGetAttributes(&attributes, p), xpuSuccess);
            EXPECT_EQ(attributes.type, xpuMemoryTypeDevice);
            EXPECT_EQ(attributes.device, i);
        }
    }

    xpuPointerAttributes attributes;
    void *host = allocHost(4096);
    ASSERT_EQ(xpuPointerGetAttributes(&attributes, host), xpuSuccess);
    EXPECT_EQ(attributes.type, xpuMemoryTypeHost);
    EXPECT_EQ(attributes.device, -1);

    std::vector<char> pageable(4096);
    ASSERT_EQ(xpuPointerGetAttributes(&attributes, pageable.data()),
              xpuSuccess);
    EXPECT_EQ(attributes.type, xpuMemoryTypeUnregistered);
}

TEST_F(XpuTest, CopiesBetweenPageableHostAndDevice) {
    constexpr size_t kSize = 3 * (1 << 20) + 77;
    constexpr size_t kOffset = 13;
    for (int i : {0, device_count_ - 1}) {
        char *dev = static_cast<char *>(allocDevice(kSize + 64, i));
        auto src = pattern(kSize, i);
        std::vector<char> back(kSize, 0);
        ASSERT_EQ(
            cudaMemcpy(dev + kOffset, src.data(), kSize, cudaMemcpyDefault),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(back.data(), dev + kOffset, kSize, cudaMemcpyDefault),
            cudaSuccess);
        EXPECT_EQ(memcmp(src.data(), back.data(), kSize), 0) << "xpu:" << i;
    }

    // Host-to-host needs no device.
    auto a = pattern(256, 1);
    std::vector<char> b(256, 0);
    ASSERT_EQ(cudaMemcpy(b.data(), a.data(), 256, cudaMemcpyDefault),
              cudaSuccess);
    EXPECT_EQ(a, b);
}

TEST_F(XpuTest, ConcurrentCopiesOnOneDevice) {
    constexpr int kThreads = 8;
    constexpr size_t kSize = 1 << 20;
    char *dev = static_cast<char *>(allocDevice(kSize * kThreads, 0));
    std::vector<int> failures(kThreads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            auto src = pattern(kSize, t);
            std::vector<char> back(kSize);
            for (int iter = 0; iter < 20; ++iter) {
                char *slot = dev + t * kSize;
                if (xpuMemcpy(slot, src.data(), kSize, xpuMemcpyDefault) !=
                        xpuSuccess ||
                    xpuMemcpy(back.data(), slot, kSize, xpuMemcpyDefault) !=
                        xpuSuccess ||
                    back != src) {
                    ++failures[t];
                }
            }
        });
    }
    for (auto &thread : threads) thread.join();
    for (int t = 0; t < kThreads; ++t) EXPECT_EQ(failures[t], 0) << t;
}

TEST_F(XpuTest, MemoryLocationNamesTheDevice) {
    for (int i = 0; i < device_count_; ++i) {
        void *ptr = allocDevice(1 << 20, i);
        auto entries = mooncake::getMemoryLocation(ptr, 1 << 20, false);
        ASSERT_EQ(entries.size(), 1u);
        EXPECT_EQ(entries[0].location, "xpu:" + std::to_string(i));
        EXPECT_EQ(entries[0].len, static_cast<size_t>(1 << 20));
    }
}

TEST_F(XpuTest, TopologyListsEveryDevice) {
    mooncake::Topology topology;
    topology.discover();
    auto matrix = topology.getMatrix();
    for (int i = 0; i < device_count_; ++i) {
        EXPECT_EQ(matrix.count("xpu:" + std::to_string(i)), 1u) << i;
    }
}

TEST_F(XpuTest, DeviceMemoryExportsDmabuf) {
    char *ptr = static_cast<char *>(allocDevice(4 << 20, 0));
    DmabufExport exp;
    ASSERT_EQ(RdmaContext::exportDmabuf(ptr + 8192, exp), 0);
    EXPECT_EQ(exp.method, DmabufExport::Method::kDmabufReg);
    ASSERT_GE(exp.fd, 0);
    EXPECT_EQ(exp.offset, 8192u);
    int fd = exp.fd;
    RdmaContext::closeDmabufExport(exp);
    EXPECT_EQ(exp.fd, -1);
    EXPECT_EQ(fcntl(fd, F_GETFD), -1);

    // The driver keeps its own descriptor, so exporting again still works
    // after the caller closed the previous one.
    DmabufExport again;
    ASSERT_EQ(RdmaContext::exportDmabuf(ptr, again), 0);
    EXPECT_EQ(again.method, DmabufExport::Method::kDmabufReg);
    EXPECT_EQ(again.offset, 0u);
    RdmaContext::closeDmabufExport(again);
}

TEST_F(XpuTest, HostMemoryKeepsHostRegistration) {
    std::vector<char> pageable(4096);
    for (void *ptr : {static_cast<void *>(pageable.data()), allocHost(4096)}) {
        DmabufExport exp;
        ASSERT_EQ(RdmaContext::exportDmabuf(ptr, exp), 0);
        EXPECT_EQ(exp.method, DmabufExport::Method::kHostReg);
        EXPECT_EQ(exp.fd, -1);
    }
    int fd = -1;
    size_t offset = 0;
    EXPECT_NE(xpuExportDmabuf(pageable.data(), &fd, &offset), xpuSuccess);
}

}  // namespace
