/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zero_copy_address_mgr.h"
#include "adapter_rts_common.h"

namespace hccl {

HcclResult ZeroCopyAddressMgr::SetMemoryRange(u32 devicePhyId, void *baseAddr, u64 length)
{
    if (baseAddr == nullptr || length == 0) {
        HCCL_ERROR("[ZeroCopyAddressMgr][SetMemoryRange] invalid input params addr[%p] len[%lu]", baseAddr, length);
        return HCCL_E_PARA;
    }

    CHK_PRT_RET(AddLocalIpc2RemoteAddr(devicePhyId, baseAddr, baseAddr, length),
        HCCL_ERROR("[ZeroCopyAddressMgr][SetMemoryRange] dev[%u] set addr [%p] failed", devicePhyId, baseAddr),
        HCCL_E_PARA);

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::UnsetMemoryRange(u32 devicePhyId, void *baseAddr)
{
    if (baseAddr == nullptr) {
        HCCL_ERROR("[ZeroCopyAddressMgr][UnsetMemoryRange] invalid input params");
        return HCCL_E_PARA;
    }

    CHK_PRT_RET(DelLocalIpc2RemoteAddr(devicePhyId, baseAddr),
        HCCL_ERROR("[ZeroCopyAddressMgr][UnsetMemoryRange] dev[%u] unset [%p] addr failed", devicePhyId, baseAddr),
        HCCL_E_PARA);

    return HCCL_SUCCESS;
}

bool ZeroCopyAddressMgr::IsAddressSet(u32 devicePhyId, void *baseAddr)
{
    std::lock_guard<std::mutex> guard(lock_);
    auto &addrMapping = reserveAddrMappings_[devicePhyId];
    return addrMapping.find(baseAddr) != addrMapping.end();
}

HcclResult ZeroCopyAddressMgr::AddLocalIpc2RemoteAddr(u32 devicePhyId, void *localIpcBase, void *remoteAddrBase, u64 length)
{
    if (devicePhyId >= MAX_MODULE_DEVICE_NUM || localIpcBase == nullptr || remoteAddrBase == nullptr) {
        HCCL_ERROR("[ZeroCopyAddressMgr][AddLocalIpc2RemoteAddr] devPhyId [%u] localIpc[%p] remoteAddr[%p] invalid params",
            devicePhyId, localIpcBase, remoteAddrBase);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> guard(lock_);
    auto &addrMapping = reserveAddrMappings_[devicePhyId];
    auto &addrRange = reserveRanges_[devicePhyId];

    // 检查地址是否已经reserve过
    CHK_PRT_RET(addrMapping.find(remoteAddrBase) != addrMapping.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][AddLocalIpc2RemoteAddr] dev[%u] remote addr %p had set", devicePhyId, remoteAddrBase), HCCL_E_PARA);

    // 检查地址reserve的地址区间是否与之前的有交叠
    AddressRange range(remoteAddrBase, length);
    CHK_PRT_RET(addrRange.find(range) != addrRange.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][AddLocalIpc2RemoteAddr] dev[%u] remote addr %p had set with overlap range",
        devicePhyId, remoteAddrBase), HCCL_E_PARA);

    ZeroCopyRingBufferItem item;
    item.type = ZeroCopyItemType::SET_MEMORY;
    item.addr.devicePhyId = devicePhyId;
    item.addr.localIpcAddr = reinterpret_cast<u64>(localIpcBase);
    item.addr.remoteAddr = reinterpret_cast<u64>(remoteAddrBase);
    item.addr.length = length;
    addrMapping.insert({remoteAddrBase, item.addr});
    addrRange.insert(range);
    CHK_RET(PushOne(item));
    HCCL_INFO("[ZeroCopyAddressMgr][AddLocalIpc2RemoteAddr] dev[%u] add set localIpc[%p] remote[%p] length[%lu]",
        devicePhyId, localIpcBase, remoteAddrBase, length);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::DelLocalIpc2RemoteAddr(u32 devicePhyId, void *remoteAddrBase)
{
    if (devicePhyId >= MAX_MODULE_DEVICE_NUM || remoteAddrBase == nullptr) {
        HCCL_ERROR("[ZeroCopyAddressMgr][DelLocalIpc2RemoteAddr] devPhyId [%u] invalid params", devicePhyId);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> guard(lock_);
    auto &addrMapping = reserveAddrMappings_[devicePhyId];
    auto &addrRange = reserveRanges_[devicePhyId];

    // 检查地址是否已经reserve过
    auto mappingIt = addrMapping.find(remoteAddrBase);
    CHK_PRT_RET(mappingIt == addrMapping.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][DelLocalIpc2RemoteAddr] dev[%u] addr %p not set", devicePhyId, remoteAddrBase), HCCL_E_PARA);

    u64 length = mappingIt->second.length;
    AddressRange range(remoteAddrBase, length);
    auto rangeIt = addrRange.find(range);
    CHK_PRT_RET(rangeIt == addrRange.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][DelLocalIpc2RemoteAddr] dev[%u] addr %p not set", devicePhyId, remoteAddrBase), HCCL_E_PARA);

    // 检查是否仍存在Activate的内存
    AddressRange localRange(mappingIt->second.localIpcAddr, length);
    auto activateIt = validAddressRanges_.find(localRange);
    CHK_PRT_RET(activateIt != validAddressRanges_.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][DelLocalIpc2RemoteAddr] dev[%u] remoteAddr %p localAddr 0x%lx still have activate memory [0x%lx, 0x%lx)",
        devicePhyId, remoteAddrBase, mappingIt->second.localIpcAddr, activateIt->start, activateIt->end), HCCL_E_PARA);

    ZeroCopyRingBufferItem item;
    item.type = ZeroCopyItemType::UNSET_MEMORY;
    item.addr = mappingIt->second;
    addrMapping.erase(mappingIt);
    addrRange.erase(rangeIt);
    CHK_RET(PushOne(item));
    HCCL_INFO("[ZeroCopyAddressMgr][DelLocalIpc2RemoteAddr] dev[%u] del set localIpc[0x%lx] remote[0x%lx] length[%lu]",
        devicePhyId, item.addr.localIpcAddr, item.addr.remoteAddr, length);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::GetLocalIpc2RemoteAddr(u32 devicePhyId, void *remoteAddr, LocalIpc2RemoteAddr &addr)
{
    if (devicePhyId >= MAX_MODULE_DEVICE_NUM || remoteAddr == nullptr) {
        HCCL_ERROR("[ZeroCopyAddressMgr][GetLocalIpc2RemoteAddr] devPhyId [%u] invalid params", devicePhyId);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> guard(lock_);
    auto &addrMapping = reserveAddrMappings_[devicePhyId];
    auto &addrRange = reserveRanges_[devicePhyId];

    AddressRange range(remoteAddr, 1);
    auto rangeIt = addrRange.find(range);
    CHK_PRT_RET(rangeIt == addrRange.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][GetLocalIpc2RemoteAddr] dev[%u] addr %p not set", devicePhyId, remoteAddr), HCCL_E_PARA);

    void *remoteAddrBase = reinterpret_cast<void *>(rangeIt->start);
    auto mapIt = addrMapping.find(remoteAddrBase);
    CHK_PRT_RET(rangeIt == addrRange.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][GetLocalIpc2RemoteAddr] dev[%u] addr %p not set", devicePhyId, remoteAddr), HCCL_E_PARA);

    addr = mapIt->second;

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::ActivateCommMemoryAddr(void *startPtr, u64 length)
{
    CHK_PRT_RET((startPtr == nullptr || length == 0),
        HCCL_ERROR("[ZeroCopyAddressMgr][ActivateCommMemoryAddr] Invalid params"), HCCL_E_PARA);
    
    AddressRange range(startPtr, length);

    std::lock_guard<std::mutex> guard(lock_);
    auto it = validAddressRanges_.find(range);
    CHK_PRT_RET((it != validAddressRanges_.end()),
        HCCL_ERROR("[ZeroCopyAddressMgr][ActivateCommMemoryAddr] overlap address exist:[0x%lx, 0x%lx) valid:[0x%lx, 0x%lx)",
        it->start, it->end, range.start, range.end), HCCL_E_PARA);
    
    validAddressRanges_.insert(range);

    ZeroCopyRingBufferItem item;
    item.type = ZeroCopyItemType::ACTIVATE_MEMORY;
    item.addr.localIpcAddr = reinterpret_cast<u64>(startPtr);
    item.addr.length = length;
    CHK_RET(PushOne(item));

    HCCL_INFO("[ZeroCopyAddressMgr][ActivateCommMemoryAddr] activate address [0x%lx, 0x%lx) success", range.start, range.end);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::DeactivateCommMemoryAddr(void *startPtr)
{
    CHK_PRT_RET((startPtr == nullptr),
        HCCL_ERROR("[ZeroCopyAddressMgr][DeactivateCommMemoryAddr] Invalid params"), HCCL_E_PARA);
    
    // 我们构造一个最小的交叠区间去做比较
    u64 litteLen = 1;
    AddressRange range(startPtr, litteLen);

    std::lock_guard<std::mutex> guard(lock_);
    auto it = validAddressRanges_.find(range);
    CHK_PRT_RET((it == validAddressRanges_.end() || it->start != range.start),
        HCCL_ERROR("[ZeroCopyAddressMgr][DeactivateCommMemoryAddr] address %p is not activate", startPtr), HCCL_E_PARA);
    
    HCCL_INFO("[ZeroCopyAddressMgr][DeactivateCommMemoryAddr] deactivate address [0x%lx, 0x%lx) success", it->start, it->end);
    validAddressRanges_.erase(it);

    ZeroCopyRingBufferItem item;
    item.type = ZeroCopyItemType::DEACTIVATE_MEMORY;
    item.addr.localIpcAddr = reinterpret_cast<u64>(startPtr);
    CHK_RET(PushOne(item));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::AddRemoteImportAddr(void *devPtr, void *handle)
{
    CHK_PRT_RET((devPtr == nullptr || handle == nullptr),
        HCCL_ERROR("[ZeroCopyAddressMgr][AddRemoteImportAddr] invalid devPtr[%p] handle[%p]",
        devPtr, handle), HCCL_E_PARA);

    std::lock_guard<std::mutex> guard(lock_);
    auto it = importAddrs_.find(devPtr);
    CHK_PRT_RET(it != importAddrs_.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][AddRemoteImportAddr] devPtr[%p] has import", devPtr), HCCL_E_PARA);

    HCCL_INFO("[ZeroCopyAddressMgr][AddRemoteImportAddr] add devPtr[%p] handle[%p]", devPtr, handle);
    importAddrs_.insert({devPtr, handle});
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::GetRemoteImportAddr(void *devPtr, void *&handle)
{
    CHK_PRT_RET((devPtr == nullptr),
        HCCL_ERROR("[ZeroCopyAddressMgr][GetRemoteImportAddr] invalid devPtr[%p]", devPtr), HCCL_E_PARA);

    std::lock_guard<std::mutex> guard(lock_);
    auto it = importAddrs_.find(devPtr);
    CHK_PRT_RET(it == importAddrs_.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][GetRemoteImportAddr] devPtr[%p] not import", devPtr), HCCL_E_PARA);

    handle = importAddrs_[devPtr];
    HCCL_INFO("[ZeroCopyAddressMgr][GetRemoteImportAddr] get devPtr[%p] handle[%p]", devPtr, handle);

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::DelRemoteImportAddr(void *devPtr)
{
    CHK_PRT_RET((devPtr == nullptr),
        HCCL_ERROR("[ZeroCopyAddressMgr][DelRemoteImportAddr] invalid devPtr[%p]", devPtr), HCCL_E_PARA);

    std::lock_guard<std::mutex> guard(lock_);
    auto it = importAddrs_.find(devPtr);
    CHK_PRT_RET(it == importAddrs_.end(),
        HCCL_ERROR("[ZeroCopyAddressMgr][GetRemoteImportAddr] devPtr[%p] not import", devPtr), HCCL_E_PARA);

    void *handle = importAddrs_[devPtr];
    HCCL_INFO("[ZeroCopyAddressMgr][GetRemoteImportAddr] del devPtr[%p] handle[%p]", devPtr, handle);
    importAddrs_.erase(it);

    return HCCL_SUCCESS;
}

bool ZeroCopyAddressMgr::IsActivateCommMemoryAddr(void *startPtr, u64 length)
{
    if (startPtr == nullptr || length == 0) {
        return false;
    }

    AddressRange range(startPtr, length);

    std::lock_guard<std::mutex> guard(lock_);
    // 我们先用最小区间去查找最前面匹配的valid内存块，后续的就依次遍历即可
    AddressRange litteRange(startPtr, 1);
    auto beginIt = validAddressRanges_.find(litteRange);
    while (beginIt != validAddressRanges_.end()) {
        // 此片内存已经是valid内存的子集了，那么是有效的
        if (range.end <= beginIt->end) {
            return true;
        }

        // 前一片内存已经匹配到了小块，把前面的内存切分掉，继续去匹配更后面的数据
        range.start = beginIt->end;
        beginIt++;
    }

    // 输入区间内有部分没有查找到，所以认为是无效的
    return false;
}

bool ZeroCopyAddressMgr::IsOverlapWithActivateAddr(void *startPtr, u64 length)
{
    if (startPtr == nullptr || length == 0) {
        return false;
    }

    AddressRange range(startPtr, length);
    std::lock_guard<std::mutex> guard(lock_);
    return validAddressRanges_.find(range) != validAddressRanges_.end();
}

bool ZeroCopyAddressMgr::IsInSetAddressRange(u32 devicePhyId, void *startPtr, u64 length)
{
    std::lock_guard<std::mutex> guard(lock_);
    auto &addrRange = reserveRanges_[devicePhyId];

    // 构造最小的数据块去寻找，如果没找到肯定没有交集
    AddressRange range(startPtr, 1);
    auto rangeIt = addrRange.find(range);
    if (rangeIt == addrRange.end()) {
        HCCL_INFO("[ZeroCopyAddressMgr][IsInSetAddressRange] not in reserve range");
        return false;
    }

    // 判断尾巴是否在当前匹配内存块中，如果不在那么不在范围内
    if (range.start + length > rangeIt->end) {
        HCCL_INFO("[ZeroCopyAddressMgr][IsInSetAddressRange] exceed reserve range");
        return false;
    }
    
    return true;
}

HcclResult ZeroCopyAddressMgr::InitRingBuffer()
{
    if (ringBuffer_.ptr() != nullptr) {
        return HCCL_SUCCESS;
    }

    ringBuffer_ = DeviceMem::alloc(ZERO_COPY_BUFFER_MAX_MAP_COUNT * sizeof(ZeroCopyRingBufferItem));
    CHK_PRT_RET(ringBuffer_.ptr() == nullptr,
        HCCL_ERROR("[ZeroCopyAddressMgr][InitRingBuffer] alloc ring buffer failed"), HCCL_E_INTERNAL);
    CHK_RET(hrtMemSet(ringBuffer_.ptr(), ringBuffer_.size(), ringBuffer_.size()));

    ringBufferCtl_ = DeviceMem::alloc(sizeof(u32) + sizeof(u32));
    CHK_PRT_RET(ringBufferCtl_.ptr() == nullptr,
        HCCL_ERROR("[ZeroCopyAddressMgr][InitRingBuffer] alloc ring buffer ctl failed"), HCCL_E_INTERNAL);
    CHK_RET(hrtMemSet(ringBufferCtl_.ptr(), ringBufferCtl_.size(), ringBufferCtl_.size()));

    devRingBufBase_ = reinterpret_cast<ZeroCopyRingBufferItem *>(ringBuffer_.ptr());
    devRingHead_ = reinterpret_cast<u32 *>(ringBufferCtl_.ptr());
    devRingTail_ = devRingHead_ + 1;

    HCCL_RUN_INFO("[ZeroCopyAddressMgr][InitRingBuffer] ringbuffer[%p] len[%lu] bufferCtl[%p] len[%lu] head[%p] tail[%p]",
        ringBuffer_.ptr(), ringBuffer_.size(), ringBufferCtl_.ptr(), ringBufferCtl_.size(), devRingHead_, devRingTail_);

    HCCL_INFO("[ZeroCopyAddressMgr][InitRingBuffer] ringbuffer[%p] head[%p] tail[%p]", devRingBufBase_, devRingHead_, devRingTail_);

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::PushOne(ZeroCopyRingBufferItem &item)
{
    if (!needPushOne) {
        HCCL_DEBUG("[ZeroCopyAddressMgr][PushOne] don't need push");
        return HCCL_SUCCESS;
    }

    // 检测RingBuffer是否已经初始化，没有的话就初始化一下
    CHK_RET(InitRingBuffer());

    u32 head = 0;
    CHK_RET(hrtMemSyncCopy(&head, sizeof(head), devRingHead_, sizeof(head), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    u32 tail = 0;
    CHK_RET(hrtMemSyncCopy(&tail, sizeof(tail), devRingTail_, sizeof(tail), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    u32 updateTail = (tail + 1) % ZERO_COPY_BUFFER_MAX_MAP_COUNT;
    CHK_PRT_RET(updateTail == head,
        HCCL_ERROR("[ZeroCopyAddressMgr][PushOne] ring buffer is full head[%u] tail[%u] capacity[%u]",
        head, tail, ZERO_COPY_BUFFER_MAX_MAP_COUNT), HCCL_E_INTERNAL);

    HCCL_INFO("[ZeroCopyAddressMgr][PushOne] type[%d] head[%u] tail[%u] updateTail[%u] tailAddr[%p]", item.type, head, tail, updateTail, devRingBufBase_ + tail);
    CHK_RET(hrtMemSyncCopy(devRingBufBase_ + tail, sizeof(ZeroCopyRingBufferItem), &item, sizeof(ZeroCopyRingBufferItem),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    CHK_RET(hrtMemSyncCopy(devRingTail_, sizeof(updateTail), &updateTail, sizeof(updateTail),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::ProcessRingBuffer(ZeroCopyRingBufferItem *ringBuffer, u32 *head, u32 *tail)
{
    if (ringBuffer == nullptr || head == nullptr || tail == nullptr) {
        HCCL_ERROR("[ZeroCopyAddressMgr][ProcessRingBuffer] invalid param ringBuff[%p] head[%p] tail[%p]",
            ringBuffer, head, tail);
        return HCCL_E_PARA;
    }

    std::lock_guard<std::mutex> guard(processRingBufferLock_);
    needPushOne = false;
    if (*head == *tail) {
        HCCL_INFO("[ZeroCopyAddressMgr][ProcessRingBuffer] ring buffer is empty, so do nothing, head[%u] tail[%u]", *head, *tail);
        return HCCL_SUCCESS;
    }

    if (*tail >= ZERO_COPY_BUFFER_MAX_MAP_COUNT || *head >= ZERO_COPY_BUFFER_MAX_MAP_COUNT) {
        HCCL_ERROR("[ZeroCopyAddressMgr][ProcessRingBuffer] invalid head/tail, head[%u] tail[%u]", *head, *tail);
        return HCCL_E_PARA;
    }

    u32 now = *head;
    while (now != *tail) {
        HCCL_INFO("[ZeroCopyAddressMgr][ProcessRingBuffer] process ringbuffer now[%u] ptr[%p] tail[%u] type[%d]",
            now, ringBuffer + now, *tail, ringBuffer[now].type);
        CHK_RET(ProcessOneAddrMap(ringBuffer[now]));
        now = (now + 1) % ZERO_COPY_BUFFER_MAX_MAP_COUNT;
    }

    // 更新所有的值
    *head = *tail;
    HCCL_INFO("[ZeroCopyAddressMgr][ProcessRingBuffer] ringbuffer head[%u] tail[%u]", *head, *tail);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::ProcessOneAddrMap(const ZeroCopyRingBufferItem &item)
{
    HCCL_INFO("[ZeroCopyAddressMgr][ProcessOneAddrMap] Item info: type[%d] dev[%u] local[0x%lx] remote[0x%lx] len[%lu]",
        item.type, item.addr.devicePhyId, item.addr.localIpcAddr, item.addr.remoteAddr, item.addr.length);
    switch (item.type) {
        case ZeroCopyItemType::SET_MEMORY:
            return AddLocalIpc2RemoteAddr(item.addr.devicePhyId, reinterpret_cast<void *>(item.addr.localIpcAddr),
                reinterpret_cast<void *>(item.addr.remoteAddr), item.addr.length);
        case ZeroCopyItemType::UNSET_MEMORY:
            return DelLocalIpc2RemoteAddr(item.addr.devicePhyId, reinterpret_cast<void *>(item.addr.remoteAddr));
        case ZeroCopyItemType::ACTIVATE_MEMORY:
            return ActivateCommMemoryAddr(reinterpret_cast<void *>(item.addr.localIpcAddr), item.addr.length);
        case ZeroCopyItemType::DEACTIVATE_MEMORY:
            return DeactivateCommMemoryAddr(reinterpret_cast<void *>(item.addr.localIpcAddr));
        default:
            HCCL_ERROR("[ZeroCopyAddressMgr][ProcessOneAddrMap] invalid type[%d]", item.type);
            return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

u32 ZeroCopyAddressMgr::GetCommRefCnt()
{
    return commRefCnt_;
}

HcclResult ZeroCopyAddressMgr::IncreCommRefCnt()
{
    commRefCnt_++;
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyAddressMgr::DecreCommRefCnt()
{
    if (commRefCnt_ == 0) {
        HCCL_WARNING("[ZeroCopyAddressMgr][%s]commRefCnt_ is 0, cannot decrement", __func__);
        return HCCL_SUCCESS;
    }
    commRefCnt_--;
    return HCCL_SUCCESS;
}

}