/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zero_copy_memory_agent.h"
#include <string>
#include "runtime/dev.h"
#include "runtime/mem.h"
#include "hccl_network_pub.h"
#include "adapter_hccp_common.h"
#include "adapter_rts_common.h"

namespace hccl {
using namespace std;

const string STR_IPC_MEM_EXCHANGE = "IpcMemExchange";
constexpr u32 IPC_MEMORY_EXCHANGE_LENGTH = 64;  // Bytes
constexpr u32 USLEEP_ONE_THOUSAND = 1000;

std::unique_ptr<ZeroCopyAddressMgr> ZeroCopyMemoryAgent::addressMgr_ = nullptr;

template <typename T>
HcclResult ConstructData(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize, T& value)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, &value, sizeof(T)));
    exchangeDataPtr += sizeof(T);
    exchangeDataBlankSize -= sizeof(T);
    return HCCL_SUCCESS;
}

/* copy 变长数据 */
HcclResult ConstructData(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize, void *ptr, size_t len)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(exchangeDataPtr, exchangeDataBlankSize, ptr, len));
    exchangeDataPtr += len;
    exchangeDataBlankSize -= len;
    return HCCL_SUCCESS;
}


template <typename T>
HcclResult ParseData(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize, T& value)
{
    CHK_PRT_RET(exchangeDataBlankSize < sizeof(T),
        HCCL_ERROR("[ParseData] blankSize is [%u] less than [%lu]", exchangeDataBlankSize, sizeof(T)), HCCL_E_INTERNAL);

    CHK_SAFETY_FUNC_RET(memcpy_s(&value, sizeof(T), exchangeDataPtr, sizeof(T)));
    exchangeDataPtr += sizeof(T);
    exchangeDataBlankSize -= sizeof(T);
    return HCCL_SUCCESS;
}

ZeroCopyMemoryAgent::ZeroCopyMemoryAgent(const std::unique_ptr<HcclSocketManager> &socketManager, u32 devicePhyId,
    s32 deviceLogicId, const HcclIpAddress &localVnicIp, const std::vector<RankInfo> &rankInfoList, RankId userRank,
    bool useSuperPodMode, const std::string &identifier)
    : initiated_(false), socketManager_(socketManager), devicePhyId_(devicePhyId), deviceLogicId_(deviceLogicId),
      localVnicIp_(localVnicIp), rankInfoList_(rankInfoList), userRank_(userRank), rankSize_(rankInfoList.size()),
      useSuperPodMode_(useSuperPodMode), identifier_(identifier)
{}

// 创建vnic socket连接，启动recv 接收线程
// 每个rank 都启动listen，并且都和对端connect
HcclResult ZeroCopyMemoryAgent::Init()
{
    isSingleRank_ = (rankInfoList_.size() == 1);
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][Init] single rank communicator"), HCCL_SUCCESS);
    std::unique_lock<std::mutex> lock(commRefCntLock_);

    CHK_RET(EstablishSockets());

    CHK_RET(InitRecvThread());
    if (!ZeroCopyMemoryAgent::IsAddressMgrInited()) {
        addressMgr_ = std::make_unique<ZeroCopyAddressMgr>();
        HCCL_RUN_INFO("[ZeroCopyMemoryAgent][%s]init addressMgr_ success.", __func__);
    }
    CHK_RET(addressMgr_->IncreCommRefCnt());
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::InitRecvThread()
{
    threadRun_ = true;
    recvThread_.reset(new (std::nothrow) std::thread(&ZeroCopyMemoryAgent::DealWithIpcMemoryRequest, this));
    CHK_SMART_PTR_NULL(recvThread_);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::EstablishSockets()
{
    CHK_PRT_RET((vnicPortCtx_ != nullptr),
        HCCL_ERROR("[ZeroCopyMemoryAgent][Init] already initd"), HCCL_E_PARA);
    CHK_RET(HcclNetOpenDev(&vnicPortCtx_, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
    CHK_PTR_NULL(vnicPortCtx_);
    CHK_RET(socketManager_->ServerInit(vnicPortCtx_, HETEROG_CCL_PORT));

    for (size_t i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].devicePhyId == devicePhyId_) {
            continue;
        }
        HcclRankLinkInfo remoteLinkInfo;
        RankInfo dstRankInfo = rankInfoList_[i];
        remoteLinkInfo.userRank = dstRankInfo.userRank;
        remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
        remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
        if (useSuperPodMode_) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(rankInfoList_[userRank_].devicePhyId,
                DeviceIdType::DEVICE_ID_TYPE_SDID, dstRankInfo.superDeviceId, remoteLinkInfo.ip));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(rankInfoList_[userRank_].devicePhyId,
                DeviceIdType::DEVICE_ID_TYPE_PHY_ID, dstRankInfo.devicePhyId, remoteLinkInfo.ip));
        }
        remoteLinkInfo.port = HETEROG_CCL_PORT;
        remoteLinkInfo.socketsPerLink = 1;
        string newTag = GenerateSocketTag(devicePhyId_, rankInfoList_[i].devicePhyId);
        std::vector<std::shared_ptr<HcclSocket> > tmpSockets;
        HcclResult ret = socketManager_->CreateSingleLinkSocket(
            newTag, vnicPortCtx_, remoteLinkInfo, tmpSockets, false, true);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Create][DestSockets]Create single link sockets failed, "
            "local rank[%u], remote rank[%u]", userRank_, i), ret);
        if (tmpSockets.size() != 1) {
            HCCL_ERROR("[ZeroCopyMemoryAgent][CreateVnic] socket number[%llu] is not 1 as expected!");
            return HCCL_E_INTERNAL;
        }
        // 设置强制断链为关闭，避免进程退出时recv失败
        tmpSockets[0]->SetForceClose(false);
        mapDevPhyIdconnectedSockets_[remoteLinkInfo.devicePhyId] = (tmpSockets[0]);
        mapDevPhyId2RankId_[remoteLinkInfo.devicePhyId] = remoteLinkInfo.userRank;
    }

    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        CHK_PRT_RET(socketManager_->WaitLinkEstablish(kv.second) != HCCL_SUCCESS,
            HCCL_ERROR("[ZeroCopyMemoryAgent][EstablishSockets] tag[%s] socket establish failed", kv.second->GetTag().c_str()),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

std::string ZeroCopyMemoryAgent::GenerateSocketTag(u32 localRank, u32 remoteRank)
{
    u32 small = localRank;
    u32 large = remoteRank;

    if (localRank > remoteRank) {
        small = remoteRank;
        large = localRank;
    }

    // Socket构造规则：前缀 + identifier + small + large
    std::string tag = STR_IPC_MEM_EXCHANGE + "_" + identifier_ 
        + "_" + std::to_string(small) + ":" + std::to_string(large);
    return tag;
}

HcclResult ZeroCopyMemoryAgent::BatchSend(const std::string &prefix, u8 *data, u64 length)
{
    HCCL_INFO("[ZeroCopyMemoryAgent][BatchSend][%s]", prefix.c_str());
    std::unique_lock<std::mutex> lock(socketMutex_);
    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        CHK_PRT_RET(kv.second->Send(data, length) != HCCL_SUCCESS,
            HCCL_ERROR("[ZeroCopyMemoryAgent][%s] send to rank[%u] failed", prefix.c_str(), mapDevPhyId2RankId_[kv.first]),
            HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::SetRemoteTgid()
{
    if (remotePids_.size() == mapDevPhyIdconnectedSockets_.size()) {
        HCCL_INFO("[ZeroCopyMemoryAgent][SetRemoteTgid] tgid exchange is ok");
        return HCCL_SUCCESS;
    }
    remotePids_.clear();

    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::SET_REMOTE_BARE_TGID;

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    CHK_RET(BatchSend(__func__, exchangeDataForSend_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

    CHK_RET(WaitForAllRemoteComplete(RequestType::SET_REMOTE_BARE_TGID_ACK));
    if (remotePids_.size() != mapDevPhyIdconnectedSockets_.size()) {
        HCCL_ERROR("[ZeroCopyMemoryAgent][SetRemoteTgid] tgid exchange failed recv pids count[%lu]", remotePids_.size());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::DeInit()
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][DeInit] single rank communicator"), HCCL_SUCCESS);
    std::unique_lock<std::mutex> lock(commRefCntLock_);
    if (!ZeroCopyMemoryAgent::IsAddressMgrInited()) {
        HCCL_ERROR("[ZeroCopyMemoryAgent][%s]addressMgr_ is nullptr, no need to deinit. local rank[u32]", __func__,
            userRank_);
        return HCCL_E_INTERNAL;
    }
    threadRun_ = false;
    if (recvThread_) {
        if (recvThread_->joinable()) {
            recvThread_->join();  // 等待线程执行后释放资源
        }
    }
    recvThread_ = nullptr;

    if (vnicPortCtx_ != nullptr) {
        socketManager_->ServerDeInit(vnicPortCtx_, HETEROG_CCL_PORT);
        HcclNetCloseDev(vnicPortCtx_);
        vnicPortCtx_ = nullptr;
    }
    CHK_RET(addressMgr_->DecreCommRefCnt());
    if (addressMgr_->GetCommRefCnt() == 0) {
        addressMgr_.reset();
        HCCL_RUN_INFO("[ZeroCopyMemoryAgent][%s]Release addressMgr_", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::SetMemoryRange(void *virPtr, size_t size, size_t alignment, uint64_t flags)
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][SetMemoryRange] single rank communicator"), HCCL_SUCCESS);
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(addressMgr_->SetMemoryRange(devicePhyId_, virPtr, size) != HCCL_SUCCESS,
        HCCL_ERROR("[ZeroCopyMemoryAgent][SetMemoryRange] invalid set ptr[%p] size[%lu] alignment[%lu] flags[%lu]",
        virPtr, size, alignment, flags), HCCL_E_PARA);

    HCCL_INFO("[ZeroCopyMemoryAgent][SetMemoryRange] basePtr[%p] size[%lu] aligment[%lu] flag[%lu]",
        virPtr, size, alignment, flags);
    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::SET_MEMORY_RANGE;

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, size));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, alignment));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, flags));

    CHK_RET(BatchSend(__func__, exchangeDataForSend_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

    CHK_RET(WaitForAllRemoteComplete(RequestType::SET_MEMORY_RANGE_ACK));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::UnsetMemoryRange(void *virPtr)
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][UnsetMemoryRange] single rank communicator"), HCCL_SUCCESS);
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(!addressMgr_->IsAddressSet(devicePhyId_, virPtr),
        HCCL_ERROR("[ZeroCopyMemoryAgent][UnsetMemoryRange] ptr[%p] is not set memory", virPtr), HCCL_E_PARA);
    CHK_RET(addressMgr_->UnsetMemoryRange(devicePhyId_, virPtr));

    HCCL_INFO("[ZeroCopyMemoryAgent][UnsetMemoryRange] basePtr[%p]", virPtr);
    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::UNSET_MEMORY_RANGE;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(BatchSend(__func__, exchangeDataForSend_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

    CHK_RET(WaitForAllRemoteComplete(RequestType::UNSET_MEMORY_RANGE_ACK));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ActivateCommMemory(void *virPtr, size_t size, size_t offset, void *memHandle, uint64_t flags)
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][ActivateCommMemory] single rank communicator"), HCCL_SUCCESS);
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(!addressMgr_->IsInSetAddressRange(devicePhyId_, virPtr, size),
        HCCL_ERROR("[ZeroCopyMemoryAgent][ActivateCommMemory] input ptr[%p] size[%lu] is not in set address range", virPtr, size), HCCL_E_PARA);
    CHK_PRT_RET(addressMgr_->IsOverlapWithActivateAddr(virPtr, size),
        HCCL_ERROR("[ZeroCopyMemoryAgent][ActivateCommMemory] input ptr[%p] size[%lu] overlap with activate memory", virPtr, size), HCCL_E_PARA);

    HCCL_INFO("[ZeroCopyMemoryAgent][ActivateCommMemory] virPtr[%p] size[%lu] offset[%lu] memHandle[%p], flags[%lu]",
        virPtr, size, offset, memHandle, flags);
    CHK_RET(SetRemoteTgid());

    uint64_t shareableHandle;
    rtDrvMemHandleType handleType = RT_MEM_HANDLE_TYPE_NONE;
    rtError_t ret = RT_ERROR_NONE;
    ret = rtMemExportToShareableHandle(memHandle, handleType, 0, &shareableHandle);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ActivateCommMemory] rtMemExportToShareableHandle handle[%p]",
        " type[%d] flags[%lu] failed, ret[%d]", memHandle, handleType, 0, ret), HCCL_E_RUNTIME);
    ret = rtMemSetPidToShareableHandle(shareableHandle, remotePids_.data(), remotePids_.size());
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ActivateCommMemory] rtMemSetPidToShareableHandle shareableHandl[%lu]",
        " failed, ret[%d]", shareableHandle, ret), HCCL_E_RUNTIME);

    HCCL_INFO("[ZeroCopyMemoryAgent][ActivateCommMemory] dev[%u] export shareableHandle[%lu]", devicePhyId_, shareableHandle);
    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::ACTIVATE_COMM_MEMORY;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, size));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, offset));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, shareableHandle));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, flags));

    CHK_RET(BatchSend(__func__, exchangeDataForSend_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

    CHK_RET(WaitForAllRemoteComplete(RequestType::ACTIVATE_COMM_MEMORY_ACK));
    CHK_RET(addressMgr_->ActivateCommMemoryAddr(virPtr, size));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::DeactivateCommMemory(void *virPtr)
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][DeactivateCommMemory] single rank communicator"), HCCL_SUCCESS);
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    CHK_PRT_RET(!addressMgr_->IsActivateCommMemoryAddr(virPtr, 1),
        HCCL_ERROR("[ZeroCopyMemoryAgent][DeactivateCommMemory] input ptr[%p] is not activate", virPtr), HCCL_E_PARA);

    HCCL_INFO("[ZeroCopyMemoryAgent][DeactivateCommMemory] virPtr[%p]", virPtr);
    CHK_RET(addressMgr_->DeactivateCommMemoryAddr(virPtr));

    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::DEACTIVATE_COMM_MEMORY;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    u64 addr = reinterpret_cast<u64>(virPtr);
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, addr));

    CHK_RET(BatchSend(__func__, exchangeDataForSend_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

    CHK_RET(WaitForAllRemoteComplete(RequestType::DEACTIVATE_COMM_MEMORY_ACK));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::BarrierClose()
{
    CHK_PRT_RET(isSingleRank_, HCCL_INFO("[ZeroCopyMemoryAgent][BarrierClose] single rank communicator"), HCCL_SUCCESS);

    HCCL_RUN_INFO("[ZeroCopyMemoryAgent][BarrierClose] [%s] ready to barrier close", identifier_.c_str());
    exchangeDataForSend_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataPtr = exchangeDataForSend_.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType = RequestType::BARRIER_CLOSE;
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, requestType));
    CHK_RET(ConstructData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId_));

    CHK_RET(BatchSend(__func__, exchangeDataForSend_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

    CHK_RET(WaitForAllRemoteComplete(RequestType::BARRIER_CLOSE_ACK));

    return HCCL_SUCCESS;
}

bool ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(void *virPtr, u64 length)
{
    if (!ZeroCopyMemoryAgent::IsAddressMgrInited()) {
        HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent is not init.", __func__);
        return false;
    }
    return addressMgr_->IsActivateCommMemoryAddr(virPtr, length);
}

HcclResult ZeroCopyMemoryAgent::GetRingBufferAddr(u64 &bufferPtr, u64 &headPtr, u64 &tailPtr)
{
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    addressMgr_->GetRingBufferAddr(bufferPtr, headPtr, tailPtr);
    return HCCL_SUCCESS;
}

bool ZeroCopyMemoryAgent::IsAddressMgrInited()
{
    return addressMgr_ != nullptr;
}

HcclResult ZeroCopyMemoryAgent::WaitForAllRemoteComplete(RequestType requestType)
{
    bool useBarrier = NeedBarrier(requestType);
    if (useBarrier) {
        reqMsgDeliverCnt_++;
    }

    u32 expectedNum = mapDevPhyIdconnectedSockets_.size();
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    while (true) {
        // reqMsgCounter：表示该类型的ACK我们收到了多少个（比如Valid的ACK收到了7个）
        // reqMsgDeliver/reqMsgFinish：表示本端收完整了多少次数据（比如两次valid）
        CHK_PRT_RET(reqMsgCounter_[static_cast<int>(requestType)] > expectedNum,
            HCCL_ERROR("[ZeroCopyMemoryAgent][WaitForAllRemoteComplete] recv request[%s] ack [%u] more than expect [%u]",
            GetReadableRequstType(requestType), reqMsgCounter_[static_cast<int>(requestType)].load(), expectedNum), HCCL_E_INTERNAL);

        if (reqMsgCounter_[static_cast<int>(requestType)] == expectedNum) {
            if (!useBarrier || (useBarrier && reqMsgDeliverCnt_ <= reqMsgFinishCnt_)) {
                reqMsgCounter_[static_cast<int>(requestType)] = 0;

                std::lock_guard<std::mutex> dfxLock(dfxMutex_);
                reqMsgFinishedRanks_[static_cast<int>(requestType)].clear();
                break;
            }
        }

        bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
        CHK_PRT_RET(bTimeout, HCCL_ERROR("[Wait][RemoteComplete %s] dev[%u] errNo[0x%016llx] timeout[%d s] completeCount[%u] %s",
            GetReadableRequstType(requestType), devicePhyId_,
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), timeout, reqMsgCounter_[static_cast<int>(requestType)].load(),
            DumpFinishInfo(requestType).c_str()), HCCL_E_TCP_TRANSFER);
        SaluSleep(USLEEP_ONE_THOUSAND);
    }
    return HCCL_SUCCESS;
}

void ZeroCopyMemoryAgent::DealWithIpcMemoryRequest()
{
    // 新线程，更新一下使用的设备
    if (hrtSetDevice(deviceLogicId_) != HCCL_SUCCESS) {
        HCCL_ERROR("[ZeroCopyMemoryAgent][DealWithIpcMemoryRequest] set device failed");
        return;
    }

    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        mapDevPhyIdReceivedLength_[kv.first] = 0;
    }

    std::vector<u8> second(IPC_MEMORY_EXCHANGE_LENGTH, 0);
    for (const auto& kv : mapDevPhyIdconnectedSockets_) {
        mapDevPhyIdReceivedData_[kv.first] = second;
    }
    HcclResult ret;
    do {
        for (const auto& kv : mapDevPhyIdconnectedSockets_) {
            if (receivedBarrierClose_.count(kv.first) && receivedBarrierCloseAck_.count(kv.first)) {
                // 该socket已经收到了BarrierClose报文，因此不允许再进行其他数据接收了
                continue;
            }

            u32 receivedLength = mapDevPhyIdReceivedLength_[kv.first];
            u32 expectedLength = IPC_MEMORY_EXCHANGE_LENGTH - receivedLength;
            u64 receivingLength = 0;
            {
                std::unique_lock<std::mutex> lock(socketMutex_);
                ret = kv.second->IRecv(mapDevPhyIdReceivedData_[kv.first].data() + receivedLength,
                    expectedLength, receivingLength);
            }
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ZeroCopyMemoryAgent][Socket][IRecv] dev[%u] failed", kv.first), ;);
            if (receivingLength != 0) {
                mapDevPhyIdReceivedLength_[kv.first] += receivingLength;
            }
            if (mapDevPhyIdReceivedLength_[kv.first] == IPC_MEMORY_EXCHANGE_LENGTH) {
                ret = ParseReceivedRequest(mapDevPhyIdReceivedData_[kv.first], mapDevPhyId2RankId_[kv.first]);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ZeroCopyMemoryAgent][Parse][ReceivedRequest] failed"), ;);
                mapDevPhyIdReceivedLength_[kv.first] = 0;
            }
        }
        SaluSleep(USLEEP_ONE_THOUSAND);
    } while (threadRun_);
    if (hrtResetDevice(deviceLogicId_) != HCCL_SUCCESS) {
        HCCL_ERROR("[ZeroCopyMemoryAgent][DealWithIpcMemoryRequest] reset device failed");
        return;
    }
}

HcclResult ZeroCopyMemoryAgent::ParseSetMemoryRange(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    size_t size;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, size));

    size_t alignment;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, alignment));

    uint64_t flags;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, flags));

    CHK_PRT_RET(devicePhyId >= MAX_MODULE_DEVICE_NUM,
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseSetMemoryRange] devicePhyId[%u] is exceed max device num[%u]", devicePhyId, MAX_MODULE_DEVICE_NUM),
        HCCL_E_PARA);

    void *remoteAddrBase = reinterpret_cast<void *>(addr);
    CHK_PRT_RET(addressMgr_->IsAddressSet(devicePhyId, remoteAddrBase),
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseSetMemoryRange] devicePhyId[%u] had set addr [%p]", devicePhyId, remoteAddrBase), HCCL_E_PARA);

    void* devPtr = nullptr;
    void* devAddr = nullptr;
    rtError_t ret = rtReserveMemAddress(&devPtr, size, alignment, devAddr, flags);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseSetMemoryRange] rtReserve Memory failed, "
        "return[%d], devPtr[%p] size[%llu] alignment[%llu] devAddr[%p] flags[%llu]",
        ret, devPtr, size, alignment, devAddr, flags), HCCL_E_RUNTIME);

    CHK_RET(addressMgr_->AddLocalIpc2RemoteAddr(devicePhyId, devPtr, reinterpret_cast<void *>(addr), size));

    CHK_RET(SendAckAfterParse(RequestType::SET_MEMORY_RANGE, RequestType::SET_MEMORY_RANGE_ACK, devicePhyId));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::SendAckAfterParse(RequestType requestType, RequestType ackType, u32 remoteDevicePhyId,
    void *extraData, u64 extraDataLen)
{
    exchangeDataForAck_.resize(IPC_MEMORY_EXCHANGE_LENGTH);
    u8 *exchangeDataAckPtr = exchangeDataForAck_.data();
    u32 exchangeDataAckBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    CHK_RET(ConstructData(exchangeDataAckPtr, exchangeDataAckBlankSize, ackType));

    CHK_RET(ConstructData(exchangeDataAckPtr, exchangeDataAckBlankSize, devicePhyId_));

    if (extraData != nullptr && extraDataLen != 0) {
        CHK_RET(ConstructData(exchangeDataAckPtr, exchangeDataAckBlankSize, extraData, extraDataLen));
    }

    // 不需要进行barrier，那么我们每处理一个请求就回复一个请求
    if (!NeedBarrier(requestType)) {
        std::unique_lock<std::mutex> lock(socketMutex_);
        CHK_PRT_RET(mapDevPhyIdconnectedSockets_.find(remoteDevicePhyId) == mapDevPhyIdconnectedSockets_.end(),
            HCCL_ERROR("[ZeroCopyMemoryAgent][SendAckAfterParse] Invalid devicePhyId [%u]", remoteDevicePhyId), HCCL_E_PARA);
        CHK_RET(mapDevPhyIdconnectedSockets_[remoteDevicePhyId]->Send(exchangeDataForAck_.data(), IPC_MEMORY_EXCHANGE_LENGTH));

        return HCCL_SUCCESS;
    }

    // 需要进行barrier的请求，我们先统计一下收到的请求数目，等于链接数才算收完所有
    u32 expectedNum = mapDevPhyIdconnectedSockets_.size();
    u32 counter = ++reqMsgCounter_[static_cast<int>(requestType)];
    HCCL_INFO("[ZeroCopyMemoryAgent][SendAckAfterParse] requestType[%d] counter %u expect %u", requestType, counter, expectedNum);
    if (counter < expectedNum) {
        return HCCL_SUCCESS;
    } else {
        reqMsgCounter_[static_cast<int>(requestType)] = 0;
        reqMsgFinishCnt_++;

        // 我们统一将所有的请求一次性都发送过去
        CHK_RET(BatchSend(__func__, exchangeDataForAck_.data(), IPC_MEMORY_EXCHANGE_LENGTH));
    }

    return HCCL_SUCCESS;
}


HcclResult ZeroCopyMemoryAgent::ParseRemoteAck(RequestType requestType, u32 remoteRank)
{
    std::lock_guard<std::mutex> dfxLock(dfxMutex_);
    reqMsgFinishedRanks_[static_cast<int>(requestType)].insert(remoteRank);

    reqMsgCounter_[static_cast<int>(requestType)]++;
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseUnsetMemoryRange(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    LocalIpc2RemoteAddr mapAddr;
    void *remoteAddr = reinterpret_cast<void *>(addr);
    CHK_PRT_RET(addressMgr_->GetLocalIpc2RemoteAddr(devicePhyId, remoteAddr, mapAddr) != HCCL_SUCCESS,
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseUnsetMemoryRange] device[%u] not set addr [%p]", devicePhyId, remoteAddr), HCCL_E_PARA);
    CHK_RET(addressMgr_->DelLocalIpc2RemoteAddr(devicePhyId, reinterpret_cast<void *>(mapAddr.remoteAddr)));

    void *devPtr = reinterpret_cast<void *>(mapAddr.localIpcAddr);
    rtError_t ret = rtReleaseMemAddress(devPtr);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseUnsetMemoryRange]rtRelease Memory failed, "\
        "return[%d], devPtr[%p]", ret, devPtr), HCCL_E_RUNTIME);

    CHK_RET(SendAckAfterParse(RequestType::UNSET_MEMORY_RANGE, RequestType::UNSET_MEMORY_RANGE_ACK, devicePhyId));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBareTgid(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    // 获取本端的ack，然后通过ack返回给对端
    u32 tgid = 0;
    rtError_t ret = rtDeviceGetBareTgid(&tgid);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseBareTgid] get tgid failed, ret[%d]", ret), HCCL_E_RUNTIME);

    HCCL_INFO("[ZeroCopyMemoryAgent][ParseBareTgid] dev[%u] tgid[%u] to remoteDev[%u]", devicePhyId_, tgid, devicePhyId);
    CHK_RET(SendAckAfterParse(RequestType::SET_REMOTE_BARE_TGID, RequestType::SET_REMOTE_BARE_TGID_ACK, devicePhyId,
        &tgid, sizeof(tgid)));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBareTgidAck(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u32 tgid;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, tgid));

    HCCL_INFO("[ZeroCopyMemoryAgent][ParseBareTgidAck] recv dev[%u] tgid[%u]", devicePhyId, tgid);
    remotePids_.emplace_back(tgid);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBarrierCloseAck(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u32 tgid;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, tgid));

    receivedBarrierCloseAck_.insert(devicePhyId);
    HCCL_RUN_INFO("[ZeroCopyMemoryAgent][ParseBarrierCloseAck] [%s] recv dev[%u] barrier close ack, so we stop this socket's recv",
        identifier_.c_str(), devicePhyId, tgid);
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseActivateCommMemory(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    size_t size;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, size));

    size_t offset;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, offset));

    size_t shareableHandle;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, shareableHandle));

    size_t flags;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, flags));

    LocalIpc2RemoteAddr mapAddr;
    void *remoteAddr = reinterpret_cast<void *>(addr);
    CHK_PRT_RET((addressMgr_->GetLocalIpc2RemoteAddr(devicePhyId, remoteAddr, mapAddr) != HCCL_SUCCESS),
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseActivateCommMemory] address may not be reseved in device[%u]", devicePhyId), HCCL_E_PARA);
    
    HCCL_INFO("[ZeroCopyMemoryAgent][ParseActivateCommMemory] prepare import from dev[%u] shareableHandle[%lu]", devicePhyId, shareableHandle);
    u64 actualAddr = mapAddr.localIpcAddr + (addr - mapAddr.remoteAddr);
    void* devPtr = reinterpret_cast<void*>(actualAddr);
    CHK_PRT_RET(actualAddr + size > mapAddr.localIpcAddr + mapAddr.length,
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseActivateCommMemory] remote addr[0x%lx] size[%lu] exceed memory range", addr, size), HCCL_E_PARA);
    CHK_PRT_RET(addressMgr_->IsOverlapWithActivateAddr(devPtr, size),
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseActivateCommMemory] remote addr[0x%lx] size[%lu] devPtr[%p] is overlap",
        addr, size, devPtr), HCCL_E_PARA);

    rtError_t ret = RT_ERROR_NONE;
    void* pHandle = nullptr;
    CHK_RET(addressMgr_->ActivateCommMemoryAddr(devPtr, size));
    ret = rtMemImportFromShareableHandle(shareableHandle, deviceLogicId_, &pHandle);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseActivateCommMemory] import shareableHandle[%lu] dev[%d] failed, ret[%d]",
        shareableHandle, deviceLogicId_, ret), HCCL_E_RUNTIME);

    ret = rtMapMem(devPtr, size, offset, pHandle, flags);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseActivateCommMemory] map dev[%p] size[%lu] offset[%lu] handle[%p]",
        " flag[%lu] failed, ret[%d]", devPtr, size, offset, pHandle, flags), HCCL_E_RUNTIME);

    CHK_RET(addressMgr_->AddRemoteImportAddr(devPtr, pHandle));

    CHK_RET(SendAckAfterParse(RequestType::ACTIVATE_COMM_MEMORY, RequestType::ACTIVATE_COMM_MEMORY_ACK, devicePhyId));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseDeactivateCommMemory(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    CHK_PRT_RET(!ZeroCopyMemoryAgent::IsAddressMgrInited(), HCCL_INFO("[ZeroCopyMemoryAgent][%s]ZeroCopyMemoryAgent "
        "is not init.", __func__), HCCL_SUCCESS);
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));

    u64 addr;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, addr));

    LocalIpc2RemoteAddr mapAddr;
    void *remoteAddr = reinterpret_cast<void *>(addr);
    CHK_PRT_RET((addressMgr_->GetLocalIpc2RemoteAddr(devicePhyId, remoteAddr, mapAddr) != HCCL_SUCCESS),
        HCCL_ERROR("[ZeroCopyMemoryAgent][ParseDeactivateCommMemory] address [%p] not be set in device[%u]",
        remoteAddr, devicePhyId), HCCL_E_PARA);

    u64 actualAddr = mapAddr.localIpcAddr + (addr - mapAddr.remoteAddr);
    void* devPtr = reinterpret_cast<void*>(actualAddr);
    CHK_RET(addressMgr_->DeactivateCommMemoryAddr(devPtr));

    void *handle = nullptr;
    CHK_RET(addressMgr_->GetRemoteImportAddr(devPtr, handle));

    rtError_t ret = RT_ERROR_NONE;
    ret = rtUnmapMem(devPtr);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseDeactivateCommMemory] rtUnmapMem dev[%p] failed, ret[%d]",
        devPtr, ret), HCCL_E_RUNTIME);
    ret = rtFreePhysical(handle);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[ZeroCopyMemoryAgent][ParseDeactivateCommMemory] rtFreePhysical handle[%lu] failed, ret[%d]",
        handle, ret), HCCL_E_RUNTIME);

    CHK_RET(addressMgr_->DelRemoteImportAddr(devPtr));

    CHK_RET(SendAckAfterParse(RequestType::DEACTIVATE_COMM_MEMORY, RequestType::DEACTIVATE_COMM_MEMORY_ACK, devicePhyId));

    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseBarrierClose(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize)
{
    u32 devicePhyId;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, devicePhyId));
    HCCL_INFO("[ZeroCopyMemoryAgent][ParseBarrierClose] recv dev[%u] barrier close", devicePhyId);

    receivedBarrierClose_.insert(devicePhyId);
    CHK_RET(SendAckAfterParse(RequestType::BARRIER_CLOSE, RequestType::BARRIER_CLOSE_ACK, devicePhyId));
    return HCCL_SUCCESS;
}

HcclResult ZeroCopyMemoryAgent::ParseReceivedRequest(std::vector<u8>& receivedData, u32 remoteRank)
{
    u8* exchangeDataPtr = receivedData.data();
    u32 exchangeDataBlankSize = IPC_MEMORY_EXCHANGE_LENGTH;

    RequestType requestType;
    CHK_RET(ParseData(exchangeDataPtr, exchangeDataBlankSize, requestType));

    HcclResult ret = HCCL_SUCCESS;
    switch (requestType) {
        case RequestType::SET_MEMORY_RANGE:
            ret = ParseSetMemoryRange(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::UNSET_MEMORY_RANGE:
            ret = ParseUnsetMemoryRange(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::ACTIVATE_COMM_MEMORY:
            ret = ParseActivateCommMemory(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::DEACTIVATE_COMM_MEMORY:
            ret = ParseDeactivateCommMemory(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::SET_REMOTE_BARE_TGID:
            ret = ParseBareTgid(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::BARRIER_CLOSE:
            ret = ParseBarrierClose(exchangeDataPtr, exchangeDataBlankSize);
            break;
        case RequestType::SET_REMOTE_BARE_TGID_ACK:
            ret = ParseBareTgidAck(exchangeDataPtr, exchangeDataBlankSize);
            ParseRemoteAck(requestType, remoteRank);
            break;
        case RequestType::SET_MEMORY_RANGE_ACK:
        case RequestType::UNSET_MEMORY_RANGE_ACK:
        case RequestType::ACTIVATE_COMM_MEMORY_ACK:
        case RequestType::DEACTIVATE_COMM_MEMORY_ACK:
            ParseRemoteAck(requestType, remoteRank);
            break;
        case RequestType::BARRIER_CLOSE_ACK:
            ret = ParseBarrierCloseAck(exchangeDataPtr, exchangeDataBlankSize);
            ParseRemoteAck(requestType, remoteRank);
            break;
        default:
            HCCL_ERROR("[Parse][ReceivedRequest] invalid RequestType[%d]", requestType);
            ret = HCCL_E_INTERNAL;
            break;
    }
    return ret;
}

std::string ZeroCopyMemoryAgent::DumpFinishInfo(RequestType requestType)
{
    std::lock_guard<std::mutex> dfxLock(dfxMutex_);
    auto &finishedRanks = reqMsgFinishedRanks_[static_cast<int>(requestType)];

    std::string msg = "Expect [";
    for (auto &info : rankInfoList_) {
        msg += std::to_string(info.userRank) + " ";
    }

    msg += "] Actual [";
    for (auto &rank : finishedRanks) {
        msg += std::to_string(rank) + " ";
    }

    msg += "]";
    finishedRanks.clear();

    return msg;
}

}  // namespace hccl