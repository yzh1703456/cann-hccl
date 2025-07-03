/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ZERO_COPY_MEMORY_AGENT_H
#define ZERO_COPY_MEMORY_AGENT_H

#include <atomic>
#include <thread>
#include <unordered_map>
#include "topoinfo_struct.h"
#include "hccl_socket_manager.h"
#include "common.h"
#include "coll_alg_param.h"
#include "zero_copy_address_mgr.h"

namespace hccl {

enum class RequestType {
    SET_MEMORY_RANGE = 0,
    SET_MEMORY_RANGE_ACK,
    UNSET_MEMORY_RANGE,
    UNSET_MEMORY_RANGE_ACK,
    ACTIVATE_COMM_MEMORY,
    ACTIVATE_COMM_MEMORY_ACK,
    DEACTIVATE_COMM_MEMORY,
    DEACTIVATE_COMM_MEMORY_ACK,
    SET_REMOTE_BARE_TGID,
    SET_REMOTE_BARE_TGID_ACK,
    BARRIER_CLOSE,
    BARRIER_CLOSE_ACK,
    RESERVED
};

const std::map<RequestType, std::string> REQUEST_TYPE_STR {
    {RequestType::SET_MEMORY_RANGE, "SET_MEMORY_RANGE"},
    {RequestType::SET_MEMORY_RANGE_ACK, "SET_MEMORY_RANGE_ACK"},
    {RequestType::UNSET_MEMORY_RANGE, "UNSET_MEMORY_RANGE"},
    {RequestType::UNSET_MEMORY_RANGE_ACK, "UNSET_MEMORY_RANGE_ACK"},
    {RequestType::ACTIVATE_COMM_MEMORY, "ACTIVATE_COMM_MEMORY"},
    {RequestType::ACTIVATE_COMM_MEMORY_ACK, "ACTIVATE_COMM_MEMORY_ACK"},
    {RequestType::DEACTIVATE_COMM_MEMORY, "DEACTIVATE_COMM_MEMORY"},
    {RequestType::DEACTIVATE_COMM_MEMORY_ACK, "DEACTIVATE_COMM_MEMORY_ACK"},
    {RequestType::SET_REMOTE_BARE_TGID, "SET_REMOTE_BARE_TGID"},
    {RequestType::SET_REMOTE_BARE_TGID_ACK, "SET_REMOTE_BARE_TGID_ACK"},
    {RequestType::BARRIER_CLOSE, "BARRIER_CLOSE"},
    {RequestType::BARRIER_CLOSE_ACK, "BARRIER_CLOSE_ACK"},
    {RequestType::RESERVED, "RESERVED"}
};

inline const char *GetReadableRequstType(RequestType type) {
    auto it = REQUEST_TYPE_STR.find(type);
    return (it != REQUEST_TYPE_STR.end()) ? it->second.c_str() : "unkown type";
}

class ZeroCopyMemoryAgent {
public:
    ZeroCopyMemoryAgent(const std::unique_ptr<HcclSocketManager> &socketManager, u32 devicePhyId,
        s32 deviceLogicId, const HcclIpAddress &localVnicIp, const std::vector<RankInfo> &rankInfoList, RankId userRank,
        bool useSuperPodMode, const std::string &identifier);
    virtual ~ZeroCopyMemoryAgent() = default;

    HcclResult Init();
    HcclResult DeInit();

    HcclResult SetMemoryRange(void *virPtr, size_t size, size_t alignment, uint64_t flags);
    HcclResult UnsetMemoryRange(void *virPtr);

    HcclResult ActivateCommMemory(void *virPtr, size_t size, size_t offset, void* memHandle, uint64_t flags);
    HcclResult DeactivateCommMemory(void *virPtr);

    HcclResult BarrierClose();

    static bool IsActivateCommMemoryAddr(void *virPtr, u64 length);
    static HcclResult GetRingBufferAddr(u64 &bufferPtr, u64 &headPtr, u64 &tailPtr);
    static bool IsAddressMgrInited();

private:
    // member functions
    std::string GenerateSocketTag(u32 localDevPhyId, u32 remoteDevPhyId);
    HcclResult BatchSend(const std::string &prefix, u8 *data, u64 length);
    // main thread functions
    HcclResult SetRemoteTgid();
    HcclResult EstablishSockets();
    HcclResult InitRecvThread();
    HcclResult WaitForAllRemoteComplete(RequestType requestType);

    // sub thread functions
    void DealWithIpcMemoryRequest();
    HcclResult ParseReceivedRequest(std::vector<u8>& receivedData, u32 remoteRank);
    HcclResult ParseSetMemoryRange(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseUnsetMemoryRange(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseBareTgid(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseBareTgidAck(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseActivateCommMemory(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseDeactivateCommMemory(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseSetMemoryRangeAck(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseBarrierClose(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult ParseBarrierCloseAck(u8* &exchangeDataPtr, u32 &exchangeDataBlankSize);
    HcclResult SendAckAfterParse(RequestType requestType, RequestType ackType, u32 remoteDevicePhyId, void *extraData = nullptr, u64 extraDataLen = 0);
    HcclResult ParseRemoteAck(RequestType requestType, u32 remoteRank);

    // 生成维测日志
    std::string DumpFinishInfo(RequestType requestType);

    // 是否操作需要barrier
    bool NeedBarrier(RequestType type)
    {
        return type == RequestType::BARRIER_CLOSE || type == RequestType::BARRIER_CLOSE_ACK;
    }

    // member variables
    bool initiated_;
    bool isSingleRank_{false};
    HcclNetDevCtx vnicPortCtx_{nullptr};
    const std::unique_ptr<HcclSocketManager> &socketManager_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    HcclIpAddress localVnicIp_;
    const std::vector<RankInfo> &rankInfoList_;
    RankId userRank_;
    u32 rankSize_;
    bool useSuperPodMode_;
    std::string identifier_{};
    std::vector<s32> remotePids_;

    std::unique_ptr<std::thread> recvThread_;
    std::mutex socketMutex_;
    std::mutex commRefCntLock_;
    std::unordered_map<u32, std::shared_ptr<HcclSocket> > mapDevPhyIdconnectedSockets_;
    std::unordered_set<u32> receivedBarrierCloseAck_;
    std::unordered_set<u32> receivedBarrierClose_{};
    std::unordered_map<u32, u32> mapDevPhyId2RankId_;   // 维测信息使用
    std::vector<u8> exchangeDataForSend_;
    std::vector<u8> exchangeDataForAck_;
    std::atomic<bool> threadRun_{false};
    std::unordered_map<u32, std::vector<u8>> mapDevPhyIdReceivedData_;
    std::unordered_map<u32, u64> mapDevPhyIdReceivedLength_;
    std::atomic<u32> reqMsgCounter_[static_cast<int>(RequestType::RESERVED)]{};
    std::mutex dfxMutex_;
    std::set<u32> reqMsgFinishedRanks_[static_cast<int>(RequestType::RESERVED)]{}; // 维测信息使用
    std::atomic<u32> reqMsgDeliverCnt_{};
    std::atomic<u32> reqMsgFinishCnt_{};

    static std::unique_ptr<ZeroCopyAddressMgr> addressMgr_;
};
}

#endif // ZERO_COPY_MEMORY_AGENT_H