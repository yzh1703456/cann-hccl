/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_MANAGER_H
#define TRANSPORT_MANAGER_H

#include <mutex>
#include <unordered_map>
#include <atomic>
#include <fstream>
#include "base.h"
#include "hccl_socket_manager.h"
#include "dispatcher.h"
#include "mem_device_pub.h"
#include "transport_pub.h"
#include "ccl_buffer_manager.h"
#include "externalinput_pub.h"
#include "sal_pub.h"
#include "thread/threads_guard.h"
#include "hccl_hash_utils.h"
#include "workflow_pub.h"
#include "comm_base_pub.h"
#include "coll_alg_param.h"

namespace hccl {

constexpr u32 AICPU_RETRY_BACKUP_PORT = 16667;     // aicpu重执行备份默认端口
constexpr u32 MULTI_QP_CONFIG_SUB_STRING_NUM = 2; // 配置信息格式为"sip,dip=sport0,sport1,...", 因此会被=分为两个子串
constexpr u32 MULTI_QP_CONFIG_IP_NUM = 2; // 有两个ip，分别为源ip和目的ip
constexpr u32 MULTI_QP_CONFIG_IP_PAIR_SHIFT_NUM = 32;
constexpr u32 MULTI_QP_CONFIG_FILE_LINE_MAX = 128 * 1024; // 配置文件最多只能配置128k行有效内容
constexpr u32 MULTI_QP_CONFIG_SRC_PORT_NUM_MAX = 32; // 一对ip对最多配置32个源端口号
constexpr u32 MULTI_QP_CONFIG_SRC_PORT_ID_MAX = 65535;

struct TransportData {
    LinkMode linkMode{LinkMode::LINK_RESERVED_MODE};
    std::vector<HcclIpAddress> remoteIpAddr;
    u32 remoteUserrank{INVALID_VALUE_RANKID};
    u32 remoteWorldRank{INVALID_VALUE_RANKID};
    s32 remoteDeviceId{-1};
    DevType deviceType{DevType::DEV_TYPE_COUNT};
    DeviceMem inputMem{DeviceMem()};
    DeviceMem outputMem{DeviceMem()};
    bool supportDataReceivedAck{false};
    u32 remoteSocketPort;

    TransportData(LinkMode linkMode,
            const std::vector<HcclIpAddress> &remoteIpAddr,
            u32 remoteUserrank,
            u32 remoteWorldRank,
            s32 remoteDeviceId,
            DevType deviceType,
            const DeviceMem &inputMem,
            const DeviceMem &outputMem,
            bool supportDataReceivedAck,
            u32 remoteSocketPort)
        : linkMode(linkMode),
        remoteIpAddr(remoteIpAddr),
        remoteUserrank(remoteUserrank),
        remoteWorldRank(remoteWorldRank),
        remoteDeviceId(remoteDeviceId),
        deviceType(deviceType),
        inputMem(inputMem),
        outputMem(outputMem),
        supportDataReceivedAck(supportDataReceivedAck),
        remoteSocketPort(remoteSocketPort) {};

    bool operator==(const TransportData &that) const
    {
        return (linkMode == that.linkMode) &&
            (remoteIpAddr == that.remoteIpAddr) &&
            (remoteUserrank == that.remoteUserrank) &&
            (remoteWorldRank == that.remoteWorldRank) &&
            (remoteDeviceId == that.remoteDeviceId) &&
            (deviceType == that.deviceType) &&
            (inputMem == that.inputMem) &&
            (outputMem == that.outputMem) &&
            (supportDataReceivedAck == that.supportDataReceivedAck) &&
            (remoteSocketPort == that.remoteSocketPort);
    }
};

struct SubCommLinkPara {
    struct SingleSubCommTransport &singleSubCommTransport;
    std::vector<std::pair<u32, u32>> remoteRankMap;
    u32 remoteRankIdStartIndex;
    u32 remoteRankIdNum;
    std::vector<std::unique_ptr<std::thread>> linkThreads;

    SubCommLinkPara(struct SingleSubCommTransport &singleSubCommTransport,
        std::vector<std::pair<u32, u32>> &remoteRankMap,
        u32 remoteRankIdStartIndex,
        u32 remoteRankIdNum)
    : singleSubCommTransport(singleSubCommTransport),
    remoteRankMap(remoteRankMap),
    remoteRankIdStartIndex(remoteRankIdStartIndex),
    remoteRankIdNum(remoteRankIdNum) {}
};
}

namespace std {

template <> class hash<hccl::TransportData> {
public:
    size_t operator()(const hccl::TransportData &transportData) const
    {
        auto linkMode = hash<s32>{}(static_cast<s32>(transportData.linkMode));
        auto remoteIpAddrFamily = hash<s32>{}(transportData.remoteIpAddr[0].GetFamily());
        auto remoteIpAddr = hash<string>{}(string(transportData.remoteIpAddr[0].GetReadableAddress()));
        auto remoteUserrank = hash<u32>{}(transportData.remoteUserrank);
        auto remoteWorldRank = hash<u32>{}(transportData.remoteWorldRank);
        auto remoteDeviceId = hash<s32>{}(transportData.remoteDeviceId);
        auto deviceType = hash<s32>{}(static_cast<s32>(transportData.deviceType));
        auto inputMemPtr = hash<u64>{}(reinterpret_cast<u64>(transportData.inputMem.ptr()));
        auto inputMemSize = hash<u64>{}(transportData.inputMem.size());
        auto outputMemPtr = hash<u64>{}(reinterpret_cast<u64>(transportData.outputMem.ptr()));
        auto outputMemSize = hash<u64>{}(transportData.outputMem.size());
        auto supportDataReceivedAck = hash<bool>{}(transportData.supportDataReceivedAck);
        auto remoteSocketPort = hash<u32>{}(transportData.remoteSocketPort);

        return hccl::HashCombine({linkMode, remoteIpAddrFamily, remoteIpAddr, remoteUserrank, remoteWorldRank,
            remoteDeviceId, deviceType, inputMemPtr, inputMemSize, outputMemPtr, outputMemSize,
            supportDataReceivedAck, remoteSocketPort});
    }
};
}  // namespace std

namespace hccl {

struct TransportIOMem {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
    DeviceMem expMem;
};

class TransportManager {
public:
    TransportManager(CCLBufferManager &cclBufferManager,
        const std::unique_ptr<HcclSocketManager> &socketManager_,
        const HcclDispatcher &dispatcher,
        const std::unique_ptr<NotifyPool> &notifyPool,
        const std::vector<RankInfo> &rankInfoList,
        RankId userRank,
        const std::string &identifier,
        s32 deviceLogicId,
        NICDeployment nicDeployment,
        bool isHaveCpuRank,
        const void *transportResourceInfoAddr,
        size_t transportResourceInfoSize,
        bool isUseRankPort,
        bool isUsedRdmaLevel0,
        const std::vector<u32> &nicRanksPort,
        const std::vector<u32> &vnicRanksPort,
        bool useSuperPodMode,
        const std::vector<HcclIpAddress> &devIpAddr,
        const HcclIpAddress &hostIp,
        const HcclIpAddress &localVnicIp,
        std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap);

    ~TransportManager();

    HcclResult CreateVirturalTransport(SingleSubCommTransport& singleSubCommTransport);
    HcclResult Alloc(const std::string &tag, const TransportIOMem &transMem, OpCommTransport &opTransportResponse,
        bool isAicpuModeEn, bool isBackup = false);
    HcclResult IncreAlloc(const std::string &tag, const TransportIOMem &transMem, OpCommTransport &opTransportReq,
        OpCommTransport &opTransportResponse, bool isAicpuModeEn, bool isBackup = false);
    HcclResult GetRemoteRankList(OpCommTransport &opTransportResponse, std::vector<u32> &rankList,
        TransportType transportType);
    HcclResult GetIncreRemoteRankList(OpCommTransport &opTransportReq,
        OpCommTransport &opTransportResponse, std::vector<u32> &rankList, TransportType transportType);
    void AddremoteUserRankToList(TransportRequest &transportRequest, std::vector<u32> &rankList,
        TransportType transportType);
    TransportManager(TransportManager const&) = delete;                 // Copy construct
    TransportManager(TransportManager&&) = delete;                      // Move construct
    TransportManager& operator=(TransportManager const&) = delete;      // Copy assign
    TransportManager& operator=(TransportManager &&) = delete;          // Move assign
    void SetQpQosAttr(u32 trafficClass, u32 serviceLevel); // 设置TC/SL配置

    HcclResult SetStopFlag(bool value);
    bool GetStopFlag();

    void SetPortConfig(bool devPortSwitchOn);
private:
    HcclResult GetIOMem(const TransportIOMem &transMem,
        const TransportMemType inputMemType, const TransportMemType outputMemType,
        DeviceMem &inputMem,  DeviceMem &outputMem, DeviceMem &expMem);
    u32 GetHostPort(s32 devicePhyId);
    u32 GetRemoteNicPort(s32 devicePhyId, u32 dstUserRank, bool isInterRdma);
    bool IsSupportInterHccs(const u32 dstRank);
    void UpdateIsInterRdma(const u32 remoteRank, bool &isInterRdma, bool forceRdma);
    HcclResult MakeRemoteLinkInfo(const u32 remoteRank, bool isInterRdma,
        u32 socketsPerLink, HcclRankLinkInfo &remoteLinkInfo);
    HcclResult CreateDestSockets(const std::string &newTag, RankId remoteRank, u64 taskNum,
        std::vector<std::shared_ptr<HcclSocket> > &connectSockets, bool &isInterRdma, bool forceRdma = false, bool isBackup = false,
        u32 subCommIndex = 0);
    u32 GetSocketsPerLink(u64 taskNum);
    HcclResult SetMachinePara(const std::string &tag, MachineType machineType, const std::string &serverId, u32 dstRank,
        const bool supportDataReceivedAck, const LinkMode linkMode,
        const std::vector<std::shared_ptr<HcclSocket> > &socketList, const DeviceMem &inputMem,
        const DeviceMem &outputMem, const DeviceMem &expMem, bool isAicpuModeEn, bool isBackup,
        u32 notifyNum, u32 trafficClass, u32 serviceLevel, MachinePara &machinePara);
    TransportType GetTransportType(const u32 dstRank, bool isUsedRdma);
    void SetTransportParam(TransportPara &para, MachinePara &machinePara);
    HcclResult TransportInit(const u32 dstRank, MachinePara &machinePara,
        std::shared_ptr<Transport> &link, bool useOneDoorbell, bool isUsedRdma);
    HcclResult CreateLink(const std::string &tag, const ErrContextPub &error_context, const MachineType machineType,
        const std::string &serverId, const u32 remoteRank, const bool supportDataReceivedAck, const LinkMode linkMode,
        const bool enableUseOneDoorbell, const std::string threadStr,
        const std::vector<std::shared_ptr<HcclSocket> > sockets, const DeviceMem inputMem, const DeviceMem outputMem,
        bool isUsedRdma, std::shared_ptr<Transport> &link, bool isAicpuModeEn,
        u32 notifyNum = 0, bool isBackup = false, const DeviceMem expMem = DeviceMem());
    HcclResult ConstructTransTag(const std::string& tag, std::string& transTag, bool isInterRdma, u32 subCommIndex = 0);
    HcclResult ExceptionHandle(const std::string &tag, OpCommTransport &opTransportResponse);

    HcclResult LoadMultiQpSrcPortFromFile();
    HcclResult GetConfigSrcPorts(MachinePara &machinePara);
    HcclResult createSubCommLinkThreads(const std::string &tag, const TransportIOMem &transMem,
        struct SubCommLinkPara &subCommLinkPara, bool isAicpuModeEn, bool isBackup, u32 subCommIndex);
    HcclResult waitSubCommLinkThreadsComplete(struct SubCommLinkPara &subCommLinkPara);
    HcclResult checkSubCommLinkThreadsStatus(const std::string &tag, struct SubCommLinkPara &subCommLinkPara, bool isBackup);
    HcclResult AllocSubCommLinks(const std::string &tag, const TransportIOMem &transMem,
        struct SingleSubCommTransport &singleSubCommTransport, bool isAicpuModeEn, bool isBackup, u32 subCommIndex);

    std::mutex mutex_;	// 用于控制互斥资源的访问
    CCLBufferManager &cclBufferManager_;
    const std::unique_ptr<HcclSocketManager> &socketManager_;
    const HcclDispatcher &dispatcher_;
    const std::unique_ptr<NotifyPool> &notifyPool_;
    const std::vector<RankInfo> &rankInfoList_;
    RankId userRank_;
    std::string identifier_;
    s32 deviceLogicId_;
    NICDeployment nicDeployment_;
    bool isHaveCpuRank_{ false };
    const void *transportResourceInfoAddr_;
    size_t transportResourceInfoSize_;
    bool isUseRankPort_{ false };
    bool isUsedRdmaLevel0_{ false };
    const std::vector<u32> &nicRanksPort_;
    const std::vector<u32> &vnicRanksPort_;
    bool useSuperPodMode_{ false };
    const std::vector<HcclIpAddress> &devIpAddr_;
    const HcclIpAddress &hostIp_;
    const HcclIpAddress &localVnicIp_;
    std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap_;
    bool devPortSwitchOn_{ false };

    std::unordered_map<TransportData, LINK> transportMap_;
    std::vector<u32> enableP2PDevices_;

    std::vector<std::string> socketTagVec_;
    std::vector<DeviceMem> extraMem_;

    std::atomic<bool> stopFlag_{false};
    HcclWorkflowMode workflowMode_{HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE};
    std::unordered_map<std::string, std::vector<u32>> mapIpPairSrcPorts_;
    bool isCfgFileRead_{ false };
    std::mutex loadCfgFileMutex_; // 控制文件资源的访问
    u64 rankConsistentDataLength_ = 0;
    u32 trafficClass_;
    u32 serviceLevel_;
};
}  // namespace hccl


#endif /* TRANSPORT_MANAGER_H */
