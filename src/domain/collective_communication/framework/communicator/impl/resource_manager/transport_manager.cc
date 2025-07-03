/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_manager.h"
#include "device_capacity.h"
#include "p2p_mgmt_pub.h"
#include <algorithm>
#include "rank_consistentcy_checker.h"
#include "env_config.h"

namespace hccl {

TransportManager::TransportManager(CCLBufferManager &cclBufferManager,
                                   const std::unique_ptr<HcclSocketManager> &socketManager,
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
                                   std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap)
    : cclBufferManager_(cclBufferManager), socketManager_(socketManager), dispatcher_(dispatcher),
    notifyPool_(notifyPool), rankInfoList_(rankInfoList), userRank_(userRank), identifier_(identifier),
    deviceLogicId_(deviceLogicId), nicDeployment_(nicDeployment), isHaveCpuRank_(isHaveCpuRank),
    transportResourceInfoAddr_(transportResourceInfoAddr), transportResourceInfoSize_(transportResourceInfoSize),
    isUseRankPort_(isUseRankPort), isUsedRdmaLevel0_(isUsedRdmaLevel0), nicRanksPort_(nicRanksPort),
    vnicRanksPort_(vnicRanksPort), useSuperPodMode_(useSuperPodMode), devIpAddr_(devIpAddr), hostIp_(hostIp),
    localVnicIp_(localVnicIp), netDevCtxMap_(netDevCtxMap), trafficClass_(HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET),
    serviceLevel_(HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET)
{
    rankConsistentDataLength_ = RankConsistentcyChecker::GetInstance().GetRankConsistentDataLength();
}

TransportManager::~TransportManager()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (enableP2PDevices_.size() != 0) {
        (void)P2PMgmtPub::DisableP2P(enableP2PDevices_);
        enableP2PDevices_.clear();
    }
}

constexpr u32 EXCEPTION_DELAY_US_COUNT = 100000;
HcclResult TransportManager::ExceptionHandle(const std::string &tag, OpCommTransport &opTransportResponse)
{
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    bool isInterRdma;
                    UpdateIsInterRdma(transportRequest.remoteUserRank, isInterRdma, transportRequest.isUsedRdma);

                    HcclRankLinkInfo remoteLinkInfo;
                    MakeRemoteLinkInfo(transportRequest.remoteUserRank, isInterRdma, 1, remoteLinkInfo);

                    HcclIpAddress ipAddr;
                    if (isInterRdma || Is310PDevice()) {
                        ipAddr = nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE ?
                            devIpAddr_[0]: hostIp_;
                    } else {
                        ipAddr = localVnicIp_;
                    }

                    std::string newTag;
                    CHK_RET(ConstructTransTag(tag, newTag, isInterRdma));
                    CHK_RET(socketManager_->AddWhiteList(newTag, netDevCtxMap_[ipAddr],
                        remoteLinkInfo));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::CreateVirturalTransport(SingleSubCommTransport& singleSubCommTransport)
{
    MachinePara machinePara;
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(
        GetExternalInputHcclLinkTimeOut());

    singleSubCommTransport.virtualLinks.clear();
    singleSubCommTransport.virtualLinks.resize(singleSubCommTransport.transportRequests.size());

    for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
        TransportPara para {};
        para.virtualFlag = true;
        para.timeout = kdefaultTimeout;
        para.index = i;
        singleSubCommTransport.virtualLinks[i].reset(new (std::nothrow) Transport(TransportType::TRANS_TYPE_RESERVED,
            para, dispatcher_, notifyPool_, machinePara));
        CHK_PRT_RET(!singleSubCommTransport.virtualLinks[i], HCCL_ERROR("[CreateVirturalTransport]In create link," \
            "new link failed"), HCCL_E_PTR);
    }

    return HCCL_SUCCESS;
}

void TransportManager::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
{
    trafficClass_ = trafficClass;
    serviceLevel_ = serviceLevel;
}

void TransportManager::AddremoteUserRankToList(TransportRequest &transportRequest, std::vector<u32> &rankList,
    TransportType transportType)
{
    if (!transportRequest.isValid) {
        return;
    }
    TransportType type = GetTransportType(transportRequest.remoteUserRank, transportRequest.isUsedRdma);
    if (type == transportType) {
        // 仅添加对应Type类型的对端
        rankList.emplace_back(transportRequest.remoteUserRank);
    }
    return;
}

HcclResult TransportManager::GetRemoteRankList(OpCommTransport &opTransportResponse, std::vector<u32> &rankList,
    TransportType transportType)
{
    // 对当前所有的transportLink做判断
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                AddremoteUserRankToList(transportRequest, rankList, transportType);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::createSubCommLinkThreads(const std::string &tag, const TransportIOMem &transMem,
    struct SubCommLinkPara &subCommLinkPara, bool isAicpuModeEn, bool isBackup, u32 subCommIndex)
{
    u32 num = subCommLinkPara.remoteRankIdNum;
    struct SingleSubCommTransport &singleSubCommTransport = subCommLinkPara.singleSubCommTransport;
    subCommLinkPara.linkThreads.resize(num);

    for (u32 i = 0; i < num; i++) {
        u32 index = subCommLinkPara.remoteRankMap[(subCommLinkPara.remoteRankIdStartIndex + i) % subCommLinkPara.remoteRankMap.size()].second;
        auto &transportRequest = singleSubCommTransport.transportRequests[index];
        auto &link = singleSubCommTransport.links[index];

        if ((!transportRequest.isValid) || (link != nullptr) || (isBackup && !transportRequest.isUsedRdma)) {
            HCCL_INFO("[%s]: no need to create p2p back link, remote UserRank[%u], userRank[%u], "
                "isUsedRdma[%u], isBackup[%d]", __func__, transportRequest.remoteUserRank, userRank_,
                transportRequest.isUsedRdma, isBackup);
            continue;
        }

        DeviceMem inputMem;
        DeviceMem outputMem;
        DeviceMem expMem;
        GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType,
            inputMem, outputMem, expMem);
        HCCL_DEBUG("transportRequest.inputMemType[%d] transportRequest.outputMemType[%d], isBackup[%d]",
            transportRequest.inputMemType, transportRequest.outputMemType, isBackup);

        std::vector<std::shared_ptr<HcclSocket>> connectSockets;
        bool isInterRdma;
        bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
        HcclResult ret = CreateDestSockets(tag, transportRequest.remoteUserRank, singleSubCommTransport.taskNum,
            connectSockets, isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex);
        HCCL_DEBUG("[%s]CreateDestSockets finished, chooseBackup[%d]", __func__, chooseBackup);
        HCCL_DEBUG("[%s]: remoteUserRank[%u], userRank[%u], isUsedRdma[%u]", __func__, transportRequest.remoteUserRank,
            userRank_, transportRequest.isUsedRdma);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc]Create dest sockets failed"), ret);

        MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank ?
            MachineType::MACHINE_SERVER_TYPE : MachineType::MACHINE_CLIENT_TYPE;
        std::string threadStr = (isInterRdma ? "HcclTerL_" : "HcclIntra_") + std::to_string(i);
        subCommLinkPara.linkThreads[i].reset(
            new (std::nothrow) std::thread(&TransportManager::CreateLink,
                this, tag, hrtErrMGetErrorContextPub(), 
                machineType, rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank, 
                singleSubCommTransport.supportDataReceivedAck, singleSubCommTransport.linkMode, 
                singleSubCommTransport.enableUseOneDoorbell, threadStr,
                connectSockets, inputMem, outputMem, transportRequest.isUsedRdma, 
                std::ref(link), isAicpuModeEn,
                transportRequest.notifyNum, chooseBackup, expMem));
        CHK_SMART_PTR_NULL(subCommLinkPara.linkThreads[i]); // 异常时其他线程待处理
        singleSubCommTransport.status[index] = TransportStatus::READY; // 建链后 transport设置为ready状态
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::waitSubCommLinkThreadsComplete(struct SubCommLinkPara &subCommLinkPara)
{
    for (u32 i = 0; i < subCommLinkPara.linkThreads.size(); i++) {
        if (subCommLinkPara.linkThreads[i] == nullptr || !subCommLinkPara.linkThreads[i]->joinable()) {
            continue;
        }
        subCommLinkPara.linkThreads[i]->join(); // 等待线程执行完毕
        CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
    }
    subCommLinkPara.linkThreads.clear();
    CHK_PRT_RET(GetStopFlag(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult TransportManager::checkSubCommLinkThreadsStatus(const std::string &tag, struct SubCommLinkPara &subCommLinkPara,
    bool isBackup)
{
    u32 num = subCommLinkPara.remoteRankIdNum;
    struct SingleSubCommTransport &singleSubCommTransport = subCommLinkPara.singleSubCommTransport;

    for (u32 i = 0; i < num; i++) {
        u32 index = subCommLinkPara.remoteRankMap[(subCommLinkPara.remoteRankIdStartIndex + i) % subCommLinkPara.remoteRankMap.size()].second;
        auto &transportRequest = singleSubCommTransport.transportRequests[index];
        auto &link = singleSubCommTransport.links[index];

        if (!transportRequest.isValid) {
            continue;
        }

        if (isBackup && !transportRequest.isUsedRdma) {
            // 备用链路不需要创建p2p
            HCCL_INFO("[%s]: no need to check p2p backup link, remoteUserRank[%u], userRank[%u], "
                "isUsedRdma[%u], isBackup[%d]", __func__, transportRequest.remoteUserRank, userRank_,
                transportRequest.isUsedRdma, isBackup);
            continue;
        }

        if (link == nullptr) {
            HCCL_ERROR("[Create]errNo[0x%016llx] transport create fail in thread, local rank[%d] remote rank[%d]",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), userRank_, transportRequest.remoteUserRank);
            SaluSleep(EXCEPTION_DELAY_US_COUNT);
            (void)notifyPool_->UnregisterOp(tag);
            return HCCL_E_NOT_FOUND;
        }   
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::AllocSubCommLinks(const std::string &tag, const TransportIOMem &transMem,
    struct SingleSubCommTransport &singleSubCommTransport, bool isAicpuModeEn, bool isBackup, u32 subCommIndex)
{
    const u32 offset = 8;
    std::vector<std::pair<u32, u32>> remoteRankMap;
    std::vector<std::unique_ptr<std::thread>> linkThreads;
    linkThreads.resize(singleSubCommTransport.transportRequests.size());
    ThreadsGuard threadsGuard(linkThreads);

    for (u32 i = 0; i< singleSubCommTransport.transportRequests.size(); i++) {
        if (singleSubCommTransport.transportRequests[i].isValid) {
            remoteRankMap.push_back(std::make_pair(singleSubCommTransport.transportRequests[i].remoteUserRank, i));
        }
    }
    if (remoteRankMap.empty()) {
        HCCL_INFO("[%s] is empty", __func__);
        return HCCL_SUCCESS;
    }

    if (singleSubCommTransport.needVirtualLink) {
        // task多线程并行下发，根据当前transport创建vtransport信息
        CHK_RET(CreateVirturalTransport(singleSubCommTransport));
    }

    // sort remoteRankMap by remoteRank
    struct LessFirstElement {
        bool operator()(const std::pair<u32, u32>& a, const std::pair<u32, u32>& b) const {
            return a.first < b.first;
        }
    };
    std::sort(remoteRankMap.begin(), remoteRankMap.end(), LessFirstElement());
    std::vector<std::pair<u32, u32>> reversedRemoteRankMap(remoteRankMap);
    std::reverse(reversedRemoteRankMap.begin(), reversedRemoteRankMap.end());

    struct SubCommLinkPara nextSubCommLinkPara(singleSubCommTransport, remoteRankMap, 0, offset);
    struct SubCommLinkPara prevSubCommLinkPara(singleSubCommTransport, reversedRemoteRankMap, 0, offset);
    auto find_greater_than_key1 = [this](const std::pair<u32, u32>& pair) {
        return pair.first >= (this->userRank_);
    };
    auto find_less_than_key1 = [this](const std::pair<u32, u32>& pair) {
        return pair.first <= (this->userRank_);
    };
    auto nextIt = find_if(remoteRankMap.begin(), remoteRankMap.end(), find_greater_than_key1);
    auto prevIt = find_if(reversedRemoteRankMap.begin(), reversedRemoteRankMap.end(), find_less_than_key1);
    u32 rankNum = remoteRankMap.size();
    nextSubCommLinkPara.remoteRankIdStartIndex = std::distance(remoteRankMap.begin(), nextIt) % rankNum;
    prevSubCommLinkPara.remoteRankIdStartIndex = std::distance(reversedRemoteRankMap.begin(), prevIt) % rankNum;

    for (u32 i = 0; i < (rankNum / (FACTOR_NUM_TWO * offset)) + 1; i++) {
        if ((i == rankNum / (FACTOR_NUM_TWO * offset)) && (rankNum % (FACTOR_NUM_TWO * offset)) != 0) {
            nextSubCommLinkPara.remoteRankIdNum = (rankNum % (FACTOR_NUM_TWO * offset)) / FACTOR_NUM_TWO + 
                ((rankNum % (FACTOR_NUM_TWO * offset)) % FACTOR_NUM_TWO);
            prevSubCommLinkPara.remoteRankIdNum = (rankNum % (FACTOR_NUM_TWO * offset)) / FACTOR_NUM_TWO;
        }

        CHK_RET(createSubCommLinkThreads(tag, transMem, nextSubCommLinkPara, isAicpuModeEn, isBackup, subCommIndex));
        CHK_RET(createSubCommLinkThreads(tag, transMem, prevSubCommLinkPara, isAicpuModeEn, isBackup, subCommIndex));
        CHK_RET(waitSubCommLinkThreadsComplete(nextSubCommLinkPara));
        CHK_RET(waitSubCommLinkThreadsComplete(prevSubCommLinkPara));
        CHK_RET(checkSubCommLinkThreadsStatus(tag, nextSubCommLinkPara, isBackup));
        CHK_RET(checkSubCommLinkThreadsStatus(tag, prevSubCommLinkPara, isBackup));
        for (auto &tmpTag : socketTagVec_) {
            (void)socketManager_->DestroySockets(tmpTag);
        }
        socketTagVec_.clear();

        nextSubCommLinkPara.remoteRankIdStartIndex += offset;
        prevSubCommLinkPara.remoteRankIdStartIndex += offset;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::Alloc(const std::string &tag, const TransportIOMem &transMem,
    OpCommTransport &opTransportResponse, bool isAicpuModeEn, bool isBackup)
{
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_RET(notifyPool_->RegisterOp(tag));
    workflowMode_ = GetWorkflowMode();  // 后续有起新的线程，因此更新一下workflowMode
    for (auto &levelNSubCommTransport : opTransportResponse) {
         u32 subCommIndex = 0;
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            subCommIndex++;
            DevType devType;
            CHK_RET(hrtGetDeviceType(devType));
            if (devType == DevType::DEV_TYPE_910_93) {
                CHK_RET(AllocSubCommLinks(tag, transMem, singleSubCommTransport, isAicpuModeEn, isBackup, subCommIndex));
                continue;
            }

            std::vector<std::unique_ptr<std::thread> > linkThreads; // 建链所需线程
            linkThreads.resize(singleSubCommTransport.transportRequests.size());
            ThreadsGuard threadsGuard(linkThreads);                 // 确保异常退出场景析构时等待线程join
            u32 threadsRapplyNum{0};                                // 线程使用计数器

            if (singleSubCommTransport.needVirtualLink) {
                // task多线程并行下发，根据当前transport创建vtransport信息
                CHK_RET(CreateVirturalTransport(singleSubCommTransport));
            }

            u32 linkIdx = 0;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid && singleSubCommTransport.links[linkIdx] == nullptr) {
                    if (isBackup && !transportRequest.isUsedRdma) {
                        // 备用链路不需要创建p2p
                        HCCL_INFO("[%s]: no need to create p2p backup link, remoteUserRank[%u], userRank[%u], "
                            "isUsedRdma[%u], isBackup[%d]", __func__, transportRequest.remoteUserRank, userRank_,
                            transportRequest.isUsedRdma, isBackup);
                        linkIdx++;
                        continue;
                    }
                    DeviceMem inputMem;
                    DeviceMem outputMem;
                    DeviceMem expMem;
                    HCCL_DEBUG("transportRequest.inputMemType[%d] transportRequest.outputMemType[%d], isBackup[%d]",
                        transportRequest.inputMemType, transportRequest.outputMemType, isBackup);
                    GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType,
                        inputMem, outputMem, expMem);

                    std::vector<std::shared_ptr<HcclSocket> > connectSockets;
                    bool isInterRdma;
                    HCCL_DEBUG("[%s]: remoteUserRank[%u], userRank[%u], isUsedRdma[%u]", __func__,
                        transportRequest.remoteUserRank, userRank_, transportRequest.isUsedRdma);
                    bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
                    HcclResult ret = CreateDestSockets(tag, transportRequest.remoteUserRank, singleSubCommTransport.taskNum,
                        connectSockets, isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex);
                    HCCL_DEBUG("[%s]CreateDestSockets finished, chooseBackup[%d]", __func__, chooseBackup);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc]Create dest sockets failed"), ret);

                    MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank?
                        MachineType::MACHINE_SERVER_TYPE : MachineType::MACHINE_CLIENT_TYPE;

                    if (transportRequest.isUsedRdma) {
                        HCCL_INFO("[%s]: create rdma link, remoteUserRank[%u], userRank[%u], "
                            "isBackup[%d], chooseBackup[%d], isInterRdma[%d]", __func__, transportRequest.remoteUserRank, 
                            userRank_, isBackup, chooseBackup, isInterRdma);
                    }

                    std::string threadStr = (isInterRdma? "HcclTerL_" : "HcclIntra_") +
                        std::to_string(threadsRapplyNum);
                    linkThreads[threadsRapplyNum].reset(
                        new (std::nothrow) std::thread(&TransportManager::CreateLink,
                            this, tag, hrtErrMGetErrorContextPub(),
                            machineType, rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
                            singleSubCommTransport.supportDataReceivedAck, singleSubCommTransport.linkMode,
                            singleSubCommTransport.enableUseOneDoorbell, threadStr, connectSockets,
                            inputMem, outputMem, transportRequest.isUsedRdma,
                            std::ref(singleSubCommTransport.links[linkIdx]), isAicpuModeEn,
                            transportRequest.notifyNum, chooseBackup, expMem));
                        CHK_SMART_PTR_NULL(linkThreads[threadsRapplyNum]); // 异常时其他线程待处理
                    singleSubCommTransport.status[linkIdx] = TransportStatus::READY; // 建链后 transport设置为ready状态
                    threadsRapplyNum++;
                }
                linkIdx++;
            }

            for (u32 index = 0; index < linkThreads.size(); index++) {
                if (linkThreads[index] == nullptr || !linkThreads[index]->joinable()) {
                    continue;
                }
                linkThreads[index]->join(); // 等待线程执行完毕
                CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
            }
            linkThreads.clear();
            CHK_PRT_RET(GetStopFlag(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

            linkIdx = 0;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    if (isBackup && !transportRequest.isUsedRdma) {
                        // 备用链路不需要创建p2p
                        HCCL_INFO("[%s]: no need to check p2p backup link, remoteUserRank[%u], userRank[%u], "
                            "isUsedRdma[%u], isBackup[%d]", __func__, transportRequest.remoteUserRank, userRank_,
                            transportRequest.isUsedRdma, isBackup);
                        linkIdx++;
                        continue;
                    }
                    if (singleSubCommTransport.links[linkIdx] == nullptr) {
                        HCCL_ERROR("[Create]errNo[0x%016llx] transport create fail in thread, local rank[%d] remote rank[%d]",
                            HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), userRank_, transportRequest.remoteUserRank);
                        (void)ExceptionHandle(tag, opTransportResponse);
                        SaluSleep(EXCEPTION_DELAY_US_COUNT);
                        (void)notifyPool_->UnregisterOp(tag);
                        return HCCL_E_NOT_FOUND;
                    }
                }
                linkIdx++;
            }
            for (auto &tmpTag : socketTagVec_) {
                (void)socketManager_->DestroySockets(tmpTag);
            }
            socketTagVec_.clear();
        }
    }
    CHK_RET(notifyPool_->UnregisterOp(tag));

    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetIncreRemoteRankList(OpCommTransport &opTransportReq,
    OpCommTransport &opTransportResponse, std::vector<u32> &rankList, TransportType transportType)
{
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            SingleSubCommTransport &reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest &transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                AddremoteUserRankToList(transportRequest, rankList, transportType);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportManager::IncreAlloc(const std::string &tag, const TransportIOMem &transMem,
    OpCommTransport &opTransportReq, OpCommTransport &opTransportResponse, bool isAicpuModeEn, bool isBackup)
{
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_RET(notifyPool_->RegisterOp(tag));

    workflowMode_ = GetWorkflowMode();
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        u32 subCommIndex = 0;
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            subCommIndex++;
            std::vector<std::unique_ptr<std::thread> > linkThreads; // 建链所需线程
            linkThreads.resize(opTransportReq[levelIndex][ringIndex].transportRequests.size());
            ThreadsGuard threadsGuard(linkThreads);                 // 确保异常退出场景析构时等待线程join
            u32 threadsRapplyNum{0};                                // 线程使用计数器
            SingleSubCommTransport &reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            SingleSubCommTransport &respSingleSubComm = opTransportResponse[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest &transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                CHK_PRT_RET(rankIndex >= respSingleSubComm.links.size(),
                    HCCL_ERROR("[IncreAlloc] The remote rank_id[%u] is larger than the existent respSingleSubComm map "\
                    "size[%u]", rankIndex, respSingleSubComm.links.size()), HCCL_E_PARA);
                if (respSingleSubComm.links[rankIndex] != nullptr &&
                    respSingleSubComm.links[rankIndex]->GetLinkType() != hccl::LinkType::LINK_RESERVED) {
                    HCCL_INFO("[IncreAlloc] The link to remote userRank[%u] has existed", transportRequest.remoteUserRank);
                    continue;
                }
                if (transportRequest.isValid) {
                    if (isBackup && !transportRequest.isUsedRdma) {
                        // 备用链路不需要创建p2p
                        HCCL_INFO("[%s]: no need to create p2p backup link, remoteUserRank[%u], userRank[%u], "
                            "isUsedRdma[%u], isBackup[%d]", __func__, transportRequest.remoteUserRank, userRank_,
                            transportRequest.isUsedRdma, isBackup);
                        continue;
                    }
                    respSingleSubComm.transportRequests[rankIndex] = transportRequest;
                    DeviceMem inputMem;
                    DeviceMem outputMem;
                    DeviceMem expMem;
                    GetIOMem(transMem, transportRequest.inputMemType, transportRequest.outputMemType,
                        inputMem, outputMem, expMem);

                    std::vector<std::shared_ptr<HcclSocket> > connectSockets;
                    bool isInterRdma;
                    bool chooseBackup = transportRequest.isUsedRdma ? isBackup : false;
                    HcclResult ret = CreateDestSockets(tag, transportRequest.remoteUserRank, reqSingleSubComm.taskNum,
                        connectSockets, isInterRdma, transportRequest.isUsedRdma, chooseBackup, subCommIndex);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[IncreAlloc]Create dest sockets failed"), ret);

                    MachineType machineType = transportRequest.localUserRank < transportRequest.remoteUserRank?
                        MachineType::MACHINE_SERVER_TYPE : MachineType::MACHINE_CLIENT_TYPE;
                    std::string threadStr = (isInterRdma? "HcclTerL_" : "HcclIntra_") +
                        std::to_string(threadsRapplyNum);
                    linkThreads[threadsRapplyNum].reset(new (std::nothrow) std::thread(&TransportManager::CreateLink,
                            this, tag, hrtErrMGetErrorContextPub(),
                            machineType, rankInfoList_[userRank_].serverId, transportRequest.remoteUserRank,
                            reqSingleSubComm.supportDataReceivedAck, reqSingleSubComm.linkMode,
                            reqSingleSubComm.enableUseOneDoorbell, threadStr, connectSockets, inputMem, outputMem,
                            transportRequest.isUsedRdma, std::ref(respSingleSubComm.links[rankIndex]), isAicpuModeEn,
                            transportRequest.notifyNum, chooseBackup, expMem));
                        CHK_SMART_PTR_NULL(linkThreads[threadsRapplyNum]); // 异常时其他线程待处理
                    respSingleSubComm.status[rankIndex] = TransportStatus::READY; // 建链后 transport设置为ready状态
                    threadsRapplyNum++;
                }
            }
            for (u32 index = 0; index < linkThreads.size(); index++) {
                if (linkThreads[index] != nullptr && linkThreads[index]->joinable()) {
                    linkThreads[index]->join();
                    CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
                }
            }
            linkThreads.clear();

            for (auto &tmpTag : socketTagVec_) {
                (void)socketManager_->DestroySockets(tmpTag);
            }
            socketTagVec_.clear();
        }
    }
    CHK_RET(notifyPool_->UnregisterOp(tag));
    return HCCL_SUCCESS;
}

HcclResult TransportManager::ConstructTransTag(const std::string& tag, std::string& transTag, bool isInterRdma, u32 subCommIndex)
{
    transTag = (Is310PDevice() || isHaveCpuRank_) ? tag : identifier_ + "_res_optimize_" + std::to_string(subCommIndex);
    std::string tmpStr = isInterRdma ? "_Inter_" : "_Intra_";
    transTag += tmpStr;
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetIOMem(const TransportIOMem &transMem,
    const TransportMemType inputMemType, const TransportMemType outputMemType,
    DeviceMem &inputMem,  DeviceMem &outputMem, DeviceMem &expMem)
{
    if (inputMemType == CCL_INPUT) {
        inputMem = transMem.cclInputMem;
    } else if (inputMemType == SCRATCH) {
        inputMem = transMem.scratchMem;
    } else if (inputMemType == PARAM_INPUT) {
        inputMem = transMem.paramInputMem;
    } else if (inputMemType == AIV_INPUT) {
        inputMem = transMem.aivInputMem;
    } else if (inputMemType == AIV_OUTPUT) {
        inputMem = transMem.aivOutputMem;
    } else if (inputMemType == CCL_OUTPUT) {
        inputMem = transMem.cclOutputMem;
    } else {
        HCCL_ERROR("inputMemType is Invalid, inputMem not set");
        return HCCL_E_INTERNAL;
    }

    if (outputMemType == CCL_OUTPUT) {
        outputMem = transMem.cclOutputMem;
    } else if (outputMemType == SCRATCH) {
        outputMem = transMem.scratchMem;
    } else if (outputMemType == PARAM_OUTPUT) {
        outputMem = transMem.paramOutputMem;
    } else if (outputMemType == AIV_INPUT) {
        outputMem = transMem.aivInputMem;
    } else if (outputMemType == AIV_OUTPUT) {
        outputMem = transMem.aivOutputMem;
    } else if (outputMemType == CCL_INPUT) {
        outputMem = transMem.cclInputMem;
    } else if (outputMemType == PARAM_INPUT) {
        outputMem = transMem.paramInputMem;
    } else {
        HCCL_ERROR("outputMemType is Invalid, inputMem not set");
        return HCCL_E_INTERNAL;
    }

    expMem = transMem.expMem;
    return HCCL_SUCCESS;
}

u32 TransportManager::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

u32 TransportManager::GetRemoteNicPort(s32 devicePhyId, u32 dstUserRank, bool isInterRdma)
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        return GetHostPort(devicePhyId);
    }
    // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
    // vnic port仅用于开启device侧端口配置时的sdma场景
    bool useVnicPort = devPortSwitchOn_ && !isInterRdma && !Is310PDevice();
    const std::vector<u32> &ranksPorts = useVnicPort ? vnicRanksPort_ : nicRanksPort_;
    return GetNicPort(devicePhyId, ranksPorts, dstUserRank, isUseRankPort_);
}

HcclResult TransportManager::CreateDestSockets(const std::string &tag, RankId remoteRank, u64 taskNum,
    std::vector<std::shared_ptr<HcclSocket> > &connectSockets, bool &isInterRdma, bool forceRdma, bool isBackup, u32 subCommIndex)
{
    // 改对端的ip和port
    UpdateIsInterRdma(remoteRank, isInterRdma, forceRdma);
    HCCL_INFO("[Create][DestSockets]UpdateIsInterRdma finished. local rank[%u], remote rank[%u],"
        "isInterRdma[%d], forceRdma[%d]", userRank_, remoteRank, isInterRdma, forceRdma);

    u32 socketsPerLink = 1;
    if (isInterRdma) {
        socketsPerLink = GetSocketsPerLink(taskNum);
    }

    HcclRankLinkInfo remoteLinkInfo;
    MakeRemoteLinkInfo(remoteRank, isInterRdma, socketsPerLink, remoteLinkInfo);
    if (isBackup) {
        remoteLinkInfo.ip = rankInfoList_[remoteRank].backupNicIp[0];
        remoteLinkInfo.port = rankInfoList_[remoteRank].backupDevicePort == HCCL_INVALID_PORT
            ? AICPU_RETRY_BACKUP_PORT : rankInfoList_[remoteRank].backupDevicePort;
    }

    HCCL_INFO("[%s] ip and port info. local rank[%u], remote rank[%u], isBackup[%d], port[%u], ip[%s]",
        __func__, userRank_, remoteRank, isBackup, remoteLinkInfo.port, remoteLinkInfo.ip.GetReadableIP());

    std::string newTag;
    CHK_RET(ConstructTransTag(tag, newTag, isInterRdma, subCommIndex));

    HcclResult ret = HCCL_SUCCESS;
    if (isInterRdma || Is310PDevice()) {
        HcclNetDevCtx netDevCtx = nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE ?
            netDevCtxMap_[devIpAddr_[0]]: netDevCtxMap_[hostIp_];
        if (isBackup && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            netDevCtx = netDevCtxMap_[rankInfoList_[userRank_].backupNicIp[0]];
            HCCL_DEBUG("[%s]refresh netDevCtx info. local rank[%u], remote rank[%u], isBackup[%d], port[%u], ip[%s]",
                __func__, userRank_, remoteRank, isBackup, remoteLinkInfo.port, 
                (rankInfoList_[userRank_].backupNicIp[0]).GetReadableIP());
        }
        ret = socketManager_->CreateSingleLinkSocket(newTag, netDevCtx, remoteLinkInfo, connectSockets, false, false);
        if (!GetExternalInputHcclIsTcpMode()) {
            std::vector<std::string>::iterator iter = std::find(socketTagVec_.begin(), socketTagVec_.end(), newTag);
            if (iter == socketTagVec_.end()) {
                socketTagVec_.push_back(newTag);
            }
        }
    } else {
        if (rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_310P3) {
            std::vector<u32> enableP2PDevices;
            enableP2PDevices.push_back(rankInfoList_[remoteRank].devicePhyId);
            HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Create][DestSockets]Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], ret[%u]",
                rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, ret), ret);
            enableP2PDevices_.push_back(rankInfoList_[remoteRank].devicePhyId);
        }
        // server内非异构场景，使能P2P
        bool isInterServer = rankInfoList_[userRank_].serverId != rankInfoList_[remoteRank].serverId;
        if (!isInterServer && !isHaveCpuRank_) {
            std::vector<u32> WaitP2PEnabledDevices;
            WaitP2PEnabledDevices.push_back(rankInfoList_[remoteRank].devicePhyId);
            HcclResult ret = P2PMgmtPub::WaitP2PEnabled(WaitP2PEnabledDevices, [this]() -> bool { return this->GetStopFlag(); });
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Create][DestSockets]Wait Enable P2P Failed, src devicePhyId[%d], dst devicePhyId[%d], ret[%u]",
                rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, ret), ret);
        }
        ret = socketManager_->CreateSingleLinkSocket(newTag, netDevCtxMap_[localVnicIp_],
            remoteLinkInfo, connectSockets, false, true);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Create][DestSockets]Create single link sockets failed, "
            "local rank[%u], remote rank[%u], isInterRdma[%d]", userRank_, remoteRank, isInterRdma), ret);
    return ret;
}

u32 TransportManager::GetSocketsPerLink(u64 taskNum)
{
    if (GetExternalInputQpSrcPortConfigPath() != "" &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return 2; // 2：多QP方式下额外创建一个socket用于同步QP状态迁移完成状态
    } else if (GetExternalInputQpsPerConnection() != HCCL_QPS_PER_CONNECTION_DEFAULT &&
               GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return 2; // 2：多QP方式下额外创建一个socket用于同步QP状态迁移完成状态
    }
    u32 socketsPerLink = 1;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (taskNum == 0) {
            taskNum = 1;
        }
        socketsPerLink = (taskNum + (HCCP_SQ_TEMPLATE_CAPACITY - 1)) / HCCP_SQ_TEMPLATE_CAPACITY;
    }
    return socketsPerLink;
}

HcclResult TransportManager::CreateLink(const std::string &tag, const ErrContextPub &error_context,
    const MachineType machineType, const std::string &serverId, const u32 remoteRank,
    const bool supportDataReceivedAck, const LinkMode linkMode,
    const bool enableUseOneDoorbell, const std::string threadStr,
    const std::vector<std::shared_ptr<HcclSocket> > sockets,
    const DeviceMem inputMem, const DeviceMem outputMem, bool isUsedRdma,
    std::shared_ptr<Transport> &link, bool isAicpuModeEn,
    u32 notifyNum, bool isBackup, const DeviceMem expMem)
{
    hrtErrMSetErrorContextPub(error_context);
    // 给当前线程添加名字
    SetThreadName(threadStr);
    link = nullptr;
    CHK_RET(hrtSetDevice(deviceLogicId_));

    SetWorkflowMode(workflowMode_); // 更新本线程的workflow

    MachinePara machinePara;
    CHK_RET(SetMachinePara(tag, machineType, serverId, remoteRank, supportDataReceivedAck, linkMode, sockets,
        inputMem, outputMem, expMem, isAicpuModeEn, isBackup, notifyNum, trafficClass_, serviceLevel_, machinePara));
    HCCL_DEBUG("inputMem[%p],outputMem[%p], inputMem size[%llu], outputMem size[%llu]", inputMem.ptr(), outputMem.ptr(),
        inputMem.size(), outputMem.size());
    HCCL_INFO("[createLink para]tag[%s], rank[%u]-localUserrank[%u]-localIpAddr[%s], linkMode[%d] "
              "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], "
              "nicDeploy[%d], isBackup[%d]",
        tag.c_str(), userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(),
        machinePara.linkMode, remoteRank, rankInfoList_[remoteRank].worldRank,
        rankInfoList_[remoteRank].serverId.c_str(), machinePara.machineType, machinePara.serverId.c_str(),
        machinePara.nicDeploy, isBackup);
    // transport初始化
    HcclResult ret = TransportInit(remoteRank, machinePara, link, enableUseOneDoorbell, isUsedRdma);
    if (ret != HCCL_SUCCESS) {
        link = nullptr;
        if (ret == HCCL_E_MEMORY) {
            std::string err_str = "[Create][DestLink]Transport init error! IPC memory allocation failed due to "
                "possible memory limit exceeded. Suggested solution: Use 3TB / (ranksize * 2) as the upper limit of "
                "HCCL_BUFFSIZE.";
            RPT_INPUT_ERR(true, "EI0009", std::vector<std::string>({"reason"}), std::vector<std::string>({err_str}));
            HCCL_ERROR("%s", err_str.c_str());
        }
        const std::string  CREATE_LINK_ERR = "[Create][DestLink]Create Dest error! createLink para:rank[" +
            std::to_string(userRank_) + "]-localUserrank[" + std::to_string(rankInfoList_[userRank_].worldRank) +
            "]-localIpAddr[" + rankInfoList_[userRank_].serverId.c_str() + "], dst_rank[" +
            std::to_string(remoteRank) + "]-remoteUserrank[" + std::to_string(rankInfoList_[remoteRank].worldRank) +
            "]-remote_ip_addr[" + rankInfoList_[remoteRank].serverId.c_str() + "]";

        RPT_INPUT_ERR(true, "EI0009", std::vector<std::string>({"reason"}),
            std::vector<std::string>({CREATE_LINK_ERR}));
        HCCL_ERROR("[Create][DestLink]Transport init error! createLink para:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
                   "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], machineType[%d], serverId[%s], linkMode[%d], "
                   "tag[%s]",
            userRank_, rankInfoList_[userRank_].worldRank, rankInfoList_[userRank_].serverId.c_str(), remoteRank,
            rankInfoList_[remoteRank].worldRank, rankInfoList_[remoteRank].serverId.c_str(),
            machinePara.machineType, machinePara.serverId.c_str(), machinePara.linkMode,
            machinePara.tag.c_str());
        return ret;
    }
    HCCL_INFO("[createLink success]:rank[%u]-localUserrank[%u]-localIpAddr[%s], "
        "dst_rank[%u]-remoteUserrank[%u]-remote_ip_addr[%s], tag[%s]", userRank_, rankInfoList_[userRank_].worldRank,
        rankInfoList_[userRank_].serverId.c_str(), remoteRank, rankInfoList_[remoteRank].worldRank,
        rankInfoList_[remoteRank].serverId.c_str(), machinePara.tag.c_str());

    return HCCL_SUCCESS;
}

HcclResult TransportManager::SetMachinePara(const std::string &tag, MachineType machineType,
    const std::string &serverId, u32 dstRank,
    const bool supportDataReceivedAck, const LinkMode linkMode,
    const std::vector<std::shared_ptr<HcclSocket> > &socketList,
    const DeviceMem &inputMem, const DeviceMem &outputMem, const DeviceMem &expMem, bool isAicpuModeEn, 
    bool isBackup, u32 notifyNum, u32 trafficClass, u32 serviceLevel, MachinePara &machinePara)
{
    machinePara.notifyNum = notifyNum;
    machinePara.linkMode = linkMode;
    machinePara.machineType = machineType;
    machinePara.serverId = serverId;
    machinePara.localUserrank = rankInfoList_[userRank_].userRank;
    machinePara.remoteUserrank = rankInfoList_[dstRank].userRank;
    machinePara.localWorldRank = rankInfoList_[userRank_].worldRank;
    machinePara.remoteWorldRank = rankInfoList_[dstRank].worldRank;
    machinePara.collectiveId = identifier_;
    machinePara.deviceType = static_cast<DevType>(rankInfoList_[dstRank].deviceType);
    machinePara.inputMem = inputMem;
    machinePara.outputMem = outputMem;
    machinePara.tc = trafficClass;
    machinePara.sl = serviceLevel;
    if(expMem.ptr() != nullptr){
        machinePara.mem.push_back(expMem);
    } else {
        machinePara.mem.clear();
    }
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */
    machinePara.tag = tag;
    if (isBackup) {
        machinePara.localIpAddr = rankInfoList_[userRank_].backupNicIp[0];
        machinePara.remoteIpAddr = rankInfoList_[dstRank].backupNicIp[0];
        u32 localDevBackUpPhyId;
        CHK_RET(hrtGetPairDevicePhyId(rankInfoList_[userRank_].devicePhyId, localDevBackUpPhyId));
        machinePara.localDeviceId = static_cast<s32>(localDevBackUpPhyId);
        u32 remoteDevBackUpPhyId;
        CHK_RET(hrtGetPairDevicePhyId(rankInfoList_[dstRank].devicePhyId, remoteDevBackUpPhyId));
        machinePara.remoteDeviceId = static_cast<s32>(remoteDevBackUpPhyId);
        HCCL_DEBUG("[%s]isBackup[%d], machinePara.localIpAddr[%s], machinePara.remoteIpAddr[%s], "
            "machinePara.localDeviceId[%d],  machinePara.remoteDeviceId[%d].", __func__,
            isBackup, machinePara.localIpAddr.GetReadableIP(), machinePara.remoteIpAddr.GetReadableIP(),
            machinePara.localDeviceId, machinePara.remoteDeviceId);
    } else {
        machinePara.localIpAddr = rankInfoList_[userRank_].nicIp[0];
        machinePara.remoteIpAddr = rankInfoList_[dstRank].nicIp[0];
        machinePara.localDeviceId = rankInfoList_[userRank_].devicePhyId;
        machinePara.remoteDeviceId = rankInfoList_[dstRank].devicePhyId;
        HCCL_DEBUG("[%s]isBackup[%d], machinePara.localIpAddr[%s], machinePara.remoteIpAddr[%s], "
            "machinePara.localDeviceId[%d],  machinePara.remoteDeviceId[%d].", __func__,
            isBackup, machinePara.localIpAddr.GetReadableIP(), machinePara.remoteIpAddr.GetReadableIP(),
            machinePara.localDeviceId, machinePara.remoteDeviceId);
    }
    // 把原来的两层vector变成一层, 方便后继调用
    if (socketList.size() > 0) {
        std::map <u32, std::vector<std::shared_ptr<HcclSocket> > > socketsMap;
        socketsMap[dstRank] = socketList;
        std::map<u32, u32> dstRankToUserRank;
        dstRankToUserRank[dstRank] = dstRank;
        CHK_RET(socketManager_->WaitLinksEstablishCompleted(socketList[0]->GetLocalRole(),
            socketsMap, dstRankToUserRank));
        machinePara.sockets = socketList;
    }
    machinePara.exchangeInfo.resize(rankConsistentDataLength_);
    CHK_RET(RankConsistentcyChecker::GetInstance().GetCheckFrame(&machinePara.exchangeInfo[0],
        rankConsistentDataLength_, tag));
    machinePara.supportDataReceivedAck = supportDataReceivedAck; /* NeedDataReceivedAck(); */
    machinePara.nicDeploy = nicDeployment_;
    machinePara.localSocketPort = rankInfoList_[userRank_].hostPort;
    machinePara.remoteSocketPort = rankInfoList_[dstRank].hostPort;
    if (isBackup) {
        u32 tempDevBackUpPhyId;
        CHK_RET(hrtGetPairDevicePhyId(rankInfoList_[userRank_].devicePhyId, tempDevBackUpPhyId));
        u32 tempDevBackUpLogicId;
        CHK_RET(hrtGetDeviceIndexByPhyId(tempDevBackUpPhyId, tempDevBackUpLogicId));
        machinePara.deviceLogicId = static_cast<s32>(tempDevBackUpLogicId);
        HCCL_DEBUG("[%s]isBackup[%d], machinePara.deviceLogicId[%d].", __func__, isBackup, machinePara.deviceLogicId);
    } else {
        machinePara.deviceLogicId = deviceLogicId_;
        HCCL_DEBUG("[%s]isBackup[%d], machinePara.deviceLogicId[%d].", __func__, isBackup, machinePara.deviceLogicId);
    }
    
    machinePara.srcPorts = std::vector<u32>(1, 0); /* 默认填充一个元素，0代表默认不配置 */
    machinePara.isAicpuModeEn = isAicpuModeEn;

    return HCCL_SUCCESS;
}

TransportType TransportManager::GetTransportType(const u32 dstRank, bool isUsedRdma)
{
    TransportType transportType;
    // 判断是否在同一个server
    if (rankInfoList_[userRank_].serverId == rankInfoList_[dstRank].serverId) {
        if (isHaveCpuRank_) {
            transportType = TransportType::TRANS_TYPE_HETEROG_P2P;
        } else {
            LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
            hrtGetPairDeviceLinkType(rankInfoList_[userRank_].devicePhyId, rankInfoList_[dstRank].devicePhyId,
                linkType);
            if (linkType == LinkTypeInServer::SIO_TYPE && GetExternalInputEnableRdmaSdmaConcurrent() && isUsedRdma
                && rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_910_93) {
                transportType = TransportType::TRANS_TYPE_P2P;
            // Server内判断是否使用rdma
            } else if (isUsedRdma) {
                transportType = TransportType::TRANS_TYPE_IBV_EXP;
            } else {
                transportType = TransportType::TRANS_TYPE_P2P;
            }
        }
    } else { // server间
        if ((!isUsedRdma) && IsSupportInterHccs(dstRank)) {
            // 超节点内节点间走HCCS通信
            transportType = TransportType::TRANS_TYPE_P2P;
        } else if (GetExternalInputHcclIsTcpMode()) {
            transportType = TransportType::TRANS_TYPE_HOST_TCP;
        } else if ((static_cast<DevType>(rankInfoList_[dstRank].deviceType) == DevType::DEV_TYPE_310P3) ||
            (static_cast<DevType>(rankInfoList_[dstRank].deviceType) == DevType::DEV_TYPE_310P1)) {
            transportType = TransportType::TRANS_TYPE_ROCE;
        } else if (isHaveCpuRank_) {
            transportType = TransportType::TRANS_TYPE_HETEROG_ROCE;
        } else if ((!isUsedRdma) && IsSupportInterHccs(dstRank)) {
            // 超节点内节点间走HCCS通信
            transportType = TransportType::TRANS_TYPE_P2P;
        } else {
            transportType = TransportType::TRANS_TYPE_IBV_EXP;
        }
    }

    HCCL_INFO("GetTransportType: srcRank[%u], dstRank[%u], transport_type[%d].",
        userRank_, dstRank, transportType);
    return transportType;
}

void TransportManager::SetTransportParam(TransportPara &para, MachinePara &machinePara)
{
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(
        GetExternalInputHcclLinkTimeOut());
    para.timeout = kdefaultTimeout;
    para.transportResourceInfoAddr = transportResourceInfoAddr_;
    para.transportResourceInfoSize = transportResourceInfoSize_;
    para.virtualFlag = false;
}

HcclResult TransportManager::TransportInit(const u32 dstRank, MachinePara &machinePara,
    std::shared_ptr<Transport> &link, bool useOneDoorbell, bool isUsedRdma)
{
    // 实例化TransportBase
    TransportPara para{};
    SetTransportParam(para, machinePara);

    TransportType type = GetTransportType(dstRank, isUsedRdma);
    if (type == TransportType::TRANS_TYPE_P2P) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_IBV_EXP) {
        if (GetExternalInputQpSrcPortConfigPath() != "" &&
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(LoadMultiQpSrcPortFromFile());
            CHK_RET(GetConfigSrcPorts(machinePara));
        }
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_HOST_TCP) {
        para.nicDeploy = nicDeployment_;
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_ROCE) {
        para.selfIp = &machinePara.localIpAddr;
        para.peerIp = &machinePara.remoteIpAddr;
        std::set<u32> listenedPort;
        CHK_SMART_PTR_NULL(socketManager_);
        CHK_RET(socketManager_->GetListenPortByIp(NICDeployment::NIC_DEPLOYMENT_DEVICE, *(para.selfIp),
            listenedPort));
        para.peerPort = *(listenedPort.begin());
        para.selfPort = para.peerPort;
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_HETEROG_P2P) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else if (type == TransportType::TRANS_TYPE_HETEROG_ROCE) {
        link.reset(new (std::nothrow) Transport(type, para, dispatcher_, notifyPool_, machinePara));
    } else {
        HCCL_ERROR("[Init][Transport]not supported transport type");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(!link, HCCL_ERROR("[Init][Transport]In create link, new link failed"), HCCL_E_PTR);

    if (useOneDoorbell) {
        link->EnableUseOneDoorbell();
    }

    CHK_RET(link->Init());
    // 算子一致性校验
    std::vector<u8> recvData = link->GetExchangeInfo();
    if (recvData.size() != 0) {
        CHK_PRT_RET(recvData.size() != machinePara.exchangeInfo.size(),
            HCCL_ERROR("[Check][ExchangeInfo]remote exchangInfo size[%zu], local exchangeInfo size[%zu]",
            recvData.size(), machinePara.exchangeInfo.size()), HCCL_E_INTERNAL);
        CHK_RET(RankConsistentcyChecker::GetInstance().CheckFrameRecv(&recvData[0],
            recvData.size(), machinePara.tag.c_str()));
    }
    return HCCL_SUCCESS;
}

bool TransportManager::IsSupportInterHccs(const u32 dstRank)
{
    // 仅判断超节点内, 兼容打平通信域同时有server内和server间, 因此不判断server_id
    bool isInterHccsDisable = GetExternalInputInterHccsDisable();
    const std::string &curSuperPodId = rankInfoList_[userRank_].superPodId;
    const std::string &dstSuperPodId = rankInfoList_[dstRank].superPodId;

    bool isInterHccs = isInterHccsDisable == false && useSuperPodMode_ == true &&
                       curSuperPodId.empty() == false && curSuperPodId == dstSuperPodId;

    HCCL_INFO("[IsSupportInterHccs] rank[%u], superPodId[%s], dstRank[%u], dstSuperPodId[%s], useSuperPodMode[%d], "\
        "isInterHccsDisable[%d], isInterHccs[%d]", userRank_, curSuperPodId.c_str(), dstRank, dstSuperPodId.c_str(),
        useSuperPodMode_, isInterHccsDisable, isInterHccs);
    return isInterHccs;
}

void TransportManager::UpdateIsInterRdma(const u32 remoteRank, bool &isInterRdma, bool forceRdma) // 待确认判断是否完善
{
    // 超节点内节点间采用HCCS通信的, 放至dstIntraClientVec_, 采用p2p建链
    bool isInterHccs = IsSupportInterHccs(remoteRank);
    bool isConcurrent = GetExternalInputEnableRdmaSdmaConcurrent();
    if (isConcurrent && forceRdma && rankInfoList_[userRank_].deviceType == DevType::DEV_TYPE_910_93) {
        LinkTypeInServer linkType;
        hrtGetPairDeviceLinkType(rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, linkType);
        if (linkType == LinkTypeInServer::SIO_TYPE) {
            isInterRdma = false;
        } else {
            isInterRdma = true;
        }
    } else if (isInterHccs && (!forceRdma)) {
        isInterRdma = false;
    } else if (rankInfoList_[userRank_].serverId != rankInfoList_[remoteRank].serverId) {
        isInterRdma = true;
    } else {
        LinkTypeInServer linkType;
        hrtGetPairDeviceLinkType(rankInfoList_[userRank_].devicePhyId, rankInfoList_[remoteRank].devicePhyId, linkType);
        isInterRdma = (isUsedRdmaLevel0_ && linkType == LinkTypeInServer::PXI_TYPE) || forceRdma;
    }
}

HcclResult TransportManager::MakeRemoteLinkInfo(const u32 remoteRank, bool isInterRdma,
    u32 socketsPerLink, HcclRankLinkInfo &remoteLinkInfo)
{
    RankInfo dstRankInfo = rankInfoList_[remoteRank];
    remoteLinkInfo.userRank = dstRankInfo.userRank;
    remoteLinkInfo.devicePhyId = dstRankInfo.devicePhyId;
    if (isInterRdma || Is310PDevice()) {
        remoteLinkInfo.ip = dstRankInfo.nicIp[0];
        remoteLinkInfo.port = GetRemoteNicPort(remoteLinkInfo.devicePhyId, dstRankInfo.userRank, isInterRdma);
        remoteLinkInfo.socketsPerLink = socketsPerLink;
    } else {
        remoteLinkInfo.ip = HcclIpAddress(dstRankInfo.devicePhyId);
        if (useSuperPodMode_) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(rankInfoList_[userRank_].devicePhyId,
                DeviceIdType::DEVICE_ID_TYPE_SDID,
                rankInfoList_[remoteRank].superDeviceId,
                remoteLinkInfo.ip));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(rankInfoList_[userRank_].devicePhyId,
                DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                rankInfoList_[remoteRank].devicePhyId,
                remoteLinkInfo.ip));
        }
        remoteLinkInfo.port = GetRemoteNicPort(rankInfoList_[remoteRank].devicePhyId,
            rankInfoList_[remoteRank].userRank, isInterRdma); // ?
        remoteLinkInfo.socketsPerLink = socketsPerLink;
    }
    HCCL_INFO("[TransportManager][MakeRemoteLinkInfo] isInterRdma[%u], is310PDevice[%u], "
        "remote rank: userRank[%u], devPhyId[%u], ip[%s], port[%u], socketsPerLink[%u]",
        isInterRdma, Is310PDevice(), remoteLinkInfo.userRank, remoteLinkInfo.devicePhyId,
        remoteLinkInfo.ip.GetReadableAddress(), remoteLinkInfo.port, remoteLinkInfo.socketsPerLink);
    return HCCL_SUCCESS;
}

HcclResult TransportManager::SetStopFlag(bool value)
{
    stopFlag_.store(value);
    return HCCL_SUCCESS;
}

bool TransportManager::GetStopFlag()
{
    return stopFlag_.load();
}

void TransportManager::SetPortConfig(bool devPortSwitchOn)
{
    devPortSwitchOn_ = devPortSwitchOn;
}

std::vector<std::string> Split(std::string &s, std::string delimiter)
{
    size_t pos_start = 0;
    size_t pos_end = s.find(delimiter, pos_start);
    std::vector<std::string> res;

    while(pos_end != std::string::npos) {
        std::string token = s.substr(pos_start, pos_end - pos_start);
        res.push_back(token);
        pos_start = pos_end + delimiter.length();
        pos_end = s.find(delimiter, pos_start);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

HcclResult GetIpPairFromString(std::string &s, std::string &ipPair, u32 lineCnt, std::string &lineAvator)
{
    std::vector<std::string> strIps = Split(s, ",");
    CHK_PRT_RET(strIps.size() != MULTI_QP_CONFIG_IP_NUM,
        HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]invalid Ip format.[%s]",
                    lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);
    
    HcclIpAddress ipAddr{};
    // 解析源ip
    auto ret = ipAddr.SetReadableAddress(strIps[0]);
    CHK_PRT_RET(ret != HCCL_SUCCESS || ipAddr.IsIPv6(),
        HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]srcIp is either in an invalid format"
                    " or is an IPv6 address.[%s]",
                    lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);

    // 解析目的ip
    ret = ipAddr.SetReadableAddress(strIps[1]);
    CHK_PRT_RET(ret != HCCL_SUCCESS || ipAddr.IsIPv6(),
        HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]dstIp is either in an invalid format" 
                    " or is an IPv6 address.[%s]",
                    lineCnt, lineAvator.c_str()),
        HCCL_E_PARA);

    // 记录ip对
    ipPair = s;
    return HCCL_SUCCESS;
}

HcclResult GetSrcPortsFromString(std::string &s, std::vector<u32> &srcPorts,
                                        u32 lineCnt, std::string &lineAvator)
{
    std::vector<std::string> strPorts = Split(s, ",");
    srcPorts.resize(strPorts.size(), 0);
    CHK_PRT_RET(strPorts.size() > MULTI_QP_CONFIG_SRC_PORT_NUM_MAX,
        HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]config ports num[%u] more than the "
        "threshold[%u].[%s]", lineCnt, strPorts.size(), MULTI_QP_CONFIG_SRC_PORT_NUM_MAX, lineAvator.c_str()),
        HCCL_E_PARA);

    for (u32 i = 0; i < strPorts.size(); i++) {
        // 检查端口号是否为全数字的字符串
        CHK_PRT_RET((IsAllDigit(strPorts[i].c_str()) != HCCL_SUCCESS)||
            (SalStrToULong(strPorts[i].c_str(), HCCL_BASE_DECIMAL, srcPorts[i]) != HCCL_SUCCESS) ||
            (srcPorts[i] == 0) || (srcPorts[i] > MULTI_QP_CONFIG_SRC_PORT_ID_MAX),
            HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]src port[%s]"
                "should be within the range of[1, %u] and configured as a valid integer.[%s]",
            lineCnt, strPorts[i].c_str(), MULTI_QP_CONFIG_SRC_PORT_ID_MAX, lineAvator.c_str()), HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult TransportManager::LoadMultiQpSrcPortFromFile()
{
    std::lock_guard<std::mutex> lock(loadCfgFileMutex_); // 加锁避免多线程访问时修改mapIpPairSrcPorts_冲突
    // 判断是否已经读取过配置文件
    if (isCfgFileRead_) {
        HCCL_DEBUG("[TransportManager][LoadMultiQpSrcPortFromFile] file has been read.");
        return HCCL_SUCCESS;
    }

    // 读取配置文件
    std::string fileStr = GetExternalInputQpSrcPortConfigPath() + "/MultiQpSrcPort.cfg";
    char realFile[PATH_MAX] = {0};
    if (realpath(fileStr.c_str(), realFile) == nullptr) {
        const std::string  CFG_FILE_PATH_ERROR = "[TransportManager][LoadMultiQpSrcPortFromFile]file path " +
            fileStr + " is invalid.";
        RPT_INPUT_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
            std::vector<std::string>({fileStr, CFG_FILE_PATH_ERROR}));
        HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile]file[%s] path invalid.", fileStr.c_str());
        return HCCL_E_PARA;
    }

    std::ifstream inFile(fileStr.c_str(), std::ifstream::in);
    if (!inFile) {
        const std::string  CFG_FILE_OPEN_ERROR = "[TransportManager][LoadMultiQpSrcPortFromFile]open file " +
            fileStr + " failed.";
        RPT_INPUT_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
            std::vector<std::string>({fileStr, CFG_FILE_OPEN_ERROR}));
        HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile]open config file[%s] failed.", fileStr.c_str());
        return HCCL_E_PARA;
    }
    HCCL_INFO("[TransportManager][LoadMultiQpSrcPortFromFile]open config file[%s] success.", fileStr.c_str());
    
    // 逐行解析配置文件
    u32 lineCnt = 1;
    std::string line;
    while(std::getline(inFile, line)) {
        std::string lineAvator = line; // 每行内容的快照, 用于dfx
        //去除空格和tab
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());

        // 去除注释
        std::string lineInfo = Split(line, "#")[0]; // 只保留#号前的内容
        if(lineInfo.empty()) {
            HCCL_DEBUG("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]commet line, do not parse.[%s]",
                        lineCnt, lineAvator.c_str());
            lineCnt++;
            continue;
        }

        // 切分字符串, 检查配置格式
        std::vector<std::string> strIpPort = Split(lineInfo, "=");
        if (strIpPort.size() != MULTI_QP_CONFIG_IP_NUM) {
            const std::string  CFG_FORMAT_ERROR = "[TransportManager][LoadMultiQpSrcPortFromFile][line: " 
            + std::to_string(lineCnt) + "] invalid format, " + 
            "Expected format per line and start with: 'srcIPN,dstIPN=srcPort0,srcPort1,...,srcPortN'";
            RPT_INPUT_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
                std::vector<std::string>({fileStr, CFG_FORMAT_ERROR}));
            HCCL_ERROR("%s, Config content[%s]",CFG_FORMAT_ERROR.c_str(), lineAvator.c_str());
            inFile.close();
            return HCCL_E_PARA;
        }

        // 解析ip对
        std::string ipPair;
        auto ret = GetIpPairFromString(strIpPort[0], ipPair, lineCnt, lineAvator);
        if (ret != HCCL_SUCCESS) {
            const std::string  IP_FORMAT_ERROR = "[TransportManager][LoadMultiQpSrcPortFromFile][line: " +
                std::to_string(lineCnt) + "] is an invalid IP or IPv6.";
            RPT_INPUT_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
                std::vector<std::string>({fileStr, IP_FORMAT_ERROR}));
            inFile.close();
            return ret;
        }

        // 解析源端口号
        std::vector<u32> srcPorts;
        ret = GetSrcPortsFromString(strIpPort[1], srcPorts, lineCnt, lineAvator);
        if (ret != HCCL_SUCCESS) {
            const std::string  PORT_FORMAT_ERROR = "[TransportManager][LoadMultiQpSrcPortFromFile][line: " +
                std::to_string(lineCnt) + "] invalid src port format.";
            RPT_INPUT_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
                std::vector<std::string>({fileStr, PORT_FORMAT_ERROR}));
            inFile.close();
            return ret;
        }

        // 配置源端口号
        if (mapIpPairSrcPorts_.find(ipPair) != mapIpPairSrcPorts_.end()) {
            const std::string  DUPLICATE_IPPAIR_ERROR = "[TransportManager][LoadMultiQpSrcPortFromFile][line: " +
                std::to_string(lineCnt) + "] ip pair: " + ipPair + " has existed.";
            RPT_INPUT_ERR(true, "EI0001", std::vector<std::string>({"env", "tips"}),
                std::vector<std::string>({fileStr, DUPLICATE_IPPAIR_ERROR}));
            HCCL_ERROR("[TransportManager][LoadMultiQpSrcPortFromFile][line: %u]ip pair[%s] has existed.[%s]",
                        lineCnt, ipPair.c_str(), lineAvator.c_str());
            inFile.close();
            return HCCL_E_PARA;
        }
        mapIpPairSrcPorts_[ipPair] = srcPorts;

        // 判断文件行数是否超过上限
        if (lineCnt >= MULTI_QP_CONFIG_FILE_LINE_MAX) {
            HCCL_RUN_INFO("[TransportManager][LoadMultiQpSrcPortFromFile]config file is too large.");
            break;
        }
        lineCnt++;
    }
    
    inFile.close();
    isCfgFileRead_ = true;
    return HCCL_SUCCESS;
}

HcclResult TransportManager::GetConfigSrcPorts(MachinePara &machinePara)
{
    std::string srcIp = std::string(machinePara.localIpAddr.GetReadableIP());
    std::string dstIp = std::string(machinePara.remoteIpAddr.GetReadableIP());
    std::string ipPair;
    std::vector<u32> &srcPorts = machinePara.srcPorts;

    // 匹配sip和dip
    ipPair = srcIp + std::string(",") + dstIp;
    auto iter = mapIpPairSrcPorts_.find(ipPair);
    CHK_PRT_RET(iter != mapIpPairSrcPorts_.end(), srcPorts = iter->second, HCCL_SUCCESS);

    // 匹配dip
    if (machinePara.localIpAddr.GetFamily() == AF_INET) {
        ipPair = std::string("0.0.0.0,") + dstIp;
    } else {
        ipPair = std::string("::/128,") + dstIp;
    }
    iter = mapIpPairSrcPorts_.find(ipPair);
    CHK_PRT_RET(iter != mapIpPairSrcPorts_.end(), srcPorts = iter->second, HCCL_SUCCESS);

    // 匹配sip
    if (machinePara.localIpAddr.GetFamily() == AF_INET) {
        ipPair = srcIp + std::string(",0.0.0.0");
    } else {
        ipPair = srcIp + std::string(",::/128");
    }
    iter = mapIpPairSrcPorts_.find(ipPair);
    CHK_PRT_RET(iter != mapIpPairSrcPorts_.end(), srcPorts = iter->second, HCCL_SUCCESS);

    // 通配
    if (machinePara.localIpAddr.GetFamily() == AF_INET) {
        ipPair = std::string("0.0.0.0,0.0.0.0");
    } else {
        ipPair = std::string("::/128,::/128");
    }
    iter = mapIpPairSrcPorts_.find(ipPair);
    CHK_PRT_RET(iter != mapIpPairSrcPorts_.end(), srcPorts = iter->second, HCCL_SUCCESS);

    // 匹配不到，直接返回
    HCCL_DEBUG("[TransportManager][GetConfigSrcPorts]ip pair[%s] not found.", ipPair.c_str());
    return HCCL_SUCCESS;
}
}  // namespace hccl
