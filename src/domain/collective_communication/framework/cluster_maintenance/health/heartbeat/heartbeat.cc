/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heartbeat.h"
#include "device_capacity.h"
#include "externalinput_pub.h"
#include "env_config.h"
#include "opexecounter_pub.h"
#include "hccl_communicator.h"
#include "task_exception_handler_pub.h"

namespace hccl {
constexpr u32 HEARTBEAT_INTERVAL = 1000; // 心跳帧发送周期为1000 ms
constexpr u32 HEARTBEAT_COUNT = HEARTBEAT_INTERVAL / BROADCAST_INTERVAL; // 心跳帧发送间隔数
constexpr u32 BASE_NUMBER = 2;
constexpr u32 RETRY_CQE_ARRAY_SIZE = 128; // 重执行时获取的CQE数组的最大数量，最大128

constexpr u32 JITTER_TIME = 300;  // 关键事件允许的误差事件范围±300s。误差来源：EVENT和NOTIFY差异、传播耗时、计时误差
constexpr u32 EVENT_MAX_CNT = 5000;  // 防止内存泄漏，同时不能太短，防止正常事件被冲掉
constexpr u32 THROUND_MILS = 1000;   // 1000ms
Heartbeat &Heartbeat::GetInstance(s32 deviceLogicID)
{
    static Heartbeat hb[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[Heartbeat][GetInstance]deviceLogicID[%d] is invalid", deviceLogicID);
        return hb[0];
    }
    return hb[deviceLogicID];
}

Heartbeat::~Heartbeat()
{
    if (!groupMap_.empty()) {
        HCCL_RUN_INFO("[Heartbeat]groupMap_ size[%llu].", groupMap_.size());
        for (auto iter = groupMap_.begin(); iter != groupMap_.end(); iter++) {
            HCCL_RUN_WARNING("[Heartbeat]UnRegister group[%s].", iter->first.c_str());
        }
    }
    (void)DeInit();
    groupMap_.clear();
    retryEnableTable_.clear();
    backupEnableTable_.clear();
}

bool Heartbeat::IsEnableBackupLink()
{
    std::lock_guard<std::mutex> lock(backupEnableMutex_);
    // 若backupEnableTable_不为空，则当前还有通信域使能借轨，需要获取备用的cqe
    auto isEmpty = backupEnableTable_.empty();
    return !isEmpty;
}

HcclResult Heartbeat::InitNic(const NicType nicType, const s32 devicePhyId, const s32 deviceLogicId,
    const hccl::HcclIpAddress ip, const u32 port, const bool isBackUp)
{
    HcclNetDevCtx nicCtx;
    CHK_RET(HcclNetOpenDev(&nicCtx, nicType, devicePhyId, deviceLogicId, ip));
    CHK_PTR_NULL(nicCtx);
    netDevCtxMap_.insert(std::make_pair(ip, nicCtx));

    if (!isBackUp) {
        std::shared_ptr<HcclSocket> tempSocket;
        EXECEPTION_CATCH((tempSocket = std::make_shared<HcclSocket>(
            nicCtx, port)), return HCCL_E_PTR);
        CHK_RET(tempSocket->Init());

        CHK_RET(tempSocket->Listen());

        listenSocketMap_.insert(std::make_pair(ip, tempSocket));
    }

    HCCL_INFO("[Heartbeat][InitNic]NicType[%d], devicePhyId[%d], deviceLogicId[%d], ip[%s], port[%u], isBackUp[%d].",
        nicType, devicePhyId, deviceLogicId, ip.GetReadableAddress(), port, isBackUp);
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::Init(const RankInfo& locRank, const bool useSuperPodMode, const bool isNeedNic, const u32 port)
{
    HCCL_INFO("[Init]heartbeat Init begin");
    devicePhyId_ = locRank.devicePhyId;
    if (IsEnableBackupLink()) {
        CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
    }
    superDeviceId_ = locRank.superDeviceId;

    if (devicePhyId_ == static_cast<u32>(HOST_DEVICE_ID)) {
        deviceLogicId_ = devicePhyId_;
        deviceBackupLogicId_ = deviceBackUpPhyId_;
    } else {
        CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId_, deviceLogicId_));
        if (IsEnableBackupLink()) {
            CHK_RET(hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackupLogicId_));
        }
    }

    std::unique_lock<std::mutex> mapLock(ctxMapMutex_);

    if (isNeedNic && locRank.nicIp.size() != 0) {
        nicIp_ = locRank.nicIp[0];
        u32 nicPort = (port == HCCL_INVALID_PORT) ? locRank.deviceNicPort : port;
        if (locRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !nicIp_.IsInvalid() &&
            netDevCtxMap_.find(nicIp_) == netDevCtxMap_.end()) {
            CHK_RET(InitNic(NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, nicIp_, nicPort));
        }
    }
    if (isNeedNic && locRank.backupNicIp.size() != 0 && IsEnableBackupLink()) {
        backupNicIp_ = locRank.backupNicIp[0];
        u32 backupPort = HCCL_INVALID_PORT; // 不初始化备用网卡上的Socket
        if (locRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE &&
            netDevCtxMap_.find(backupNicIp_) == netDevCtxMap_.end()) {
            CHK_RET(InitNic(NicType::DEVICE_NIC_TYPE, deviceBackUpPhyId_, deviceBackupLogicId_,
                backupNicIp_, backupPort, true));
        }
    }
    if (isNeedNic && locRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST &&
            netDevCtxMap_.find(locRank.hostIp) == netDevCtxMap_.end()) {
        u32 hostPort = GetHostPort(devicePhyId_);
        CHK_RET(InitNic(NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, locRank.hostIp, hostPort));
    }
    mapLock.unlock();

    uid_ = GetUId(locRank);
    nicDeploy_ = locRank.nicDeploy;

    startSendRecvTask_ = true;
    sendRecvThread_.reset(new (std::nothrow) std::thread(&Heartbeat::HeartbeatStatusMonitor, this));
    CHK_SMART_PTR_NULL(sendRecvThread_);

    lostThreshold_ = 30;  // 心跳丢失阈值为30s
    initialized_ = true;
    isDeInit_ = false;
    HCCL_INFO("[Init]heartbeat Init end");
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::DeInit()
{
    HCCL_INFO("[DeInit]heartbeat deinit begin");
    isDeInit_ = true;
    startSendRecvTask_ = false;
    {
        std::unique_lock<std::mutex> lock(ProcessLock_);
        for (auto iter = rankId2SocketMap_.begin(); iter != rankId2SocketMap_.end(); iter++) {
            if (iter->second.socket->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_SERVER) {
                CHK_PRT_RET(listenSocketMap_.find(iter->second.socket->GetLocalIp()) == listenSocketMap_.end(),
                    HCCL_ERROR("ip[%s] listenSocketMap is not found",
                    iter->second.socket->GetLocalIp().GetReadableAddress()), HCCL_E_NOT_FOUND);
                listenSocketMap_[iter->second.socket->GetLocalIp()]->DelWhiteList(iter->second.wlistInfosVec);
            }
            iter->second.socket->Close();
        }
        rankId2SocketMap_.clear();
        rankId2StatusMap_.clear();
    }
    std::queue<HeartBeatFrame> empty;
    std::swap(errStatusQueue_, empty);
    if (sendRecvThread_) {
        if (sendRecvThread_->joinable()) {
            sendRecvThread_->join();
        }
    }

    std::unique_lock<std::mutex> mapLock(ctxMapMutex_);
    listenSocketMap_.clear();
    for (auto &iter : netDevCtxMap_) {
        HcclNetCloseDev(iter.second);
    }
    vnicIp_.clear();
    nicIp_.clear();
    backupNicIp_.clear();

    netDevCtxMap_.clear();
    mapLock.unlock();

    initialized_ = false;
    HCCL_INFO("[DeInit]heartbeat deinit end");
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::PrepareConnect(ConnInfo &info)
{
    CHK_SMART_PTR_NULL(info.socket);
    if (info.socket->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_SERVER) {
        CHK_PRT_RET(listenSocketMap_.find(info.socket->GetLocalIp()) == listenSocketMap_.end(),
            HCCL_ERROR("ip[%s] listenSocketMap is not found",
            info.socket->GetLocalIp().GetReadableAddress()), HCCL_E_NOT_FOUND);
        CHK_RET(listenSocketMap_[info.socket->GetLocalIp()]->AddWhiteList(info.wlistInfosVec));
    } else {
        if (info.socket->GetStatus() != HcclSocketStatus::SOCKET_OK) {
            CHK_RET(info.socket->Connect());
        }
    }

    return HCCL_SUCCESS;
}

HcclResult Heartbeat::RegisterRanks(const RankInfo& locRank, std::vector<RankInfo> &rankInfos, const u32 port,
    const bool isNeedNic, const std::string &group, bool useSuperPodMode, bool isUsedRdma)
{
    HCCL_INFO("[RegisterRanks]group[%s] isUsedRdma[%d], isNeedNic[%d], RegisterRanks Start.",
        group.c_str(), isUsedRdma, isNeedNic);
    auto iter = groupMap_.find(group);
    if (iter != groupMap_.end()) {
        HCCL_INFO("group[%s] has Registered, skip.", group.c_str());
        return HCCL_SUCCESS;
    }

    if (!initialized_) {
        CHK_RET(Init(locRank, useSuperPodMode, isNeedNic, port));
    }
    // 刷新uid_，防止不同通信域下serverId不一致问题
    uid_ = GetUId(locRank);

    std::unique_lock<std::mutex> mapLock(ctxMapMutex_);
    if (devicePhyId_ != static_cast<u32>(HOST_DEVICE_ID) && rankInfos.size() > 1 && vnicIp_.IsInvalid()) {
        vnicIp_ = HcclIpAddress(useSuperPodMode ? superDeviceId_ : devicePhyId_);
        u32 vnicPort = (port == HCCL_INVALID_PORT) ? locRank.deviceVnicPort : port;
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(devicePhyId_,
            (useSuperPodMode ? DeviceIdType::DEVICE_ID_TYPE_SDID : DeviceIdType::DEVICE_ID_TYPE_PHY_ID),
            (useSuperPodMode ? superDeviceId_ : devicePhyId_), vnicIp_));
        if (netDevCtxMap_.find(vnicIp_) == netDevCtxMap_.end()) {
            CHK_RET(InitNic(NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, vnicIp_, vnicPort));
        }
    }

    // 防止首次没有读到nicIp, 后续注册心跳的时候刷新上
    if (isNeedNic && nicIp_.IsInvalid() && locRank.nicIp.size() != 0) {
        nicIp_ = locRank.nicIp[0];
        u32 nicPort = (port == HCCL_INVALID_PORT) ? locRank.deviceNicPort : port;
        if (locRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && !nicIp_.IsInvalid() &&
            netDevCtxMap_.find(nicIp_) == netDevCtxMap_.end()) {
            CHK_RET(InitNic(NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, nicIp_, nicPort));
        }
    }

    if (isNeedNic && backupNicIp_.IsInvalid() && locRank.backupNicIp.size() != 0) {
        backupNicIp_ = locRank.backupNicIp[0];
        u32 backupPort = HCCL_INVALID_PORT; // 不初始化备用网卡上的Socket
        if (IsEnableBackupLink()) {
            CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
            CHK_RET(hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackupLogicId_));
            if (locRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE &&
                netDevCtxMap_.find(backupNicIp_) == netDevCtxMap_.end()) {
                CHK_RET(InitNic(NicType::DEVICE_NIC_TYPE, deviceBackUpPhyId_, deviceBackupLogicId_,
                    backupNicIp_, backupPort, true));
            }
        }
    }

    if (isNeedNic && locRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST &&
            netDevCtxMap_.find(locRank.hostIp) == netDevCtxMap_.end()) {
        u32 hostPort = GetHostPort(devicePhyId_);
        CHK_RET(InitNic(NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, locRank.hostIp, hostPort));
    }
    mapLock.unlock();

    std::unique_lock<std::mutex> lock(ProcessLock_);
    for (const auto remRank : rankInfos) {
        UIDType rem = GetUId(remRank);
        rankId2StatusMap_.insert(rem, Status());
        groupMap_[group].insert(std::make_pair(rem, NO_CONN));
        HCCL_INFO("[RegisterRanks]group[%s] remote:%s", group.c_str(), FormatUId(rem).c_str());
    }
    lock.unlock();

    std::map<UIDType, ConnInfo> needConnectRank;
    CHK_RET(GetConnectRank(locRank, rankInfos, needConnectRank, useSuperPodMode, isUsedRdma));
    for (auto iter = needConnectRank.begin(); iter != needConnectRank.end(); iter++) {
        if (iter->second.newConn) {
            CHK_RET(PrepareConnect(iter->second));
        }
    }

    auto startTime = std::chrono::steady_clock::now();
    bool isLoop = false;
    do {
        isLoop = false;
        CHK_RET(CreateHeartConnect(group, needConnectRank, isLoop, startTime));
        if (isDeInit_) {
            break;
        }
        if (isLoop) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        }
    } while (isLoop);
    HCCL_INFO("[RegisterRanks]group[%s] isUsedRdma[%d], netDevCtxMap size[%llu], nicIp IsInvalid[%d],"
        " RegisterRanks Completed", group.c_str(), isUsedRdma, netDevCtxMap_.size(), nicIp_.IsInvalid());
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::CreateHeartConnect(const std::string &group, std::map<UIDType, ConnInfo> &needConnectRank,
    bool &isLoop, std::chrono::time_point<std::chrono::steady_clock> &startTime)
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    lock.unlock();
    auto HEART_CREATE_LINK_TIMEOUT = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    for (auto iter = needConnectRank.begin(); iter != needConnectRank.end(); iter++) {
        if (iter->second.newConn) {
            if ((std::chrono::steady_clock::now() - startTime) >= HEART_CREATE_LINK_TIMEOUT) {
                HCCL_ERROR("establish rank[%s] to rank[%s] connection, get rasocket timeout, timeout[%lld s], the "
                           "HCCL_CONNECT_TIMEOUT may be insufficient. group[%s]",
                    FormatUId(uid_).c_str(),
                    FormatUId(iter->first).c_str(),
                    HEART_CREATE_LINK_TIMEOUT,
                    group.c_str());
                return HCCL_E_TIMEOUT;
            }

            if (iter->second.socket->GetStatus() == HcclSocketStatus::SOCKET_CONNECTING) {
                isLoop = true;
                continue;
            }
            // 心跳socket建链完成后，需要立即及激活其心跳收发能力
            lock.lock();
            rankId2SocketMap_.insert(iter->first, iter->second);
            CHK_RET(
                rankId2SocketMap_[iter->first].recvBuffer.Init(2 * sizeof(HeartBeatFrame)));  // 2倍帧长，确报不会溢出
            lock.unlock();
            iter->second.newConn = false;
            HCCL_RUN_INFO("group:[%s], establish rank[%s] to rank[%s] connection success",
                group.c_str(),
                FormatUId(uid_).c_str(),
                FormatUId(iter->first).c_str());
        } else if (groupMap_[group].find(iter->first) == groupMap_[group].end() ||
                   (groupMap_[group].count(iter->first) && groupMap_[group][iter->first] == NO_CONN)) {
            lock.lock();
            rankId2SocketMap_.ref(iter->first);
            HCCL_RUN_INFO("group:[%s], establish rank[%s] to rank[%s] connection success",
                group.c_str(),
                FormatUId(uid_).c_str(),
                FormatUId(iter->first).c_str());
            lock.unlock();
        }
        groupMap_[group][iter->first] = HAS_CONN;
    }
    return HCCL_SUCCESS;
}

void Heartbeat::RegisterRetryInfo(const std::string &commIdentifier, bool retryEnable, bool backupEnable)
{
    {
        std::lock_guard<std::mutex> retryEnablelock(retryEnableMutex_);
        auto search = retryEnableTable_.find(commIdentifier);
        if (search != retryEnableTable_.end()) {
            HCCL_INFO("[RegisterToHeartBeat]register identifier[%s] retryEnable[%d] has been registered",
                commIdentifier.c_str(), search->second);
        } else {
            retryEnableTable_.insert({commIdentifier, retryEnable});
            HCCL_RUN_INFO("[RegisterToHeartBeat]register identifier[%s] retryEnable[%d]", commIdentifier.c_str(), retryEnable);
        }
    }
    if (backupEnable) {
        // 若当前通信域使能借轨，则加入到backupEnableTable_中
        std::lock_guard<std::mutex> backupEnablelock(backupEnableMutex_);
        if (backupEnableTable_.find(commIdentifier) == backupEnableTable_.end()) {
            backupEnableTable_.insert(commIdentifier);
            HCCL_RUN_INFO("[RegisterToHeartBeat]register identifier[%s] backupEnable[%d]", commIdentifier.c_str(), backupEnable);
        }
    }
    return ;
}

HcclResult Heartbeat::RegisterToHeartBeat(u32 userRank, DevType devType, std::vector<RankInfo> &rankInfoList,
    const u32 port, const bool isNeedNic, const std::string &commIdentifier, bool useSuperPodMode, bool isUsedRdmaLevel0,
    bool retryEnable, bool backupEnable)
{
    if (Is310PDevice() || devType == DevType::DEV_TYPE_310P3) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(rankInfoList.size() == 1,
        HCCL_WARNING("[RegisterToHeartBeat]Identifier[%s] rankSize[%llu] needn't to register.",
        commIdentifier.c_str(), rankInfoList.size()), HCCL_SUCCESS);

    RankInfo locRank;
    for (auto rank : rankInfoList) {
        if (userRank == rank.userRank) {
            locRank = rank;
            break;
        }
    }

    RegisterRetryInfo(commIdentifier, retryEnable, backupEnable);
    CHK_RET(RegisterRanks(locRank, rankInfoList, port, isNeedNic, commIdentifier, useSuperPodMode, isUsedRdmaLevel0));
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::RegisterToHeartBeat(u32 userRank, DevType devType, std::vector<RankInfo> &rankInfoList,
    const u32 port, const bool isNeedNic, u32 peerRankId, const std::string &commIdentifier, const std::string& tag,
    bool useSuperPodMode, bool isUsedRdmaLevel0, bool retryEnable, bool backupEnable)
{
    if (Is310PDevice() || devType == DevType::DEV_TYPE_310P3 ||
        (rankInfoList[userRank].devicePhyId == HOST_DEVICE_ID) ||
        (rankInfoList[peerRankId].devicePhyId == HOST_DEVICE_ID)) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(rankInfoList.size() == 1,
        HCCL_WARNING("[RegisterToHeartBeat]Identifier[%s] rankSize[%llu] needn't to register.",
        commIdentifier.c_str(), rankInfoList.size()), HCCL_SUCCESS);

    RankInfo locRank;
    std::vector<RankInfo> peerRankInfoList;
    bool findLoc = false;
    bool findPeer = false;

    if (userRank == peerRankId) { // batchSendRecv 注册，不建链
        for (auto rank : rankInfoList) {
            if (userRank == rank.userRank) {
                locRank = rank;
                peerRankInfoList.push_back(rank);
                findLoc = true;
            }
            if (findLoc) {
                break;
            }
        }
    } else {// sendRecv 注册，对端建链
        for (auto rank : rankInfoList) {
            if (userRank == rank.userRank) {
                locRank = rank;
                peerRankInfoList.push_back(rank);
                findLoc = true;
            }

            if (peerRankId == rank.userRank) {
                peerRankInfoList.push_back(rank);
                findPeer = true;
            }

            if (findLoc && findPeer) {
                break;
            }
        }
    }
    RegisterRetryInfo(commIdentifier, retryEnable, backupEnable);
    const std::string groupName = commIdentifier + "_" + tag;
    CHK_RET(RegisterRanks(locRank, peerRankInfoList, port, isNeedNic, groupName, useSuperPodMode, isUsedRdmaLevel0));
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::UnRegisterRanks(const std::string &group)
{
    HCCL_INFO("[UnRegisterRanks]group[%s] UnregisterRanks Start.", group.c_str());
    CHK_PRT_RET(initialized_ == false, HCCL_WARNING("Heartbeat has been destroyed"), HCCL_SUCCESS);

    {
        std::unique_lock<std::mutex> lock(ProcessLock_);
        auto iter = groupMap_.find(group);
        if (iter == groupMap_.end()) {
            HCCL_INFO("group[%s] hasn't Registered, skip", group.c_str());
            return HCCL_SUCCESS;
        }

        for (const auto remRank : groupMap_[group]) {
            UIDType rem = remRank.first;
            rankId2StatusMap_.erase(rem);
            if (remRank.second == HAS_CONN) {
                if (rankId2SocketMap_.count(rem) == 1) {
                    if (rankId2SocketMap_[rem].socket->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_SERVER) {
                        CHK_PRT_RET(listenSocketMap_.find(rankId2SocketMap_[rem].socket->GetLocalIp()) ==
                            listenSocketMap_.end(), HCCL_ERROR("ip[%s] listenSocketMap is not found",
                            rankId2SocketMap_[rem].socket->GetLocalIp().GetReadableAddress()), HCCL_E_NOT_FOUND);
                        listenSocketMap_[rankId2SocketMap_[rem].socket->GetLocalIp()]->DelWhiteList(
                            rankId2SocketMap_[rem].wlistInfosVec);
                    }
                    rankId2SocketMap_[rem].socket->Close();
                }
                HCCL_INFO("[UnRegisterRanks]group[%s] socket erase remote:%s", group.c_str(), FormatUId(rem).c_str());
                rankId2SocketMap_.erase(rem);
            }
            HCCL_INFO("[UnRegisterRanks]group[%s] status erase remote:%s", group.c_str(), FormatUId(rem).c_str());
        }
        groupMap_.erase(iter);
        HCCL_INFO("[UnRegisterRanks]group[%s] UnregisterRanks Completed.", group.c_str());
    }

    if (groupMap_.size() == 0) {
        HCCL_RUN_INFO("[UnRegisterRanks]Entry HeartBeat DeInit.");
        CHK_RET(DeInit());
    }

    return HCCL_SUCCESS;
}

void Heartbeat::UnRegisterToHeartBeat(DevType devType, const std::string &commIdentifier)
{
    if (Is310PDevice() || devType == DevType::DEV_TYPE_310P3) {
        return;
    }
    ClearRetryEnableMapItem(commIdentifier);
    HcclResult ret = UnRegisterRanks(commIdentifier);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("UnRegisterToHeartBeat failed");
    }
}
void Heartbeat::UnRegisterToHeartBeat(DevType devType, const std::string &commIdentifier, const std::string &tag)
{
    if (Is310PDevice() || devType == DevType::DEV_TYPE_310P3) {
        return;
    }
    ClearRetryEnableMapItem(commIdentifier);
    const std::string groupName = commIdentifier + "_" + tag;
    HcclResult ret = UnRegisterRanks(groupName);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("UnRegisterToHeartBeat failed");
    }
}

UIDType Heartbeat::GetUId(const RankInfo &rankInfo) const
{
    UIDType uid;
    s32 ret = snprintf_s(uid.id,
        sizeof(uid.id),
        sizeof(uid.id) - 1,
        "%s%s%s",
        rankInfo.serverId.c_str(),
        "/",
        std::to_string(rankInfo.devicePhyId).c_str());
    if (ret == -1) {
        HCCL_WARNING("[Heartbeat][GetUId] snprintf_s falied");
    }
    return uid;
}

std::string Heartbeat::FormatUId(const UIDType &uid) const
{
    return uid.id;
}

std::string Heartbeat::GetConnTag(HcclSocketRole role, UIDType &rem)
{
    std::string tag;
    if (role == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        tag = "HeartBeat_" + FormatUId(uid_) + "_to_" + FormatUId(rem);
    } else {
        tag = "HeartBeat_" + FormatUId(rem) + "_to_" + FormatUId(uid_);
    }

    return tag;
}

HcclResult Heartbeat::GetConnInfo(RankInfo &remRank, bool useSuperPodMode, HcclSocketRole role, HcclSocketType type,
    std::map<UIDType, ConnInfo> &needConnectRank)
{
    bool newConn = true;
    UIDType rem = GetUId(remRank);
    {
        std::unique_lock<std::mutex> lock(ProcessLock_);
        if (rankId2SocketMap_.has(rem)) {
            newConn = false;
        }
    }
    std::string tag = GetConnTag(role, rem);
    HcclIpAddress remNicIp;
    if (remRank.nicIp.size() > 0) {
        remNicIp = remRank.nicIp[0];
    }

    if (type == HcclSocketType::SOCKET_NIC && (nicIp_.IsInvalid() || remNicIp.IsInvalid())) {
        HCCL_INFO("No Invalid Nic, Skip");
        return HCCL_SUCCESS;
    }

    u32 remoteDeviceId;
    u32 localDeviceId;
    DeviceIdType deviceIdType;
    if (useSuperPodMode) {
        remoteDeviceId = remRank.superDeviceId;
        localDeviceId = superDeviceId_;
        deviceIdType = DeviceIdType::DEVICE_ID_TYPE_SDID;
    } else {
        remoteDeviceId = remRank.devicePhyId;
        localDeviceId = devicePhyId_;
        deviceIdType = DeviceIdType::DEVICE_ID_TYPE_PHY_ID;
    }

    HcclIpAddress locNicIp = nicIp_;
    if (type == HcclSocketType::SOCKET_VNIC) {
        // 获取本端vnic ip
        locNicIp = HcclIpAddress(localDeviceId);
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(devicePhyId_, deviceIdType, localDeviceId, locNicIp));
        // 获取远端vnic ip
        remNicIp =  HcclIpAddress(remoteDeviceId);
        CHK_RET(hrtRaGetSingleSocketVnicIpInfo(devicePhyId_, deviceIdType, remoteDeviceId, remNicIp));
    }

    u32 port = HCCL_INVALID_PORT;
    if (remRank.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        port = GetHostPort(remoteDeviceId);
    } else {
        port = GetPort(type, remRank.userRank, remoteDeviceId);
    }

    HCCL_INFO("remote userRank[%u], connet port[%u].", remRank.userRank, port);

    std::shared_ptr<HcclSocket> tempSocket;
    std::unique_lock<std::mutex> mapLock(ctxMapMutex_);
    CHK_PRT_RET(netDevCtxMap_.find(locNicIp) == netDevCtxMap_.end(),
        HCCL_ERROR("ip[%s] netDevCtx is not found, socket type[%d]", locNicIp.GetReadableAddress(), type),
        HCCL_E_NOT_FOUND);
    HcclNetDevCtx devCtx = netDevCtxMap_[locNicIp];
    mapLock.unlock();
    ConnInfo conn(newConn, tempSocket);
    if (role == HcclSocketRole::SOCKET_ROLE_SERVER) {
        SocketWlistInfo wlistInfo;
        wlistInfo.connLimit = 1;
        CHK_SAFETY_FUNC_RET(memcpy_s(&wlistInfo.tag[0],sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1));

        wlistInfo.remoteIp.addr = remNicIp.GetBinaryAddress().addr;
        wlistInfo.remoteIp.addr6 = remNicIp.GetBinaryAddress().addr6;
        conn.wlistInfosVec.push_back(wlistInfo);
    }

    EXECEPTION_CATCH((tempSocket = std::make_shared<HcclSocket>(tag,
        devCtx, remNicIp, port, role)), return HCCL_E_PTR);
    CHK_RET(tempSocket->Init());

    conn.socket = tempSocket;

    needConnectRank.insert(std::make_pair(rem, conn));
    return HCCL_SUCCESS;
}

void GetSocketTypeIn91093(const std::vector<RankInfo> &rankInfos, bool useSuperPodMode, u32 index,
    u32 nextOrPrevIndex, HcclSocketType& type)
{
    // 910_93 Type要动态改一下 1. 同server vnic 2. 不同server 超结点内vnic 超结点间nic
    auto locRank = rankInfos[index];
    auto rankInfo = rankInfos[nextOrPrevIndex];
    bool localUseSuporPodModel = useSuperPodMode && locRank.superPodId.empty() == false;
    bool needSuperModeHb =
        localUseSuporPodModel && useSuperPodMode && rankInfo.superPodId.empty() == false;
    if (needSuperModeHb) {
        if (locRank.serverId == rankInfo.serverId) {  // serverId相同表示同超结点同server
            type = HcclSocketType::SOCKET_VNIC;
        } else if (locRank.superPodId == rankInfo.superPodId) {  // 同超结点
            type = (GetExternalInputInterHccsDisable() == true || GetExternalInputInterVnicDisable() == true)
                        ? HcclSocketType::SOCKET_NIC
                        : HcclSocketType::SOCKET_VNIC;
        } else {  // 表示不同超结点
            type = HcclSocketType::SOCKET_NIC;
        }
    }
}

template <typename T>
HcclResult Heartbeat::GetSamePlaneConnInfo(HcclSocketType type, std::vector<std::pair<T, u32>> &connVec, T &locId,
    std::vector<RankInfo> &rankInfos, std::map<UIDType, ConnInfo> &needConnectRank, bool useSuperPodMode, u32 worldRank)
{
    u32 index = 0;
    for (; index < connVec.size(); index++) {
        if (connVec[index].first == locId) {
            break;
        }
    }

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    u32 connCount = connVec.size();
    if (connCount <= 1) {
        HCCL_INFO("nothing need to connect");
    } else if (connCount == 2) {  // 2个rank, 只需建链一条连接
        u32 nextIndex = connVec[(index + 1) % connCount].second;
        if (devType == DevType::DEV_TYPE_910_93) {
            GetSocketTypeIn91093(rankInfos, useSuperPodMode, connVec[index].second, nextIndex, type);
        }
        HCCL_INFO("[GetSamePlaneConnInfo]local rank[%u], remote rank[%u], type[%d]",
            worldRank,
            rankInfos[nextIndex].worldRank,
            type);
        if (index == 0) {
            CHK_RET(GetConnInfo(rankInfos[nextIndex], useSuperPodMode, HcclSocketRole::SOCKET_ROLE_CLIENT,
                type, needConnectRank));
        } else {
            CHK_RET(GetConnInfo(rankInfos[nextIndex], useSuperPodMode, HcclSocketRole::SOCKET_ROLE_SERVER,
                type, needConnectRank));
        }
    } else {
        u32 nextIndex = connVec[(index + 1) % connCount].second;
        if (devType == DevType::DEV_TYPE_910_93) {
            GetSocketTypeIn91093(rankInfos, useSuperPodMode, connVec[index].second, nextIndex, type);
        }
        HCCL_INFO("[GetSamePlaneConnInfo][nextIndex]local rank[%u], remote rank[%u], type[%d]",
            worldRank,
            rankInfos[nextIndex].worldRank,
            type);
        CHK_RET(GetConnInfo(rankInfos[nextIndex], useSuperPodMode, HcclSocketRole::SOCKET_ROLE_CLIENT, type,
            needConnectRank));

        u32 prevIndex = connVec[(index + connCount - 1) % connCount].second;
        if (devType == DevType::DEV_TYPE_910_93) {
            GetSocketTypeIn91093(rankInfos, useSuperPodMode, connVec[index].second, prevIndex, type);
        }
        HCCL_INFO("[GetSamePlaneConnInfo][prevIndex]local rank[%u], remote rank[%u], type[%d]",
            worldRank,
            rankInfos[prevIndex].worldRank,
            type);
        CHK_RET(GetConnInfo(rankInfos[prevIndex], useSuperPodMode, HcclSocketRole::SOCKET_ROLE_SERVER,
            type, needConnectRank));
    }

    return HCCL_SUCCESS;
}

HcclResult Heartbeat::GetConnectRank(const RankInfo &locRank, std::vector<RankInfo> &rankInfos,
    std::map<UIDType, ConnInfo> &needConnectRank, bool useSuperPodMode, bool isUsedRdma)
{
    std::vector<std::pair<u32, u32>> devVec;
    std::vector<std::pair<std::string, u32>> serVec;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));

    for (u32 index = 0; index < rankInfos.size(); index++) {
        auto rankInfo = rankInfos[index];
        if (rankInfo.serverId == locRank.serverId) {
            devVec.push_back(std::make_pair(rankInfo.devicePhyId, index));
        }
        if (rankInfo.devicePhyId == locRank.devicePhyId) {
            serVec.push_back(std::make_pair(rankInfo.serverId , index));
        }
    }
    // server内单环dev排布, 为兼容310P(devId为0, 2, 4...), 扩展为16
    int *ringConfig;
    int ringConfig910A[16] = {0, 3, 1, 2, 7, 4, 6, 5, 4, 6, 2, 0, 3, 1, 5, 7};
    int ringConfig910B[16] = {0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8};

    ringConfig = ringConfig910A;
    if (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_310P3 ||
        devType == DevType::DEV_TYPE_910_93) {
        ringConfig = ringConfig910B;
    }
    std::sort(devVec.begin(), devVec.end(), [&](const std::pair<u32, u32> p1, const std::pair<u32, u32> p2) {
        return ringConfig[p1.first] < ringConfig[p2.first];
    });

    std::sort(serVec.begin(), serVec.end(),
        [](const std::pair<std::string, u32> &p1, const std::pair<std::string, u32> &p2) {
            return p1.first < p2.first;
        });
    u32 locDevId = locRank.devicePhyId;
    u32 worldRank = locRank.worldRank;

    HcclSocketType devSocketType =
        ((devType == DevType::DEV_TYPE_910B) && isUsedRdma) ? HcclSocketType::SOCKET_NIC : HcclSocketType::SOCKET_VNIC;
    CHK_RET(GetSamePlaneConnInfo(devSocketType, devVec, locDevId, rankInfos, needConnectRank,
        useSuperPodMode, worldRank));

    auto nodeId = locRank.serverId;
    CHK_RET(GetSamePlaneConnInfo(HcclSocketType::SOCKET_NIC, serVec, nodeId, rankInfos, needConnectRank,
        useSuperPodMode, worldRank));
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::SendFrame(UIDType &dst, UIDType &crimer, UIDType &informer, HeartBeatStatus status)
{
    HeartBeatFrame bf(uid_, dst, crimer, informer, status);
    if (rankId2SocketMap_[dst].sendBuffer.size() > 0) {
        if (status != HeartBeatStatus::HEARTBEAT_OK) {
            rankId2SocketMap_[dst].sendBuffer.push(bf);
        }
        while (rankId2SocketMap_[dst].sendBuffer.size() > 0) {
            HeartBeatFrame bf = rankId2SocketMap_[dst].sendBuffer.front();
            u64 sendDis = sizeof(HeartBeatFrame) - rankId2SocketMap_[dst].restSize;
            u64 compSize = 0;
            HcclResult ret = rankId2SocketMap_[dst].socket->ISend(
                reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(&bf) + sendDis),
                rankId2SocketMap_[dst].restSize,
                compSize);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            if (rankId2SocketMap_[dst].restSize == compSize) {
                rankId2SocketMap_[dst].sendBuffer.pop();
                rankId2SocketMap_[dst].restSize = sizeof(HeartBeatFrame);
                HCCL_DEBUG("[Heartbeat][SendFrame] Send Success, from [%s] to [%s] about [%s] by [%s] status[%d]",
                    FormatUId(uid_).c_str(),
                    FormatUId(dst).c_str(),
                    FormatUId(crimer).c_str(),
                    FormatUId(informer).c_str(),
                    status);
            } else {
                rankId2SocketMap_[dst].restSize = rankId2SocketMap_[dst].restSize - compSize;
                break;
            }
        }
    } else {
        u64 compSize = 0;
        u32 expectSize = sizeof(HeartBeatFrame);
        HcclResult ret = rankId2SocketMap_[dst].socket->ISend(&bf, expectSize, compSize);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        if (compSize == expectSize) {
            HCCL_DEBUG("[Heartbeat][SendFrame] Send Success, from [%s] to [%s] about [%s] by [%s] status[%d]",
                FormatUId(uid_).c_str(),
                FormatUId(dst).c_str(),
                FormatUId(crimer).c_str(),
                FormatUId(informer).c_str(),
                status);
        } else {
            HCCL_WARNING("[Heartbeat][SendFrame] Send Not Complete, from [%s] to [%s] about [%s] by [%s] status[%d]",
                FormatUId(uid_).c_str(),
                FormatUId(dst).c_str(),
                FormatUId(crimer).c_str(),
                FormatUId(informer).c_str(),
                status);
            rankId2SocketMap_[dst].restSize = expectSize - compSize;
            rankId2SocketMap_[dst].sendBuffer.push(bf);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult Heartbeat::RecvFrame(UIDType &src)
{
    HeartBeatFrame bf;
    u64 compSize = 0;
    u64 expectSize = sizeof(HeartBeatFrame);
    while (true) {
        compSize = 0;
        HcclResult retVal = rankId2SocketMap_[src].socket->IRecv(&bf, expectSize, compSize);
        if (retVal == HCCL_SUCCESS && compSize > 0) {
            rankId2SocketMap_[src].recvBuffer.PushSeg(reinterpret_cast<u8 *>(&bf), compSize);
            if (rankId2SocketMap_[src].recvBuffer.Size() >= expectSize) {
                rankId2SocketMap_[src].recvBuffer.GetSeg(reinterpret_cast<u8 *>(&bf), expectSize);
                rankId2SocketMap_[src].recvBuffer.PopSeg(expectSize);
                CHK_RET(ParseFrame(bf, src));
            }
        } else if (retVal == HCCL_E_INTERNAL) {
            return HCCL_E_INTERNAL;
        } else {
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::ParseFrame(HeartBeatFrame &bf, UIDType &src)
{
    if (bf.src != src || bf.dst != uid_) {
        HCCL_WARNING("rank[%s] recv wrong frame", FormatUId(uid_).c_str());
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[Heartbeat][RecvFrame] Recv Success, from [%s] to [%s] about [%s] by [%s] state[%d]",
        FormatUId(bf.src).c_str(),
        FormatUId(bf.dst).c_str(),
        FormatUId(bf.crimer).c_str(),
        FormatUId(bf.informer).c_str(),
        bf.status);

    // 能够收到进程卡住表示心跳是正常的
    if (bf.status == HeartBeatStatus::HEARTBEAT_OK || bf.status == HeartBeatStatus::HEARTBEAT_STUCK) {
        rankId2SocketMap_[src].lostNum = 0;
    }

    // 只有心跳非正常时才需要打印TRACE
    if (bf.status != HeartBeatStatus::HEARTBEAT_OK) {
        SetStatus(bf.crimer, bf.informer, bf.status);
    }
    return HCCL_SUCCESS;
}

void Heartbeat::SetStatus(UIDType &crimer, UIDType &informer, HeartBeatStatus status, bool needBroadcast)
{
    if (rankId2StatusMap_[crimer].status != status) {
        rankId2StatusMap_[crimer].informer = informer;
        rankId2StatusMap_[crimer].status = status;
        rankId2StatusMap_[crimer].needBroadcast = needBroadcast;
        if (needBroadcast) {
            errRankQueue_.push(crimer);
        }

        errStatusQueue_.push(HeartBeatFrame(crimer, informer, status, TIME_NOW(), std::chrono::system_clock::now()));
        if (errStatusQueue_.size() > EVENT_MAX_CNT) {
            errStatusQueue_.pop();
        }
        HCCL_RUN_INFO("[HCCL_TRACE]rank [%s]: rank [%s] status[%s] by rank [%s]",
            FormatUId(uid_).c_str(),
            FormatUId(crimer).c_str(),
            GetHeartBeatStatusStr(status).c_str(),
            FormatUId(informer).c_str());
    }
}
bool Heartbeat::IsKeyEvent(HeartBeatFrame &event, HcclUs curTime)
{
    bool ret = false;
    s64 intervalTime = DURATION_US(curTime - event.TOARelative).count() / (TIME_S_TO_MS * ONE_MILLISECOND_OF_USLEEP);
    s64 execTimeout = GetExternalInputHcclExecTimeOut();
    s64 detectionTime = 0;
    switch (event.status) {
        case HeartBeatStatus::HEARTBEAT_LOST:
            detectionTime = (lostThreshold_ * HEARTBEAT_INTERVAL) / TIME_S_TO_MS;
            break;
        case HeartBeatStatus::HEARTBEAT_CQE_ERR:
        case HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT:
            detectionTime = 0;
            break;
        case HeartBeatStatus::HEARTBEAT_STUCK:
            detectionTime = 2 * GetExternalInputStuckDetectTime();  // 最长探测时间为2倍的卡住检测时间
            break;
        case HeartBeatStatus::HEARTBEAT_NOTIFY:
        default:
            return false;  // 当前不支持的事件，不做处理和展现
    }
    ret = ((execTimeout - intervalTime - detectionTime) < JITTER_TIME) &&
          ((intervalTime + detectionTime - execTimeout) < JITTER_TIME);
    return ret;
}

void Heartbeat::MakeErrMsg(std::queue<HeartBeatFrame> &keyEvents, std::vector<std::string> &errStatusVec)
{
    while (keyEvents.size() > 0) {
        auto &tmp = keyEvents.front();
        std::string crimerStr = FormatUId(tmp.crimer);
        std::string informerStr = FormatUId(tmp.informer);

        std::string headStr = "Cluster Exception Location[IP/ID]:[";

        time_t tm = std::chrono::system_clock::to_time_t(tmp.TOASystem);
        std::string timeStr(ctime(&tm));
        if (!timeStr.empty()) {  // ctime()函数自带换行符，需要去掉
            timeStr.pop_back();
        }
        timeStr = ", Arrival Time:[" + timeStr + "]";

        std::string errStr = ", ExceptionType:";
        std::string reasonStr = ", Possible Reason:";
        switch (tmp.status) {
            case HeartBeatStatus::HEARTBEAT_LOST:
                errStr = errStr + "[Heartbeat Lost Occurred]";
                reasonStr = reasonStr + "1. Process has exited, 2. Network Disconnected";
                errStr =
                    headStr + crimerStr + "]" + timeStr + ", Discoverer:[" + informerStr + "]" + errStr + reasonStr;
                break;
            case HeartBeatStatus::HEARTBEAT_NOTIFY:
                errStr = errStr + "[Notify Wait Error Occurred]";
                errStr = headStr + crimerStr + "]" + timeStr + errStr;
                break;
            case HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT:
                errStr = errStr + "[OpRetry Not Supported Occurred]";
                reasonStr = reasonStr + "OpRetry is not supported";
                errStr = headStr + crimerStr + "]" + timeStr + errStr + reasonStr;
                break;
            case HeartBeatStatus::HEARTBEAT_CQE_ERR:
                errStr = errStr + "[Error cqe Occurred]";
                reasonStr = reasonStr + "1.Network Disconnected, 2.Remote Rank Coredown";
                errStr = headStr + crimerStr + "]" + timeStr + errStr + reasonStr;
                break;
            case HeartBeatStatus::HEARTBEAT_STUCK:
                errStr = errStr + "[Stuck Occurred]";
                reasonStr = reasonStr + "1.Host process is stuck, 2.Device task is stuck";
                errStr = headStr + crimerStr + "]" + timeStr + errStr + reasonStr;
                break;
            default:
                errStr = " Unknown";
        }
        errStatusVec.emplace_back(errStr);
        keyEvents.pop();
    }
}
std::vector<std::string> Heartbeat::PrintEvents(std::map<HeartBeatStatus, std::queue<HeartBeatFrame>> &keyEvents)
{
    std::vector<std::string> errStatusVec;
    // 打印优先级 opretry not support > error cqe > stuck > lost
    MakeErrMsg(keyEvents[HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT], errStatusVec);
    MakeErrMsg(keyEvents[HeartBeatStatus::HEARTBEAT_CQE_ERR], errStatusVec);
    MakeErrMsg(keyEvents[HeartBeatStatus::HEARTBEAT_STUCK], errStatusVec);
    MakeErrMsg(keyEvents[HeartBeatStatus::HEARTBEAT_LOST], errStatusVec);
    return errStatusVec;
}
std::vector<std::string> Heartbeat::GetErrStatusVec()
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    HcclUs curTime = TIME_NOW();
    std::map<HeartBeatStatus, std::queue<HeartBeatFrame>> keyEvents;
    while (errStatusQueue_.size() > 0) {
        auto &tmp = errStatusQueue_.front();
        if (IsKeyEvent(tmp, curTime)) {  // 非关键事件不处理
            keyEvents[tmp.status].push(tmp);
        }
        errStatusQueue_.pop();
    }
    return PrintEvents(keyEvents);
}

void Heartbeat::ProcessExceptionEvent()
{
    while (errRankQueue_.size() > 0) {
        UIDType cur = errRankQueue_.front();
        rankId2StatusMap_[cur].needBroadcast = false;
        for (auto iterRem = rankId2SocketMap_.begin(); iterRem != rankId2SocketMap_.end(); iterRem++) {
            UIDType rem = iterRem->first;
            if (rem != rankId2StatusMap_[cur].informer &&
                rankId2StatusMap_[rem].status == HeartBeatStatus::HEARTBEAT_OK) {
                (void)SendFrame(rem, cur, rankId2StatusMap_[cur].informer, rankId2StatusMap_[cur].status);
            }
        }
        errRankQueue_.pop();
    }
}

void Heartbeat::HeartbeatStatusMonitor()
{
    // 给当前线程添加名字
    SetThreadName("Hccl_HeartBeat");

    u32 count = 0;
    if (deviceLogicId_ != static_cast<u32>(HOST_DEVICE_ID)) {
        hrtSetDevice(deviceLogicId_);
    }
    uint64_t cnt = 0;
    HcclResult ret;
    auto counterStat = CounterStat();
    InitStuckDetection(counterStat);
    while (startSendRecvTask_) {
        ProcessLock_.lock();
        count++;
        if (count >= HEARTBEAT_COUNT) {
            count = 0;
            for (auto iter = rankId2SocketMap_.begin(); iter != rankId2SocketMap_.end(); iter++) {
                UIDType rem = iter->first;
                HCCL_DEBUG(
                    "rank[%s] Try to Send HeartBeat to rank[%s]", FormatUId(uid_).c_str(), FormatUId(rem).c_str());
                rankId2SocketMap_[rem].lostNum++;
                ret = SendFrame(rem,
                    uid_,
                    uid_,
                    (counterStat.issueCnt != 0) ? HeartBeatStatus::HEARTBEAT_STUCK : HeartBeatStatus::HEARTBEAT_OK);
                ret == HCCL_E_INTERNAL ? errorSocket_.push_back(rem) : void(0);
            }
            DelErrorSocket();
            ProcessCqeErrInfo();
        }

        for (auto iter = rankId2SocketMap_.begin(); iter != rankId2SocketMap_.end(); iter++) {
            UIDType rem = iter->first;
            HCCL_DEBUG("rank[%s] Try to Recv from rank[%s]", FormatUId(uid_).c_str(), FormatUId(rem).c_str());
            ret = RecvFrame(rem);
            if (ret == HCCL_E_INTERNAL) {
                errorSocket_.push_back(rem);
            } else if (rankId2SocketMap_[rem].lostNum >= lostThreshold_) {
                SetStatus(rem, uid_, HeartBeatStatus::HEARTBEAT_LOST);
            }
        }
        DelErrorSocket();
        StuckDetection(cnt, counterStat);
        ProcessExceptionEvent();
        ProcessLock_.unlock();

        std::this_thread::sleep_for(std::chrono::milliseconds(BROADCAST_INTERVAL));
    }
    if (deviceLogicId_ != static_cast<u32>(HOST_DEVICE_ID)) {
        hrtResetDevice(deviceLogicId_);
    }
}

void Heartbeat::InitStuckDetection(CounterStat &counterStat)
{
    counterStat.isNeedDetect = (GetExternalInputOpCounter() == true) ? true : false;
    counterStat.couterPrintInter = GetExternalInputStuckDetectTime() * THROUND_MILS / BROADCAST_INTERVAL;
}

void Heartbeat::StuckDetection(uint64_t &cnt, CounterStat &counterStat)
{
    HCCL_DEBUG("cnt: %d, isNeedDetect: %d, issueCnt:%llu, interTimes:%d",
        cnt,
        counterStat.isNeedDetect,
        counterStat.issueCnt,
        counterStat.couterPrintInter);
    cnt++;
    HcclResult ret = HCCL_SUCCESS;
    if (counterStat.isNeedDetect && cnt % counterStat.couterPrintInter == 0) {
        if (counterStat.isFirst) {
            OpExeCounter::GetInstance(deviceLogicId_).GetCounter(counterStat.oldCounter);
            counterStat.isFirst = false;
        } else {
            ret = OpExeCounter::GetInstance(deviceLogicId_).GetCounter(counterStat.newCounter);
            if (ret == HCCL_SUCCESS && counterStat.newCounter.first == counterStat.oldCounter.first &&
                counterStat.newCounter.first == counterStat.oldCounter.second &&
                counterStat.newCounter.first == counterStat.newCounter.second) {
                HCCL_RUN_INFO("[HCCL_TRACE]rank:%s, count of currently executed operators:%d",
                    FormatUId(uid_).c_str(),
                    counterStat.newCounter.first);
                counterStat.couterPrintInter *= (BASE_NUMBER << counterStat.issueCnt);  // 检测卡住后，把检测周期放长
                counterStat.issueCnt++;
            } else {
                // 检测不卡之后，检测间隔恢复到默认间隔
                counterStat.couterPrintInter = GetExternalInputStuckDetectTime() * THROUND_MILS / BROADCAST_INTERVAL;
                counterStat.issueCnt = 0;
            }
            counterStat.oldCounter = counterStat.newCounter;  // 更新旧的计数器
        }
    }
}

void Heartbeat::PrintAndBroadCastErrorCqe(const ErrCqeInfo &info)
{
    time_t tmpt;
    struct tm *now;
    if (info.cqeInfo.status == 0) {
        return;
    }

    SetStatus(uid_, uid_, HeartBeatStatus::HEARTBEAT_CQE_ERR);
    tmpt = static_cast<time_t>(info.cqeInfo.time.tv_sec);
    now = localtime(&tmpt);
    if (now == nullptr) {
        HCCL_ERROR("[Heartbeat]localtime fail, cqe error status[%u],"
            "localIP[%s], remoteIP:[%s]", info.cqeInfo.status, nicIp_.GetReadableAddress(),
            info.cqeInfo.remoteIp.GetReadableAddress());
    } else {
        HCCL_ERROR("[Heartbeat]cqe error status[%u], time:[%04u-%02d-%02d %02d:%0d:%02d.%06u],"
                "localIP[%s], remoteIP:[%s]",
            info.cqeInfo.status,
            now->tm_year + TIME_FROM_1900,
            now->tm_mon + 1,
            now->tm_mday,
            now->tm_hour,
            now->tm_min,
            now->tm_sec,
            static_cast<u32>(info.cqeInfo.time.tv_usec),
            nicIp_.GetReadableAddress(),
            info.cqeInfo.remoteIp.GetReadableAddress());
    }

    std::unique_lock<std::mutex> lock(remoteIpMutex_);
    auto search = remoteIpMap.find(info.identifier);
    if (search != remoteIpMap.end()) {
        remoteIpMap[info.identifier].insert(info.cqeInfo.remoteIp);
    } else {
        std::set<HcclIpAddress> remoteIpSet;
        remoteIpSet.insert(info.cqeInfo.remoteIp);
        remoteIpMap.insert(std::pair<std::string, std::set<HcclIpAddress>>(info.identifier, remoteIpSet));
    }
}

void Heartbeat::SaveQpnForOpRetry(const ErrCqeInfo &info)
{
    if (info.cqeInfo.status == 0) {
        return;
    }

    HCCL_RUN_INFO("[Heartbeat][SaveQpnForOpRetry]receive a cqe error [%u][%u], dstrank[%u] identifier[%s]",
            info.cqeInfo.status, info.qpn, info.remoteRank, info.identifier.c_str());
    auto identiSearch = rankMapForRetryAgent.find(info.identifier);
    if (identiSearch != rankMapForRetryAgent.end()) {
        auto rankSearch = identiSearch->second.find(info.remoteRank);
        if (rankSearch != identiSearch->second.end()) {
            (*rankSearch).second.insert(ErrQpnInfo(info.cqeInfo, info.qpn));
        } else {
            identiSearch->second.insert({info.remoteRank, {ErrQpnInfo(info.cqeInfo, info.qpn)}});
        }
    } else {
        std::map<u32, std::set<ErrQpnInfo>> rankExtendMap;
        rankExtendMap[info.remoteRank] = {ErrQpnInfo(info.cqeInfo, info.qpn)};
        rankMapForRetryAgent.insert(std::make_pair(info.identifier, rankExtendMap));
    }
}

void Heartbeat::OpRetryCQEHandle(const HcclNetDevCtx netDevCtx)
{
    u32 cqeNum = RETRY_CQE_ARRAY_SIZE;
    do {
        cqeNum = RETRY_CQE_ARRAY_SIZE;

        std::vector<ErrCqeInfo> infos;
        HcclResult ret = HcclCommunicator::GetTransportCqeErrors(netDevCtx, infos, cqeNum);
        if (ret != HCCL_SUCCESS || cqeNum == 0) {
            return;
        }
        for (auto &info : infos) {
            if (!GetRetryEnable(info)) {
                PrintAndBroadCastErrorCqe(info);
            } else {
                SaveQpnForOpRetry(info);
            }
        }
    } while (cqeNum == RETRY_CQE_ARRAY_SIZE);
}


bool Heartbeat::GetRetryEnable(const ErrCqeInfo &info)
{
    std::lock_guard<std::mutex> retryEnablelock(retryEnableMutex_);
    auto search = retryEnableTable_.find(info.identifier);
    if (search != retryEnableTable_.end()) {
        return search->second;
    }
    return false;
}
HcclResult Heartbeat::ClearRetryEnableMapItem(const std::string &identifier)
{
    CHK_PRT_RET(initialized_ == false, HCCL_WARNING("Heartbeat has been destroyed"), HCCL_SUCCESS);
    u32 delRes = 0;
    {
        std::lock_guard<std::mutex> retryEnablelock(retryEnableMutex_);
        delRes = retryEnableTable_.erase(identifier);
        if (delRes != 0) {
            HCCL_INFO("[Heartbeat][ClearRetryEnableMapItem] del identifier[%s] succ", identifier.c_str());
        } else {
            HCCL_DEBUG("[Heartbeat][ClearRetryEnableMapItem] identifier[%s] is not found.", identifier.c_str());
        }
    }
    std::lock_guard<std::mutex> bakcupEnablelock(backupEnableMutex_);
    delRes = backupEnableTable_.erase(identifier);
    if (delRes != 0) {
        HCCL_INFO("[Heartbeat][ClearRetryEnableMapItem] del backup identifier[%s] succ", identifier.c_str());
    } else {
        HCCL_DEBUG("[Heartbeat][ClearRetryEnableMapItem] identifier[%s] is not found.", identifier.c_str());
    }
    return HCCL_SUCCESS;
}
void Heartbeat::ProcessCqeErrInfoByNetDevCtx(const HcclIpAddress &nicIp)
{
    std::unique_lock<std::mutex> mapLock(ctxMapMutex_);
    auto iter = netDevCtxMap_.find(nicIp);
    if (iter == netDevCtxMap_.end() || netDevCtxMap_[nicIp] == nullptr) {
        return;
    }
    mapLock.unlock();
    const HcclNetDevCtx netDevCtx = iter->second;
    std::vector<ErrCqeInfo> infos;
    u32 cqeNum = 1;
    HcclResult ret = HcclCommunicator::GetTransportCqeErrors(netDevCtx, infos, cqeNum);
    if (ret != HCCL_SUCCESS || infos.size() == 0) {
        return;
    }
    if (!GetRetryEnable(infos[0])) {
        PrintAndBroadCastErrorCqe(infos[0]);
    } else {
        SaveQpnForOpRetry(infos[0]);
    }
    // infoList 处理
    OpRetryCQEHandle(netDevCtx);
}

void Heartbeat::ProcessCqeErrInfo()
{
    ProcessCqeErrInfoByNetDevCtx(nicIp_);
    if (IsEnableBackupLink()) {
        ProcessCqeErrInfoByNetDevCtx(backupNicIp_);
    }
}

void Heartbeat::DelErrorSocket()
{
    for (auto rem : errorSocket_) {
        HCCL_RUN_INFO(
            "rank[%s] Try to Send/recv HeartBeat to rank[%s]", FormatUId(uid_).c_str(), FormatUId(rem).c_str());
        rankId2StatusMap_.erase(rem);
        if(rankId2SocketMap_.has(rem)) {
            if (rankId2SocketMap_[rem].socket->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_SERVER &&
                listenSocketMap_.find(rankId2SocketMap_[rem].socket->GetLocalIp()) != listenSocketMap_.end()) {
                listenSocketMap_[rankId2SocketMap_[rem].socket->GetLocalIp()]->DelWhiteList(
                    rankId2SocketMap_[rem].wlistInfosVec);
            }
            rankId2SocketMap_[rem].socket->Close();
            while (rankId2SocketMap_.erase(rem)) {};
        }
    }
    errorSocket_.clear();
}

HcclResult Heartbeat::GetQpnErr(const std::string &identifier, std::set<std::tuple<u32, u32, u32>> &qpErrSet)
{
    std::unique_lock<std::mutex> lock(qpnMapMutexForRetry_);
    auto search = rankMapForRetryAgent.find(identifier);
    if (search == rankMapForRetryAgent.end()) {
        HCCL_INFO("[GetQpnErr]identifier[%s] is not found", identifier.c_str());
        return HCCL_SUCCESS;
    }
    if (search->second.size() > 0) {
        for (auto iter: search->second) {
            u32 dstRank = iter.first;
            for (auto qpnInfo: iter.second){
                u32 status = qpnInfo.cqeInfo.status;
                qpErrSet.insert(std::make_tuple(dstRank, status, qpnInfo.qpn));
            }
        }
    }
    HCCL_INFO("[GetQpnErr]identifier[%s] is found, qpErrSet size is %u", identifier.c_str(), qpErrSet.size());
    return HCCL_SUCCESS;
}
// OpRetry 失败后，将进行广播操作
HcclResult Heartbeat::BroadcastCqeErr(const std::string &identifier)
{
    u32 cqeSize = 0;
    std::unique_lock<std::mutex> qpnMaplock(qpnMapMutexForRetry_);
    auto search = rankMapForRetryAgent.find(identifier);
    if (search != rankMapForRetryAgent.end()) {
        if (search->second.size() > 0) {
            cqeSize = search->second.size();
            for (auto &qpInfo: search->second) {
                for (auto qpnset:qpInfo.second) {
                    PrintAndBroadCastErrorCqe(ErrCqeInfo(qpnset.cqeInfo, identifier, qpInfo.first, qpnset.qpn));
                    HCCL_RUN_INFO("[BroadcastCqeErr][item]remoteIp[%s] remoteRank[%u] status[%u] qpn[%u]",
                        qpnset.cqeInfo.remoteIp.GetReadableAddress(), qpInfo.first, qpnset.cqeInfo.status, qpnset.qpn);
                }
            }
            search->second.clear();
        }
    }
    // 查询剩余量，一般为0
    HCCL_RUN_INFO("[Heartbeat][BroadcastCqeErr]clear qpn err size from [%u] to [%u], identifier[%s] ",
        cqeSize, search->second.size(), identifier.c_str());
    return HCCL_SUCCESS;
}

/* 非点对点通信 重执行成功后进行调用*/
HcclResult Heartbeat::ClearAllCqeErr(const std::string &identifier)
{
    std::unique_lock<std::mutex> qpnMaplock(qpnMapMutexForRetry_);
    u32 cqeSize = 0;
    auto search = rankMapForRetryAgent.find(identifier);
    if (search != rankMapForRetryAgent.end()) {
        if (search->second.size() > 0) {
            cqeSize = search->second.size();
            search->second.clear();
        }
    }
    // 查询剩余量，一般为0
    HCCL_RUN_INFO("[Heartbeat][ClearAllCqeErr]clear qpn err size from [%u] to [%u], identifier[%s]",
        cqeSize, search->second.size(), identifier.c_str());
    return HCCL_SUCCESS;
}
/* 点对点通信 重执行成功后进行调用
*/
HcclResult Heartbeat::ClearCqeErr(const std::string &identifier, u32 remoteRank, u32 qpn)
{
    HCCL_RUN_INFO("[Heartbeat][ClearCqeErr] identifier[%s] remoteRank[%u] qpn[%u].", 
        identifier.c_str(), remoteRank, qpn);
    std::unique_lock<std::mutex> qpnMaplock(qpnMapMutexForRetry_);
    const auto &search = rankMapForRetryAgent.find(identifier);
    if (search == rankMapForRetryAgent.end()) {
        return HCCL_SUCCESS;
    }

    auto &ranksearch = rankMapForRetryAgent[identifier];
    // 删除指定通信域内的固定 remoteRank 固定qpn的cqe err
    if (ranksearch.find(remoteRank) != ranksearch.end()){
        if (ranksearch[remoteRank].size() == 1){
            //remotrank只有一个qpn err，直接删除map
            ranksearch.erase(remoteRank);
            HCCL_RUN_INFO("[ClearCqeErr][qpnClear] clear dstRank[%u] qpn[%u] now", remoteRank, qpn);
        } else if (ranksearch[remoteRank].size() > 1){
            for (auto iter = ranksearch[remoteRank].begin(); iter !=ranksearch[remoteRank].end();) {
                if (iter->qpn == qpn) {
                    iter = ranksearch[remoteRank].erase(iter);
                    HCCL_RUN_INFO("[ClearCqeErr][qpnClear] clear dstRank[%u] qpn[%u] now", remoteRank, qpn);
                } else {
                    ++iter;
                }
            }
        }
    }

    // 查询指定通信域剩余的 QP ERROR 数量
    HCCL_RUN_INFO("[ClearCqeErr][qpnClear]identifier qpn err left [%u] now.", search->second.size());
    return HCCL_SUCCESS;
}

HcclResult Heartbeat::CheckErrorCqe(const std::string &identifier, HcclResult &result)
{
    HcclIpAddress ip;
    result = HCCL_SUCCESS;

    std::unique_lock<std::mutex> lock(remoteIpMutex_);
    auto search = remoteIpMap.find(identifier);
    if (search == remoteIpMap.end()) {
        if (qpnDissociativeSet.size()!=0) { // 如果没有发生error cqe异常的通信域，则确认是否存在游离(被提前destory的)qpn
            HCCL_ERROR("[Heartbeat]find cqe error [%d] num[%llu] dissociative. maybe its qp has already been destoryed",
                result, qpnDissociativeSet.size());
            qpnDissociativeSet.clear();
            return HCCL_E_REMOTE;
        }
        return HCCL_SUCCESS;
    }
    if (search->second.size() > 0) {
        result = HCCL_E_REMOTE;
        HCCL_ERROR("[Heartbeat]find cqe error [%d], in comm [%s]", result, identifier.c_str());
        for (auto &it : search->second) {
            HCCL_ERROR("[Heartbeat]find cqe error, localIP[%s], remoteIP[%s]",
                nicIp_.GetReadableAddress(), it.GetReadableAddress());
        }
    }
    lock.unlock();

    return HCCL_SUCCESS;
}

HcclResult Heartbeat::SetRankPortInfo(bool isUseRankPort,
    std::vector<u32> &nicRanksPorts, std::vector<u32> &vnicRanksPorts, bool devPortSwitchOn)
{
    isUseRankPort_ = isUseRankPort;
    nicRanksPorts_ = nicRanksPorts;
    vnicRanksPorts_ = vnicRanksPorts;
    devPortSwitchOn_ = devPortSwitchOn;
    return HCCL_SUCCESS;
}

void Heartbeat::SetOpretryErr()
{
    // 重执行约束场景，给errStatusQueue添加重执行失败心跳帧
    SetStatus(uid_, uid_, HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT);
}

u32 Heartbeat::GetPort(HcclSocketType type, u32 remoteUserRank, u32 remoteDeviceId)
{
    u32 port = HCCL_INVALID_PORT;
    if (isUseRankPort_) {
        if (devPortSwitchOn_ && type == HcclSocketType::SOCKET_VNIC
            && remoteUserRank < vnicRanksPorts_.size() && vnicRanksPorts_[remoteUserRank] != HCCL_INVALID_PORT) {
            port = vnicRanksPorts_[remoteUserRank];
            HCCL_INFO("[Heartbeat][GetPort] use vnic ranks port[%u]", port);
        } else if (remoteUserRank < nicRanksPorts_.size() && nicRanksPorts_[remoteUserRank] != HCCL_INVALID_PORT) {
            port = nicRanksPorts_[remoteUserRank];
            HCCL_INFO("[Heartbeat][GetPort] use nic ranks port[%u]", port);
        } else {
            port = HETEROG_CCL_PORT;
        }
    } else if (!GetExternalInputHcclDeviceNicDisable() && !GetExternalInputHcclHostRdmaEnable()) {
            port = HETEROG_CCL_PORT;
    } else {
            port = GetHostPort(remoteDeviceId);
    }
    return port;
}

u32 Heartbeat::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

HcclResult RegisterToHeartBeat(s32 deviceLogicID, u32 userRank, DevType devType, std::vector<RankInfo> &rankInfoList,
    const u32 port, const bool isNeedNic, u32 peerRankId, const std::string &commIdentifier, const std::string& tag,
    bool useSuperPodMode, bool isUsedRdmaLevel0)
{
    return peerRankId == INVALID_VALUE_RANKID ?
        Heartbeat::GetInstance(deviceLogicID).RegisterToHeartBeat(userRank, devType, rankInfoList, port, isNeedNic,
        commIdentifier, useSuperPodMode, isUsedRdmaLevel0) :
        Heartbeat::GetInstance(deviceLogicID).RegisterToHeartBeat(userRank, devType, rankInfoList, port, isNeedNic,
        peerRankId, commIdentifier, tag, useSuperPodMode, isUsedRdmaLevel0);
}

void UnRegisterRanks(s32 deviceLogicID, DevType devType, const std::string &commIdentifier,
    const std::string &tag)
{
    return tag.empty() ?
        Heartbeat::GetInstance(deviceLogicID).UnRegisterToHeartBeat(devType, commIdentifier) :
        Heartbeat::GetInstance(deviceLogicID).UnRegisterToHeartBeat(devType, commIdentifier, tag);
}

HcclResult SetRankPortInfo(s32 deviceLogicID, bool isUseRankPort, std::vector<u32> &ranksPort)
{
    return Heartbeat::GetInstance(deviceLogicID).SetRankPortInfo(isUseRankPort, ranksPort, ranksPort, false);
}

std::vector<std::string> GetErrStatusVec(s32 deviceLogicID)
{
    return Heartbeat::GetInstance(deviceLogicID).GetErrStatusVec();
}

__attribute__((constructor)) void HeartBeatCallBackInit()
{
    RegisterHeartBeatCallBack(RegisterToHeartBeat, UnRegisterRanks, SetRankPortInfo);
    RegisterGetErrStatusVecCallBack(GetErrStatusVec);
}
}  // namespace hccl
