/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator_attrs.h"
#include "device_capacity.h"
#include "config.h"
#include "externalinput_pub.h"
#include "env_config.h"

using namespace std;

namespace hccl {
HcclCommunicatorAttrs::HcclCommunicatorAttrs()
{
}
HcclCommunicatorAttrs::~HcclCommunicatorAttrs()
{
}
HcclResult HcclCommunicatorAttrs::Init(HcclCommParams &params, const RankTable_t &rankTable)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfo(rankTable));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
        WorldGroupInfo &groupCommonData)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(InitRankInfoSubGroup(rankList, groupCommonData));
    return HCCL_SUCCESS;
}
bool HcclCommunicatorAttrs::IsStandardCard()
{
    if (Is310P3Common()) {
        HCCL_INFO("The current device just support this StandardCard case.");
        return true;
    }
    return ((pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) &&
           (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size() == 0) &&
           (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size() == 0));
}
bool HcclCommunicatorAttrs::Is310PDuoCard()
{
    return (Is310P3Common() && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == userRankSize_));
}
bool HcclCommunicatorAttrs::IsCommon310P3DUO(const std::vector<RankInfo_t> &rankList)
{
    std::vector<u32> devIdList;
    std::vector<std::vector<u32>> checkDevList;
    checkDevList.resize(FACTOR_NUM_TWO);

    for (RankInfo_t rankInfo : rankList) {
        u32 curId = rankInfo.deviceInfo.devicePhyId;
        devIdList.push_back(curId);
    }
    if (devIdList.size() == DEVICE_PER_MODULE) {
        return true;
    }
    std::sort(devIdList.begin(), devIdList.end());
    for (u32 i = 0; i < devIdList.size(); i++) {
        if (devIdList[i] % FACTOR_NUM_TWO == 0) {
            checkDevList[0].push_back(devIdList[i]); // 主die
        } else {
            checkDevList[1].push_back(devIdList[i]); // 从die
        }
    }
    if (devIdList.size() == (DEVICE_PER_MODULE / FACTOR_NUM_TWO) && checkDevList[0].size() == checkDevList[1].size()) {
        return (checkDevList[1][0] - checkDevList[0][0]) == 1 &&
            (checkDevList[1][1] - checkDevList[0][1]);
    } else {
        return false;
    }
}
bool HcclCommunicatorAttrs::Is310P3Common()
{
    return !isHaveCpuRank_ && !Is310PDevice() && deviceType_ == DevType::DEV_TYPE_310P3;
}
bool HcclCommunicatorAttrs::CompareWithUserRank(const RankInfo &left, const RankInfo &right)
{
    return left.userRank < right.userRank;
}
HcclResult HcclCommunicatorAttrs::CheckDeviceType(const DevType deviceType) const
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR("[Check][DeviceType]errNo[0x%016llx] deivce Type[%d] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::GetNicInfo(const NICDeployment &nicDeploy, const u32 curRankIndex,
    const std::vector<RankInfo_t> &servRankList, RankInfo &rankInfo) const
{
    CHK_PRT_RET(servRankList.empty(), HCCL_ERROR("[Get][NicInfo]errNo[0x%016llx] server rank list is empty",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    rankInfo.nicDeploy = nicDeploy;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        // 检查网卡个数
        // 网卡挂载位置在host时，按rank index从网卡列表中获取
        const RankInfo_t &curRankInfo = servRankList[curRankIndex];
        rankInfo.nicIp.push_back(curRankInfo.hostIp);
    } else {
        CHK_PRT_RET(curRankIndex >= servRankList.size(), HCCL_ERROR("[Get][NicInfo]rankindex[%u] invalid,rank list "\
            "size is[%zu]", curRankIndex, servRankList.size()), HCCL_E_PARA);

        const RankInfo_t &curRankInfo = servRankList[curRankIndex];
        CHK_PRT_RET(curRankInfo.deviceInfo.deviceIp.size() == 0,
            HCCL_ERROR("[Get][NicInfo]rankindex[%u] invalid,deviceIp is zero", curRankIndex), HCCL_E_PARA);
        rankInfo.nicIp.push_back(curRankInfo.deviceInfo.deviceIp[0]);
        if (curRankInfo.deviceInfo.backupDeviceIp.size() == 0) {
            HcclIpAddress invalidAddr;
            rankInfo.backupNicIp.push_back(invalidAddr);
        } else {
            rankInfo.backupNicIp.push_back(curRankInfo.deviceInfo.backupDeviceIp[0]);
        }
        rankInfo.deviceNicPort = curRankInfo.deviceInfo.port;
        rankInfo.deviceVnicPort = curRankInfo.deviceInfo.vnicPort;
        rankInfo.backupDevicePort = curRankInfo.deviceInfo.backupPort;
        HCCL_INFO("[Get][NicInfo]serverId[%s], serverIdx[%u], rankIndex[%u], nicIp[%s], backupNicIp[%s], "
            "deviceNicPort[%u], deviceVnicPort[%u], backupDevicePort[%u]",
            rankInfo.serverId.c_str(), rankInfo.serverIdx, curRankIndex,
            rankInfo.nicIp[0].GetReadableIP(), rankInfo.backupNicIp[0].GetReadableIP(),
            rankInfo.deviceNicPort, rankInfo.deviceVnicPort, rankInfo.backupDevicePort);
    }

    return HCCL_SUCCESS;
}

// private
HcclResult HcclCommunicatorAttrs::InitCommParams(HcclCommParams &params)
{
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    deviceType_ = params.deviceType;

    identifier_ = params.identifier;
    collectiveId_ = params.id.internal;
    commWorkMode_ = params.commWorkMode;
    HCCL_DEBUG(
        "userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
        userRank_, realUserRank_, userRankSize_, deviceLogicId_, deviceType_, commWorkMode_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetServerId(const RankTable_t &rankTable)
{
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            serverId_ = rankTable.rankList[i].serverId;
            superPodId_ = rankTable.rankList[i].superPodId;
            superDeviceId_ = rankTable.rankList[i].superDeviceId;
            break;
        }
    }

    if (serverId_.empty()) {
        HCCL_ERROR("[Set][ServerId]SetServerId fail");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SetServerNum(const std::vector<RankInfo_t> &ranks)
{
    std::vector<std::string> serverIds;
    for (u32 index = 0; index < ranks.size(); index++) {
        std::vector<std::string>::iterator found = find(serverIds.begin(), serverIds.end(), ranks[index].serverId);
        if (found == serverIds.end()) {
            serverIds.push_back(ranks[index].serverId);
        }
    }
    serverNum_ = serverIds.size();
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice(const RankTable_t &rankTable)
{
    deviceNumPerServer_ = 0;
    bool isConnectedWithHCCS = true;
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID && isHaveCpuRank_ != true) {
            isHaveCpuRank_ = true;
        }
        if (serverId_ == rankTable.rankList[i].serverId &&
            rankTable.rankList[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            deviceNumPerServer_++;
        } else {
            continue;
        }
        if (Is310PDevice()) {
            continue;
        }
        CHK_RET(GetPairDeviceLinkType(rankTable, i, isConnectedWithHCCS, linkType));
    }
    if (deviceType_ == DevType::DEV_TYPE_910B && !isConnectedWithHCCS) {
        deviceNumPerAggregation_ = deviceNumPerServer_ / FACTOR_NUM_TWO;
    } else {
        deviceNumPerAggregation_ = deviceNumPerServer_;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::GetPairDeviceLinkType(const RankTable_t &rankTable, u32 i,
    bool &isConnectedWithHCCS, LinkTypeInServer &linkType)
{
    for (u32 j = i + 1; j < rankTable.rankList.size(); j++) {
        if (rankTable.rankList[i].serverId == rankTable.rankList[j].serverId) {
            bool isValidRanki = rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID;
            bool isValidRankj = rankTable.rankList[j].deviceInfo.devicePhyId == HOST_DEVICE_ID;
            if ( isValidRanki || isValidRankj) {
                continue;
            }
            CHK_RET(hrtGetPairDeviceLinkType(rankTable.rankList[i].deviceInfo.devicePhyId,
                rankTable.rankList[j].deviceInfo.devicePhyId, linkType));
        }
        if (linkType != LinkTypeInServer::HCCS_TYPE) {
            isConnectedWithHCCS = false;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::GetMixInnerLinkInfo(std::unordered_map<u32, u32> &pairLinkCounter,
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> &pairLinkInfo)
{
    pairLinkInfo.clear();
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::PXI_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 0;
    for (auto &it_local : nicList_) {
        for (auto &it_dest : nicList_) {
            if (it_local == it_dest || static_cast<s32>(it_local) == HOST_DEVICE_ID ||
                static_cast<s32>(it_dest) == HOST_DEVICE_ID) {
                continue;
            }
            LinkTypeInServer linkType;
            CHK_RET(hrtGetPairDeviceLinkType(it_local, it_dest, linkType));
            pairLinkInfo[static_cast<u32>(linkType)][it_local].push_back(it_dest);
            pairLinkCounter[static_cast<u32>(linkType)]++;
        }
    }
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        for (auto it : pairLinkInfo) {
            HCCL_DEBUG("pair link information linkType[%u], size[%llu]", it.first, it.second.size());
        }
        for (auto it : pairLinkCounter) {
            HCCL_DEBUG("pair link counter information linkType[%u], size[%llu]", it.first, it.second);
        }
    }

    return HCCL_SUCCESS;
}

// sub group适配获取server内设配数
HcclResult HcclCommunicatorAttrs::SetInnerServerAverageDevice(const std::vector<RankInfo> &rankList)
{
    deviceNumPerServer_ = 0;
    bool isConnectedWithHCCS = true;
    LinkTypeInServer linkType = LinkTypeInServer::HCCS_TYPE;
    for (u32 i = 0; i < rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (serverId_ == rankList[i].serverId && rankList[i].devicePhyId != HOST_DEVICE_ID) {
            deviceNumPerServer_++;
        } else {
            continue;
        }
        if (Is310PDevice() || isHaveCpuRank_) {
            // 异构场景无需获取链路类型并校验
            continue;
        }
        for (u32 j = i + 1; j < rankList.size(); j++) {
            if (rankList[i].serverId == rankList[j].serverId) {
                CHK_RET(hrtGetPairDeviceLinkType(rankList[i].devicePhyId, rankList[j].devicePhyId, linkType));
            }
            if (linkType != LinkTypeInServer::HCCS_TYPE) {
                isConnectedWithHCCS = false;
            }
        }
    }
    if (deviceType_ == DevType::DEV_TYPE_910B && !isConnectedWithHCCS) {
        deviceNumPerAggregation_ = deviceNumPerServer_ / FACTOR_NUM_TWO;
    } else {
        deviceNumPerAggregation_ = deviceNumPerServer_;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::TransformRankInfoByServerId(
    const std::vector<RankInfo_t> &rankList, ServRankInfo &servRankInfo) const
{
    for (size_t index = 0; index < rankList.size(); ++index) {
        const RankInfo_t &rankInfo = rankList[index];
        std::string serverId = SalTrim(rankInfo.serverId);
        ServRankInfo::iterator itr = servRankInfo.find(serverId);
        if (itr != servRankInfo.end()) {
            itr->second.push_back(rankInfo);
        } else {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair(serverId, rankInfoList);
            servRankInfo.insert(rankInfoPair);
        }
    }
    // 每个server下的rank列表按  设备Id 从小到大的顺序排序
    for (auto &iter : servRankInfo) {
        std::sort(iter.second.begin(), iter.second.end(), CompareWithDevicePhyId);
    }
    return HCCL_SUCCESS;
}
bool HcclCommunicatorAttrs::CompareWithDevicePhyId(const RankInfo_t &left, const RankInfo_t &right)
{
    return left.deviceInfo.devicePhyId < right.deviceInfo.devicePhyId;
}
HcclResult HcclCommunicatorAttrs::SetModuleInfo(const std::vector<RankInfo_t> &rankList)
{
    isDiffDeviceType_ = IsDiffDeviceType(rankList);
    isDiffDeviceModule_ = IsDiffDeviceModule(rankList);
    HCCL_DEBUG("[SetModuleInfo]isDiffDeviceModule_[%u] isDiffDeviceType_[%u]", isDiffDeviceModule_, isDiffDeviceType_);
    multiModuleDiffDeviceNumMode_ = false;
    moduleNum_ = serverNum_;

    std::map<u32, std::vector<RankInfo_t>> moduleMap;
    for (RankInfo_t rankInfo : rankList) {
        u32 moduleIdx = INVALID_UINT;
        CHK_RET(GetModuleIdx(rankInfo, moduleIdx)); // 这里不判断混合组网，只提取每个server实际的moduleidx
        if (static_cast<s32>(rankInfo.deviceInfo.devicePhyId) == HOST_DEVICE_ID) {
            continue;
        }
        auto iter = moduleMap.find(moduleIdx);
        if (iter == moduleMap.end()) {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            moduleMap.insert(std::make_pair(moduleIdx, rankInfoList));
        } else {
            iter->second.push_back(rankInfo);
        }
    }
    if (moduleMap.size() == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<u32> moduleDeviceNumVec;

    moduleNum_ = moduleMap.size();
    u32 preDeviceNum = moduleMap.begin()->second.size();
    u32 curDeviceNum = preDeviceNum;
    for (auto &moduleInfo: moduleMap) {
        curDeviceNum = moduleInfo.second.size();
        if (curDeviceNum != preDeviceNum) {
            multiModuleDiffDeviceNumMode_ = true;
        }

        moduleDeviceNumVec.push_back(curDeviceNum);

        HCCL_INFO("module[%d] contains [%d]devices", moduleInfo.first, moduleInfo.second.size());
        for (auto &rankInfo : moduleInfo.second) {
            HCCL_INFO("moduleIdx[%d] Info: rankId[%d], serverId[%s], serverIdx[%d], devicePhyId[%d]",
                      moduleInfo.first, rankInfo.rankId, rankInfo.serverId.c_str(), rankInfo.serverIdx,
                      rankInfo.deviceInfo.devicePhyId);
        }
    }
    
    if (isDiffDeviceType_) {
        gcdDeviceNumPerAggregation_ = CalGCD(moduleDeviceNumVec);
        multiModuleDiffDeviceNumMode_ = false;
        deviceNumPerAggregation_ = gcdDeviceNumPerAggregation_;
        useSuperPodMode_ = false;
        HCCL_INFO("[HcclCommunicatorAttrs][SetModuleInfo]mix mode, set multiModuleDiffDeviceNumMode to false, "
            "gcdDeviceNumPerAggregation [%u] deviceNumPerAggregation [%u]",
            gcdDeviceNumPerAggregation_, deviceNumPerAggregation_);
    }
    
    HCCL_RUN_INFO("different module contains different numbers of cards:[%d]", multiModuleDiffDeviceNumMode_);
    HCCL_RUN_INFO("different module contains different type of cards:[%d]", isDiffDeviceType_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetSuperPodInfo(const std::vector<RankInfo_t> &rankList)
{
    //1.超节点数目 2.超节点间server数是否一致 3.
    superPodNum_ = 0;
    multiSuperPodDiffServerNumMode_ = false;
    std::map<std::string, std::set<u32>> superPodToServerNum; //记录每个超节点中的server数目
    for (RankInfo_t rankInfo : rankList) {
        // superPodId为空时, 返回超节点数量为0, 按照非超节点模式处理
        CHK_PRT_RET(rankInfo.superPodId.empty(),
            HCCL_DEBUG("ranks[%u] superPodId[%s] is empty, set superPodNum to zero", rankInfo.rankId,
            rankInfo.superPodId.c_str()),
            HCCL_SUCCESS);

        superPodToServerNum[rankInfo.superPodId].insert(rankInfo.serverIdx);
    }
    superPodNum_ = superPodToServerNum.size();
    u32 preServerNum = superPodToServerNum.begin()->second.size();
    u32 curServerNum = preServerNum;
    for (auto superPodItem: superPodToServerNum) {
        curServerNum = superPodItem.second.size();
        if (curServerNum != preServerNum) {
            multiSuperPodDiffServerNumMode_ = true;
        }
        HCCL_INFO("[Set][SuperPodInfo]SuperPod[%s] contains [%d]servers", superPodItem.first.c_str(), superPodItem.second.size());
    }
    HCCL_RUN_INFO("[Set][SuperPodInfo]different surperPod contains different numbers of servers:[%d]",
        multiSuperPodDiffServerNumMode_);
    if (isDiffDeviceType_) {
        multiSuperPodDiffServerNumMode_ = false;
        HCCL_RUN_INFO("mix mode, set multiSuperPodDiffServerNumMode to false");
    }
    return HCCL_SUCCESS;
}
// 集群中存在910B A+X时，0-7卡: moduleIdx = 2 * serverIdx; 8-15卡: moduleIdx = 2 * serverIdx + 1
// 集群中不存在910B A+X时，moduleIdx = serverIdx
HcclResult HcclCommunicatorAttrs::GetModuleIdx(const RankInfo_t &rankInfo, u32 &moduleIdx)
{
    CHK_PRT_RET(rankInfo.serverIdx == INVALID_UINT,
        HCCL_ERROR("serverIdx is invalid:[%u], rankId:[%u]", rankInfo.serverIdx, rankInfo.rankId), HCCL_E_PARA);
    CHK_PRT_RET(deviceType_ == DevType::DEV_TYPE_COUNT,
        HCCL_ERROR("deviceType_ is invalid:[%d], rankId:[%u]", deviceType_, rankInfo.rankId), HCCL_E_PARA);
    u32 serverIdx = rankInfo.serverIdx;
    if (GetRankInfoDevType(rankInfo) == DevType::DEV_TYPE_910B && isDiffDeviceModule_) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO + rankInfo.deviceInfo.devicePhyId / DEVICE_PER_MODULE;
    } else if (isDiffDeviceType_) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO;
    } else {
        moduleIdx = serverIdx;
    }
    CHK_PRT_RET(moduleIdx == INVALID_UINT,
        HCCL_ERROR("GetModuleIdx failed. moduleIdx:[%d], rankId:[%u]", moduleIdx, rankInfo.rankId), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// 用于标识集群中是否存在 不同芯片形态
bool HcclCommunicatorAttrs::IsDiffDeviceType(const std::vector<RankInfo_t> &rankList) const
{
    if (rankList.size() <= 1 || isHaveCpuRank_) {
        return false;
    }
    for (const RankInfo_t &rankInfo : rankList) {
        if (GetRankInfoDevType(rankInfo) != deviceType_) {
            HCCL_INFO("[IsDiffDeviceType] deviceType_[%d], and ranktable contains devicePhyId[%d]-deviceType[%d]",
                deviceType_, rankInfo.deviceInfo.devicePhyId, rankInfo.deviceInfo.deviceType);
            return true;
        }
    }
    return false;
}

// 用于标识集群中是否存在 910B A+X形态
bool HcclCommunicatorAttrs::IsDiffDeviceModule(const std::vector<RankInfo_t> &rankList) const
{
    bool minDevice = false;
    bool maxDevice = false;
    bool isDiffMeshAggregation = false;
    if (!isDiffDeviceType_ && (deviceType_ != DevType::DEV_TYPE_910B || rankList.size() == 0)) {
        HCCL_INFO("[IsDiffDeviceModule] deviceType_[%d], rankList.size[%u]", deviceType_, rankList.size());
        return false;
    }
    for (const RankInfo_t &rankInfo : rankList) {
        if (GetRankInfoDevType(rankInfo) == DevType::DEV_TYPE_910B) {
            if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                minDevice = true;
            } else {
                maxDevice = true;
            }
        }
    }
    if (minDevice && maxDevice) {
        isDiffMeshAggregation = true;
    }
    return isDiffMeshAggregation;
}
HcclResult HcclCommunicatorAttrs::SetNiclistInfo(){
    for (auto &iter : servRankInfo_[serverId_]) {
        if (((!iter.hostIp.IsInvalid()) || (!iter.deviceInfo.deviceIp[0].IsInvalid())) &&
            (iter.deviceInfo.devicePhyId != HOST_DEVICE_ID)) {
            if (isDiffDeviceType_) {
                u32 gcdIdx = userRank_ / gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMin = gcdIdx * gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMax = (gcdIdx + 1) * gcdDeviceNumPerAggregation_ - 1;
                if (iter.rankId < gcdUserRankMin || iter.rankId > gcdUserRankMax) {
                    continue;
                }
            }
            nicList_.push_back(iter.deviceInfo.devicePhyId);
        }
    }
    std::sort(nicList_.begin(), nicList_.end());
    HCCL_DEBUG("nic isDiffDeviceType[%u] userRank[%u] nicList size[%d]", isDiffDeviceType_, userRank_, nicList_.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitHccsPortNum()
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910_93) {
        CHK_RET(hrtGetHccsPortNum(deviceLogicId_, hccsPortNum_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitTopoInfo(const RankTable_t &rankTable)
{
    topoInfoParse_.reset(new (std::nothrow) TopoInfoParse());
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->Init(rankTable, serverId_, deviceNumPerServer_));
    if (isDiffDeviceType_) {
        CHK_RET(GetMixInnerLinkInfo(pairLinkCounter_, pairLinkInfo_));   // 获取混合组网场景上HCCS、PXI链接的数目
    } else {
        CHK_RET(topoInfoParse_->GetServerInnerLinkInfo(pairLinkCounter_, pairLinkInfo_));   // 获取本Server上HCCS、PXI链接的数目
    }
    // 初始化阶段判断组网状态
    CHK_RET(topoInfoParse_->IsSingleMeshAggregation(isSingleMeshAggregation_));         // 确认集群中只有一个MeshAggregation
    CHK_RET(topoInfoParse_->IsAllRankSamePlane(isAllRankSamePlane_));                   // 确认集群所有卡在一个平面上
    isStandardCard_ = IsStandardCard();
    is310PDuoCard_ = Is310PDuoCard();
    if (is310PDuoCard_) {
        isCommon310P3DUO_ = IsCommon310P3DUO(rankTable.rankList);
    }
    CHK_RET(InitHccsPortNum());
    CHK_RET(topoInfoParse_->ParseAndCheck(nicList_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitTopoInfo(const std::vector<RankInfo> &rankList)
{
    topoInfoParse_.reset(new (std::nothrow) TopoInfoParse());
    CHK_SMART_PTR_NULL(topoInfoParse_);
    CHK_RET(topoInfoParse_->Init(rankList, serverId_, deviceNumPerServer_));
    if (isDiffDeviceType_) {
        CHK_RET(GetMixInnerLinkInfo(pairLinkCounter_, pairLinkInfo_));   // 获取混合组网场景上HCCS、PXI链接的数目
    } else {
        CHK_RET(topoInfoParse_->GetServerInnerLinkInfo(pairLinkCounter_, pairLinkInfo_));   // 获取本Server上HCCS、PXI链接的数目
    }
    // 初始化阶段判断组网状态
    CHK_RET(topoInfoParse_->IsSingleMeshAggregation(isSingleMeshAggregation_));         // 确认集群中只有一个MeshAggregation
    CHK_RET(topoInfoParse_->IsAllRankSamePlane(isAllRankSamePlane_));                   // 确认集群所有卡在一个平面上
    isStandardCard_ = IsStandardCard();
    is310PDuoCard_ = Is310PDuoCard();
    CHK_RET(InitHccsPortNum());
    if (!isStandardCard_) {
        CHK_RET(topoInfoParse_->Check());
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SetInterModeInSuperPod()
{
    // 硬件配置为非超节点模式或软件（ranktable）中未配置sdid，后面按照非超节点形态处理
    if (!useSuperPodMode_) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[Set][InterModeInSuperPod]: serverNum[%u], superPodNum[%u].", serverNum_, superPodNum_);
    // 超节点HCCS模式
    if (GetExternalInputInterHccsDisable() == false && serverNum_ > 1 && superPodNum_ > 0) {
        isUsedInterHccsMode_ = true;
        HCCL_RUN_INFO("[Set][InterModeInSuperPod]: will use inter HCCS Mode, superPodId[%s], superDeviceId[0x%x], "
                      "superPodNum[%u], serverNum[%u], userRank[%u].",
            superPodId_.c_str(), superDeviceId_, superPodNum_, serverNum_, userRank_);
    }
    return HCCL_SUCCESS;
}
// 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
HcclResult HcclCommunicatorAttrs::CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const
{
    if (serverNum_ == 1 && moduleNum_ == HCCL_MODULE_NUM_TWO && GetExternalInputIntraRoceSwitch() == 0) {
        std::vector<u32> devIdList0;
        std::vector<u32> devIdList1;
        for (RankInfo_t rankInfo : rankList) {
            if (rankInfo.deviceInfo.devicePhyId == HOST_DEVICE_ID) {
                HCCL_ERROR("[Check][SingleServerComm]not support cpu rank");
                return HCCL_E_NOT_SUPPORT;
            }
            if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                devIdList0.push_back(rankInfo.deviceInfo.devicePhyId);
            } else {
                devIdList1.push_back(rankInfo.deviceInfo.devicePhyId);
            }
        }
        std::sort(devIdList0.begin(), devIdList0.end());
        std::sort(devIdList1.begin(), devIdList1.end());

        if (devIdList0.size() != devIdList1.size()) {
            HCCL_ERROR("[Check][SingleServerComm]errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "\
                "deviceNum in module0:[%d] not equal to deviceNum in module1:[%d]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),  serverNum_, moduleNum_, devIdList0.size(), devIdList1.size());
            return HCCL_E_NOT_SUPPORT;
        }
        for (size_t i = 0; i < devIdList0.size(); i++) {
            if (devIdList0[i] % DEVICE_PER_MODULE != devIdList1[i] % DEVICE_PER_MODULE) {
                HCCL_ERROR("[Check][SingleServerComm]errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "\
                    "deviceId[%d] in module0 and deviceId[%d] in module1 are not on the same plane",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), serverNum_, moduleNum_, devIdList0[i], devIdList1[i]);
                return HCCL_E_NOT_SUPPORT;
            }
        }
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SetRankInfoList(const RankTable_t &rankTable)
{
    // 检查rank table入参正确性
    CHK_RET(CheckRankTable(rankTable, servRankInfo_));
    // 获取芯片类型
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));

    // 遍历rank table获取rank信息
    rankInfoList_.clear();
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t &orgRankInfo = iter->second[index];
            // 构建comm 使用的rank 信息
            RankInfo rankInfo;
            rankInfo.userRank = orgRankInfo.rankId;
            rankInfo.worldRank = orgRankInfo.rankId;

            rankInfo.deviceType = GetRankInfoDevType(orgRankInfo);
            CHK_RET(CheckDeviceType(rankInfo.deviceType));

            if (rankInfo.deviceType != DevType::DEV_TYPE_910B || rankInfo.deviceType != DevType::DEV_TYPE_910_93) {
                // 910B、910_93形态不做devicePhyId最大值的判断
                CHK_RET(CheckDevPhyId(orgRankInfo.deviceInfo.devicePhyId));
            }
            rankInfo.devicePhyId = orgRankInfo.deviceInfo.devicePhyId;
            rankInfo.deviceNicPort = orgRankInfo.deviceInfo.port;
            rankInfo.deviceVnicPort = orgRankInfo.deviceInfo.vnicPort;

            rankInfo.serverId = orgRankInfo.serverId;
            rankInfo.serverIdx = orgRankInfo.serverIdx;
            rankInfo.hostIp = orgRankInfo.hostIp;
            rankInfo.hostPort = orgRankInfo.hostPort;
            rankInfo.localRank = orgRankInfo.localRank;
            rankInfo.superDeviceId = orgRankInfo.superDeviceId;
            rankInfo.superPodId = orgRankInfo.superPodId;
            rankInfo.superPodIdx = orgRankInfo.superPodIdx;
            CHK_RET(GetNicInfo(rankTable.nicDeploy, index, iter->second, rankInfo));
            rankInfo.nicIdx.assign(nicList_.begin(), nicList_.end());
            rankInfoList_.push_back(rankInfo);
        }
    }
    // 将rank id从小到大的顺序返回
    CHK_RET(SortRankInfoList());
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::CheckRankTable(const RankTable_t &rankTable, const ServRankInfo &servRankInfo)
{
    // 检查网卡挂载位置
    if (CheckNicDeploy(rankTable.nicDeploy, deviceType_) != HCCL_SUCCESS) {
        HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] nicDeploy[%d] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), rankTable.nicDeploy,
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_HOST),
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_DEVICE));
        return HCCL_E_PARA;
    }

    if (Is310PDevice()) {
        // 异构场景无需检查server内device个数
        return HCCL_SUCCESS;
    }

    if (CheckSuperDeviceId(rankTable) != HCCL_SUCCESS) {
        HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] super_device_id is invalid in ranktable, "
            "ranktable config vaule: rankId[%u], superDeviceId[0x%x]",
            HCCL_ERROR_CODE(HCCL_E_PARA), userRank_, superDeviceId_);
        return HCCL_E_PARA;
    }

    // 检查服务器上的设备信息
    ServRankInfo::const_iterator iterBegin = servRankInfo.begin();
    u32 devNum = 0;
    CHK_RET(GetDevNum(iterBegin->second, devNum));

    bool multiServerDiffDeviceNumMode = false;
    for (ServRankInfo::const_iterator iter = iterBegin; iter != servRankInfo.end(); ++iter) {
        // 检测每个服务器内的设备数是否相等，如果不相同即为多server不同卡模式
        u32 curServerDevNum = 0;
        CHK_RET(GetDevNum(iter->second, curServerDevNum));
        if (devNum != curServerDevNum) {
            HCCL_WARNING("[Check][RankTable] devnum isn't same,(serverA:[%s],serverB:[%s])"\
                "devNum(%u, %u)", iterBegin->first.c_str(), iter->first.c_str(), devNum, curServerDevNum);
            multiServerDiffDeviceNumMode = true;
        }
    }

    // 非多server不同卡模式下，判断实际设备数目和userRank_table中的记录一致
    if (multiServerDiffDeviceNumMode == false && rankTable.deviceNum != devNum * servRankInfo.size()) {
        HCCL_WARNING("[Check][RankTable]errNo[0x%016llx] devnum  isn't same, number in rankTable:[%u], actual:[%llu]",
            HCCL_ERROR_CODE(HCCL_E_PARA), rankTable.deviceNum, devNum * servRankInfo.size());
        return HCCL_E_PARA;
    }

    // 910模组：服务器内设备的数目必须是2的次幂,在此check(非模组形态无此限制不check)
    // 910B、910_93模组形态未定，服务器内设备的数目校验规则后续补充
    if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() > 0 && devNum > HCCL_DEVICE_NUM_TWO &&
        (deviceType_ != DevType::DEV_TYPE_910B && deviceType_ != DevType::DEV_TYPE_910_93 && !Is310P3Common())) {
        CHK_PRT_RET(CheckDevCount(devNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Check][RankTable]errNo[0x%016llx] devnum  is invaild in server.",
                HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::CheckDevPhyId(const s32 &devicePhyId) const
{
    if (devicePhyId > COMM_MAX_DEVICE_ID && devicePhyId != HOST_DEVICE_ID) {
        HCCL_ERROR("[Check][DevPhyId]errNo[0x%016llx] devicePhyId[%d] out of range[-1, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), devicePhyId, COMM_MAX_DEVICE_ID);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SortRankInfoList()
{
    // 按rank id从小到大的顺序返回
    std::sort(rankInfoList_.begin(), rankInfoList_.end(), CompareWithUserRank);

    for (u32 index = 0; index < rankInfoList_.size(); ++index) {
        CHK_PRT_RET((index != rankInfoList_[index].userRank),
            HCCL_ERROR("[HcclCommunicatorAttrs][SortRankInfoList]errNo[0x%016llx] index[%u] != rankInfoList.userRank[%u]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, rankInfoList_[index].userRank), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SethbRankInfo(const std::vector<RankInfo> &rankList, 
    WorldGroupInfo &groupCommonData)
{
    // 记录serverId
    serverId_ = groupCommonData.serverId;
    useSuperPodMode_ = groupCommonData.useSuperPodMode;

    for (auto &rankInfo : rankList) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::CheckNicDeploy(NICDeployment nicDeploy, DevType deviceType) const
{
    (void)deviceType;
    if (nicDeploy >= NICDeployment::NIC_DEPLOYMENT_RESERVED) {
        HCCL_ERROR("[Check][NicDeploy]errNo[0x%016llx] nicDeploy[%u] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), nicDeploy,
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_HOST),
            static_cast<int32_t>(NICDeployment::NIC_DEPLOYMENT_DEVICE));
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::CheckSuperDeviceId(const RankTable_t &rankTable)
{
    // 非910_93/910_93非超节点形态 || 用户配置非超节点模式，无需校验SDID合法性
    if (!useSuperPodMode_) {
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            s64 drvSuperDeviceID = 0;
            CHK_RET(hrtGetDeviceInfo(deviceLogicId_, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
                HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, drvSuperDeviceID));
            if (superDeviceId_ != static_cast<u32>(drvSuperDeviceID)) {
                RPT_INPUT_ERR(true, "EI0004", std::vector<std::string>({ "error_reason", "ranktable_path" }),
                    std::vector<std::string>({ "the 'super_device_id' in the ranktable is invalid",
                    "Please check the 'super_device_id' in ranktable" }));
                HCCL_ERROR("[Check][SuperDeviceId]errNo[0x%016llx] super_device_id is invalid, " \
                    "expect value [0x%x], ranktable config vaule [0x%x]",
                    HCOM_ERROR_CODE(HCCL_E_PARA), drvSuperDeviceID, superDeviceId_);
                return HCCL_E_PARA;
            }
            break;
        }
    }
    HCCL_RUN_INFO("[Check][SuperDeviceId]: superDevice check success, superPodId[%s], " \
        "superDeviceId[0x%x], userRank[%u].", superPodId_.c_str(), superDeviceId_, userRank_);
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::CheckDevCount(const u32 devNum)
{
    if (devNum > HCCL_AISERVER_DEVICE_NUM) {
        HCCL_ERROR("[Check][DevCount]errNo[0x%016llx] devNum[%u] out of range[%u, %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            devNum, 0, HCCL_AISERVER_DEVICE_NUM);
        return HCCL_E_PARA;
    }
    // 其他拓扑算法设备数目: 1 server: 1, 2, 4, 8
    //                     n server: 1*n, 2*n, 4*n, 8*n
    if (!Check2N(devNum)) {
        const std::string devnumError("devNum must be divisible by 8, or equal to 1, 2 or 4, please check devNum");
        RPT_ENV_ERR(true,
            "EI0004",
            std::vector<std::string>({"error_reason", "ranktable_path"}),
            std::vector<std::string>({
                devnumError,
                "The ranktable path configured in the training can be "
                "found in the plogs."
                }));

        HCCL_ERROR("[Check][DevCount]errNo[0x%016llx] devNum[%u] devNum must be divisible by 8, or equal to 1, 2 or 4",
            HCCL_ERROR_CODE(HCCL_E_PARA),
            devNum);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
bool HcclCommunicatorAttrs::Check2N(u32 num) const
{
    if (num < 1) {
        return false;
    } else {
        return ((num & (num - 1)) == 0);
    }
}
HcclResult HcclCommunicatorAttrs::UpdateNicList()
{
    std::vector<u32> subCommNicList;
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].serverId == serverId_ &&
            std::find(nicList_.begin(), nicList_.end(), rankInfoList_[i].devicePhyId) != nicList_.end()) {
            if (isDiffDeviceType_) {
                u32 gcdIdx = userRank_ / gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMin = gcdIdx * gcdDeviceNumPerAggregation_;
                u32 gcdUserRankMax = (gcdIdx + 1) * gcdDeviceNumPerAggregation_ - 1;
                if (rankInfoList_[i].userRank < gcdUserRankMin || rankInfoList_[i].userRank > gcdUserRankMax) {
                    continue;
                }
            }
            subCommNicList.push_back(rankInfoList_[i].devicePhyId);
        }
    }
    nicList_ = subCommNicList;
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        // 打印更新后的nicList_
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = nicList_.begin(); it != nicList_.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string nicListString = stringRepresentation.str();
        const char *charNicList = nicListString.c_str();
        HCCL_DEBUG("[HcclCommunicatorAttrs][Init] The subcommunication domain related nicList_: %s", charNicList);
    }
    // 将更新的nicList_刷新到rankInfoList_中
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        rankInfoList_[i].nicIdx.assign(nicList_.begin(), nicList_.end());
    }
    return  HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SetLocalRankInfo()
{
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        HCCL_DEBUG(" host ip: %s host port: %u dev phy id: %d.", rankInfoList_[i].hostIp.GetReadableAddress(),
            rankInfoList_[i].hostPort, rankInfoList_[i].devicePhyId);
        if (rankInfoList_[i].userRank == userRank_) {
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            devIpAddr_ = rankInfoList_[i].nicIp;
            devBackupIpAddr_ = rankInfoList_[i].backupNicIp;
            devBackupPort_ = rankInfoList_[i].backupDevicePort;
            hostIp_ = rankInfoList_[i].hostIp;
            hostPort_ = rankInfoList_[i].hostPort;
            localRank_ = rankInfoList_[i].localRank;
            HCCL_DEBUG("localRank_[%u].", localRank_);
            break;
        }
    }
    // 在确定 servRankInfo_ 和 serverId_ 信息后，就完成初始判断
    HCCL_DEBUG("[HcclCommunicatorAttrs][Init]deviceType[%u].", deviceType_);
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[HcclCommunicatorAttrs][Init]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    } else {
        HCCL_DEBUG("[HcclCommunicatorAttrs][Init]devicePhyId[%u] != HOST_DEVICE_ID", devicePhyId_);
        CHK_RET(hrtGetDevice(&deviceLogicId_));
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::SetLocalRankInfoSubGroup(const std::vector<RankInfo> &rankList)
{
    rankInfoList_.assign(rankList.begin(), rankList.end());
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        if (rankInfoList_[i].userRank == userRank_) {
            devIpAddr_ = rankInfoList_[i].nicIp;
            devBackupIpAddr_ = rankInfoList_[i].backupNicIp;
            devBackupPort_ = rankInfoList_[i].backupDevicePort;
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            superPodId_ = rankInfoList_[i].superPodId;
            superDeviceId_ = rankInfoList_[i].superDeviceId;
            hostIp_ = rankInfoList_[i].hostIp;
            hostPort_ = rankInfoList_[i].hostPort;
            nicList_.assign(rankInfoList_[i].nicIdx.begin(), rankInfoList_[i].nicIdx.end());
            nicDeployment_ = rankInfoList_[i].nicDeploy;
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::CheckLocalRankInfo()
{
    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        if (userRank_ == rankInfoList_[i].userRank) {
            CHK_PRT_RET(static_cast<s32>(devicePhyId_) != rankInfoList_[i].devicePhyId,
                HCCL_ERROR("[Init][Para]errNo[0x%016llx] parameter check failed, "
                "userrank[%u] == rankInfoList.userrank[%u], phyid[%d] != rankInfoList.devid[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), userRank_, rankInfoList_[i].userRank,
                static_cast<s32>(devicePhyId_), rankInfoList_[i].devicePhyId),
                HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetRanksPort(const std::vector<RankInfo_t> &rankList)
{
    bool devicePortSwitchOn = GetExternalInputNpuPortSwitch();
    if (devicePortSwitchOn) {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        vnicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto &rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT
                ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
            vnicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.vnicPort == HCCL_INVALID_PORT
                ? HETEROG_CCL_PORT : rankInfo.deviceInfo.vnicPort;
        }
    } else {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto &rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT
                || rankInfo.deviceInfo.port == 0 ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
        }
    }
    isUseRankPort_ = ((devicePortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) || isHaveCpuRank_
        || nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ? true : isUseRankPort_;
    HCCL_INFO("[HcclCommunicatorAttrs][SetRanksPort] devicePortSwitchOn[%u], isHaveCpuRank[%u], isUseRankPort[%u], "
        "nicRanksPort size[%u], vnicRanksPort size[%u].",
        devicePortSwitchOn, isHaveCpuRank_, isUseRankPort_, nicRanksPort_.size(), vnicRanksPort_.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitRankInfo(const RankTable_t &rankTable)
{
    // 获取serverId
    CHK_RET(SetServerId(rankTable));
    // 获取server数
    CHK_RET(SetServerNum(rankTable.rankList));
    CHK_PRT_RET(serverNum_ != rankTable.serverNum,
        HCCL_ERROR("[HcclCommunicatorAttrs][InitRankInfo]calculated serverNum[%u] is not equal to ranktable serverNum[%u]",
        serverNum_, rankTable.serverNum), HCCL_E_PARA);
    // 本节点的sdid配置有效(ranktable v1.2)且环境配置server id有效时, 使能superPod
    if (superDeviceId_ != INVALID_UINT) {
        CHK_RET(IsSuperPodMode(useSuperPodMode_)); // 使能superPod
    }
    // 获取server内设备数, 赋值 ishavecpurank_
    CHK_RET(SetInnerServerAverageDevice(rankTable));
    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankTable.rankList, servRankInfo_));
    // 获取module相关信息，moduleNum_, isDiffDeviceModule_, multiModuleDiffDeviceNumMode_;
    CHK_RET(SetModuleInfo(rankTable.rankList));
    // 获取超节点相关信息，superPodNum_, multiSuperPodDiffServerNumMode_
    CHK_RET(SetSuperPodInfo(rankTable.rankList));
    // 生成nicList
    CHK_RET(SetNiclistInfo());
    // 解析拓扑信息
    CHK_RET(InitTopoInfo(rankTable));
    // 设置超节点内节点间模式，包括是否使用sdid获取vnicip、节点间是否使能HCCS
    CHK_RET(SetInterModeInSuperPod());
    // 解析ranktable信息(生成rankInfoList_)，供给commfactory使用
    CHK_RET(SetRankInfoList(rankTable));
    // 解析当前Rank信息
    CHK_RET(SetLocalRankInfo());
    // 解析rank和port的映射信息
    CHK_RET(SetRanksPort(rankTable.rankList));

    interServer_ = rankTable.serverNum > 1; // serverNum为1时，不进行roce初始化
    nicDeployment_ = rankTable.nicDeploy;
    return HCCL_SUCCESS;
}

void HcclCommunicatorAttrs::GenCollectiveId(HcclCommParams &params, const RankTable_t &rankTable)
{
    collectiveId_ = rankTable.collectiveId.empty() ? params.id.internal : rankTable.collectiveId;
}
u32 HcclCommunicatorAttrs::CalMeshAggRankSize(int halfDevNum) const
{
    u32 size = INVALID_VALUE_RANKSIZE;
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        u32 aggregationRankSize0 = 0;
        u32 aggregationRankSize1 = 0;
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t &orgRankInfo = iter->second[index];
            if (orgRankInfo.deviceInfo.devicePhyId < halfDevNum) {
                aggregationRankSize0++;
            } else {
                aggregationRankSize1++;
            }
        }
        u32 tmpsize = INVALID_VALUE_RANKSIZE;
        if (aggregationRankSize0 && aggregationRankSize1) {
            tmpsize = aggregationRankSize0;
        } else {
            tmpsize = iter->second.size();
        }
        size = size > tmpsize ? tmpsize : size;
    }
    return size;
}
HcclResult HcclCommunicatorAttrs::SetMeshAggregationRankSize(u32 size)
{
    HCCL_INFO("[Set][HcclCommunicatorAttrs][MeshAggregationRankSize]set MeshAggregationRankSize[%u].", size);
    meshAggregationRankSize_ = size;
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::CalAndSetMeshAggRankSize()
{
    u32 size = INVALID_VALUE_RANKSIZE;
    if ((deviceType_ == DevType::DEV_TYPE_910B) && isDiffDeviceModule_) { // 910B 16p场景
        size = CalMeshAggRankSize(HCCL_DEVICE_NUM_EIGHT);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) { // 标卡
            size = 1;
        } else { // 模组
            size = CalMeshAggRankSize(HCCL_DEVICE_NUM_FOUR);
        }
    } else { // 910B的8卡、310P 直接返回server内的size数量
        size = servRankInfo_.begin()->second.size();
    }
    CHK_RET(SetMeshAggregationRankSize(size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::SetWorldGroupInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> &phyIdNicInfoMap,
    std::vector<RankInfo> &worldRankInfoList, std::vector<u32> &nicRanksPort, std::vector<u32> &vnicRanksPort)
{
    for (auto &ipInfo : phyIdNicInfoMap) {
        for (auto &devInfo : ipInfo.second) {
            rankDevicePhyIdNicInfoMap_[ipInfo.first][devInfo.first] = devInfo.second;
            HCCL_DEBUG("phyIdNicInfoMap print hostIp[%s] devId[%u] devIp[%s]",
                ipInfo.first.c_str(), devInfo.first, devInfo.second.GetReadableAddress());
        }
    }

    for (auto &rankInfo : worldRankInfoList) {
        worldRankInfoList_.push_back(rankInfo);
    }

    for (auto &port : nicRanksPort) {
        nicRanksPort_.push_back(port);
        HCCL_DEBUG("nicRanksPort port[%u]", port);
    }
    for (auto &port : vnicRanksPort) {
        vnicRanksPort_.push_back(port);
        HCCL_DEBUG("vnicRanksPort port[%u]", port);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicatorAttrs::InitRankInfoSubGroup(const std::vector<RankInfo> &rankList,
    WorldGroupInfo &groupCommonData)
{
    //填充心跳信息
    SethbRankInfo(rankList,groupCommonData);
    // 获取server内平均device数
    CHK_RET(SetInnerServerAverageDevice(rankList));
    // 将子通信域的ranklist结构体形式转换成全局通信域的
    std::vector<RankInfo_t> rankListNew;
    CHK_RET(TransformRankList(rankList, rankListNew));
    // 获取server数
    CHK_RET(SetServerNum(rankListNew));
    // 获取module相关信息，moduleNum_, isDiffDeviceModule_, multiModuleDiffDeviceNumMode_;
    CHK_RET(SetModuleInfo(rankListNew));
    // 获取超节点相关信息，superPodNum_, multiSuperPodDiffServerNumMode_
    CHK_RET(SetSuperPodInfo(rankListNew));
    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankListNew, servRankInfo_));
    // 解析拓扑信息
    CHK_RET(InitTopoInfo(rankList));
    //  inline reduce 开关
    inlineReduceSwitchOn_ = groupCommonData.inlineReduceSwitchOn;
    // 设置rank关联信息
    CHK_RET(SetLocalRankInfoSubGroup(rankList));
    // 设置超节点内节点间模式，包括是否使用sdid获取vnicip、节点间是否使用HCCS
    CHK_RET(SetInterModeInSuperPod());

    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        // 打印原来的nicList_
        std::ostringstream stringRepresentation;
        for (std::vector<uint32_t>::iterator it = nicList_.begin(); it != nicList_.end(); it++) {
            stringRepresentation << *it << " ";
        }
        std::string nicListString = stringRepresentation.str();
        const char *charNicList = nicListString.c_str();
        HCCL_DEBUG("[HcclCommunicatorAttrs][Init] The original nicList_: %s", charNicList);
    }
    interServer_ = serverNum_ > 1; // serverNum为1时，不进行roce初始化
    // 更新成跟子通信域相关的nicList_
    CHK_RET(UpdateNicList());
    // 检查当前user_rank 对应的devid和rt查到的一致
    CHK_RET(CheckLocalRankInfo());
    CHK_RET(CalAndSetMeshAggRankSize());

    if (IsEnableRoce()) {
        isUsedRdmaLevel0_ = IsUsedRdmaLevel0AndIpInvalid();
    }

    CHK_RET(SetWorldGroupInfo(groupCommonData.phyIdNicInfoMap, groupCommonData.worldRankInfoList,
        groupCommonData.ranksPort, groupCommonData.vnicRanksPort));
    for (auto &rankInfo : worldRankInfoList_) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isUseRankPort_ = true;
            break;
        }
    }
    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

    groupNicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
    if (nicRanksPort_.size()) {
        for (auto &rankInfo : rankInfoList_) {
            groupNicRanksPort_[rankInfo.userRank] = nicRanksPort_[rankInfo.worldRank];
            HCCL_INFO("hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                "nic port[%u], devicePhyId[%d]",
                rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(),
                rankInfo.userRank, rankInfo.worldRank, groupNicRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
        }
    }
    bool devicePortSwitchOn = groupCommonData.devPortSwitchOn;
    if (devicePortSwitchOn) {
        groupVnicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
        if (vnicRanksPort_.size()) {
            for (auto &rankInfo : rankInfoList_) {
                groupVnicRanksPort_[rankInfo.userRank] = vnicRanksPort_[rankInfo.worldRank];
                HCCL_INFO("hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                    "vnic port[%u], devicePhyId[%d]",
                    rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(),
                    rankInfo.userRank, rankInfo.worldRank, groupVnicRanksPort_[rankInfo.userRank],
                    rankInfo.devicePhyId);
            }
        }
    }
    isUseRankPort_ = ((devicePortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) || isHaveCpuRank_
        || nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ? true : isUseRankPort_;
    HCCL_INFO("[InitRankInfoSubGroup]:isUsedRdmaLevel0_[%d]", isUsedRdmaLevel0_);
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicatorAttrs::TransformRankList(
    const std::vector<RankInfo> &rankListIn, std::vector<RankInfo_t> &rankListOut)
{
    for (size_t index = 0; index < rankListIn.size(); ++index) {
        RankInfo_t rankInfoTmp;
        rankInfoTmp.serverId = rankListIn[index].serverId;
        rankInfoTmp.deviceInfo.devicePhyId = rankListIn[index].devicePhyId;
        rankInfoTmp.deviceInfo.deviceType = rankListIn[index].deviceType;
        rankInfoTmp.serverIdx = rankListIn[index].serverIdx;
        rankInfoTmp.rankId = rankListIn[index].userRank;
        rankInfoTmp.hostIp = rankListIn[index].hostIp;
        rankInfoTmp.hostPort = rankListIn[index].hostPort;
        rankInfoTmp.localRank = rankListIn[index].localRank;
        rankInfoTmp.superDeviceId = rankListIn[index].superDeviceId;
        rankInfoTmp.superPodId = rankListIn[index].superPodId;
        rankInfoTmp.superPodIdx = rankListIn[index].superPodIdx;
        rankListOut.push_back(rankInfoTmp);
    }
    return HCCL_SUCCESS;
}
bool HcclCommunicatorAttrs::IsEnableRoce()
{
    // 910B单机两种使能roce场景：1、a+x同时使用两module  2.标卡
    bool roceSwitch = IsSupportEnableRoce();
    bool isInterServerVnic = false;
    // 910_93超节点内节点间走HCCS通信 && Vnic建链, 不需要使能NIC
    if (useSuperPodMode_ && superPodNum_ == 1 &&
        GetExternalInputInterHccsDisable() == false && GetExternalInputInterVnicDisable() == false) {
        isInterServerVnic = true;
    }
    bool ret = (interServer_ && !isInterServerVnic) || roceSwitch;
    HCCL_INFO("IsEnableRoce ret: %d, interServer_: %d, isInterServerVnic: %d, roceSwitch: %d, "\
        "isSingleMeshAggregation_: %u", ret, interServer_, isInterServerVnic, roceSwitch, isSingleMeshAggregation_);
    return ret;
}

// a+x mesh间需要同时保证ip有效和roce开关打开才能走rdma
bool HcclCommunicatorAttrs::IsUsedRdmaLevel0AndIpInvalid()
{
    u32 nicNum = devIpAddr_.size();
    bool ipInvalid = true;
    for (u32 i = 0; i < nicNum; i++) {
        if (devIpAddr_[i].IsInvalid()) {
            HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
            ipInvalid = false;
            continue;
        }
    }
    // 机间卡数不一致场景下，IP有效情况下就走RDMA
    // 机间卡数一致场景下，需环境变量ROCE打开(IsEnableRoce在针对多机下未对此开关进行拦截)且IP有效情况下走RDMA
    return ((GetExternalInputIntraRoceSwitch() || multiModuleDiffDeviceNumMode_ || isDiffDeviceType_) && ipInvalid);
}
bool HcclCommunicatorAttrs::IsSupportEnableRoce()
{
    // 910B单机两种使能roce场景：1、a+x同时使用两module  2.标卡
    bool roceSwitch = false;
    HCCL_INFO("[HcclCommunicator]IsSupportEnableRoce");
    if (isDiffDeviceType_) {
        roceSwitch = true;
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        roceSwitch = (GetExternalInputIntraRoceSwitch() && (!isSingleMeshAggregation_ || isStandardCard_)) ||
                     multiModuleDiffDeviceNumMode_;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        roceSwitch = GetExternalInputEnableRdmaSdmaConcurrent() || multiSuperPodDiffServerNumMode_ || 
                     (multiModuleDiffDeviceNumMode_ && superPodNum_ > 1);
    } else { // 其他单机场景为了防止用户误用roce开关
        roceSwitch = isStandardCard_ ? GetExternalInputIntraRoceSwitch() : false;
    }
    return roceSwitch;
}

void HcclCommunicatorAttrs::GetTopoAttr(HcclTopoAttr &topoAttr)
{
    topoAttr.serverNum = serverNum_;
    topoAttr.superPodNum = superPodNum_;
    topoAttr.moduleNum = moduleNum_;
    topoAttr.deviceNumPerServer = deviceNumPerServer_;
    topoAttr.deviceNumPerAggregation = deviceNumPerAggregation_;
    topoAttr.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;
    topoAttr.multiSuperPodDiffServerNumMode = multiSuperPodDiffServerNumMode_;
    topoAttr.meshAggregationRankSize = meshAggregationRankSize_;
    topoAttr.isDiffDeviceModule = isDiffDeviceModule_;
    topoAttr.isDiffDeviceType = isDiffDeviceType_;
    topoAttr.gcdDeviceNumPerAggregation = gcdDeviceNumPerAggregation_;
    topoAttr.isSingleMeshAggregation = isSingleMeshAggregation_;
    topoAttr.isAllRankSamePlane = isAllRankSamePlane_;
    topoAttr.userRank = userRank_;
    topoAttr.realUserRank = realUserRank_;
    topoAttr.userRankSize = userRankSize_;
    topoAttr.devicePhyId = devicePhyId_;
    topoAttr.useSuperPodMode = useSuperPodMode_;
    topoAttr.deviceLogicId = deviceLogicId_;
    topoAttr.deviceType = deviceType_;
    topoAttr.isStandardCard = isStandardCard_;
    topoAttr.is310PDuoCard = is310PDuoCard_;
    topoAttr.isCommon310P3DUO = isCommon310P3DUO_;
    topoAttr.hccsPortNum = hccsPortNum_;
    topoAttr.nicList = nicList_;
    topoAttr.pairLinkCounter = pairLinkCounter_;
    topoAttr.pairLinkInfo = pairLinkInfo_;
    topoAttr.rankInfoList = rankInfoList_;
    topoAttr.isSupportRdmaLite = isSupportRdmaLite_;
    topoAttr.localNicPort = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
    topoAttr.isNeedInitNic = isNeedInitNic_;
}

void HcclCommunicatorAttrs::GetAlgoAttr(HcclAlgoAttr &algoAttr)
{
    algoAttr.isHaveCpuRank = isHaveCpuRank_;
    algoAttr.inlineReduceSwitchOn = inlineReduceSwitchOn_;
    algoAttr.isUsedRdmaLevel0 = isUsedRdmaLevel0_;
    HCCL_INFO("[CollectAlgoAttr]:isUsedRdmaLevel0:[%d]", isUsedRdmaLevel0_);
    algoAttr.isUsedInterHccsMode = isUsedInterHccsMode_;
    algoAttr.identifier = identifier_;
    algoAttr.collectiveId = collectiveId_;
    algoAttr.nicDeployment = nicDeployment_;
    algoAttr.commWorkMode = commWorkMode_;
}
void HcclCommunicatorAttrs::GenUsedRdmaLevel0()
{
    isUsedRdmaLevel0_ = IsSupportEnableRoce();
}

void HcclCommunicatorAttrs::GenSupportRdmaLite()
{
    isSupportRdmaLite_ = IsSupportRDMALite(deviceLogicId_);
}

bool HcclCommunicatorAttrs::GetUsedRdmaLevel0()
{
    return isUsedRdmaLevel0_;
}
bool HcclCommunicatorAttrs::GetSupportRdmaLite()
{
    return isSupportRdmaLite_;
}
std::string HcclCommunicatorAttrs::GetServerId()
{
    return serverId_;
}
u32 HcclCommunicatorAttrs::GetServerNum()
{
    return serverNum_;
}
std::string HcclCommunicatorAttrs::GetSuperPodId()
{
    return superPodId_;
}
u32 HcclCommunicatorAttrs::GetSuperDeviceId()
{
    return superDeviceId_;
}
bool HcclCommunicatorAttrs::GetSuperPodMode()
{
    return useSuperPodMode_;
}
u32 HcclCommunicatorAttrs::GetSuperPodNums()
{
    return superPodNum_;
}
u32 HcclCommunicatorAttrs::GetDeviceNumPerAggregation()
{
    return deviceNumPerAggregation_;
}
u32 HcclCommunicatorAttrs::GetDeviceNumPerServer()
{
    return deviceNumPerServer_;
}
DevType HcclCommunicatorAttrs::GetRankInfoDevType(const RankInfo_t &rankInfo) const
{
    if (rankInfo.deviceInfo.deviceType == DevType::DEV_TYPE_NOSOC) {
        return deviceType_; // 兼容非混合组网场景，rankInfo中deviceType字段可能没有赋值
    }
    return rankInfo.deviceInfo.deviceType;
}
bool HcclCommunicatorAttrs::GetDiffDeviceType()
{
    return isDiffDeviceType_;
}
u32 HcclCommunicatorAttrs::GetGcdDeviceNumPerAggregation()
{
    return gcdDeviceNumPerAggregation_;
}
ServRankInfo HcclCommunicatorAttrs::GetServRankInfo()
{
    return servRankInfo_;
}
bool HcclCommunicatorAttrs::GetDiffDeviceModule()
{
    return isDiffDeviceModule_;
}
u32 HcclCommunicatorAttrs::GetModuleNum()
{
    return moduleNum_;
}
bool HcclCommunicatorAttrs::GetMultiModuleDiffDeviceNumMode()
{
    return multiModuleDiffDeviceNumMode_;
}
bool HcclCommunicatorAttrs::GetMultiSuperPodDiffServerNumMode()
{
    return multiSuperPodDiffServerNumMode_;
}
std::vector<u32> HcclCommunicatorAttrs::GetNicList()
{
    return nicList_;
}
bool HcclCommunicatorAttrs::GetSingleMeshAggregation()
{
    return isSingleMeshAggregation_;
}
bool HcclCommunicatorAttrs::GetAllRankSamePlane()
{
    return isAllRankSamePlane_;
}
bool HcclCommunicatorAttrs::GetStandardCard()
{
    return isStandardCard_;
}
bool HcclCommunicatorAttrs::Get310PDuoCard()
{
    return is310PDuoCard_;
}
bool HcclCommunicatorAttrs::GetIsCommon310P3DUO()
{
    return isCommon310P3DUO_;
}
s32 HcclCommunicatorAttrs::GetHccsPortNum()
{
    return hccsPortNum_;
}
void HcclCommunicatorAttrs::GetPairLinkCounter(std::unordered_map<u32, u32> &pairLinkCounter)
{
    pairLinkCounter = pairLinkCounter_;
}
void HcclCommunicatorAttrs::GetPairLinkInfo(
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> &pairLinkInfo)
{
    pairLinkInfo = pairLinkInfo_;
}
bool HcclCommunicatorAttrs::GetUsedInterHccsMode()
{
    return isUsedInterHccsMode_;
}
std::vector<RankInfo> HcclCommunicatorAttrs::GetRankInfoList()
{
    return rankInfoList_;
}
std::vector<HcclIpAddress> HcclCommunicatorAttrs::GetDevIpAddr()
{
    return devIpAddr_;
}
std::vector<HcclIpAddress> HcclCommunicatorAttrs::GetDevBackupIpAddr()
{
    return devBackupIpAddr_;
}
u32 HcclCommunicatorAttrs::GetBackupDevPort()
{
    return devBackupPort_;
}
u32 HcclCommunicatorAttrs::GetDevicePhyId()
{
    return devicePhyId_;
}
HcclIpAddress HcclCommunicatorAttrs::GetHostIp()
{
    return hostIp_;
}
u32 HcclCommunicatorAttrs::GetHostPort()
{
    return hostPort_;
}
u32 HcclCommunicatorAttrs::GetLocalRank()
{
    return localRank_;
}
std::string HcclCommunicatorAttrs::GetCollectiveId()
{
    return collectiveId_;
}
s32 HcclCommunicatorAttrs::GetDeviceLogicId()
{
    return deviceLogicId_;
}
bool HcclCommunicatorAttrs::GetInterServe()
{
    return interServer_;
}
NICDeployment HcclCommunicatorAttrs::GetNicDeployment()
{
    return nicDeployment_;
}
bool HcclCommunicatorAttrs::GetHaveCpuRank()
{
    return isHaveCpuRank_;
}
u32 HcclCommunicatorAttrs::GetMeshAggregationRankSize()
{
    return meshAggregationRankSize_;
}
bool HcclCommunicatorAttrs::GetInlineReduceSwitchOn()
{
    return inlineReduceSwitchOn_;
}

u32 HcclCommunicatorAttrs::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

u32 HcclCommunicatorAttrs::GetLocalNicPort(NicType nicType)
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        return GetHostPort(devicePhyId_);
    }
    // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
    // groupRanksPort_为空说明此时处于全局通信域，要从ranksPort_取监听端口；否则取groupRanksPort_
    if (nicType == NicType::HOST_NIC_TYPE) {
        return GetHostPort(devicePhyId_);
    } if (nicType == NicType::VNIC_TYPE && GetExternalInputNpuPortSwitch()) {
        // vnic ports仅在开启device侧端口配置时单独配置
        std::vector<u32> &ranksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
        return GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    } else {
        // 1. 开启device侧端口配置时的nic port时使用ranksPorts
        // 2. 异构场景使用ranksPorts
        // 3. 其余场景场景isUseRankPort_应当为false，使用默认port
        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        return GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    }
}
void HcclCommunicatorAttrs::SetNeedInitNicFlag(const bool isNeedInitNic)
{
    isNeedInitNic_ = isNeedInitNic;
}
}
