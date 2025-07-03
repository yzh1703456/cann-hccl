/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topoinfo_ranktable_partition.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace hccl {

TopoinfoRanktablePartition::TopoinfoRanktablePartition(hccl::HcclCommParams &globalParams,
    hccl::RankTable_t &globalRankTable)
    : globalParams_(globalParams), globalRankTable_(globalRankTable)
{
}

TopoinfoRanktablePartition::~TopoinfoRanktablePartition()
{
}

HcclResult TopoinfoRanktablePartition::GenerateSubRankTable(const uint32_t rankNum, const uint32_t *rankIds,
    hccl::RankTable_t &subRankTable)
{
    subRankTable.nicDeploy = globalRankTable_.nicDeploy;
    std::unordered_map<uint32_t, size_t> rankInfoMap;
    for (size_t i = 0; i < globalRankTable_.rankList.size(); i++) {
        auto rankId = globalRankTable_.rankList[i].rankId;
        rankInfoMap[rankId] = i;
    }
    std::unordered_map<std::string, u32> serverIdMap;
    std::unordered_map<std::string, u32> superPodIdMap;
    std::unordered_set<uint32_t> rankIdSet;
    subRankTable.deviceNum = 0;
    for (size_t i = 0; i < rankNum; i++) {
        CHK_PTR_NULL(rankIds + i);
        uint32_t rankId = rankIds[i];
        CHK_PRT_RET(
            rankIdSet.find(rankId) != rankIdSet.end(),
            HCCL_ERROR("[TopoinfoRanktablePartition][GenerateSubRankTable]errNo[0x%016llx], " \
                "duplicated rankId[%u] in rankIds.",
                HCCL_ERROR_CODE(HCCL_E_PARA), rankId),
            HCCL_E_PARA);

        auto iter = rankInfoMap.find(rankId);
        CHK_PRT_RET(
            iter == rankInfoMap.end(),
            HCCL_ERROR("[TopoinfoRanktablePartition][GenerateSubRankTable]errNo[0x%016llx], " \
                "fail to find target rank[%u] in the global communicator.",
                HCCL_ERROR_CODE(HCCL_E_PARA), rankId),
            HCCL_E_PARA);

        hccl::RankInfo_t rankInfo = globalRankTable_.rankList[iter->second];
        serverIdMap.emplace(rankInfo.serverId, serverIdMap.size());
        superPodIdMap.emplace(rankInfo.superPodId, superPodIdMap.size());

        rankInfo.rankId = i;
        rankInfo.serverIdx = serverIdMap[rankInfo.serverId];
        rankInfo.superPodIdx = superPodIdMap[rankInfo.superPodId];
        subRankTable.rankList.emplace_back(rankInfo);

        if (rankInfo.deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            subRankTable.deviceNum++;
        }
        HCCL_INFO(
            "[TopoinfoRanktablePartition][GenerateSubRankTable]" \
            "Pick rank[%u] from global comm as rank[%u] in sub comm, " \
            "severId[%s], serverIdx[%u], superPodId[%s], superDeviceId[%u], devicePhyId[%d].",
            rankId, i, rankInfo.serverId.c_str(), rankInfo.serverIdx, rankInfo.superPodId.c_str(),
            rankInfo.superDeviceId, rankInfo.deviceInfo.devicePhyId);
    }

    subRankTable.serverNum = serverIdMap.size();
    subRankTable.superPodNum = superPodIdMap.size();
    subRankTable.rankNum = rankNum;

    return HCCL_SUCCESS;
}


HcclResult TopoinfoRanktablePartition::GenerateSubParams(const hccl::RankTable_t &subRankTable,
    const uint32_t subCommRankId, hccl::HcclCommParams &subParams)
{
    subParams.rank = subCommRankId;
    subParams.userRank = subRankTable.rankList[subCommRankId].rankId;
    subParams.totalRanks = subRankTable.rankList.size();
    subParams.logicDevId = globalParams_.logicDevId;
    subParams.serverId = subRankTable.rankList[subCommRankId].serverId;
    subParams.deviceType = globalParams_.deviceType;
    subParams.commPortConfig.devPortSwitchOn = globalParams_.commPortConfig.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::GetRankTableStr(const hccl::RankTable_t &subRankTable, std::string &rankTableStr)
{
    nlohmann::json basicJson;
    HcclResult ret = Struct2JsonRankTable(subRankTable, globalParams_.deviceType, basicJson);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_RUN_WARNING("cluster info to json failed, ret[%d].", ret), HCCL_E_INTERNAL);
    rankTableStr = std::move(basicJson.dump());
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::TransformRankInfo(const RankTable_t &clusterInfo,
    nlohmann::json &perRankJson, u32 rankIndex)
{
    auto rankInfo = clusterInfo.rankList[rankIndex];
    perRankJson[PROP_HOST_IP] = std::string(rankInfo.hostIp.GetReadableIP());
    perRankJson[PROP_DEV_ID] = std::to_string(rankInfo.deviceInfo.devicePhyId);
    perRankJson[PROP_DEV_NIC_PORT] = std::to_string(rankInfo.deviceInfo.port);
    perRankJson[PROP_DEV_VNIC_PORT] = std::to_string(rankInfo.deviceInfo.vnicPort);
    perRankJson[PROP_BACKUP_DEV_PORT] = std::to_string(rankInfo.deviceInfo.backupPort);
    perRankJson[PROP_RANK_ID] = std::to_string(rankInfo.rankId);
    perRankJson[PROP_SERVER_ID] = rankInfo.serverId;
    perRankJson[PROP_SUPER_POD_ID] = rankInfo.superPodId;
    perRankJson[PROP_SUPER_DEVICE_ID] = std::to_string(rankInfo.superDeviceId);
    if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE && rankInfo.deviceInfo.deviceIp.size() != 0 &&
        !rankInfo.deviceInfo.deviceIp[0].IsInvalid()) {
        perRankJson[PROP_DEV_IP] = std::string(rankInfo.deviceInfo.deviceIp[0].GetReadableIP());
    }
    if (clusterInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE &&
        rankInfo.deviceInfo.backupDeviceIp.size() != 0 && !rankInfo.deviceInfo.backupDeviceIp[0].IsInvalid()) {
        perRankJson[PROP_BACKUP_DEV_IP] = std::string(rankInfo.deviceInfo.backupDeviceIp[0].GetReadableIP());
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::TransformServerList(const RankTable_t &clusterInfo,
    nlohmann::json &rankListJson)
{
    for (size_t i = 0; i < clusterInfo.rankList.size(); i++) {
        nlohmann::json perRankJson;
        CHK_RET(TransformRankInfo(clusterInfo, perRankJson, i));
        perRankJson[PROP_RANK_ID] = perRankJson;
        rankListJson.push_back(perRankJson);
    }
    return HCCL_SUCCESS;
}

HcclResult TopoinfoRanktablePartition::Struct2JsonRankTable(const RankTable_t &clusterInfo, const DevType deviceType,
    nlohmann::json& ClusterJson)
{
    ClusterJson[PROP_SERVER_COUNT] = std::to_string(clusterInfo.serverNum);
    ClusterJson[PROP_SUPER_POD_NUM] = std::to_string(clusterInfo.superPodNum);
    ClusterJson[PROP_RANK_NUM] = std::to_string(clusterInfo.rankNum);
    ClusterJson[PROP_DEV_NUM] = std::to_string(clusterInfo.deviceNum);

    nlohmann::json rankListJson;
    CHK_RET(TransformServerList(clusterInfo, rankListJson));
    ClusterJson[PROP_RANK_LIST] = rankListJson;

    ClusterJson[PROP_STATUS] = "completed";
    ClusterJson[PROP_VERSION] = (deviceType == DevType::DEV_TYPE_910_93) ? "1.2" : "1.0";
    return HCCL_SUCCESS;
}
}  // namespace hccl