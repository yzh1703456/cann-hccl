/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCHER_H
#define TOPO_MATCHER_H

#include <condition_variable>
#include "dispatcher.h"
#include "comm_base_pub.h"
#include "externalinput_pub.h"
#include "coll_alg_param.h"
#include "comm_factory_pub.h"
#include "common_pub.h"
#include "hccl_common.h"
#include "calc_impl.h"

namespace hccl {
constexpr u32 COMM_LEVEL1_INDEX = COMM_LEVEL1;
using HcclAlgoInfo = struct HcclAlgoInfoDef {
    bool inlineReduceSwitchOn;       // 收到数量时同时完成Reduce计算
    std::string identifier;
    bool isUsedRdmaLevel0;

    HcclAlgoInfoDef()
        : inlineReduceSwitchOn(true),
        identifier(""),
        isUsedRdmaLevel0(false)
    {}
};

struct HcclTopoInfo {
    u32 userRank;                    // 通信域 RankID
    u32 userRankSize;                // 通信域的 Rank数量
    u32 devicePhyId;
    s32 deviceLogicId;
    std::vector<u32> nicList;
    bool isSingleMeshAggregation;
    u32 deviceNumPerAggregation;     // 每个module中的Device数量
    u32 superPodNum;                 // 集群中总的超节点数
    DevType deviceType;
    TopoType topoType;
    bool is310P3Common;
    u32 serverNum;
    u32 meshAggregationRankSize;
    u32 multiModuleDiffDeviceNumMode;
    u32 multiSuperPodDiffServerNumMode;
    bool isDiffDeviceType;
    u32 gcdDeviceNumPerAggregation;
    u32 realUserRank;
    bool isDiffDeviceModule;
    u32 moduleNum;
    bool useSuperPodMode;
    std::unordered_map<u32, bool> isUsedRdmaMap;
    std::unordered_map<u32, u32> pairLinkCounter; // server内所有device间的链路类型计数

    std::vector<std::vector<std::vector<std::vector<u32>>>> CommPlaneSubGroupVector; // 保存所有 level 的通信分组信息
    std::map<std::string, std::string> ahcAlgOption;
    std::vector<bool> isAsymPlanVector; // 对应 level 是否为非对称

    HcclTopoInfo()
        : userRank(0),
        userRankSize(0),
        devicePhyId(0),
        deviceLogicId(0),
        nicList(0),
        isSingleMeshAggregation(false),
        deviceNumPerAggregation(0),
        superPodNum(0),
        deviceType(DevType::DEV_TYPE_COUNT),
        topoType(TopoType::TOPO_TYPE_COMMON),
        is310P3Common(false),
        serverNum(0),
        meshAggregationRankSize(0),
        multiModuleDiffDeviceNumMode(0),
        multiSuperPodDiffServerNumMode(0),
        isDiffDeviceType(false),
        realUserRank(0),
        isDiffDeviceModule(false),
        moduleNum(0),
        useSuperPodMode(false)
    {}
};

using HcclExternalEnable = struct HcclExternalEnableDef {
    u32 enableRdmaSdmaConcurrent;
    u32 enableFfts;
    u32 deterministic;
    u32 highPerfEnable;
    u32 intraRoceSwitch;
    u32 dumpDebug;
    u32 interHccsDisable;
    bool aivMode;
    bool aicpuUnfold;

    HcclExternalEnableDef()
        : enableRdmaSdmaConcurrent(0),
        enableFfts(1),
        deterministic(0),
        highPerfEnable(0),
        intraRoceSwitch(0),
        dumpDebug(0),
        interHccsDisable(0),
        aivMode(false),
        aicpuUnfold(false)
    {}
};

bool CheckRankNeighbors(const std::vector<u32> &nicList);
bool CheckSdmaWithRohTopo(const std::vector<u32> &nicList, std::vector<u32> &topoList);

class TopoMatcher {
public:
    explicit TopoMatcher(const std::vector<std::vector<std::vector<u32>>> CommPlaneRanks,
                         const std::vector<bool> isBridgeVector,
                         HcclTopoInfo& topoInfo,
                         HcclAlgoInfo& algoInfo,
                         HcclExternalEnable& externalEnable,
                         std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank);
    HcclResult CalcCommPlaneInfo(const std::string &tag, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport, TransportMemType inPutMemType,
        TransportMemType outPutMemType);
    HcclTopoInfo GetTopoInfo();
    HcclAlgoInfo GetAlgoInfo();
    u32 GetExternalInputEnableRdmaSdmaConcurrent();
    u32 GetExternalInputHcclEnableFfts();
    u32 GetExternalInputHcclDeterministic();
    u32 GetExternalInputHcclHighPerfEnable();
    u32 GetExternalInputIntraRoceSwitch();
    u32 GetExternalInputHcclDumpDebug();
    u32 GetExternalInputInterHccsDisable();
    bool CheckSdmaWithRohTopo(const std::vector<u32> &nicList, std::vector<u32> &topoList);
    HcclResult GetSubRootForScatter(const u32 root, u32& subRoot);
    u32 GetSubRootUserRank(const u32 userRank, const u32 rootUserRank);
    u32 GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank);
    u32 GetSubRootWithSuperPod(const u32 userRank, const u32 rootUserRank);
    HcclResult GetLocalSuperPodRankSize(const u32 userRank, u32& devNumInlocalPod, u32& rankIdxInPod);
    HcclResult GetLocalServerRankSize(const u32 userRank, u32& devNumInlocalServer, u32& rankIdxInServer);
    HcclResult SetDeterministicConfig(const u8 deterministic);
    HcclResult SetAivModeConfig(const bool aivMode);
    HcclResult SetAicpuUnfoldConfig(const bool aicpuUnfold);
    u8 GetDeterministicConfig() const;
    bool GetAivModeConfig() const;
    bool GetAicpuUnfoldConfig() const;
    HcclResult GetGlobalSubGroups(const CommPlane level, std::vector<std::vector<std::vector<u32>>> &globalSubGroups);
    HcclResult SetGlobalSubGroups(const CommPlane level, std::vector<std::vector<std::vector<u32>>> &globalSubGroups);
    HcclResult GetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>> &commPlaneSubGroupVector);
    HcclResult SetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>> &commPlaneSubGroupVector);
    HcclResult GetAHCAlgOption(std::map<std::string, std::string> &ahcAlgOption);

protected:

private:

    HcclResult GetRankMap(const CommParaInfo &commParaInfo, std::vector<SingleSubCommTransport> &commTransport);

    HcclResult SetRankMap();

    HcclResult SetIsUsedRdma(const CommParaInfo &commParaInfo, std::vector<SingleSubCommTransport> &commTransport);

    HcclResult GetSub2UserRankMap(CommPlane commPlane, u32 ringIndex, std::map<u32, u32> &subCommRank2UserRank);

    HcclResult GetUserRank2SubMap(CommPlane commPlane, u32 ringIndex, std::map<u32, u32> &userRank2subCommRank);

    HcclResult GetIsUsedRdma(const CommParaInfo &commParaInfo, bool &isUsedRdma);

    const u32 GetSubCollectiveRank(const std::vector<u32> &vecPara) const;

    std::vector<std::vector<std::vector<u32>>> CommPlaneVector_;
    std::vector<bool> isBridgeVector_;
    HcclTopoInfo topoInfo_;
    HcclAlgoInfo algoInfo_;
    HcclExternalEnable externalEnable_;
    u32 userRank_;
    std::vector<std::vector<std::map<u32, u32>>> subCommRank2UserRank_;
    std::vector<std::vector<std::map<u32, u32>>> userRank2subCommRank_;

    // serverAndsuperPodToRank_[0]: 通信域在当前superPod内, 按照serverIdx划分的所有rank信息
    // serverAndsuperPodToRank_[1]: 通信域所有rank的信息, 按照superPodId -> RankInfo 的结构划分
    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank_;

    u32 userRankIdx_ = 0;
};
}  // namespace hccl

#endif /* * TOPO_MATCHER_H */