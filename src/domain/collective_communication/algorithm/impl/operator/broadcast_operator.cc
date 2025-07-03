/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_operator.h"
#include "device_capacity.h"
#include "rank_consistentcy_checker.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "coll_alg_op_registry.h"

namespace hccl {
BroadCastOperator::BroadCastOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_BROADCAST)
{
    // 由于bcast暂不支持server间ring，需继续使用HD或NHR
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) &&
        !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) &&
        !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
        HCCL_WARNING("[BroadCastOperator][BroadCastOperator] do not support ring in AlgoLevel1 yet, reset algType=HD.");
    }
}
BroadCastOperator::~BroadCastOperator()
{
}

HcclResult BroadCastOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    if (isDiffDeviceType_) {
        ret = SelectAlgforMix(param, algName);
    } else if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (Is310PDevice() && topoType_ == TopoType::TOPO_TYPE_2P_MESH) {
        ret = SelectAlgfor310P(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        ret = SelectAlgfor91093(param, algName);
    } else {
        HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastSelector][SelectAlg]tag[%s], broadcast failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        newTag = tag;
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        newTag = tag;
        u32 part1Size = 2 * (moduleNum_ - (1 << static_cast<u32>(log2(moduleNum_))));
        u32 rootId = param.root / deviceNumPerAggregation_;
        std::string appendTag = std::to_string((rootId >= part1Size) || ((rootId % 2) == 0));
        newTag = newTag + '_' + appendTag;
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(appendTag, param.tag));
        }
    } else if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType1), HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] broadcast newTag is [%s]", newTag.c_str());
    return ret;
}

HcclResult BroadCastOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{

    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[BroadCastOperator][SelectAlgforMix] only support NHR in AlgoLevel1 yet, "\
            "default is algType=NHR.");
        algName = "BroadCastMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        HCCL_WARNING("[BroadCastOperator][SelectAlgforMix] only support ring in AlgoComm yet, "\
            "default is algType=ring.");
        algName = "BroadCastComm";
    }

    HCCL_INFO("[SelectAlgforMix] broadcast SelectAlgforMix is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "BroadCastCommFor310P";
    HCCL_INFO("[SelectAlgfor310P3] broadcast SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor310P(const OpParam& param, std::string& algName)
{
    algName = "BroadcastPlusBroadcast";
    HCCL_INFO("[SelectAlgfor310P] broadcast SelectAlgfor310P is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910A] broadcast SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "BroadCastMeshExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_4P_RING) {
        algName = "BroadCast4pRingExecutor";
    } else if (isRingTopo) {
        algName = "BroadCastRingExecutor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor910B] broadcast SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult BroadCastOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    // level 1重定向为NHR, 因scatter && broadcast只支持nhr/nb
    if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) &&
        !(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[BroadCastOperator][BroadCastOperator] do not support ring in AlgoLevel1 yet, reset algType=NHR.");
    }

    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING("The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable"\
            "to be greater than the current data volume[%llu] bytes to improve the performance of the 91093 environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }

    bool smallCountOptimSingleServer =
        (serverNum_ == 1) &&
        ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ||
        (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode)) &&
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_2_MB * userRankSize_);
    bool smallCountOptimMultiServer =
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && (serverNum_ != 1) && (superPodNum_ == 1) &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_256_KB * userRankSize_);
    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "BroadCastComm";
    } else if (smallCountOptimMultiServer) {
        algName = "BroadCastComm";
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    } else if (smallCountOptimSingleServer) {
        algName = "BroadCastSmallCountExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        algName = "BroadCastRingFor91093Executor";
    } else {
        algName = "BroadCastComm";
    }
    HCCL_INFO("[SelectAlgfor91093] broadcast SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_BROADCAST, Broadcast, BroadCastOperator);
}