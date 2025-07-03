/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_operator.h"
#include "device_capacity.h"
#include "hccl_aiv.h"

namespace hccl {
ReduceScatterVOperator::ReduceScatterVOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher) :
    CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V)
{
}

ReduceScatterVOperator::~ReduceScatterVOperator()
{
}

HcclResult ReduceScatterVOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    HcclResult ret;

    if(deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if(deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg] ReduceScatterV only support A2 and 310P.");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlg]tag[%s], reduce_scatter_v failed, return[%d]",
            tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = algType_.algoLevel1;
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
            CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
                HCCL_ERROR("level1: algType1[%u] is invalid.", algType1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.VDataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_V_NO_INLINE = "_no_inline";
        newTag = isInlineReduce ? newTag : newTag + REDUCE_SCATTER_V_NO_INLINE;
    }

    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] reduce_scatter_v newTag is [%s]", newTag.c_str());

    return ret;
}

HcclResult ReduceScatterVOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    if(!isSingleMeshAggregation_) {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlgforA2] ReduceScatterV only support one module.");
        return HCCL_E_NOT_SUPPORT;
    }

    const auto *countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 maxDataSize = maxCount * unitSize; // 单位：字节
    // 910B单机AIV模式下ReduceScatterV算子当前仅支持单卡数据量不大于256M的场景，大于256M暂不支持
    bool isAivMode = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_ && maxDataSize <= AIV_BIG_SIZE
        && IsSupportAIVReduce(param.VDataDes.dataType, param.reduceType)
        && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE;
    HCCL_INFO("[ReduceScatterVOperator][SelectAlgfor910B]isAivMode[%d], maxCount[%llu], maxDataSize[%llu], "
        "deterministic[%u], isSingleMeshAggregation[%d].", isAivMode, maxCount, maxDataSize,
        topoMatcher_->GetDeterministicConfig(), isSingleMeshAggregation_);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isAivMode) {
        if (maxDataSize > AIV_REDUCE_SCATTER_MID_SIZE) {
            algName = "ReduceScatterVAIVBigCountExecutor";
        } else {
            algName = "ReduceScatterVMeshAivSmallCountExecutor";
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshOpbaseExecutor";
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.VDataDes.dataType, param.reduceType)) {
        algName = "ReduceScatterVMeshExecutor";
    } else {
        HCCL_ERROR("[ReduceScatterVOperator][SelectAlgforA2] ReduceScatterV only support inlinereduce.");
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[SelectAlgforA2] reduce_scatter_v SelectAlgforA2 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterVOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "ReduceScatterVFor310PRing";
    HCCL_INFO("[SelectAlgfor310P3] reduce_scatter_v SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, ReduceScatterV, ReduceScatterVOperator);

}