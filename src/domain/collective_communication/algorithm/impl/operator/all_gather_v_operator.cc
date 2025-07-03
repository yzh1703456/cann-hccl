/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_v_operator.h"
#include "device_capacity.h"
#include "executor_impl.h"
#include "coll_alg_op_registry.h"
#include "hccl_aiv.h"

namespace hccl {
AllGatherVOperator::AllGatherVOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER_V)
{
}

AllGatherVOperator::~AllGatherVOperator()
{
}

HcclResult AllGatherVOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;

    if(deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else if(deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else {
        HCCL_ERROR("[AllGatherVOperator][SelectAlg] all_gatherv only support A2 and 310P.");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherVOperator][SelectAlg]tag[%s], all_gatherv failed, return[%d]", tag.c_str(), ret), ret);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = algType_.algoLevel1;
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType1), HCCL_E_INTERNAL);

        newTag = tag + level1Iter->second + algName;
    }

    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] all_gatherv newTag is [%s]", newTag.c_str());
    return ret;
}

HcclResult AllGatherVOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    const auto *countsPtr = static_cast<const u64*>(param.VDataDes.counts);
    auto countsPerRank = std::vector<u64>(countsPtr, countsPtr + userRankSize_);
    u64 maxCount = *std::max_element(countsPerRank.begin(), countsPerRank.end());
    u32 unitSize = SIZE_TABLE[param.VDataDes.dataType];
    u64 dataSize = maxCount * unitSize;
    bool isBigData = false;

    if (dataSize > AIV_ALL_GATHER_SMALL_SIZE) {
        isBigData = true;
    } 

    if (!isSingleMeshAggregation_) {
        HCCL_ERROR("[AllGatherVOperator][SelectAlgfor910B] AllGatherV only support one module");
        return HCCL_E_NOT_SUPPORT;
    }
    
    bool isAivMode = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_ &&
                     IsSupportAIVCopy(param.VDataDes.dataType) && dataSize <= AIV_BIG_SIZE;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (isAivMode) {
            if (isBigData){
                algName = "AllGatherVMeshAivExecutor";
            }else{
                algName = "AllGatherVMeshAivSmallCountExecutor";
            }
        } else {
            algName = "AllGatherVMeshOpbaseExecutor";
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        algName = "AllGatherVMeshExecutor";
    }
    HCCL_INFO("[SelectAlgforA2] all_gather_v SelectAlgforA2 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherVOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "AllGatherVFor310PExecutor";
    HCCL_INFO("[SelectAlgfor310P3] all_gatherv SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER_V, AllGatherV, AllGatherVOperator);

}