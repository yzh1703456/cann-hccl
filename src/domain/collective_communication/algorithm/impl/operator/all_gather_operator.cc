/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_operator.h"
#include "device_capacity.h"
#include "rank_consistentcy_checker.h"
#include "executor_impl.h"
#include "coll_alg_op_registry.h"
#include "hccl_aiv.h"

namespace hccl {
AllGatherOperator::AllGatherOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER)
{
}

AllGatherOperator::~AllGatherOperator()
{
}

HcclResult AllGatherOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    if (userRankSize_ == 1 && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        algName = "AllGatherSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret;

    ret = SelectAlgforNew(param,algName);
    
    // if (isDiffDeviceType_) {
    //     ret = SelectAlgforMix(param, algName);
    // } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
    //     ret = SelectAlgfor310P3(param, algName);
    // } else if (deviceType_ == DevType::DEV_TYPE_910) {
    //     ret = SelectAlgfor910A(param, algName);
    // } else if (deviceType_ == DevType::DEV_TYPE_910B) {
    //     ret = SelectAlgfor910B(param, algName);
    // } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
    //     ret = SelectAlgfor91093(param, algName);
    // }  else {
    //     HCCL_ERROR("[SelectAlg] device type[%d] is out of range for selector.", deviceType_);
    //     return HCCL_E_NOT_SUPPORT;
    // }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllGatherSelector][SelectAlg]tag[%s], all_gather failed, return[%d]", tag.c_str(), ret), ret);
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
    HCCL_INFO("[SelectAlg] all_gather newTag is [%s]", newTag.c_str());
    return ret;
}

//直接选择我们的算法
HcclResult AllGatherOperator::SelectAlgforNew(const OpParam& param, std::string& algName)
{

    algName = "AllGatherRingFor91093Executor";
    HCCL_INFO("[SelectAlgfor91093] Current topoType_ = [%d], serverNum = [%u], deviceNumPerAggregation = [%u], workflowMode = [%d], aicpuUnfoldMode = [%d]",
        topoType_, serverNum_, deviceNumPerAggregation_, workflowMode_, param.aicpuUnfoldMode);

    HCCL_INFO("[SelectAlgforRing] all_gather SelectAlgforNew is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgforMix(const OpParam& param, std::string& algName)
{

    if (gcdDeviceNumPerAggregation_ > 1) {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
        HCCL_WARNING("[AllGatherOperator][SelectAlgforMix]only support NHR in AlgoLevel1 yet, "\
            "default is algType=NHR.");
        algName = "AllGatherMixExecutor";
    } else {
        algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        HCCL_WARNING("[AllGatherOperator][SelectAlgforMix]only support ring in AlgoComm yet, "\
            "default is algType=ring.");
        algName = "AllGatherComm";
    }

    HCCL_INFO("[SelectAlgforMix] all_gather SelectAlgforMix is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "AllGatherFor310PExecutor";
    HCCL_INFO("[SelectAlgfor310P3] all_gather SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "AllGatherMeshExecutor";
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910A] all_gather SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;
    
    bool isAivMode = topoMatcher_->GetAivModeConfig() && isSingleMeshAggregation_ &&
        IsSupportAIVCopy(param.DataDes.dataType) && dataSize <= AIV_BIG_SIZE;
    if (isAivMode) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && dataSize <= AIV_ALL_GATHER_SMALL_SIZE) {
            algName = "AllGatherMeshAivSmallCountExecutor"; 
            HCCL_INFO("[SelectAlgfor910BAIV] AllGather SelectAlgfor910B is algName [%s]", algName.c_str());
        } else {
            algName = "AllGatherMeshAivExecutor"; 
            HCCL_INFO("[SelectAlgfor910BAIV] AllGather SelectAlgfor910B is algName [%s]", algName.c_str());
        }
        return HCCL_SUCCESS;
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLGATHER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor910B] context num[%u] is out of capacityof FFTS+ graph[%u],"
                "reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    // 多机场景下aiv支持情况
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize = 0;
    u64 commOutputSize = 0;

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    bool isServNumPowOfTwo = (serverNum_ > 0) && ((serverNum_ & (serverNum_ - 1)) == 0);
    bool isSupportAivRdmaCount = !isSingleMeshAggregation_ && !multiModuleDiffDeviceNumMode_ &&
        (isServNumPowOfTwo || dataSize <= HCCL_SMALL_COUNT_128_KB) &&
        dataSize * userRankSize_ <= HCCL_MID_COUNT_16_MB && dataSize <= HCCL_SMALL_COUNT_256_KB;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    // 暂只支持单算子模式
    bool isCCLBufferGE16M = isOpbase &&
        (commInputSize >= HCCL_MID_COUNT_16_MB && commOutputSize >= HCCL_MID_COUNT_16_MB);
    bool isAivRdmaMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVCopy(param.DataDes.dataType) &&
        isMeshTopo && isCCLBufferGE16M && isSupportAivRdmaCount;
    if (isAivRdmaMode) {
        algName = "AllGatherAivRdmaExecutor";
    } else if (isMeshTopo) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (isSingleMeshAggregation_) {
                algName = "AllGatherMeshOpbaseExecutor";
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
                algName = "AllGatherMeshOpbasePipelineExecutor";
            }
        }
        if (algName.empty()) {
			if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || dataSize > HCCL_SMALL_COUNT_1_MB) {
				algName = "AllGatherMeshExecutor";
			} else {
				algName = "AllGatherMeshGraphExecutor";
			}
        }
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910B] all_gather SelectAlgfor910B is algName [%s], current mode is [%u]", algName.c_str(), workflowMode_);
    return HCCL_SUCCESS;
}

bool AllGatherOperator::SmallCountOptimMultiServer(const OpParam& param)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 totalSize = param.DataDes.count * unitSize * userRankSize_;
    void *commInputPtr = nullptr;
    u64 commInputSize = 0;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    bool dmaReduceLimit= (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        (((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_FOUR == 0) && (commInputSize * HCCL_DEVICE_NUM_FOUR < totalSize)) ||
        ((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_TWO == 0) && (commInputSize * HCCL_DEVICE_NUM_TWO < totalSize)) ||
        ((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_TWO != 0) && (commInputSize < totalSize)) || retryEnable_);
    bool smallCountOptimMultiServer =
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && (serverNum_ != 1) && (superPodNum_ == 1) &&
        (((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_FOUR == 0) && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_4_MB)) ||
        ((deviceNumPerAggregation_ % HCCL_DEVICE_NUM_FOUR != 0) && (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_1_MB))) &&
        !dmaReduceLimit && !GetExternalInputInterHccsDisable();
    return smallCountOptimMultiServer;
}

HcclResult AllGatherOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    if (dataSize >= cclBufferManager_.GetInCCLbufferSize()) {
        HCCL_WARNING("The current inCCLbufferSize is [%llu] bytes, change the HCCL_BUFFSIZE environment variable"\
            "to be greater than the current data volume[%llu] bytes to improve the performance of the 91093 environment.",
            cclBufferManager_.GetInCCLbufferSize(), dataSize);
    }
    //判断当前是否是“单算子模式”，用于后续通信策略或算法路径的选择
    bool isOpbase = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    bool isAivMode = topoMatcher_->GetAivModeConfig() && IsSupportAIVCopy(param.DataDes.dataType) && serverNum_ == 1 &&
        ((isOpbase && dataSize <= AIV_ALL_GATHER_A3_ENTRY_SIZE) ||
        (!isOpbase && dataSize <= AIV_ALL_GATHER_A3_GRAPH_ENTRY_SIZE));
    if (isAivMode) {
        //区分是否为小数据量优化路径
        if ((isOpbase && dataSize <= AIV_ALL_GATHER_SMALL_SIZE)
            || (!isOpbase && dataSize <= AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE)) {
            algName = "AllGatherMeshAivSmallCountExecutor"; // 目前a3 aivmode下单算子模式正好全走小数据
        } else {
            algName = "AllGatherMeshAivExecutor"; 
        }
        HCCL_INFO("[SelectAlgfor91093] all_gather SelectAlgfor91093 is algName [%s]", algName.c_str());
        return HCCL_SUCCESS;
    }
    //判断是否使用Small Count优化路径（不同于AIV）
    bool smallCountOptimSingleServer = (serverNum_ == 1) &&
        ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ||
        (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode)) &&
        (param.DataDes.count * unitSize <= HCCL_SMALL_COUNT_2_MB) &&
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) && !GetExternalInputInterHccsDisable();
    bool smallCountOptimMultiServer = SmallCountOptimMultiServer(param);
        
    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "AllGatherComm"; //通用通信实现
    } else if (smallCountOptimMultiServer) {
        algName = "AllGatherSmallCount"; //多机小数据量优化路径
    } else if (smallCountOptimSingleServer) {  //单机小数据量优化路径
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            algName = "AllGatherMeshOpbaseExecutor";  //单算子模式
        } else {
            algName = "AllGatherMeshExecutor";  //图模式
        }
        //RDMA/SDMA并发+双环拓扑场景
    } else if (GetExternalInputEnableRdmaSdmaConcurrent() && topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        !param.aicpuUnfoldMode && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] concurrent only support ring or NB in AlgoLevel1 "\
                "yet, default is ring.");
        }
        algName = "AllGatherDoubleRingConcurrentExecutor";
    } else {
        if (GetExternalInputEnableRdmaSdmaConcurrent()) {
            if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB)) {
                algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
            }
        } else if (!(algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB ||
            algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING)) {
            algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet, "\
                "default is algType=NHR.");
        }
        if (IsSupportUnifiedMarch(param, topoType_, serverNum_, superPodNum_)) {
            algName = "AllGatherSemiRingExecutor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            algName = "AlignedAllGatherDoubleRingFor91093Executor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING){
            algName = "AllGatherRingFor91093Executor";
        } else {
            algName = "AllGatherComm";
        }
    }
    HCCL_INFO("[SelectAlgfor91093] all_gather SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER, AllGather, AllGatherOperator);

}