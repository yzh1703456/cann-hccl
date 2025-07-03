/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "coll_alg_utils.h"
#include "workflow_pub.h"

namespace hccl {
    
bool IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0)
{
    return originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_2P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_1P_MESH;
}

bool IsAlltoAllvcSatisfyBufferSize(const OpParam& param, u32 userRankSize) {
    for (u32 i = 0; i < userRankSize; i++) {
        u64 maxSendLength = 0;
        u64 maxRecvLength = 0;
        // 计算每个rank需使用的中转内存大小是否满足cclbuffer大小
        for (u32 j = 0; j < userRankSize; j++) {
            u64 curSendCounts =
                *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i * userRankSize + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];

            u64 curRecvCounts =
                *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i + userRankSize * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];

            maxSendLength += curSendLength;
            maxRecvLength += curRecvLength;
        }
        if ((maxSendLength <= GetExternalInputCCLBuffSize()) || (maxRecvLength <= GetExternalInputCCLBuffSize())) {
            return false;
        }
    }
    return true;
}

bool IsSupportUnifiedMarch(const OpParam& param, const TopoType& topoType, u32 serverNum, u32 superPodNum)
{
    bool isGraphMode = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    bool isDoubleRing = topoType == TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    bool isSingleServer = (serverNum == 1) && (superPodNum == 1);
    return (param.aicpuUnfoldMode) && isDoubleRing && isGraphMode && isSingleServer;
}

bool IsSupportDirectFullmeshForAlltoallv(const OpParam& param, DevType deviceType, bool useSuperPodMode, u32 serverNum,
    bool isSingleMeshAggregation, u32 userRankSize)
{
    bool isDeviceType = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B);
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isHCCS = false;
    bool isSatisfyBuffer = true;
    if (deviceType == DevType::DEV_TYPE_910_93) {
        isHCCS = (serverNum > 1) ?
            (!GetExternalInputInterHccsDisable() && useSuperPodMode) : (!GetExternalInputInterHccsDisable());
    } else if (deviceType == DevType::DEV_TYPE_910B) {
        isHCCS = (isSingleMeshAggregation) ? (true) : (false);
        if (isHCCS && (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC ||
                       param.opType == HcclCMDType::HCCL_CMD_ALLTOALL)) {
            // 910B场景下alltoall和alltoallvc需满足数据量大于cclbuffer大小条件
            isSatisfyBuffer = IsAlltoAllvcSatisfyBufferSize(param, userRankSize);
        }
    }
    HCCL_DEBUG("[IsSupportDirectFullmeshForAlltoallv]isDevice91093[%u], isOpbase[%u], isHCCS[%u], isSatisfyBuffer[%u]",
        isDeviceType, isOpbase, isHCCS, isSatisfyBuffer);
    return isDeviceType && isOpbase && isHCCS && isSatisfyBuffer;
}

bool SatisfyIntraSuperPod(DevType deviceType, u32 rankSize, bool useSuperPodMode, u32 superPodNum)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    bool isSingleSuperPod = superPodNum == 1;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    return (isDevice91093 && rankSizeSupport && isHCCS && isSingleSuperPod && isOpbase);
}

bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool twoLevelIntraUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    HCCL_DEBUG("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition]isDevice91093 %u twoLevelIntraUseMesh %u isHCCS %u",
        isDevice91093, twoLevelIntraUseMesh, isHCCS);
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isDevice91093),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 910_93 device type, use default algorithm type"));
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isHCCS),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91093 && twoLevelIntraUseMesh && rankSizeSupport && isHCCS);
}

template<typename keyType>
std::string GetAlgoString(const std::map<keyType, std::string>& levelMap, keyType key) {
    auto iter = levelMap.find(key);
    if (iter == levelMap.end()) {
        return "invalid algo type";
    } else {
        return iter->second;
    }
}

std::string AlgTypeToStr(const AlgType algType)
{
    AlgTypeLevel0 algTypeLevel0 = algType.algoLevel0;
    AlgTypeLevel1 algTypeLevel1 = algType.algoLevel1;
    AlgTypeLevel2 algTypeLevel2 = algType.algoLevel2;
    std::string algStrLevel0 = GetAlgoString(HCCL_ALGO_LEVEL0_NAME_MAP, algTypeLevel0);
    std::string algStrLevel1 = GetAlgoString(HCCL_ALGO_LEVEL1_NAME_MAP, algTypeLevel1);
    std::string algStrLevel2 = GetAlgoString(HCCL_ALGO_LEVEL2_NAME_MAP, algTypeLevel2);
    std::string algStr;
    algStr.append("level0:").append(algStrLevel0).append(",level1:").append(algStrLevel1).append(",level2:").append(algStrLevel2);
    return algStr;
}

bool Is310P3Common(bool isHaveCpuRank, DevType deviceType)
{
    return !isHaveCpuRank && !Is310PDevice() && deviceType == DevType::DEV_TYPE_310P3;
}

u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum)
{
    u64 piplineSliceNum = 0;
    bool isInterRing = false;
    if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        isInterRing = true;
    } else {
        isInterRing = false;
    }

    do {
        if (!GetExternalInputHcclEnablePipline()) {
            break;
        }
        /* 不支持pipline流水的场景 */
        // 支持的硬件场景
        if (deviceType != DevType::DEV_TYPE_910B || deviceNumPerAggregation < HCCL_DEVICE_NUM_TWO ||
            moduleNum < HCCL_DEVICE_NUM_TWO) {
            break;
        }
        // 支持的算子和算法场景
        if (opType != HcclCMDType::HCCL_CMD_ALLREDUCE ||
           (isInterRing && moduleNum > MAX_RING_PIPLINE_SERVER_NUM)) {
            break;
        }
        u64 sliceNumTemp = std::min(dataSize / deviceNumPerAggregation / MIN_PER_LINK_DATA_SIZE, MAX_PIPLINE_SLICE_NUM);
        // 图模式切分数量 <= 1时, 不做切分
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            sliceNumTemp <= MIN_PIPLINE_SLICE_NUM) {
            break;
        }

        /* 支持pipline流水, 但数据量不足以进行切分的场景 */
        // Server间使用Ring算法, 且单Server数据量<64KB时, 不做切分
        if ((isInterRing && dataSize / moduleNum < MIN_RING_DATA_SIZE)) {
            sliceNumTemp = 1;
        }
        // 支持pipline但数据量不满足切分条件时, 返回1, 用于单算子场景预申请流资源
        piplineSliceNum = (sliceNumTemp == 0) ? 1 : sliceNumTemp;
    } while (0);
    return piplineSliceNum;
}

bool HcclOpInplaceDefaultCase(const OpParam &param, u8 &isInplaceStatus)
{
    // unknown op
    if (param.inputPtr != param.outputPtr) {
        // 可以走重执行
        HCCL_DEBUG("[CollAlgOperator][IsHcclOpInplace]param.inputPtr[%p] != param.outputPtr[%p]. They do not overlap.",
            param.inputPtr, param.outputPtr);
        isInplaceStatus = 0;
        return false;
    } else {
        HCCL_DEBUG("[CollAlgOperator][IsHcclOpInplace]param.inputPtr[%p] == param.outputPtr[%p]. They overlap.",
            param.inputPtr, param.outputPtr);
        isInplaceStatus = 1;
        return true;
    }
}

bool IsInputOutputOverlap(const OpParam &param, u64 inputDataSize, u64 outputDataSize, u8 &isInplaceStatus)
{
    if (inputDataSize == 0 || outputDataSize == 0) {
        // 不存在overlap情况
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They do not overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 0;
        return false;
    }
    u64 inputStart = reinterpret_cast<u64>(param.inputPtr);
    u64 inputEnd = reinterpret_cast<u64>(param.inputPtr) + inputDataSize - 1;
    u64 outputStart = reinterpret_cast<u64>(param.outputPtr);
    u64 outputEnd = reinterpret_cast<u64>(param.outputPtr) + outputDataSize - 1;

    if (inputStart <= outputEnd && outputStart <= inputEnd) {
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 2; // The status 2 is overlap with dataSize.
        return true;
    } else {
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They do not overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 0;
        return false;
    }
}

bool IsInputOutPtrNotNullPtr(const OpParam &param, u8 &isInplaceStatus)
{
    if (param.inputPtr == nullptr || param.outputPtr == nullptr) {
        // 不存在overlap情况
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]param.tag[%s], the inputPtr[%p], the outputPtr[%p]."
            "They do not overlap.", param.tag.c_str(), param.inputPtr, param.outputPtr);
        isInplaceStatus = 0;
        return false;
    } else {
        return true;
    }
}

u32 InplaceDataUnitSize(const HcclCMDType &opType, const OpParam &param)
{
    u32 unitSize = 0;
    if (opType != HcclCMDType::HCCL_CMD_ALLTOALLV && opType != HcclCMDType::HCCL_CMD_ALLTOALLVC &&
        opType != HcclCMDType::HCCL_CMD_ALLTOALL) {
        if (param.DataDes.dataType >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_WARNING("[InplaceDataUnitSize] out of range[%d, %d]",
                HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
            return 0;
        }
        unitSize = SIZE_TABLE[param.DataDes.dataType];
    }
    return unitSize;
}

bool IsHcclOpInplace(const HcclCMDType &opType, const OpParam &param, u32 userRank, u32 userRankSize,
    u8 &isInplaceStatus)
{
    if (!IsInputOutPtrNotNullPtr(param, isInplaceStatus)) {
        return false;
    }
    u32 unitSize = InplaceDataUnitSize(opType, param);
    u64 inputDataSize = 0;
    u64 outputDataSize = 0;
    switch (opType) {
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
            isInplaceStatus = 0;
            return false;
            break;
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            inputDataSize = param.DataDes.count * unitSize;
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE:
            inputDataSize = param.DataDes.count * unitSize;
            if (userRank == param.root) {
                outputDataSize = param.DataDes.count * unitSize;
            }
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            inputDataSize = param.DataDes.count * unitSize;
            outputDataSize = param.DataDes.count * unitSize * userRankSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            inputDataSize = param.DataDes.count * unitSize * userRankSize;
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_GATHER:
            inputDataSize = param.DataDes.count * unitSize;
            if (userRank == param.root) {
                outputDataSize = param.DataDes.count * unitSize * userRankSize;
            }
            break;
        case HcclCMDType::HCCL_CMD_SCATTER:
            if (userRank == param.root) {
                inputDataSize = param.DataDes.count * unitSize * userRankSize;
            }
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        default:
            return HcclOpInplaceDefaultCase(param, isInplaceStatus);
            break;
    }
    return IsInputOutputOverlap(param, inputDataSize, outputDataSize, isInplaceStatus);
}

bool CheckUserInMemNotLargerThanCCLInMem(const HcclCMDType &opType, OpParam &param,
    u64 commInputSize, u32 userRankSize)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = 0;
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        dataSize = param.DataDes.count * unitSize * userRankSize;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        dataSize = param.DataDes.count * unitSize;
    }

    if (dataSize <= commInputSize) {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU] UserInMem[%llu] <= CCLInMem[%llu]", dataSize, commInputSize);
    } else {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU] UserInMem[%llu] > CCLInMem[%llu]", dataSize, commInputSize);
    }
    return dataSize <= commInputSize;
}

bool ExecutorOnlySupportDMAReduce(const std::string& algName)
{
    return (algName == "AllReduceMeshSmallCountExecutor") || (algName == "ReduceScatterDeterExecutor");
}

bool ExecutorCanSupportDMAReduce(const std::string& algName)
{
    const std::set<std::string> executorCanSupportDMAReduceSet = {
        "AllReduceRingFor91093Executor", "AllReduceDoubleRingConcurrentExecutor",
        "AllReduceFastDoubleRingFor91093Executor", "AlignedAllReduceDoubleRingFor91093Executor",
        "ReduceScatterRingFor91093Executor", "ReduceScatterDoubleRingConcurrentExecutor",
        "ReduceScatterFastDoubleRingFor91093Executor", "AlignedReduceScatterDoubleRingFor91093Executor"
        };
    if (executorCanSupportDMAReduceSet.find(algName) != executorCanSupportDMAReduceSet.end()) {
        return true;
    }
    return false;
}

bool ExecutorNoSupportDMAReduce(const std::string& algName)
{
    return (algName == "AllReduceComm") || (algName == "ReduceScatterComm");
}

bool ExecutorSupportInPlace(OpParam &param, const std::string& algName, bool retryEnable,
    InplaceSupportRetryStatus &inPlaceSupportRetryStatus)
{
    // case 2.2
    if (ExecutorOnlySupportDMAReduce(algName)) {
        if (retryEnable) {
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorOnlySupportDMAReduce[%s] is not allowed"
                " for inplace case, the executor without DMAReduce will be applied.", algName.c_str());
            inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1;
            return true;
        }
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorOnlySupportDMAReduce[%s] is not allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE1;
        return false;
    } else if (ExecutorNoSupportDMAReduce(algName)) {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorNoSupportDMAReduce[%s] is allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::ALWAYS_NO_DMA_REDUCE;
        return true;
    } else if (ExecutorCanSupportDMAReduce(algName)) {
        if (retryEnable) {
            // 对应的executor会感应RetryEnable环境变量，走非DMA削减逻辑
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorCanSupportDMAReduce[%s] is not allowed"
                " for inplace case, the executor without DMAReduce will be applied.", algName.c_str());
            inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2;
            return true;
        }
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorCanSupportDMAReduce[%s] is not allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE2;
        return false;
    } else {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The unknown executor[%s] does not support "
            "for an inplace case yet.", algName.c_str());
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::UNKONWN_EXECUTOR;
        return false;
    }
}

bool FitRetryConditionforInPlaceOp(
    const HcclCMDType &opType, OpParam &param, const std::string& algName, u64 commInputSize, u32 userRankSize,
    bool retryEnable,
    InplaceSupportRetryStatus &inPlaceSupportRetryStatus)
{
    // case 1 allgather or broadcast
    if (opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
        opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        inPlaceSupportRetryStatus = InplaceSupportRetryStatus::AG_BD_CASE;
        return true;
    }
    // case 2 reducescatter or allreduce
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // case 2.1
        if (CheckUserInMemNotLargerThanCCLInMem(opType, param, commInputSize, userRankSize)) {
            // case 2.4: 在hccl_communicator.cc的ExecOp之前已经该让图模式走单算子模式了，理论上不会进入此条件
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The retry with inplace case is expected to be supported, "
                "therefore HcclWorkflowMode is set to [%u]",
                static_cast<u8>(GetWorkflowMode()));
            return ExecutorSupportInPlace(param, algName, retryEnable, inPlaceSupportRetryStatus);
        } else {
            // case 2.3 UsrIn > CCLIn
            inPlaceSupportRetryStatus = InplaceSupportRetryStatus::USER_LARGER_THAN_CCL;
            return false;
        }
    }
    // 其他算子类型不支持
    inPlaceSupportRetryStatus = InplaceSupportRetryStatus::NOT_BASIC_OP_CASE;
    return false;
}

u32 CalGCD(std::vector<u32> &nums)
{
    if (nums.size() == 0) {
        return 1;
    }
    std::sort(nums.begin(), nums.end(), [](const u32 &num1, const u32 &num2) {
        return num1 > num2;
    });

    u32 curGcd = nums[0];
    for (u32 i = 1; i < nums.size(); i++) {
        curGcd = CalGCD(curGcd, nums[i]);
    }
    HCCL_DEBUG("[CalGCD]size[%u], gcd[%u]", nums.size(), curGcd);
    return curGcd;
}

u32 CalGCD(u32 a, u32 b)
{
    if (a == 0 || b == 0) {
        return 1;
    }

    u32 gcd = b;
    while (a % b != 0) {
        gcd = a % b;
        a = b;
        b = gcd;
    }
    HCCL_DEBUG("[CalGCD]a[%u] b[%u], gcd[%u]", a, b, gcd);
    return gcd;
}
}