/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_operator.h"
#include "device_capacity.h"
#include "executor_impl.h"
#include "stream_active_manager.h"
#include "all_gather_operator.h"
#include <vector>
#include "coll_alg_exec_registry.h"
#include "coll_alg_op_registry.h"
#include "coll_all_to_all_executor.h"
#include "hccl_aiv.h"

namespace hccl {

constexpr u64 ALLTOALL_PIPELINE_MIN_CCL_SIZE = 80 * 1024 * 1024;
constexpr u64 MAX_RMDA_RANK_SIZE = 8;

AlltoAllOperator::AlltoAllOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLTOALL)
{
}

AlltoAllOperator::~AlltoAllOperator()
{
}

void AlltoAllOperator::SetVirtualDispatcher(const HcclDispatcher vDispatcher)
{
    vDispatcher_ = vDispatcher;
    return;
}

void AlltoAllOperator::SetParallelTaskLoader(ParallelTaskLoader* parallelTaskLoader)
{
    parallelTaskLoader_ = parallelTaskLoader;
    return;
}

HcclResult AlltoAllOperator::CheckSendRecvParams(
    const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u32 rankSize = allMeshAggregationSendRecvInfo.size();
    for (u32 i = 0; i < rankSize; i++) {
        u32 sendsSize = allMeshAggregationSendRecvInfo[i].sendLength.size();
        u32 recvsSize = allMeshAggregationSendRecvInfo[i].recvLength.size();
        if (rankSize != sendsSize || rankSize != recvsSize) {
            HCCL_ERROR(
                "[AlltoAllV][CheckSendRecvParam] rankSize[%u], sendsSize[%u], recvsSize[%u] are not match Index[%u]",
                rankSize, sendsSize, recvsSize, i);
            return HCCL_E_PARA;
        }
        for (u32 j = 0; j < sendsSize; j++) {
            if (allMeshAggregationSendRecvInfo[i].sendLength[j] != allMeshAggregationSendRecvInfo[j].recvLength[i]) {
                HCCL_ERROR("SendLength[%u][%u]: %llu and recvLength[%u][%u]: %llu are not match", i, j,
                    allMeshAggregationSendRecvInfo[i].sendLength[j], j, i,
                    allMeshAggregationSendRecvInfo[j].recvLength[i]);
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllvcSendRecvInfo(const void *sendCountMatrix, HcclDataType sendType,
    HcclDataType recvType)
{
    allMeshAggregationSendRecvInfo_.clear();
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendCounts.resize(userRankSize_);
        sendRecvInfo.sendDispls.resize(userRankSize_);
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        u64 curSendDispls = 0;
        u64 curSendOffset = 0;

        sendRecvInfo.recvCounts.resize(userRankSize_);
        sendRecvInfo.recvDispls.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        u64 curRecvDispls = 0;
        u64 curRecvOffset = 0;
        // sendCountMatrix[i * userRankSize_ + j] 代表rank i发送到rank j的count参数
        for (u32 j = 0; j < userRankSize_; j++) {
            u64 curSendCounts = *(static_cast<const u64 *>(sendCountMatrix) + i * userRankSize_ + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[sendType];
            sendRecvInfo.sendCounts[j] = curSendCounts;
            sendRecvInfo.sendDispls[j] = curSendDispls;
            sendRecvInfo.sendLength[j] = curSendLength;
            sendRecvInfo.sendOffset[j] = curSendOffset;
            curSendDispls += curSendCounts;
            curSendOffset += curSendLength;

            u64 curRecvCounts = *(static_cast<const u64 *>(sendCountMatrix) + i + userRankSize_ * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
            sendRecvInfo.recvCounts[j] = curRecvCounts;
            sendRecvInfo.recvDispls[j] = curRecvDispls;
            sendRecvInfo.recvLength[j] = curRecvLength;
            sendRecvInfo.recvOffset[j] = curRecvOffset;
            curRecvDispls += curRecvCounts;
            curRecvOffset += curRecvLength;

            HCCL_DEBUG("GetAlltoAllvcSendRecvInfo rank[%u], sendCounts[%llu], sendDispls[%llu] "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[j], sendRecvInfo.sendDispls[j],
                sendRecvInfo.recvCounts[j], sendRecvInfo.recvDispls[j]);
        }
        allMeshAggregationSendRecvInfo_.push_back(sendRecvInfo);
    }
    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));
    return HCCL_SUCCESS;
}

void AlltoAllOperator::UpdateAlltoAllCopyMode(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    std::string& copyMode)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 maxSendSize = 0;
        u64 maxRecvSize = 0;
        for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo) {
            for (u32 i = 0; i < userRankSize_; i++) {
                u64 curSendSize = sendRecvInfo.sendLength[i] + sendRecvInfo.sendOffset[i];
                maxSendSize = std::max(maxSendSize, curSendSize);
                u64 curRecvSize = sendRecvInfo.recvLength[i] + sendRecvInfo.recvOffset[i];
                maxRecvSize = std::max(maxRecvSize, curRecvSize);
            }
        }
        bool isAlltoAllZCopyMode = (maxSendSize <= GetExternalInputCCLBuffSize()) &&
                                   (maxRecvSize <= GetExternalInputCCLBuffSize());
        if (isAlltoAllZCopyMode) {
           copyMode = "ZCopy";
        }
        HCCL_INFO("[AlltoAllOperator][UpdateAlltoAllCopyMode] maxSendSize[%llu], maxRecvSize[%llu], "\
            "cclBufferSize[%llu], CopyMode[%s]", maxSendSize, maxRecvSize,
            GetExternalInputCCLBuffSize(), copyMode.c_str());
    } else {
        // 图模式走ZCopy实现
        copyMode = "ZCopy";
    }
}

HcclResult AlltoAllOperator::GetAlltoAllvSendRecvInfo(const OpParam& param, const HostMem &alltoallAddrInfoGathered)
{
    allMeshAggregationSendRecvInfo_.clear();
    u64 stepSize = sizeof(u64) * userRankSize_;
    const u32 addrItemNum = 4;
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    for (u32 i = 0; i < userRankSize_; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendLength.resize(userRankSize_);
        sendRecvInfo.sendOffset.resize(userRankSize_);
        sendRecvInfo.recvLength.resize(userRankSize_);
        sendRecvInfo.recvOffset.resize(userRankSize_);
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + 0 * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.sendOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvLength.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvLengthStep * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(sendRecvInfo.recvOffset.data(),
            stepSize,
            static_cast<u8 *>(alltoallAddrInfoGathered.ptr()) + i * stepSize * addrItemNum + recvOffsetStep * stepSize,
            stepSize));
        allMeshAggregationSendRecvInfo_.push_back(std::move(sendRecvInfo));
    }

    for (auto &sendRecvInfo : allMeshAggregationSendRecvInfo_) {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendRecvInfo.sendCounts.push_back(sendRecvInfo.sendLength[i] / SIZE_TABLE[param.All2AllDataDes.sendType]);
            sendRecvInfo.sendDispls.push_back(sendRecvInfo.sendOffset[i] / SIZE_TABLE[param.All2AllDataDes.sendType]);
            sendRecvInfo.recvCounts.push_back(sendRecvInfo.recvLength[i] / SIZE_TABLE[param.All2AllDataDes.recvType]);
            sendRecvInfo.recvDispls.push_back(sendRecvInfo.recvOffset[i] / SIZE_TABLE[param.All2AllDataDes.recvType]);
            HCCL_INFO("[GetAlltoAllvSendRecvInfo] rank[%u], sendCounts[%llu], sendDispls[%llu], "\
                "recvCounts[%llu], recvDispls[%llu]", i, sendRecvInfo.sendCounts[i], sendRecvInfo.sendDispls[i],
                sendRecvInfo.recvCounts[i], sendRecvInfo.recvDispls[i]);
            HCCL_INFO("[GetAlltoAllvSendRecvInfo] rank[%u], sendLength[%llu], sendOffset[%llu], "\
                "recvLength[%llu], recvOffset[%llu]", i, sendRecvInfo.sendLength[i], sendRecvInfo.sendOffset[i],
                sendRecvInfo.recvLength[i], sendRecvInfo.recvOffset[i]);
        }
    }

    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlgforAiv(const OpParam& param, std::string& algName)
{
    if (deviceType_ == DevType::DEV_TYPE_910B && param.opType == HcclCMDType::HCCL_CMD_ALLTOALL &&
        !isSingleMeshAggregation_) {
        // aiv模式下910A2多server场景 alltoall算子
        algName = "AlltoAllStagedAIVRdmaExecutor";
    } else if (deviceType_ == DevType::DEV_TYPE_910_93 && serverNum_ > 1) {
        algName = "AlltoAllMeshAivFor91093Executor";
    } else {
        algName = "AlltoAllMeshAivExecutor";
    }
    HCCL_INFO("[SelectAlgforAlltoAll] all_to_all algName is [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlgforAlltoAll(const OpParam& param, std::string& algName, std::string& copyMode)
{
    if (IsSatisfyAlltoAllAivCondition(param)) {
        CHK_RET(SelectAlgforAiv(param, algName));
        return HCCL_SUCCESS; // alltoall aiv不需要后面操作，直接返回
    }

    bool useOneLevelAlgorithm =
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE;
        // 用户配置打平 alltoall

    // NA+pairwise算法不支持A+X跨mesh两卡
    bool isSingleDeviceModuleP2p = (userRankSize_ <= HCCL_ALLTOALLV_P2P_SIZE);
    if (userRankSize_ == 1 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode) {
        algName = "RunAlltoAllSingleExecutor";
        return HCCL_SUCCESS ;
    } else if (isCommon310P3DUO_) {
        algName = "RunAlltoAllVFor310PExecutor";
    } else if (IsSupportDirectFullmeshForAlltoallv(param, deviceType_, useSuperPodMode_, serverNum_,
        isSingleMeshAggregation_, userRankSize_) || param.aicpuUnfoldMode || deviceType_ == DevType::DEV_TYPE_310P3) {
        algName = "RunAlltoAllDirectFullmesh";
        HCCL_INFO("[SelectAlgforAlltoAll] all_to_all algName is [%s]", algName.c_str());
        return HCCL_SUCCESS;
    } else if (IsSatisfyAlltoallPipelineCondition()) {
        algName = "RunAlltoAllVTwoLevelPipeline";
    } else if (SatisfyIntraSuperPod(deviceType_, userRankSize_, useSuperPodMode_, superPodNum_) ||
        useOneLevelAlgorithm || isAllRankSamePlane_ || isSingleDeviceModuleP2p || multiModuleDiffDeviceNumMode_ ||
        multiSuperPodDiffServerNumMode_) {
        algName = "RunAlltoAllVFullMesh";   //910B卡数不一致走这
    } else {
        algName = "RunAlltoAllVStaged";
    }

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        // alltoallv
        CHK_RET(GetAlltoAllvSendRecvInfo(param, hostCollectBuffer_));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALL){
        // alltoallvc&&alltoall
        CHK_RET(GetAlltoAllvcSendRecvInfo(param.All2AllDataDes.sendCountMatrix, param.All2AllDataDes.sendType,
            param.All2AllDataDes.recvType));
    } else {
        HCCL_ERROR("[AlltoAllOperator][SelectAlgforAlltoAll] get wrong opType");
        return HCCL_E_PARA;
    }
    UpdateAlltoAllCopyMode(allMeshAggregationSendRecvInfo_, copyMode);

    HCCL_INFO("[SelectAlgforAlltoAll] alltoall algName is [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    HcclResult ret;
    std::string copyMode = "BCopy";

    ret = SelectAlgforAlltoAll(param, algName, copyMode);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[SelectAlgforAlltoAll][SelectAlg]tag[%s], Alltoall failed, return[%d]", tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (IsSupportDirectFullmeshForAlltoallv(param, deviceType_, useSuperPodMode_, serverNum_,
            isSingleMeshAggregation_, userRankSize_) || param.aicpuUnfoldMode) {
            newTag = tag + algName;
        } else {
            newTag = tag + algName + copyMode;
        }
        newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    } else {
        newTag = tag;
    }
    HCCL_INFO("[SelectAlg] Alltoall operator newTag is [%s]", newTag.c_str());

    if (!IsSatisfyAlltoAllAivCondition(param) &&
         !IsSupportDirectFullmeshForAlltoallv(param, deviceType_, useSuperPodMode_, serverNum_,
            isSingleMeshAggregation_, userRankSize_) && !param.aicpuUnfoldMode) {
        CHK_RET(SetExcutorExtraInfo(algName, param));
    }
    return ret;
}

HcclResult AlltoAllOperator::GetAlltoAllvAllAddrInfo(u64 *sendLength, u64 *sendOffset,
    u64 *recvLength, u64 *recvOffset, std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    const u32 addrItemNum = 4;
    u64 stepSize = sizeof(u64) * userRankSize_;

    std::vector<u64> alltoallAddrInfo(userRankSize_ * addrItemNum, 0);
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;

    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[0], stepSize, sendLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[userRankSize_], stepSize, sendOffset, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvLengthStep * userRankSize_], stepSize, recvLength, stepSize));
    CHK_SAFETY_FUNC_RET(memcpy_s(&alltoallAddrInfo[recvOffsetStep * userRankSize_], stepSize, recvOffset, stepSize));


    preMetaInfo->inputData = alltoallAddrInfo;
    preMetaInfo->inputSize = stepSize * addrItemNum;
    preMetaInfo->outputSize = userRankSize_ * stepSize * addrItemNum;

    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PrepareAlltoAllAddrInfo(const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    std::vector<u64> vctSendLength(userRankSize_, 0);
    std::vector<u64> vctSendOffset(userRankSize_, 0);
    std::vector<u64> vctRecvLength(userRankSize_, 0);
    std::vector<u64> vctRecvOffset(userRankSize_, 0);

    for (u32 i = 0; i < userRankSize_; i++) {
        vctSendLength[i] = *(static_cast<const u64 *>(sendCounts) + i) * SIZE_TABLE[sendType];
        vctSendOffset[i] = *(static_cast<const u64 *>(sdispls) + i) * SIZE_TABLE[sendType];
        vctRecvLength[i] = *(static_cast<const u64 *>(recvCounts) + i) * SIZE_TABLE[recvType];
        vctRecvOffset[i] = *(static_cast<const u64 *>(rdispls) + i) * SIZE_TABLE[recvType];

        HCCL_DEBUG("[PrepareAlltoAllAddrInfo] rank[%u], SendLength[%llu], SendOffset[%llu], "\
            "RecvLength[%llu], RecvOffset[%llu]", i, vctSendLength[i], vctSendOffset[i], vctRecvLength[i],
            vctRecvOffset[i]);
    }
    CHK_RET(GetAlltoAllvAllAddrInfo(vctSendLength.data(), vctSendOffset.data(),
        vctRecvLength.data(), vctRecvOffset.data(), preMetaInfo));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::PreparePreOpParam(OpParam& preProcessOpParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
{
    u64 stepSize = sizeof(u64) * userRankSize_;
    u32 perDataSize = SIZE_TABLE[HCCL_DATA_TYPE_UINT64];

    preProcessOpParam.tag = HCCL_ALLTOALL_PARA_ALLGATHER;
    preProcessOpParam.inputPtr = cclBufferManager_.GetInAlltoAllvParaBuffer().ptr();
    preProcessOpParam.inputSize = (preMetaInfo->outputSize / stepSize) * perDataSize;
    preProcessOpParam.outputPtr = cclBufferManager_.GetOutAlltoAllvParaBuffer().ptr();
    preProcessOpParam.outputSize = (preMetaInfo->outputSize / stepSize) * perDataSize * userRankSize_;
    preProcessOpParam.DataDes.count = (preMetaInfo->outputSize / stepSize);
    preProcessOpParam.DataDes.dataType = HCCL_DATA_TYPE_UINT64;
    preProcessOpParam.stream = preProcessStream;
    preProcessOpParam.aicpuUnfoldMode = false;
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::JudgeIfNeedPreProcessAndGetParam(const OpParam& param,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV && !IsSatisfyAlltoAllAivCondition(param)) {
        if (IsSupportDirectFullmeshForAlltoallv(param, deviceType_, useSuperPodMode_, serverNum_,
            isSingleMeshAggregation_, userRankSize_) || param.aicpuUnfoldMode) {
            return false;
        }
        CHK_RET(PrepareAlltoAllAddrInfo(param.All2AllDataDes.sendCounts, param.All2AllDataDes.sdispls,
            param.All2AllDataDes.sendType, param.All2AllDataDes.recvCounts, param.All2AllDataDes.rdispls,
            param.All2AllDataDes.recvType, preMetaInfo));
        preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        return true;
    }
    return false;
}

void AlltoAllOperator::SetPreProcessResult(HostMem hostCollectBuffer)
{
    hostCollectBuffer_ = std::move(hostCollectBuffer);
}

HcclResult AlltoAllOperator::SetExcutorExtraInfo(const std::string& algName, const OpParam& param)
{
    HCCL_DEBUG("[AlltoAllOperator][SetExcutorExtraInfo]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[AlltoAllOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }

    CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor *>(executor_.get());
    return alltoAllExecutor->SetExcutorExtraInfo(allMeshAggregationSendRecvInfo_);
}

HcclResult AlltoAllOperator::SetExecutorAttr(const OpParam& param)
{
    CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor *>(executor_.get());
    CHK_RET(alltoAllExecutor->SetAlgType(algType_));
    CHK_RET(alltoAllExecutor->SetVirtualDispatcher(vDispatcher_));
    CHK_RET(alltoAllExecutor->SetCCLInBuffer(cclBufferManager_.GetInCCLbufferSize()));
    CHK_RET(alltoAllExecutor->SetParallelTaskLoader(parallelTaskLoader_));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::CheckNeedRecreateComm(const std::string& algName, const OpParam& param,
    u64 lastScratchMemSize, bool& needRecreateAlltoallComm)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[AlltoAllOperator][CheckNeedRecreateComm]Fail to find executor for algName[%s]",
            algName.c_str()), HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }
    CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor *>(executor_.get());
    CHK_RET(alltoAllExecutor->CheckNeedRecreateComm(lastScratchMemSize, needRecreateAlltoallComm));
    return HCCL_SUCCESS;
}

bool AlltoAllOperator::IsSatisfyAlltoallPipelineCondition()
{
    bool cclBigEnough = GetExternalInputCCLBuffSize() >= ALLTOALL_PIPELINE_MIN_CCL_SIZE;
    bool multiRankPerServer = meshAggregationRankSize_ > 1;
    bool isMultiServer = ((userRankSize_ > meshAggregationRankSize_) &&
        (userRankSize_ % meshAggregationRankSize_) == 0);
    auto autoAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool satisfyAlgType = (static_cast<AlgTypeLevel1>(autoAlgTypeLevel1) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) &&
        CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLTOALL) <= HCCL_FFTS_CAPACITY;
    HCCL_DEBUG("[AlltoAllOperator][IsSatisfyAlltoallPipelineCondition]multiRankPerServer %u, "
        "isMultiServer %u, satisfyAlgType, %u, multiModuleDiffDeviceNumMode_ %u", multiRankPerServer,
        isMultiServer, satisfyAlgType, multiModuleDiffDeviceNumMode_);
    bool res = (deviceType_ == DevType::DEV_TYPE_910B && satisfyAlgType && multiRankPerServer &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isMultiServer &&
        !multiModuleDiffDeviceNumMode_ && cclBigEnough);
    if (satisfyAlgType && !res) {
        HCCL_WARNING("alltoall algo type is set to pipeline, but cclBigEnough is %u, multiRankPerServer is %u, "
            "isMultiServer is %u", cclBigEnough, multiRankPerServer, isMultiServer);
    }
    return res;
}

bool AlltoAllOperator::IsSatisfy91093OffloadCondition()
{
    bool isOffload = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB;
    bool isDeviceType = deviceType_ == DevType::DEV_TYPE_910_93;
    bool isAicpuUnfoldMode = topoMatcher_->GetAicpuUnfoldConfig();
    return isAicpuUnfoldMode && isDeviceType && isOffload;
}

bool AlltoAllOperator::IsSatisfyAlltoAllAivCondition(const OpParam& param)
{
    bool isOpbase = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    bool isBufferEnough = !isOpbase ||
        cclBufferManager_.GetInCCLbufferSize() >= AIV_ALL_TO_ALL_BIG_SIZE * MAX_RANK_SIZE;
    bool isSupportAiv = topoMatcher_->GetAivModeConfig() && IsSupportAIVCopy(param.All2AllDataDes.sendType) &&
        userRankSize_ > 1 && isBufferEnough;
    if (deviceType_ == DevType::DEV_TYPE_910B) {
        bool isModuleSatisfy = false;
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            // alltoall算子支持单机和多机场景
            if (isSingleMeshAggregation_) {
                isModuleSatisfy = true;
            } else {
                bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
                // 多机场景下当前不支持module间卡数不一致场景，集群中总的服务器数需要满足条件，cclbuffer大小需要满足
                isModuleSatisfy = isOpbase && !multiModuleDiffDeviceNumMode_ && (moduleNum_ <= MAX_RMDA_RANK_SIZE) &&
                                IsBufferSatisfyAlltoAllAivCondition(param);
            }
        } else {
            // 其他算子只有单机场景支持aiv
            isModuleSatisfy = isSingleMeshAggregation_;
        }
        bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
            topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
        return isSupportAiv && isModuleSatisfy && isMeshTopo;
    } else if (deviceType_ == DevType::DEV_TYPE_910_93) {
        bool isSupportInterHccs = (superPodNum_ == 1 && serverNum_ > 1 && !GetExternalInputInterHccsDisable());
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            u64 dataSize = param.All2AllDataDes.sendCount * SIZE_TABLE[param.All2AllDataDes.sendType];
            return isSupportAiv && ((serverNum_ == 1 && dataSize <= AIV_ALL_TO_ALL_A3_ENTRY_SIZE) ||
                isSupportInterHccs);
        } else {
            return isSupportAiv && (serverNum_ == 1 || isSupportInterHccs);
        }
    }
    return false;
}

bool AlltoAllOperator::IsBufferSatisfyAlltoAllAivCondition(const OpParam& param)
{
    u64 sendCount = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix));
    u64 dataSize = SIZE_TABLE[param.All2AllDataDes.sendType];
    u64 scratchMemSize = sendCount * dataSize * userRankSize_;

    // 每个rank的数据需要满足小于190K
    if (!(sendCount * dataSize <= HCCL_SMALL_COUNT_190_KB)) {
        HCCL_WARNING("[AlltoAllOperator]dataSize[%u] > [%u], doesn't meet the aiv condition, select default algorithm",
            sendCount * dataSize, HCCL_SMALL_COUNT_190_KB);
        return false;
    }

    // cclbuffer是否足够存储每个rank的中转数据
    if (!(scratchMemSize <= cclBufferManager_.GetInCCLbufferSize())) {
        HCCL_WARNING("[AlltoAllOperator]cclbuffer[%u] < scratchMemSize[%u]+32K, don't meet the aiv condition, "
            "please set HCCL_BUFFSIZE to increase cclbuffer", cclBufferManager_.GetInCCLbufferSize(), scratchMemSize);
        return false;
    }
    return true;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(const OpParam& param, u64 &memSize)
{
    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        memSize = 0;
        HCCL_INFO("[Get][AlltoAllStagedWorkSpaceMemSize]Asym scene, No workSpaceMem required. "\
                  "multiModuleDiffDeviceNumMode[%d], multiSuperPodDiffServerNumMode[%d], memSize:[%llu]",
                  multiModuleDiffDeviceNumMode_, multiSuperPodDiffServerNumMode_, memSize);
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(hostCollectBuffer_.ptr());
    CHK_RET(GetAlltoAllvSendRecvInfo(param, hostCollectBuffer_));

    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo_,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

HcclResult AlltoAllOperator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        memSize = 0;
        HCCL_INFO("[Get][AlltoAllStagedWorkSpaceMemSize]Asym scene, No workSpaceMem required. "\
                  "multiModuleDiffDeviceNumMode[%d], multiSuperPodDiffServerNumMode[%d], memSize:[%llu]",
                  multiModuleDiffDeviceNumMode_, multiSuperPodDiffServerNumMode_, memSize);
        return HCCL_SUCCESS;
    }
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRank = userRank_;
    userRankInfo.userRankSize = userRankSize_;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allMeshAggregationSendRecvInfo,
        memSize, meshAggregationRankSize_);

    HCCL_INFO("Calculate workSpace MemSize done, memSize[%llu]", memSize);

    // 计算结果
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALLV, AlltoAllV, AlltoAllOperator);
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALL, AlltoAll, AlltoAllOperator);
REGISTER_OP(HcclCMDType::HCCL_CMD_ALLTOALLVC, AlltoAllVC, AlltoAllOperator);

}