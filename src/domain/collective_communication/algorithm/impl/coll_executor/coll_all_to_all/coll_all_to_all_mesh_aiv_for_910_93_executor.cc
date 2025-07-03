/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_mesh_aiv_for_910_93_executor.h"
#include "alg_profiling.h"

namespace hccl {

CollAlltoAllMeshAivFor91093Executor::CollAlltoAllMeshAivFor91093Executor(const HcclDispatcher dispatcher,
                                                         std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAlltoAllMeshAivFor91093Executor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[CollAlltoAllMeshAivFor91093Executor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u].",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 - level1 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    }
    HCCL_INFO("[CollAlltoAllMeshAivFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh 超节点内建SDMA链路
HcclResult CollAlltoAllMeshAivFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    commCombinePara.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));

    LevelNSubCommTransport &commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto &transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = false;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    execMem.inputMem = (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ?
        algRes.paramInputMem : algRes.cclInputMem);
    execMem.outputMem = algRes.aivOutputMem;
    HcclResult ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllMeshAivFor91093Executor][Orchestrate]errNo[0x%016llx] tag[%s] "
            "excutor kernel run failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AlltoAll executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAlltoAllMeshAivFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAlltoAllMeshAivFor91093Executor][KernelRun]alltoall aiv enter.");

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    u32 localRank = level0CommInfo.localRank;
    u32 localRankSize = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollAlltoAllMeshAivFor91093Executor][KernelRun] userRank [%u] localRank [%u] localRankSize[%u]",
        topoAttr_.userRank, localRank, localRankSize);

    HcclResult ret;
    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[0] = execMem.inputMem.ptr();
    buffersOut[0] = execMem.outputMem.ptr();

    if (aivClearEnable_) {
        ClearAivSyncBuf(buffersOut, localRank, localRankSize, param.stream.ptr());
    }

    AivTopoArgs topoArgs { localRank, localRankSize, MAX_RANK_SIZE, 0, topoAttr_.serverNum, topoAttr_.deviceType };
    AivResourceArgs resourceArgs { param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size() };
    AivAlgArgs algArgs {};
    AivProfilingInfo aivProfilingInfo;
    
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE){
        HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, workflowMode_);
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);
    }

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        AivOpArgs opArgs {
            HcclCMDType::HCCL_CMD_ALLTOALL, execMem.inputPtr, execMem.outputPtr, param.All2AllDataDes.sendCount,
            param.All2AllDataDes.sendType, HCCL_REDUCE_RESERVED, 0, isOpbase
        };
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo);
    } else {
        ExtraArgsV2 extraArgs;
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            for (u32 i = 0; i < localRankSize; i++) {
                extraArgs.sendCounts[i] = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) +
                    localRank * localRankSize + i);
                extraArgs.recvCounts[i] = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) +
                    i * localRankSize + localRank);
                if (i == 0) {
                    extraArgs.sendDispls[i] = 0;
                    extraArgs.recvDispls[i] = 0;
                } else {
                    extraArgs.sendDispls[i] = extraArgs.sendDispls[i - 1] + extraArgs.sendCounts[i - 1];
                    extraArgs.recvDispls[i] = extraArgs.recvDispls[i - 1] + extraArgs.recvCounts[i - 1];
                }
            }
        } else {
            for (u32 i = 0; i < localRankSize; i++) {
                extraArgs.sendCounts[i] = *(static_cast<const u64 *>(param.All2AllDataDes.sendCounts) + i);
                extraArgs.recvCounts[i] = *(static_cast<const u64 *>(param.All2AllDataDes.recvCounts) + i);
                extraArgs.sendDispls[i] = *(static_cast<const u64 *>(param.All2AllDataDes.sdispls) + i);
                extraArgs.recvDispls[i] = *(static_cast<const u64 *>(param.All2AllDataDes.rdispls) + i);
            }
        }

        AivOpArgs opArgs {
            HcclCMDType::HCCL_CMD_ALLTOALLV, execMem.inputPtr, execMem.outputPtr, 0,
            param.All2AllDataDes.sendType, HCCL_REDUCE_RESERVED, 0, isOpbase
        };
        ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    }

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE){ 
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
        HCCL_PROFILER_DEL_TAG(param.tag);
    }
    blockDim_ = aivProfilingInfo.blockDim;

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAlltoAllMeshAivFor91093Executor][KernelRun]alltoall aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[CollAlltoAllMeshAivFor91093Executor][KernelRun]alltoall aiv run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlltoAllMeshAivFor91093Executor", AlltoAllMeshAivFor91093, CollAlltoAllMeshAivFor91093Executor);

} // namespace hccl