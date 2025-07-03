/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "alg_profiling.h"
#include "coll_reduce_scatter_v_aiv_big_count_executor.h"

namespace hccl {
CollReduceScatterVAIVBigCountExecutor::CollReduceScatterVAIVBigCountExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterVExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceScatterVAIVBigCountExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    // AIV通信需要AIV buffer
    needAivBuffer = true;
    HCCL_INFO("[CollReduceScatterVAIVBigCountExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u].",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVAIVBigCountExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVAIVBigCountExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    // ReduceScatterV 大数据量场景下不支持图模式
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::AIV_OUTPUT;
    }

    HCCL_INFO("[CollReduceScatterVAIVBigCountExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVAIVBigCountExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}


HcclResult CollReduceScatterVAIVBigCountExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HcclResult ret = HCCL_SUCCESS;
    ExecMem execMem;

    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // ReduceScatterV 大数据量场景下不支持图模式
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.aivOutputMem;
        ret = KernelRun(param, execMem);
    }

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterVAIVBigCountExecutor][Orchestrate]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], ReduceScatterV executor orchestrate success, take time [%lld]us",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVAIVBigCountExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterVAIVBigCountExecutor][KernelRun]ReduceScatterV aiv enter.");
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;
    HCCL_DEBUG("[CollReduceScatterVAIVBigCountExecutor][KernelRun] userRank [%u] localRank [%u]", topoAttr_.userRank, localRank);

    ExtraArgs extraArgs;
    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersIn[i])));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersOut[i])));
        } else {
            buffersIn[i] = execMem.inputMem.ptr();
            buffersOut[i] = execMem.outputMem.ptr();
        }
        extraArgs.sendCounts[i] = *(static_cast<const u64 *>(param.VDataDes.counts) + i);
        extraArgs.sendDispls[i] = *(static_cast<const u64 *>(param.VDataDes.displs) + i);
        extraArgs.maxCount = std::max(extraArgs.maxCount, extraArgs.sendCounts[i]);
    }

    if (aivClearEnable_) {
        ClearAivSyncBuf(buffersOut, localRank, localRankSize, param.stream.ptr());
    }

    bool isOpbase = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    execMem.count = (static_cast<const u64 *>(param.VDataDes.counts))[topoAttr_.userRank];

    AivOpArgs opArgs {
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, execMem.inputPtr, execMem.outputPtr, extraArgs.maxCount,
            param.VDataDes.dataType, param.reduceType, param.root, isOpbase
    };
    AivTopoArgs topoArgs { localRank, localRankSize };
    AivResourceArgs resourceArgs { param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size() };
    AivAlgArgs algArgs {};
    struct AivProfilingInfo aivProfilingInfo;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE){
        HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, workflowMode_);
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);
    }

    HcclResult ret = ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, extraArgs, aivProfilingInfo);
    
    TaskAivProfiler(opArgs.cmdType, aivProfilingInfo.tag, opArgs.count * sizeof(opArgs.dataType),
        aivProfilingInfo.blockDim, topoArgs.rankSize, resourceArgs.buffersOut[topoArgs.rank], resourceArgs.stream,
        algArgs.step, aivProfilingInfo.beginTime);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE){ 
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
        HCCL_PROFILER_DEL_TAG(param.tag);
    }
    blockDim_ = aivProfilingInfo.blockDim;
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[CollReduceScatterVAIVBigCountExecutor][KernelRun]"
        "reducescatterv aiv failed, return[%d]", ret), ret);

    HCCL_INFO("[CollReduceScatterVAIVBigCountExecutor][KernelRun]reducescatterv aiv run success.");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterVAIVBigCountExecutor", ReduceScatterVAIVBigCount, CollReduceScatterVAIVBigCountExecutor);
} // namespace hccl