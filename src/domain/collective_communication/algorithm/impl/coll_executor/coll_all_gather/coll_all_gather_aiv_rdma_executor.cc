/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_aiv_rdma_executor.h"
#include "alg_profiling.h"

namespace hccl {
constexpr u32 A_X_SIZE = 16;

CollAllGatherAivRdmaExecutor::CollAllGatherAivRdmaExecutor(const HcclDispatcher dispatcher,
                                                                           std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum=0;
    HCCL_INFO("[CollAllGatherAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherAivRdmaExecutor::GetIfNeedAivBuffer(bool &needAivBuffer)
{
    needAivBuffer = true;
    HCCL_INFO("[CollAllGatherAivRdmaExecutor][GetIfNeedAivBuffer]tag[%s] needAivBuffer is [%u]",
        tag_.c_str(), needAivBuffer);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_OUTPUT, TransportMemType::AIV_INPUT, opTransport));
    CHK_RET(CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherAivRdmaExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherAivRdmaExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;
    u64 totalSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.scratchMem = algRes.scratchMem;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        execMem.inputMem = algRes.cclOutputMem;
        execMem.outputMem = algRes.aivInputMem;  // 使用aivIn的第33M作为Flag区
    } else {
        execMem.inputMem = algRes.aivInputMem;
        execMem.outputMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize * topoAttr_.userRankSize);
    }

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherAivRdmaExecutor]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllReduce executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherAivRdmaExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllGatherAivRdmaExecutor][KernelRun]AllGather aiv enter");
    HcclWorkflowMode workflow = workflowMode_;
    bool isOpbase = (workflow == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    //获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 outerRankSize = outerCommInfo.localRankSize;
    u32 commIndex = outerCommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo innerCommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = innerCommInfo.localRank;
    u64 inputMemSize = perDataSize * param.DataDes.count;

    auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    bool hugeData = IsHugeData(inputMemSize);
    auto opMeta = HcclOpMetaInfo::GetOneForAllGather(autoSelectedAlgTypeLevel1, hugeData, false, CopyPattern::BCOPY, 
        false, true);

    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));
    DeviceMem dstMem;
    u64 baseOffset = serverIndex * inputMemSize;
    if (isOpbase) {
        dstMem = execMem.inputMem.range(baseOffset, inputMemSize);
        CHK_SMART_PTR_NULL(dstMem);
    }
    else {
        baseOffset = serverIndex * inputMemSize * outerRankSize;
        u64 outerOffset = commIndex * inputMemSize;
         dstMem = execMem.inputMem.range(baseOffset + outerOffset, inputMemSize);
        CHK_SMART_PTR_NULL(dstMem);
    }

    // STP1，将数据从userinput内存拷贝到ccloutput内存的对应位置
    DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(param.inputPtr), inputMemSize);
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherMeshExecutor][KernelRun]all gather 4PmeshHD memcpy Failed, Offset[%llu], Size[%llu].",
        baseOffset, inputMemSize), ret);
    //  STP2， AI server 间 recursive halving doubling all gather
    u64 hdCount = inputMemSize / perDataSize;
    std::unique_ptr<ExecutorBase> innerExecutor;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
        // 1-单server-SDMA
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("allgather mesh: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("allgather mesh: using nhr algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
        HCCL_INFO("allgather mesh: using nhr_v1 algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
        HCCL_INFO("allgather mesh: using nonuniform-bruck algo inter-server.");
    } else {
        innerExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("allgather mesh: using halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(innerExecutor);

    DeviceMem rdmadstMem;
    if (isOpbase) {
        rdmadstMem = execMem.inputMem.range(0, inputMemSize * innerCommInfo.localRankSize);
        CHK_SMART_PTR_NULL(rdmadstMem);
    }
    else {
        hdCount = inputMemSize*outerRankSize / perDataSize;
        rdmadstMem = execMem.inputMem;
    }
 
    CHK_RET(innerExecutor->Prepare(rdmadstMem, rdmadstMem, execMem.inputMem, hdCount,
        param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
        std::vector<Slice>(COMM_INDEX_0), 0));
 
    u32 rankSize = innerCommInfo.localRankSize;
    CHK_RET(innerExecutor->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(innerExecutor, innerCommInfo));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));

    //STP3 机内重排
    void *buffersIn[MAX_RANK_SIZE];
    void *buffersOut[MAX_RANK_SIZE];

    u32 localRank = outerCommInfo.localRank;
    u32 localRankSize = outerCommInfo.localRankSize;

    void* tmpAivBufferData;
    void* tmpAivBufferFlag;

    for (u32 i = 0; i < localRankSize; i++) {
        if (i != localRank) {
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(tmpAivBufferData)));
            CHK_RET(outerCommInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(tmpAivBufferFlag)));
            buffersIn[i] = static_cast<u8 *>(tmpAivBufferData);
            buffersOut[i] = static_cast<u8 *>(tmpAivBufferFlag) + HCCL_MID_COUNT_32_MB;
        } else {
            buffersIn[i] = static_cast<u8 *>(execMem.inputMem.ptr());
            buffersOut[i] = static_cast<u8 *>(execMem.outputMem.ptr()) + HCCL_MID_COUNT_32_MB;
        }
    }

    if (aivClearEnable_) {
        ClearAivSyncBuf(buffersOut, localRank, localRankSize, param.stream.ptr());
    }
    
    u32 serverNum = innerCommInfo.localRankSize;

    AivOpArgs opArgs {
        HcclCMDType::HCCL_CMD_ALLGATHER, execMem.inputPtr, execMem.outputPtr, param.DataDes.count,
        param.DataDes.dataType, HCCL_REDUCE_RESERVED, 0, isOpbase
    };
    AivTopoArgs topoArgs {
        localRank, localRankSize, topoAttr_.isDiffDeviceModule ? topoAttr_.devicePhyId : A_X_SIZE,
        0, serverNum, topoAttr_.deviceType
    };
    AivResourceArgs resourceArgs { param.tag, param.stream.ptr(), buffersIn, buffersOut, execMem.inputMem.size() };
    AivAlgArgs algArgs {0};
    struct AivProfilingInfo aivProfilingInfo;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE){
        HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, workflowMode_);
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);
    }

    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));

    TaskAivProfiler(opArgs.cmdType, aivProfilingInfo.tag, opArgs.count * sizeof(opArgs.dataType),
        aivProfilingInfo.blockDim, topoArgs.rankSize, resourceArgs.buffersOut[topoArgs.rank], resourceArgs.stream,
        algArgs.step, aivProfilingInfo.beginTime);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE){ 
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
        HCCL_PROFILER_DEL_TAG(param.tag);
    }

    blockDim_ = aivProfilingInfo.blockDim;

    HCCL_INFO("[CollAllGatherAivRdmaExecutor][KernelRun]allGather aiv run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherAivRdmaExecutor", AllGatherAivRdma, CollAllGatherAivRdmaExecutor);

} // namespace hccl