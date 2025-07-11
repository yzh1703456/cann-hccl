/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_oneshot_executor.h"

namespace hccl {

CollAllReduceMeshOneshotExecutor::CollAllReduceMeshOneshotExecutor(const HcclDispatcher dispatcher,
                                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    CCLMemSlice_ = false;
    DMAReduceFlag_ = true;
}

HcclResult CollAllReduceMeshOneshotExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        totalStreamNum = topoAttr_.deviceNumPerAggregation - 1U;
    } else {
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    }
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllReduceMeshOneshotExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOneshotExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOneshotExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceMeshOneshotExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOneshotExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollAllReduceMeshOneshotExecutor::IsHugeData(const u64 curSize)
{
    // 只有server内通信，多QP哈希散列下不刷新子图
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllReduceMeshOneshotExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    bool smallData = totalSize <= HCCL_SMALL_COUNT_128_KB;
    return smallData;
}

HcclResult CollAllReduceMeshOneshotExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    CHK_RET(ActiveSlaveStreams(param.stream));

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    if (level0CommInfo.localRankSize == DEVICE_TWO) {
        level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_MESH_DIRECT_ONESHOT, dispatcher_);
    } else {
        level0TempAlg = 
            AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_CHUNK_MESH, dispatcher_);
    }

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        level0CommInfo.localRank, level0CommInfo.localRankSize, topoAttr_.userRank, &opInfo));
        
    CHK_RET(level0TempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("allreduce mesh oneshot run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMeshOneshotLoopExecutor", AllReduceMeshOneshot, CollAllReduceMeshOneshotExecutor);

} // namespace hccl