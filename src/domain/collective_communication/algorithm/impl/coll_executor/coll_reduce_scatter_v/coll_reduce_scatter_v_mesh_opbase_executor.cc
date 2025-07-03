/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_v_mesh_opbase_executor.h"

namespace hccl {

CollReduceScatterVMeshOpbaseExecutor::CollReduceScatterVMeshOpbaseExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

void CollReduceScatterVMeshOpbaseExecutor::ParseParam(const OpParam& param)
{
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterVMeshOpbaseExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCommInfo(
    std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollReduceScatterVMeshOpbaseExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollReduceScatterVMeshOpbaseExecutor::IsHugeData(const u64 curSize)
{
    // 只有server内通信，多QP哈希散列下不刷新子图
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::CalcCurCountsAndCurDispls(const u64 maxTotalCount,
    std::vector<u64> &countsLeft, std::vector<u64> &displs, std::vector<u64> &curCounts, std::vector<u64> &curDispls,
    bool &finished)
{
    finished = true;

    curCounts.resize(countsLeft.size(), 0);
    curDispls.resize(displs.size(), 0);

    // 先设置本轮的displacements，等于入参displs
    std::copy(displs.begin(), displs.end(), curDispls.begin());    

    // 分配好每个rank的counts
    for (auto i = 0U; i < countsLeft.size(); ++i) {
        const auto curCount = countsLeft[i] < maxTotalCount ? countsLeft[i] : maxTotalCount;
        curCounts[i] = curCount;
        countsLeft[i] -= curCount;
        displs[i] += curCount;
        
        if(curCount != 0) {
            finished = false;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterVMeshOpbaseExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HcclDataType dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo subCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 rankSize = subCommInfo.localRankSize;

    /* *******************节点内reducescatter ******************************************/
    // reduce_scatter_v 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> inputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < rankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        inputSlices.emplace_back(std::move(userslice));
    }

    HcomCollOpInfo *opInfoPtr = nullptr;
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, 0, dataType,
        param.root, param.reduceType};
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType,
        param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, inputSlices, 0, reduceAttr,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, opInfoPtr));

    CHK_RET(TempAlg->RegisterProfiler(
        (subCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, subCommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterVMeshOpbaseExecutor",
    ReduceScatterVMeshOpbase, CollReduceScatterVMeshOpbaseExecutor);
}