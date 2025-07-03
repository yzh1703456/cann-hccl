/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_v_mesh_opbase_executor.h"

#include <algorithm> 

namespace hccl {
CollAllGatherVMeshOpbaseExecutor::CollAllGatherVMeshOpbaseExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherVExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllGatherVMeshOpbaseExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherVMeshOpbaseExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshOpbaseExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshOpbaseExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherVMeshOpbaseExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshOpbaseExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherVMeshOpbaseExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = (cclBuffSize - HCCL_MIN_SLICE_ALIGN_910B) / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

bool CollAllGatherVMeshOpbaseExecutor::IsHugeData(const u64 curSize)
{
    // 该算法只涉及mesh内，不对RDMA多qp做强制刷新
    bool hugeData = curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllGatherVMeshOpbaseExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    dataType = param.VDataDes.dataType;
    const u32 unitSize = SIZE_TABLE[dataType];

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 rankSize = level0CommInfo.localRankSize;

    // DMA消减后仅使用ccl out通信，ccl out根据实际使用大小重新申请内存空间
    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = 0;
    DeviceMem curOutputMem = execMem.outputMem.range(baseOffset, inputMemSize);
    CHK_SMART_PTR_NULL(curOutputMem);

    // allgatherv 计算slice，数据分成ranksize份，每份的起始偏移和大小
    std::vector<Slice> outputSlices;
    const auto counts = static_cast<u64*>(param.VDataDes.counts);
    const auto displs = static_cast<u64*>(param.VDataDes.displs);
    for (u32 rank = 0; rank < rankSize; ++rank) {
        Slice userslice;
        userslice.offset = displs[rank] * unitSize;
        userslice.size = counts[rank] * unitSize;
        outputSlices.emplace_back(std::move(userslice));
    }

    // DMA消减场景，打包opInfo
    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, execMem.count, dataType,
        param.root, param.reduceType};
    
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_MESH_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    CHK_RET(tempAlg->Prepare(algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, &opInfo, level0CommInfo.localRank, level0CommInfo.localRankSize));

    CHK_RET(tempAlg->Prepare(curOutputMem, curOutputMem, execMem.inputMem, execMem.count,
        dataType, param.stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, outputSlices, baseOffset));

    CHK_RET(tempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(tempAlg, level0CommInfo));
    HCCL_INFO("allgatherv mesh for A2 run success");

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherVMeshOpbaseExecutor::CalcCurCountsAndCurDispls(const u64 maxTotalCount,
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

REGISTER_EXEC("AllGatherVMeshOpbaseExecutor", AllGatherVMeshOpbase, CollAllGatherVMeshOpbaseExecutor);
} // namespace hccl
