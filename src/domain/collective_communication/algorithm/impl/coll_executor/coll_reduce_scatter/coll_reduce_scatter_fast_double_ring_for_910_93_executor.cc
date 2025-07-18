/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_reduce_scatter_fast_double_ring_for_910_93_executor.h"

namespace hccl {
CollReduceScatterFastDoubleRingFor91093Executor::CollReduceScatterFastDoubleRingFor91093Executor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlignedReduceScatterDoubleRingFor91093Executor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

HcclResult CollReduceScatterFastDoubleRingFor91093Executor::DoubleRingReduceScatter(const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
    const u64 count, const HcclDataType dataType, const HcclReduceOp reductionOp,
    const std::vector<std::vector<Slice> > multRingsSliceZero, Stream stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> multRingsUserMemSlice, const bool disableDMAReduce)
{
    (void)tag;
    HCCL_INFO("[CollReduceScatterFastDoubleRingFor91093Executor][DoubleRingReduceScatter] DoubleRingReduceScatter starts");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();
    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    // 拿到ring环映射关系
    SubCommInfo level0ZeroCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder =
        GetRingsOrderByTopoType(level0ZeroCommInfo.localRankSize, topoType_, nicList);

    u64 reduceAttr = GetReduceAttr(inputMem, outputMem, dataType, reductionOp);

    SubCommInfo level0RingCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    // 生成两个ring上的userMemIn_上对应的slices
    std::vector<std::vector<Slice>> userMemInputSlicesOfDoubleRing;
    CHK_RET(CollectMultiRingsUserMemSlices(ringNum, dataType,
        opInfo, multRingsSliceZero,
        multiRingsOrder, multRingsUserMemSlice,
        userMemInputSlicesOfDoubleRing));
    // 生成两个ring上的rankOrder
    std::vector<std::vector<u32>> rankOrders;
    CHK_RET(CollectMultiRingsRankOrder(ringNum, multiRingsOrder, rankOrders));
    // 初始化executor
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_DB_RING_SLC, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);
    ret = tempAlg->Prepare(inputMem, inputMem, outputMem, count, dataType, stream, multRingsSliceZero,
        reductionOp, LEVEL0_BRIDGE_RANK_ID, baseOffset, disableDMAReduce, reduceAttr, opInfo,
        topoAttr_.userRank, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        rankOrders, userMemInputSlicesOfDoubleRing);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterFastDoubleRingFor91093Executor][DoubleRingReduceScatter] Double ring reduce scatter failed"
        "failed,return[%d]", ret), ret);
    u32 ringIndexOp = COMM_INDEX_0;
    u32 rankSize = level0RingCommInfo.localRankSize;
    ret = tempAlg->RegisterProfiler(
        ((ringIndexOp + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0RingCommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterFastDoubleRingFor91093Executor][DoubleRingReduceScatter] Double ring reduce scatter failed "
        "failed,return[%d]", ret), ret);

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    ret = RunTemplate(tempAlg, level0RingCommInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterFastDoubleRingFor91093Executor][DoubleRingReduceScatter] Double ring reduce scatter failed "
        "failed,return[%d]", ret), ret);

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem, outputMem, stream, dispatcher_));
    return HCCL_SUCCESS;
}
REGISTER_EXEC("ReduceScatterFastDoubleRingFor91093Executor", ReduceScatterFastDoubleRingFor91093, CollReduceScatterFastDoubleRingFor91093Executor);
}
