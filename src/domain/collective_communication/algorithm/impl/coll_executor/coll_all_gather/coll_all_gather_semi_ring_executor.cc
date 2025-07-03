/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_semi_ring_executor.h"

namespace hccl {

CollAllGatherSemiRingExecutor::CollAllGatherSemiRingExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherRingFor91093Executor(dispatcher, topoMatcher)
{
}

HcclResult CollAllGatherSemiRingExecutor::CalcNotifyNum(u32 streamNum, u32 &notifyNum)
{
    // notify数量是从流的两倍 + 新增带notifyId的notify资源
    notifyNum = 2U * streamNum + (topoAttr_.deviceNumPerAggregation + 4U);
    HCCL_INFO("[CollAllGatherSemiRingExecutor][CalcNotifyNum]tag[%s] notifyNum_ is [%u]", tag_.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSemiRingExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE + 1U;

    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    }

    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherSemiRingExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSemiRingExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));

    LevelNSubCommTransport &commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto &transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.notifyNum = topoAttr_.deviceNumPerAggregation + 4U; //只传递额外的notify个数
            HCCL_INFO("[CollAllGatherSemiRingExecutor][CalcLevel0CommInfo] set extral notifyNum[%u]",
                transportRequest.notifyNum);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSemiRingExecutor::DoubleRingMidCountAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    HCCL_INFO("[CollAllGatherSemiRingExecutor][KernelRun]AllGatherDoubleRingConcurrentExecutor starts.");
    
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 执行
    std::unique_ptr<AlgTemplateBase> executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_UNIFIED_MARCH, dispatcher_);
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(stream, level0CommInfo, algResResp_->paramInputMem, algResResp_->paramOutputMem,
        inputMem, outputMem, count, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        multRingsUserMemSlice));

    HcclResult ret = executor->RegisterProfiler(
        ((COMM_INDEX_0 + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID) +
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherSemiRingExecutor][DoubleRingMidCountAllGather]Double ring "
        "all gather failed, return[%d]", ret), ret);

    CHK_RET(executor->RunAsync());

    HCCL_INFO("[CollAllGatherSemiRingExecutor] all gather double ring level1 run success");
    return ret;
}

HcclResult CollAllGatherSemiRingExecutor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(DoubleRingMidCountAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherSemiRingExecutor", AllGatherDoubleRingMidCount,
    CollAllGatherSemiRingExecutor);

} // namespace hccl