/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_ring_for_910_93_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollAllReduceRingFor91093Executor::CollAllReduceRingFor91093Executor(const HcclDispatcher dispatcher,
                                                                 std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollAllReduceRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && 
        GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceRingFor91093Executor][CalcStreamNum] tag[%s] streamNum_[%u].",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollAllReduceRingFor91093Executor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    (void)totalSize;
    bool smallData = IsAllReduceSmallData(curSize);
    return smallData;
}

bool CollAllReduceRingFor91093Executor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllReduceRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s]Calc HDCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollAllReduceRingFor91093Executor::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    bool isLargeSize = (curSize / topoAttr_.deviceNumPerAggregation >= HCCL_SPLIT_SIZE_INTER_SERVER);
    return GetExternalInputEnableRdmaSdmaConcurrent() && (topoAttr_.serverNum > 1) && isLargeSize;
}

HcclResult CollAllReduceRingFor91093Executor::RunIntraSeverReduceScatter(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const HcclReduceOp &reductionOp,
    const std::vector<std::vector<Slice>> &multRingsSliceZero, const Stream &stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice, const bool disableDMAReduce)
{
    CHK_RET(MultiRingReduceScatter(tag, inputMem, outputMem, count, dataType, reductionOp,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingFor91093Executor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceRingFor91093Executor][Run]The CollAllReduceRingFor91093Executor starts");
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    // 构造ring algorithm对应的reduce-scatter实例

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }

    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
    HcomCollOpInfo reduceScatterOpInfo = {
        "", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo reduceScatterGraphModeOpInfo = {
        "", execMem.inputMem.ptr(), nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        reduceScatterOpInfoPtr = &reduceScatterGraphModeOpInfo;
    }
    if (DMAReduceFlag_) {
        reduceScatterOpInfoPtr = &reduceScatterOpInfo;
    }
    bool disableDMAReduce = algOpContext_.opRetryHandler.retryEnable &&
        (algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1 ||
        algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2);
    const std::vector<std::vector<Slice>> multRingsUserMemSliceDefault = std::vector<std::vector<Slice>>(0);
    CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multRingsSliceZero, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr, multRingsUserMemSliceDefault, disableDMAReduce));
    HCCL_INFO("allreduce double ring stage0 run success.");

    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    
    /* 三步算法step2: 内层 - 节点间 allreduce */
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, hdSize, level0CommInfo, multRingsSliceZero, param.tag));

    u64 hdCount = hdSize / perDataSize;
    if (topoAttr_.superPodNum <= 1 || isSelectAHC) {
        bool isConcurrent = (topoAttr_.moduleNum > 1) && !isSelectAHC && !aicpuUnfoldMode_;
        if (GetExternalInputEnableRdmaSdmaConcurrent() && (hdSize >= HCCL_SPLIT_SIZE_INTER_SERVER) && isConcurrent) {
            u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
            BEST_SPLIT_VALUE_SR;
            CHK_RET(Level1AllReduceConcurrent(execMem.inputMem, execMem.outputMem, execMem.count,
                param.DataDes.dataType, param.reduceType, param.stream, PROF_STAGE_1,
                dataSegsSlice, segmentIdx, commIndex, hdSize, syncTrans));
        } else {
            DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
            CHK_SMART_PTR_NULL(allreduceInput);
            DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
            CHK_SMART_PTR_NULL(allreduceOutput);

            CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
            CHK_RET(CheckCommSize(commPlaneLevel1, commIndex + 1));
            SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, commIndex);

            u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);
            std::unique_ptr<AlgTemplateBase> level1TempAlg;
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, 
                    dispatcher_);
                HCCL_INFO("allreduce ring: using ring algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
                HCCL_INFO("allreduce ring: using nhr_v1 algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                // 获取通信域分组信息
                std::vector<std::vector<std::vector<u32>>> gloableSubGroups;
                CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, gloableSubGroups));
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC, dispatcher_);
                HCCL_INFO("allreduce ring: using ahc algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr, execMem.count, gloableSubGroups[0]));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
                // 获取通信域分组信息
                std::vector<std::vector<std::vector<u32>>> gloableSubGroups;
                CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, gloableSubGroups));
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_AHC_BROKE, dispatcher_);
                HCCL_INFO("allreduce ring: using ahc-broke algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr, execMem.count, gloableSubGroups[0]));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, 
                    dispatcher_);
                HCCL_INFO("allreduce ring: using nonuniform-bruck algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType]; // 单位 byte
                HCCL_DEBUG("allreduce ring: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]",
                    curSize, topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
                if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
                    level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
                } else {
                    level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
                }
                HCCL_INFO("allreduce ring: using nhr algo inter-server.");
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
            } else {
                HCCL_ERROR("allreduce ring: algType_[%u] is not supported.", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1TempAlg);
            u32 rankSize = level1CommInfo.localRankSize;
            // 节点间的hd 使用环0来记录
            CHK_RET(level1TempAlg->Prepare(
                allreduceInput, allreduceOutput, allreduceOutput, hdCount,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID,
                std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));

            HCCL_INFO("allreduce double ring stage1 run success");
        }
    } else {
        // 超节点内做reducescatter
        CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
        u32 level1RankSize = level1CommInfo.localRankSize;
        u64 level1Offset = dataSegsSlice[segmentIdx].offset;

        // 根据数据量计算每个环上数据的偏移和大小
        CHK_RET(AlgTemplateBase::PrepareSliceData(hdCount, perDataSize, level1RankSize, 0, dataSegsSlice));
        DeviceMem reducescatterInput = execMem.inputMem.range(level1Offset, hdSize);
        CHK_SMART_PTR_NULL(reducescatterInput);
        DeviceMem reducescatterOutput = execMem.outputMem.range(level1Offset, hdSize);
        CHK_SMART_PTR_NULL(reducescatterOutput);
        if (level1RankSize > 1) {
            u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput,
                param.DataDes.dataType, param.reduceType);
            std::unique_ptr<AlgTemplateBase> level1RSTempAlg;

            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, 
                    dispatcher_);
                CHK_SMART_PTR_NULL(level1RSTempAlg);
                CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, 
                    dispatcher_);
                CHK_SMART_PTR_NULL(level1RSTempAlg);
                CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, 
                    dispatcher_);
                CHK_SMART_PTR_NULL(level1RSTempAlg);
                CHK_RET(level1RSTempAlg->Prepare(reduceAttr, false));
                HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
            } else {
                HCCL_ERROR("reducescatter ring: algType_[%u] is not supported.", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_RET(level1RSTempAlg->Prepare(
                reducescatterInput, reducescatterInput, reducescatterOutput, hdCount, param.DataDes.dataType,
                param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));

            CHK_RET(level1RSTempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1RSTempAlg, level1CommInfo));
            HCCL_INFO("allreduce double ring [superpod] level1 reducescatter run success");
        }

        // 超节点间做allreduce
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        u32 rankSize = level2CommInfo.localRankSize;
        u32 localRank = level1CommInfo.localRank;

        DeviceMem allreduceInput =
            reducescatterInput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
        CHK_SMART_PTR_NULL(allreduceInput);
        DeviceMem allreduceOutput =
            reducescatterOutput.range(dataSegsSlice[localRank].offset, dataSegsSlice[localRank].size);
        CHK_SMART_PTR_NULL(allreduceOutput);

        u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

        std::unique_ptr<AlgTemplateBase> level2ARTempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
            HCCL_INFO("allreduce ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_INFO("allreduce ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_INFO("allreduce ring: using ring algo inter-superPod.");
        } else {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("allreduce ring: using halving-doubling algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level2ARTempAlg);
        CHK_RET(level2ARTempAlg->Prepare(reduceAttr));

        u64 arCount = dataSegsSlice[localRank].size / perDataSize;
        CHK_RET(level2ARTempAlg->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, arCount,
            param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID,
            std::vector<Slice>(0), dataSegsSlice[localRank].offset + level1Offset));
        CHK_RET(level2ARTempAlg->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2ARTempAlg, level2CommInfo));
        HCCL_INFO("allreduce double ring [superpod] level2 allreduce run success");

        // 超节点内做allgather
        if (level1RankSize > 1) {
            std::unique_ptr<AlgTemplateBase> level1AGTempAlg;
            DeviceMem allgatherInput = execMem.outputMem.range(level1Offset, hdSize);
            DeviceMem allgatherOutput = execMem.outputMem.range(level1Offset, hdSize);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, 
                    dispatcher_);
                HCCL_INFO("allgather ring: using ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, 
                    dispatcher_);
                HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, 
                    dispatcher_);
                HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
            } else {
                HCCL_ERROR("allgather ring: algType_[%u] is not supported.", algType_.algoLevel1);
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1AGTempAlg);
            CHK_RET(level1AGTempAlg->Prepare(allgatherInput, allgatherOutput, allgatherOutput, arCount,
                param.DataDes.dataType, param.stream,
                HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, level1Offset));
            CHK_RET(level1AGTempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1AGTempAlg, level1CommInfo));
            HCCL_INFO("allreduce double ring [superpod] level1 allgather run success");
        }
    }
    /* 三步算法step3：外层 - 节点内 allgather */
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo allgatherOpInfoGraphModeOpInfo = {
        "", nullptr, execMem.outputMem.ptr(), execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        allgatherOpInfoPtr = &allgatherOpInfoGraphModeOpInfo;
    }
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, hdCount,
        param.DataDes.dataType, multRingsSliceZero, param.stream,
        PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("allreduce double ring stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceRingFor91093Executor", AllReduceRingFor91093, CollAllReduceRingFor91093Executor);

} // namespace hccl
