/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_reduce_scatter_ring_for_910_93_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterRingFor91093Executor::CollReduceScatterRingFor91093Executor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

void CollReduceScatterRingFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    // 是否需要scratch memory
    if ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) &&
        isSupportSDMAReduce_ && IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_;
        } else {
            scratchMemSize = totalSize_;
        }
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO("[CollReduceScatterRingFor91093Executor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]",
        tag_.c_str(), scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcStreamNum(u32& streamNum)
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
    HCCL_INFO("[CollReduceScatterRingFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO("[CollReduceScatterRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType,
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
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterRingFor91093Executor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

bool CollReduceScatterRingFor91093Executor::IsHugeData(const u64 curSize, OpParam *param)
{
    // 多QP哈希散列开启且RDMA通信下，强制刷新子图
    // 这里如果CheckCommSize返回ERROR，相当于HugeData true，防止GetSubCommInfo越界
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;
    const u64 TBE_REDUCE_MAX_COUNT = INT32_MAX;

    u64 curCount = curSize / SIZE_TABLE[param->DataDes.dataType];
    bool issupportRDMAInlineReduce = IsSupportRDMAReduce(param->DataDes.dataType, param->reduceType);
    bool hugeData =
        (curSize * level2RankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE) ||
        (curSize > SDMA_SEND_MAX_SIZE) ||
        ((!isSupportSDMAReduce_) && (curCount > TBE_REDUCE_MAX_COUNT)) ||
        ((!issupportRDMAInlineReduce) && (curCount * level2RankSize / HCCL_INTERNODE_MAX_DATA_RATE > TBE_REDUCE_MAX_COUNT));
    return hugeData;
}

bool CollReduceScatterRingFor91093Executor::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    bool isLargeSize = (curSize >= HCCL_SPLIT_SIZE_INTER_SERVER);
    return GetExternalInputEnableRdmaSdmaConcurrent() && (topoAttr_.serverNum > 1) && isLargeSize;
}

HcclResult CollReduceScatterRingFor91093Executor::RunIntraSeverReduceScatter(
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

void CollReduceScatterRingFor91093Executor::FillMultiRingSlice(
    const ExecMem &execMem,
    const std::vector<std::vector<Slice>> &multiStreamSlice,
    u32 sliceNum, u32 level1RankSize, u32 level2RankSize,
    const u32 ringIndex, std::vector<Slice> &dataSlice)
{
    for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
        Slice sliceTemp;
        for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
                sliceTemp.offset = multiStreamSlice[ringIndex][level0Idx].offset +
                    level1Idx * sliceNum * execMem.outputMem.size() +
                    level2Idx * sliceNum * level1RankSize * execMem.outputMem.size();
                dataSlice.push_back(sliceTemp);
                HCCL_DEBUG("rank[%u] sliceTemp.size[%zu]，sliceTemp.offset[%llu]", topoAttr_.userRank,
                    sliceTemp.size, sliceTemp.offset);
            }
        }
    }
}

void CollReduceScatterRingFor91093Executor::CalLevel0DataSegsSlice(
    const ExecMem &execMem,
    const std::vector<std::vector<Slice>> &multiStreamSlice,
    u32 sliceNum, u32 level1RankSize, u32 level2RankSize,
    std::vector<std::vector<Slice>> &level0DataSegsSlice)
{
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        FillMultiRingSlice(execMem, multiStreamSlice, sliceNum, level1RankSize, level2RankSize, ringIndex, dataSlice);
        level0DataSegsSlice.push_back(dataSlice);
    }
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel1DataSegsSlice(
    const ExecMem &execMem, const u32 &commIndex,
    u32 sliceNum, u32 level1RankSize, u32 level2RankSize,
    std::vector<Slice> &level1DataSegsSlice)
{
    for (u32 i = 0; i < level1RankSize; i++) {
        Slice sliceTemp;
        u32 level1UserRank;
        CHK_RET(GetUserRankByRank(COMM_LEVEL1, commIndex, i, level1UserRank));
        if (level2RankSize <= 1) {
            sliceTemp.size = execMem.outputMem.size();
            sliceTemp.offset = level1UserRank * execMem.outputMem.size();
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                sliceTemp.offset, sliceTemp.size);
        } else {
            for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
                sliceTemp.size = execMem.outputMem.size();
                sliceTemp.offset = (level1UserRank % (level1RankSize * sliceNum)) * execMem.outputMem.size() +
                        level2Idx * sliceNum * level1RankSize * execMem.outputMem.size();
                level1DataSegsSlice.push_back(sliceTemp);
                HCCL_DEBUG("rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                    sliceTemp.offset, sliceTemp.size);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollReduceScatterRingFor91093Executor][KernelRun] The ReduceScatterDoubleRingExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 ringNum;\
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING && !
        IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum)) {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }

    u32 sliceNum = level0CommInfo.localRankSize;
    Slice sliceTemp;
    u32 commIndex = level0CommInfo.localRank;
    commIndex = RefreshCommIdx(commIndex, topoAttr_.nicList, topoAttr_.devicePhyId);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移

    // 节点内reduce scatter
    CHK_RET(ActiveSlaveStreams(param.stream));
    u32 level1RankSize = level1CommInfo.localRankSize;

    // 计算slice
    std::vector<std::vector<Slice>> level0DataSegsSlice;
    bool useInlineRduce = false;
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType,
        param.reduceType);
    useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    multiStreamSlice = ReduceScatterRingSlicePrepare(ringNum, sliceNum, useInlineRduce, execMem.outputMem,
        dataSegsSlice, param.tag);

    CalLevel0DataSegsSlice(execMem, multiStreamSlice, sliceNum, level1RankSize, level2RankSize, level0DataSegsSlice);

    std::vector<std::vector<Slice>> multRingsUserMemSlice;

    HcomCollOpInfo opInfo = {"", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType,
        param.root, param.reduceType, param.DataDes.strideCount};
    HCCL_DEBUG("[CollReduceScatterRingFor91093Executor][KernelRun] execMem.inputPtr[%p], execMem.outputPtr[%p], execMem.inputMem[%p], execMem.outputMem[%p], param.DataDes.strideCount[%llu]",
        execMem.inputPtr, execMem.outputPtr, execMem.inputMem.ptr(), execMem.outputMem.ptr(), param.DataDes.strideCount);
    HcomCollOpInfo *opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }
    bool disableDMAReduce = algOpContext_.opRetryHandler.retryEnable &&
        (algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1 ||
        algOpContext_.opRetryHandler.inPlaceSupportRetryStatus == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2);

    CHK_PRT_RET(0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR("[CollReduceScatterRingFor91093Executor][KernelRun]strideCount[%llu] is smaller than opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    HCCL_DEBUG("[CollReduceScatterRingFor91093Executor][KernelRun]strideCount[%llu], opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count);
    if (opInfoPtr == nullptr &&
        (!(topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB || disableDMAReduce)))) {
        multRingsUserMemSlice = level0DataSegsSlice;
        //图模式，根据strideCount更新slice的offset
        if (param.DataDes.strideCount != 0) {
            CHK_RET(UpdateOffsetBasedOnStrideCount(param, multRingsUserMemSlice));
        }
    } else {
        for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
            std::vector<Slice> level1UserMemSlice;
            for (auto &cclSlice : level0DataSegsSlice[ringIndex]) {
                Slice tmpSlice;
                u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
                tmpSlice.size = cclSlice.size;
                CHK_PRT_RET(execMem.outputMem.size() == 0,
                    HCCL_ERROR("[CollReduceScatterRingFor91093Executor][KernelRun]cclout memsize[%llu] is zero",
                    execMem.outputMem.size()), HCCL_E_PARA);
                tmpSlice.offset =
                    (cclSlice.offset / execMem.outputMem.size()) * count * perDataSize +
                    multiStreamSlice[ringIndex][0].offset;
                level1UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level1UserMemSlice);
        }
    }
    // 区分消减拷贝场景
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        // 图模式opinfo不为空
        HcomCollOpInfo graphModeOpInfo = {
            "", execMem.inputMem.ptr(), nullptr, param.DataDes.count, param.DataDes.dataType,
            param.root, param.reduceType, param.DataDes.strideCount};
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType, level0DataSegsSlice,
            param.stream, PROF_STAGE_1, 0, &graphModeOpInfo, multRingsUserMemSlice, disableDMAReduce));
    } else if (opInfoPtr != nullptr && (level1RankSize > 1 || level2RankSize > 1)) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
        opInfoByReduceScatterDMAreduce.outputAddr      = nullptr;
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType, level0DataSegsSlice,
            param.stream, PROF_STAGE_1, 0, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice, disableDMAReduce));
    } else {
        CHK_RET(RunIntraSeverReduceScatter(param.tag, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.reduceType,
            level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, opInfoPtr, multRingsUserMemSlice, disableDMAReduce));
    }
    // 对于单server图模式的最后一步需要把数据从ccl input拷贝到ccl output上
    if (level1RankSize == 1 && level2RankSize == 1 && opInfoPtr == nullptr) {
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    if  (level1RankSize > 1) {
        // 节点间做reduce scatter(ring/NHR/NB)
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1TempAlg;

        // 计算slice
        std::vector<Slice> level1DataSegsSlice;

        CHK_RET(CalLevel1DataSegsSlice(execMem, commIndex, sliceNum, level1RankSize, level2RankSize,
            level1DataSegsSlice));

        if (GetExternalInputEnableRdmaSdmaConcurrent() && (execMem.outputMem.size() >= HCCL_SPLIT_SIZE_INTER_SERVER)
            && !aicpuUnfoldMode_) {
            u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
                BEST_SPLIT_VALUE_SR;
            CHK_RET(Level1ReduceScatterConcurrent(execMem.inputMem, execMem.scratchMem, execMem.count,
                param.DataDes.dataType, param.reduceType, param.stream, PROF_STAGE_2,
                level1DataSegsSlice, syncTrans, reduceAttr));
        } else {
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
                HCCL_INFO("reducescatter ring: using ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr));
                HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-server.");
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
                CHK_SMART_PTR_NULL(level1TempAlg);
                CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
                HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-server.");
            }

            CHK_RET(level1TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
                param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, level1DataSegsSlice));
            CHK_RET(level1TempAlg->RegisterProfiler(
                (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
                PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
            CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
        }
    }

    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);

        // 计算slice
        std::vector<Slice> level2DataSegsSlice;
        for (u32 i = 0; i < level2RankSize; i++) {
            sliceTemp.size = execMem.outputMem.size();
            u32 level2UserRank;
            CHK_RET(GetUserRankByRank(COMM_LEVEL2, COMM_INDEX_0, i, level2UserRank));
            sliceTemp.offset = level2UserRank * execMem.outputMem.size();
            level2DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG("rank[%u], level2DataSegsSlice[%u].offset=%llu, size=[%llu], level2RankSize[%u]",
                topoAttr_.userRank, i, sliceTemp.offset, sliceTemp.size, level2RankSize);
        }

        std::unique_ptr<AlgTemplateBase> level2TempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr));
            HCCL_INFO("reducescatter ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr, false));
            HCCL_INFO("reducescatter ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr));
            HCCL_INFO("reducescatter ring: using ring algo inter-superPod.");
        }

        CHK_RET(level2TempAlg->Prepare(execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, level2DataSegsSlice));
        CHK_RET(level2TempAlg->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    }

    if (level1RankSize > 1 || level2RankSize > 1) {
        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(),
            execMem.outputMem.size());
        if (opInfoPtr != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8 *>(opInfoPtr->outputAddr), execMem.outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    HCCL_INFO("reducescatter double ring run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterRingFor91093Executor", ReduceScatterRingFor91093, CollReduceScatterRingFor91093Executor);
}
