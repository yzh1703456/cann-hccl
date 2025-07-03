/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_double_ring_concurrent_executor.h"

namespace hccl {

CollAllReduceDoubleRingConcurrentExecutor::CollAllReduceDoubleRingConcurrentExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher): CollAllReduceExecutor(dispatcher, topoMatcher)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        DMAReduceFlag_ = true;
    } else {
        DMAReduceFlag_ = false;
    }
}

HcclResult CollAllReduceDoubleRingConcurrentExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    // DoubleRing只支持910_93场景
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    } else { //图模式增加两条流用于机内并发
        totalStreamNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
        totalStreamNum += RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
    }

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceDoubleRingConcurrentExecutor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingConcurrentExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingConcurrentExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllReduceDoubleRingConcurrentExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceDoubleRingConcurrentExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0_ANYPATH_SDMA, CommType::COMM_TAG_RING_INNER);
    commParaLevel0.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0_ANYPATH_SDMA], inputType, outputType));
    CommParaInfo commParaLevel0Rdma(COMM_LEVEL0_ANYPATH_RDMA, CommType::COMM_TAG_RING_INNER);
    commParaLevel0Rdma.forceRdma = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0Rdma, opTransport[COMM_LEVEL0_ANYPATH_RDMA], inputType, outputType));

    return HCCL_SUCCESS;
}

bool CollAllReduceDoubleRingConcurrentExecutor::IsSmallData(const u64 totalSize, const u64 curSize)
{
    return false;
}

bool CollAllReduceDoubleRingConcurrentExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
        curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllReduceDoubleRingConcurrentExecutor::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_HALVING_DOUBLING;
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollAllReduceDoubleRingConcurrentExecutor::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    bool isLargeSize = (curSize / topoAttr_.deviceNumPerAggregation >= HCCL_SPLIT_SIZE_INTER_SERVER);
    return GetExternalInputEnableRdmaSdmaConcurrent() && isLargeSize;
}

HcclResult CollAllReduceDoubleRingConcurrentExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollAllReduceDoubleRingConcurrentExecutor][Run]The CollAllReduceDoubleRingConcurrentExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllReduceDoubleRingConcurrentExecutor][KernelRun]perDataSize size is zero."),
        HCCL_E_PARA);
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice> > multi2RingsSlice; // 数据基于该rank上环0的偏移
    std::vector<std::pair<bool, std::vector<Slice>>> multi4RingsSlice; // 基于2环数据切分2环SDMA+2环ROH bool = true表示SDMA
    u32 ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    CHK_RET(CheckCommSize(COMM_LEVEL0_ANYPATH_SDMA, ringNum));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0_ANYPATH_SDMA, COMM_INDEX_0);
    u32 sliceNum = level0CommInfo.localRankSize;
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    // 构造ring algorithm对应的reduce-scatter实例
    multi2RingsSlice = AnyPathPrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    CHK_PRT_RET(multi2RingsSlice.size() != ringNum, HCCL_ERROR("[CollAllReduceDoubleRingConcurrentExecutor][Run]"\
        "ringNum[%u] != multRingsSliceZero size[%zu]", ringNum, multi2RingsSlice.size()),
        HCCL_E_INTERNAL);

    // 根据数据量计算每个环上数据的偏移和大小
    u32 syncTrans = BEST_SPLIT_VALUE_SR;
    u64 totalDataSize = execMem.count * perDataSize;
    if ((totalDataSize / sliceNum) < HCCL_SPLIT_SIZE_INTER_SERVER) {
        syncTrans = MAX_SPLIT_VALUE;
    }
    multi4RingsSlice.resize(multi2RingsSlice.size() * SLICES_FACTOR);
    for (u32 ringIndex = 0; ringIndex < multi2RingsSlice.size(); ringIndex++) {
        std::vector<Slice> sdmaSlice;
        std::vector<Slice> rdmaSlice;
        for (u32 segsIndex = 0; segsIndex < multi2RingsSlice[ringIndex].size(); segsIndex++) {
            auto totalSize = multi2RingsSlice[ringIndex][segsIndex].size;
            auto sdmaSliceOffset = multi2RingsSlice[ringIndex][segsIndex].offset;
            auto sdmaSliceSize = ((totalSize <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans == MAX_SPLIT_VALUE)) ?
                totalSize : ((syncTrans * totalSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) *
                HCCL_MIN_SLICE_ALIGN_910_93;
            Slice sdmaSliceTmp;
            sdmaSliceTmp.offset = sdmaSliceOffset;
            sdmaSliceTmp.size = sdmaSliceSize;
            Slice rdmaSliceTmp;
            rdmaSliceTmp.offset = sdmaSliceOffset + sdmaSliceSize;
            rdmaSliceTmp.size = totalSize - sdmaSliceSize;
            sdmaSlice.push_back(sdmaSliceTmp);
            rdmaSlice.push_back(rdmaSliceTmp);
            HCCL_DEBUG("Ring index:%u, segId:%u, Orignal [offset %llu, size %llu], sdma "
                "[offset %llu, size %llu], rdma [offset %llu, size %llu]",
                ringIndex, segsIndex, sdmaSliceOffset, totalSize, sdmaSliceTmp.offset,
                sdmaSliceTmp.size, rdmaSliceTmp.offset, rdmaSliceTmp.size);
        }
        multi4RingsSlice[ringIndex] = std::make_pair(true, sdmaSlice); // true表示使用sdma
        multi4RingsSlice[ringIndex + multi2RingsSlice.size()] = std::make_pair(false, rdmaSlice); // false表示rdma
    }
    if (syncTrans == MAX_SPLIT_VALUE) {
        multi4RingsSlice.erase(multi4RingsSlice.end() - multi2RingsSlice.size(), multi4RingsSlice.end());
    }

    HcomCollOpInfo *reduceScatterOpInfoPtr = nullptr;
    // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
    HcomCollOpInfo reduceScatterOpInfo = {
        "", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    if (DMAReduceFlag_) {
        reduceScatterOpInfoPtr = &reduceScatterOpInfo;
    }
    CHK_RET(MultiRingReduceScatterConcurrent(param.tag, execMem.inputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.reduceType, multi4RingsSlice, param.stream,
        PROF_STAGE_0, 0, reduceScatterOpInfoPtr));
    HCCL_INFO("allreduce double ring stage0 run success");

    /* 三步算法step2: 内层 - 节点间 allreduce */
    u64 hdSize;
    u32 segmentIdx;
    u32 commIndex;
    CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, hdSize,
        level0CommInfo, multi2RingsSlice, param.tag));
    auto nicList = topoAttr_.nicList;
    auto devicePhyId = topoAttr_.devicePhyId;
    commIndex = RefreshCommIdx(commIndex, nicList, devicePhyId);
    u64 hdCount = hdSize / perDataSize;
    u32 syncTrans1 = BEST_SPLIT_VALUE_DR;
    if (hdSize < HCCL_SPLIT_SIZE_INTER_SERVER) {
        syncTrans1 = MAX_SPLIT_VALUE;
    }
    if (topoAttr_.superPodNum <= 1) {
        DeviceMem allreduceInput;
        DeviceMem allreduceOutput;
        CHK_RET(CheckCommSize(COMM_LEVEL1_ANYPATH_SDMA, commIndex + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, commIndex);
        u32 level1RankSize = level1CommInfo.localRankSize;
        std::vector<std::pair<bool, Slice>> level1MultSlice;
        level1MultSlice.resize(RDMA_PLANE_NUM_IN_NPRING_DOUBLE);
        Slice sdmaSlice;
        Slice rdmaSlice;
        auto sdmaSliceSize = ((hdSize <= HCCL_MIN_SLICE_ALIGN_910_93) || (syncTrans1 == MAX_SPLIT_VALUE)) ? hdSize:
                ((syncTrans1 * hdSize / MAX_SPLIT_VALUE) / HCCL_MIN_SLICE_ALIGN_910_93) * HCCL_MIN_SLICE_ALIGN_910_93;
        sdmaSlice.size = sdmaSliceSize;
        sdmaSlice.offset = dataSegsSlice[segmentIdx].offset;
        rdmaSlice.size = hdSize - sdmaSlice.size;
        rdmaSlice.offset = dataSegsSlice[segmentIdx].offset + sdmaSlice.size;
        HCCL_DEBUG("Level1 Total[offset:%llu, size:%llu], sdma[offset %llu, size %llu], rdma[offset %llu, size %llu]",
            hdSize, sdmaSlice.offset, sdmaSlice.offset, sdmaSlice.size, rdmaSlice.offset, rdmaSlice.size);
        level1MultSlice[0] = std::make_pair(true, sdmaSlice);
        level1MultSlice[1] = std::make_pair(false, rdmaSlice);
        if (syncTrans1 == MAX_SPLIT_VALUE) {
            level1MultSlice.erase(level1MultSlice.end() - 1, level1MultSlice.end());
        }
        // SDMA和RDMA通信域
        u32 commPlaneNum = level1MultSlice.size();
        for (u32 planeIndex = 0; planeIndex < commPlaneNum; planeIndex++) {
            HcclResult ret = HCCL_SUCCESS;
            Slice dmaSlice = level1MultSlice[planeIndex].second;
            SubCommInfo level1RdmaCommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_RDMA, commIndex);
            SubCommInfo level1TempCommInfo = level1MultSlice[planeIndex].first ? level1CommInfo : level1RdmaCommInfo;
            allreduceInput = execMem.inputMem.range(dmaSlice.offset, dmaSlice.size);
            CHK_SMART_PTR_NULL(allreduceInput);
            allreduceOutput = execMem.outputMem.range(dmaSlice.offset, dmaSlice.size);
            CHK_SMART_PTR_NULL(allreduceOutput);
            u64 SliceCount = dmaSlice.size / perDataSize;
            u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);
            std::unique_ptr<AlgTemplateBase> level1TempAlg;
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1TempAlg = 
                    AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
                HCCL_INFO("allreduce ring: using nonuniform-bruck algo inter-server.");
            } else {
                level1TempAlg = 
                    AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
                HCCL_INFO("allreduce ring: using ring algo inter-server.");
            }
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            
            if (planeIndex != (commPlaneNum - 1)) {
                HCCL_INFO("level1TempCommInfo planeIndex step 0");
                ret = LocalNotify::Wait(algResResp_->slaveStreams[planeIndex], dispatcher_,
                                        algResResp_->notifiesAux[planeIndex], PROF_STAGE_1);
                CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("stream[%u] wait failed", planeIndex), ret);

                CHK_RET(level1TempAlg->Prepare(allreduceInput, allreduceOutput, allreduceOutput, SliceCount, 
                    param.DataDes.dataType, algResResp_->slaveStreams[planeIndex], param.reduceType, 
                    LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), dmaSlice.offset));

                CHK_RET(level1TempAlg->RegisterProfiler(
                    (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1TempCommInfo.localRank,
                    PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, algResResp_->slaveStreams[planeIndex]));

                CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
                ret = LocalNotify::Post(algResResp_->slaveStreams[planeIndex], dispatcher_,
                    algResResp_->notifiesMain[planeIndex], PROF_STAGE_1);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[collAllReduce]level1 stream[%u] record failed", planeIndex), ret);
                //主环record启动从环
                ret = LocalNotify::Post(const_cast<Stream&>(param.stream), dispatcher_,
                    algResResp_->notifiesAux[planeIndex], PROF_STAGE_1);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[collAllReduce]level1 stream[%u] record failed", planeIndex), ret);
            } else {
                HCCL_INFO("level1TempCommInfo planeIndex step 1");
                CHK_RET(level1TempAlg->Prepare(allreduceInput, allreduceOutput, allreduceOutput, SliceCount, 
                    param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID,
                    std::vector<Slice>(0), dmaSlice.offset));
                CHK_RET(level1TempAlg->RegisterProfiler(
                    (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1TempCommInfo.localRank,
                    PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

                CHK_RET(RunTemplate(level1TempAlg, level1TempCommInfo));
                for (u32 ring = 0; ring < (commPlaneNum - 1); ring++) {
                    ret = LocalNotify::Wait(const_cast<Stream&>(param.stream), dispatcher_,
                        algResResp_->notifiesMain[ring], PROF_STAGE_1);
                    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("param.stream[%u] wait failed",ring), ret);
                }
            }
        }
        allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        CHK_RET(AlgTemplateBase::ExecEmptyTask(allreduceInput, allreduceOutput, const_cast<Stream&>(param.stream),
            dispatcher_));
        HCCL_INFO("allreduce double ring stage1 run success");
    } else {
        // 超节点内做reducescatter
        CHK_RET(CheckCommSize(COMM_LEVEL1_ANYPATH_SDMA, commIndex + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, commIndex);
        SubCommInfo level1ZeroCommInfo = GetSubCommInfo(COMM_LEVEL1_ANYPATH_SDMA, COMM_INDEX_0);
        u32 sliceNum = level1ZeroCommInfo.localRankSize;
        // 根据数据量计算每个环上数据的偏移和大小
        CHK_RET(AlgTemplateBase::PrepareSliceData(hdCount, perDataSize, sliceNum, 0, dataSegsSlice));
        DeviceMem reducescatterInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);
        DeviceMem reducescatterOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, hdSize);

        u64 reduceAttr = GetReduceAttr(reducescatterInput, reducescatterOutput,
            param.DataDes.dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1RSTempAlg;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1RSTempAlg);
            CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
            CHK_RET(level1RSTempAlg->Prepare(
                reducescatterInput, reducescatterInput, reducescatterOutput, hdCount,
                param.DataDes.dataType, param.stream, param.reduceType,
                LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        } else {
            level1RSTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
            CHK_SMART_PTR_NULL(level1RSTempAlg);
            CHK_RET(level1RSTempAlg->Prepare(reduceAttr));
            CHK_RET(level1RSTempAlg->Prepare(
                reducescatterInput, reducescatterOutput, reducescatterOutput, hdCount,
                param.DataDes.dataType, param.stream, param.reduceType,
                LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, dataSegsSlice[segmentIdx].offset));
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_RET(level1RSTempAlg->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1RSTempAlg, level1CommInfo));
        HCCL_INFO("allreduce double ring [superpod] level1 reducescatter run success");

        // 超节点间做allreduce
        u64 arSize;
        std::vector<std::vector<Slice> > rdSlice;
        rdSlice.push_back(dataSegsSlice);
        CHK_RET(PrepareLevel1CommInfo(segmentIdx, commIndex, arSize, level1ZeroCommInfo, rdSlice, param.tag));
        auto nicList = topoAttr_.nicList;
        auto devicePhyId = topoAttr_.devicePhyId;
        commIndex = RefreshCommIdx(commIndex, nicList, devicePhyId);
        u64 arCount = arSize / perDataSize;

        CHK_RET(CheckCommSize(COMM_LEVEL2, commIndex + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, commIndex);
        u32 rankSize = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0).localRankSize;

        DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[segmentIdx].offset, arSize);
        DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, arSize);
        reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

        std::unique_ptr<AlgTemplateBase> level2ARTempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
            HCCL_INFO("reducescatter ring: using ring algo inter-server.");
        } else {
            level2ARTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_INFO("reducescatter ring: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level2ARTempAlg);
        CHK_RET(level2ARTempAlg->Prepare(reduceAttr));

        CHK_RET(level2ARTempAlg->Prepare(
            allreduceInput, allreduceOutput, allreduceOutput, arCount,
            param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID,
            std::vector<Slice>(0), dataSegsSlice[segmentIdx].offset));
        CHK_RET(level2ARTempAlg->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2ARTempAlg, level2CommInfo));
        HCCL_INFO("allreduce double ring [superpod] level2 allreduce run success");
        // 超节点内做allgather
        std::unique_ptr<AlgTemplateBase> level1AGTempAlg;
        DeviceMem allgatherInput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, arSize);
        DeviceMem allgatherOutput = execMem.outputMem.range(dataSegsSlice[segmentIdx].offset, arSize*sliceNum);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        } else {
            level1AGTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        }
        CHK_SMART_PTR_NULL(level1AGTempAlg);
        CHK_RET(level1AGTempAlg->Prepare(allgatherOutput, allgatherOutput, allgatherOutput, arCount,
            param.DataDes.dataType, param.stream,
            HcclReduceOp::HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice,
            dataSegsSlice[segmentIdx].offset));
        CHK_RET(level1AGTempAlg->RegisterProfiler(
            (sliceNum << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1AGTempAlg, level1CommInfo));
        HCCL_INFO("allreduce double ring [superpod] level1 allgather run success");
    }

    /* 三步算法step3：外层 - 节点内 allgather */
    HcomCollOpInfo *allgatherOpInfoPtr = nullptr;
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    HcomCollOpInfo allgatherOpInfo = {
        "", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };
    if (DMAReduceFlag_) {
        allgatherOpInfoPtr = &allgatherOpInfo;
    }
    CHK_RET(MultiRingAllGatherConcurrent(param.tag, execMem.inputMem, execMem.outputMem, hdCount,
        param.DataDes.dataType, multi4RingsSlice, param.stream,
        PROF_STAGE_2, 0, allgatherOpInfoPtr));
    HCCL_INFO("allreduce double ring stage2 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceDoubleRingConcurrentExecutor", AllReduceDoubleRingConcurrent,
    CollAllReduceDoubleRingConcurrentExecutor);

} // namespace hccl