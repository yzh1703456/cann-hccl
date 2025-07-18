/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_executor.h"

namespace hccl {
CollAllGatherExecutor::CollAllGatherExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollCommExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAllGatherExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    HCCL_PROFILER_ADD_TAG(param.tag, algoAttr_.identifier, workflowMode_);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);
    CHK_RET(AddSubStreamToProfiling());
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_ADD_OPDATA_OP(param.tag, param.DataDes.count, param.inputPtr, param.outputPtr,
            param.DataDes.dataType, INVALID_VALUE_RANKID, algoAttr_.identifier, HcclReduceOp::HCCL_REDUCE_RESERVED);
        HCCL_PROFILER_ADD_GROUPRANK(algoAttr_.identifier, topoAttr_.userRankSize, topoAttr_.userRank);
    }

    HcclResult ret = HCCL_SUCCESS;
    // 图模式和单卡场景下不需要Loop
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u64 totalSize = param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputMem = DeviceMem::create(algRes.paramInputMem.ptr(), totalSize);
        execMem.outputMem = DeviceMem::create(algRes.paramOutputMem.ptr(), totalSize * topoAttr_.userRankSize);
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        HCCL_DEBUG("[CollAllGatherExecutor][Orchestrate]offload inputMem[%p][%llu], outputMem[%p][%llu]," \
            "scratchMem[%p][%llu], inputPtr[%p] outputPtr[%p], count[%llu]",
            execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
            execMem.scratchMem.ptr(), execMem.scratchMem.size(), execMem.inputPtr, execMem.outputPtr, execMem.count);
        ret = KernelRun(param, execMem);
    } else if (topoAttr_.userRankSize == 1) {
        ExecMem execMem;
        execMem.count = param.DataDes.count;
        execMem.inputMem = algRes.cclInputMem;
        execMem.outputMem = algRes.cclOutputMem;
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = param.inputPtr;
        execMem.outputPtr = param.outputPtr;
        ret = KernelRun(param, execMem);
    } else {
        ret = RunLoop(param, algRes);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherExecutor][Orchestrate]errNo[0x%016llx]all gather excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !is310P3Common_) {
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
        HCCL_PROFILER_DEL_TAG(param.tag);
        HCCL_PROFILER_DEL_OPDATA(param.tag);
        HCCL_PROFILER_DEL_GROUPRANK(algoAttr_.identifier);
    }
    HCCL_INFO("tag[%s], Allgather executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}


u64 CollAllGatherExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    HCCL_WARNING("[CollAllGatherExecutor][CalcLoopMaxCount]" \
        "using default maxCountPerLoop[%llu] as CCLBuffSize / unitSize.", maxCountPerLoop);
    return maxCountPerLoop;
}

bool CollAllGatherExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE ||
            curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

bool CollAllGatherExecutor::IsSmallData(const u64 size)
{
    HCCL_INFO("[CollAllGatherExecutor][IsSmallData]opMeta is using the default option: not small data.");
    return false;
}

bool CollAllGatherExecutor::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    HCCL_INFO("[CollAllGatherExecutor]opMeta is using the default option: not data split.");
    return false;
}

// 基于性能考量，合并RunLoop和RunLoopInner
HcclResult CollAllGatherExecutor::RunLoop(OpParam &param, AlgResourceResponse &algRes)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    u8 *curInputPtr = static_cast<u8 *>(param.inputPtr);
    u8 *curOutputPtr = static_cast<u8 *>(param.outputPtr);
    void *commInputPtr = algRes.cclInputMem.ptr();
    u8 *commOutputPtr = static_cast<u8 *>(algRes.cclOutputMem.ptr());
    CHK_PTR_NULL(curInputPtr);
    CHK_PTR_NULL(curOutputPtr);
    CHK_PTR_NULL(commInputPtr);
    CHK_PTR_NULL(commOutputPtr);

    u64 maxCountPerLoop = CalcLoopMaxCount(algRes.cclInputMem.size(), unitSize);   // override
    CHK_PRT_RET(maxCountPerLoop == 0,
        HCCL_ERROR("[CollAllGatherExecutor][RunLoop]tag[%s], userRankSize is [%u], maxCountPerLoop is [%llu].",
            param.tag.c_str(), topoAttr_.userRankSize, maxCountPerLoop),
        HCCL_E_PARA);

    bool smallData = IsSmallData(param.DataDes.count * unitSize);
    for (u64 countLeft = param.DataDes.count, curCount = 0, inputOffset = 0, outputOffset = 0;
            countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        // 判断剩余数据量对应的output size是否大于中转output size
        curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]tag[%s], inputOffset[%llu], outputOffset[%llu], " \
            "sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%d].",
            param.tag.c_str(), inputOffset, outputOffset, curInputPtr, curOutputPtr, curCount, param.DataDes.dataType);

        if (!is310P3Common_) {
            /* 设置子图复用标志 */
            auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
            bool hugeData = IsHugeData(curSize);    // override
            bool dataSplit = IsDataSplitForRdmaSdmaConcurrent(curSize);
            auto opMeta = HcclOpMetaInfo::GetOneForAllGather(autoSelectedAlgTypeLevel1, hugeData, smallData,
                CopyPattern::BCOPY, dataSplit);
            CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));
        }

        // 执行
        if (!DMAReduceFlag_) {
            // 如果使用in CCL buffer，需要将user buffer in中的结果拷贝到CCL buffer in
            DeviceMem srcMem = DeviceMem::create(curInputPtr, curSize);
            DeviceMem dstMem = DeviceMem::create(commInputPtr, curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            HCCL_DEBUG("[CollAllGatherExecutor][RunLoop]copy from user in to ccl in.");
        }

        // 使用当前Loop偏移到的地址作为当前的inputPtr和outputPtr
        ExecMem execMem;
        execMem.count = curCount;
        execMem.inputMem = DeviceMem::create(commInputPtr, curSize);
        execMem.outputMem = DeviceMem::create(commOutputPtr, curSize * topoAttr_.userRankSize);
        execMem.scratchMem = algRes.scratchMem;
        execMem.inputPtr = curInputPtr;
        execMem.outputPtr = curOutputPtr;
        HcclResult ret = KernelRun(param, execMem);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherExecutor][RunLoop]errNo[0x%016llx]kernel run error, tag[%s], " \
            "inputMem ptr[%p], outputMem ptr[%p], count[%llu], dataType[%d]",
            HCCL_ERROR_CODE(ret), param.tag.c_str(), commInputPtr, commOutputPtr,
            curCount, param.DataDes.dataType),
            ret);

        if (!DMAReduceFlag_) {
            // 如果使用CCL buffer，需要将CCL buffer out中的结果拷贝到user buffer out
            for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
                // 拷贝中转output上每个slice的数据到output内存，目的端中每个slice的size固定为output的size
                DeviceMem dstMem = DeviceMem::create(curOutputPtr + param.DataDes.count * unitSize * i, curSize);
                DeviceMem srcMem = DeviceMem::create(commOutputPtr + curSize * i, curSize);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, param.stream));
            }
        }

        if (!is310P3Common_) {
            CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
        }

        inputOffset = curSize;
        outputOffset = curSize;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::PrepareAllgatherSlice(u32 sliceNum, u64 inputMemSize,
    std::vector<Slice> &dataSegsSlice) const
{
    Slice sliceTemp;
    for (u32 i = 0; i < sliceNum; i++) { // 根据数据量计算每个环上数据的偏移和大小
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize * i;
        dataSegsSlice.push_back(sliceTemp);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::CalculateLevel1AllgatherSlice(u64 inputMemSize, u32 level0RankSize, u32 level1RankSize,
    std::vector<std::vector<Slice>> multRingsSliceZero, std::vector<std::vector<Slice>> &multRingsSlice) const
{
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level1DataSlice;
        for (u32 level0Idx = 0; level0Idx < level0RankSize; level0Idx++) {
            CHK_PRT_RET(multRingsSliceZero[ringIndex].size() < level0RankSize,
                HCCL_ERROR("[CalculateLevel1AllgatherSlice]multRingsSliceZero[ringIndex]" \
                "size is smaller than level0RankSize."), HCCL_E_INTERNAL);
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                Slice tmpSlice;
                tmpSlice.size = multRingsSliceZero[ringIndex][level0Idx].size;
                tmpSlice.offset =
                    multRingsSliceZero[ringIndex][level0Idx].offset + level1Idx * level0RankSize * inputMemSize;
                level1DataSlice.push_back(tmpSlice);
            }
        }
        multRingsSlice.push_back(level1DataSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::CalculateLevel2AllgatherSlice(u64 inputMemSize, u32 level0RankSize,
    u32 level1RankSize, u32 level2RankSize, std::vector<std::vector<Slice>> multRingsSliceZero,
    std::vector<Slice> &level2DataSlice, u32 ringIndex) const
{
    for (u32 level0Idx = 0; level0Idx < level0RankSize; level0Idx++) {
        for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                Slice tmpSlice;
                tmpSlice.size = multRingsSliceZero[ringIndex][level0Idx].size;
                tmpSlice.offset = multRingsSliceZero[ringIndex][level0Idx].offset +
                    (level1Idx * level0RankSize + level2Idx * level0RankSize * level1RankSize) *inputMemSize;
                level2DataSlice.push_back(tmpSlice);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherExecutor::AllGatherLevel2(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    u64 count, HcclDataType dataType, Stream &stream, HcomCollOpInfo *opInfo)
{
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0));
    SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    u64 inputMemSize = inputMem.size();
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 level1RankSize = level1CommInfo.localRankSize;
    u32 level2RankSize = level2CommInfo.localRankSize;
    u32 level0ServerIndex = level0CommInfo.localRank;
    u32 level1ServerIndex = level1CommInfo.localRank;

    std::unique_ptr<AlgTemplateBase> level2AGExecutor;
    level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    HCCL_INFO("allgather ring: using ring algo inter-server.");
    CHK_SMART_PTR_NULL(level2AGExecutor);

    // 计算slice, 不同超节点相同slice
    std::vector<Slice> level2DataSegsSlice;
    Slice sliceTemp;
    for (u32 i = 0; i < level2RankSize; i++) {
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = i * level1RankSize * level0RankSize * inputMemSize;
        level2DataSegsSlice.push_back(sliceTemp);
    }
    //  outputMem传整块，通过baseOffset偏移
    u64 level2BaseOffset = (level0ServerIndex + level1ServerIndex * level1RankSize) * inputMemSize;
    CHK_RET(level2AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, level2BaseOffset));

    CHK_RET(level2AGExecutor->RegisterProfiler((
        level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, stream));

    CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
    HCCL_INFO("allgather double ring [superpod] level2 allgather run success");

    // 第二步，各个AI Server 间 all gather (ring/NHR)
    HCCL_INFO("commIdx:%u Tag[%s].commLevel1.size():%u", commIndex, tag.c_str(),
        level1RankSize);

    if (level1RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level1AGExecutor;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("allgather ring: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
        } else {
            HCCL_ERROR("allgather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGExecutor);

        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice;
        for (u32 j = 0; j < level2RankSize; j++) {
            for (u32 i = 0; i < level1RankSize; i++) {
                sliceTemp.size = inputMemSize;
                sliceTemp.offset =
                    (i * level0RankSize +  j * level1RankSize * level0RankSize + level0ServerIndex) *inputMemSize;
                level1DataSegsSlice.push_back(sliceTemp);
            }
        }

        CHK_RET(level1AGExecutor->Prepare(outputMem, outputMem, inputMem, count, dataType, stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

        CHK_RET(level1AGExecutor->RegisterProfiler((
            level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, stream));

        CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo));
        HCCL_INFO("allgather double ring [superpod] level1 allgather run success");
    }

    // 节点内做all gather double ring
    std::vector<Slice> dataSegsSlice;
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    //  多环数据切分
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    std::vector<std::vector<Slice>> multRingsSlice;
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize, level2RankSize,
            multRingsSliceZero, level2DataSlice, ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    if (!DMAReduceFlag_) {
        multRingsUserMemSlice = multRingsSlice;
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> level2UserMemSlice;
            for (auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset =
                    (cclSlice.offset / inputMemSize) * count * perDataSize +
                    multRingsSliceZero[ringIndex][0].offset;
                level2UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level2UserMemSlice);
        }
    }

    CHK_RET(ActiveSlaveStreams(stream));
    if (DMAReduceFlag_ && level1RankSize > 1) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo->inputAddr = nullptr;
    }
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count,
        dataType, multRingsSlice, stream, PROF_STAGE_2, 0, opInfo, multRingsUserMemSlice));

    HCCL_INFO("allgather double ring [superpod] level2 allgather run success");
    return HCCL_SUCCESS;
}

} // namespace hccl