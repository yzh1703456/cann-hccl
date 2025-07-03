/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"
using namespace AscendC;

class AivAllGatherSmallGraph91093 : public AivCommBase {
public:
    __aicore__ inline AivAllGatherSmallGraph91093() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllGatherSmallGraph91093::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    localSetTensor.SetValue(0, tag);
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;

    uint32_t blockNumPerGroup = block_num / rankSize_; // block_num需要能被rankSize_整除
    uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;

    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    uint64_t blockOffset = blockIdxInGroup * avgLengthPerSlice;
    uint32_t dstRank = block_idx / blockNumPerGroup;

    uint32_t flagOffsetBase = BASE_FLAG_OFFSET  * AIV_ALL_GATHER_91093_SMALLDATA_GRAPH;
    uint32_t flagXOffset = blockIdxInGroup * FLAG_SIZE + rank_ * blockNumPerGroup * FLAG_SIZE + flagOffsetBase;
    uint32_t flagOffset = block_idx * FLAG_SIZE + flagOffsetBase;

    __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[dstRank] + flagXOffset);
    __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset);
    GlobalTensor<int32_t> globalSet;
 
    if (dstRank == rank_) {
        CpGM2GM(outputGM + rank_ * len + blockOffset, (__gm__ T *)(inputGM + blockOffset), count);
    } else {
        globalSet.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
        CheckFlagNew(ctrlFlagsGM, tag);
        PipeBarrier<PIPE_ALL>();

        CpGM2GM(outputGM + dstRank * len + blockOffset, (__gm__ T *)(GM_IN[dstRank]) + blockOffset, count);
        
        ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[dstRank] + block_num * FLAG_SIZE + flagXOffset);
        ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + block_num * FLAG_SIZE + flagOffset);
        PipeBarrier<PIPE_MTE3>();
        globalSet.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);
        CheckFlagNew(ctrlFlagsGM, tag);
    }
    return;
}

template<typename T>
__aicore__ inline void aiv_all_gather_91093_smalldata_graph(KERNEL_ARGS_DEF)
{
    AivAllGatherSmallGraph91093 op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, len, tag);
}