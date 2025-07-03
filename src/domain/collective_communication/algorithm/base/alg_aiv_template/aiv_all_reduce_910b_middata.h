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

class AivAllReduceMid910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceMid910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivAllReduceMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t padCount = 32 / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = 0;

    // 用9个flag
    uint32_t flagOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_MIDDATA;
    flagOffset += (tag % 2 == 0) ? 0 : (rankSize_ + 1) * FLAG_SIZE;

    GM_ADDR flagAddrSelf = GM_OUT[rank_] + flagOffset;
    GM_ADDR flagAddrOther = GM_OUT[block_idx] + flagOffset;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    uint32_t dataOffset = (tag % 2 == 0) ? 0 : AIV_PING_PONG_SIZE;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[block_idx] + dataOffset);

    if (block_idx == rank_) {
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf), 0);
        PipeBarrier<PIPE_ALL>();
    }

    // LocalCopy
    uint64_t gmOffset = block_idx * avgLengthPerSlice;

    count = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);
    CpGM2GM(cclGmSelf + gmOffset, inputGm + gmOffset, count);
    PipeBarrier<PIPE_ALL>();

    SetFlagNew((__gm__ int32_t*)(flagAddrSelf + FLAG_SIZE + block_idx * FLAG_SIZE), tag);

    // ReduceScatter
    if (block_idx != rank_) {
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + FLAG_SIZE + rank_ * FLAG_SIZE), tag);
        CheckFlagNew((__gm__ int32_t *)(flagAddrOther + FLAG_SIZE + rank_ * FLAG_SIZE), tag);

        count = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);

        uint64_t gmOffset = rank_ * avgLengthPerSlice;

        PipeBarrier<PIPE_ALL>();
        CpGM2GM(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, true, reduceOp_);

        PipeBarrier<PIPE_MTE3>();
        
        // 本aiv reduce完成
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf), tag, true);
    }

    // 每个aiv读相应对端的flag
    CheckFlagNew((__gm__ int32_t *)(flagAddrOther), (rankSize_ - 1) * tag);

    // AllGather
    gmOffset = block_idx * avgLengthPerSlice;
    count = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);

    PipeBarrier<PIPE_ALL>();
    CpGM2GM(outputGm + gmOffset, cclGmOther + gmOffset, count);

    return;
}

template<typename T>
__aicore__ inline void aiv_all_reduce_910b_middata(KERNEL_ARGS_DEF)
{
    AivAllReduceMid910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, len, tag);
}
