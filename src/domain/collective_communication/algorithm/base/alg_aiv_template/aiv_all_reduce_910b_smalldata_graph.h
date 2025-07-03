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

class AivAllReduceSmallGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceSmallGraph910B() {}

    template <typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint32_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllReduceSmallGraph910B::Process(GM_ADDR input, GM_ADDR output, uint32_t len,
    int32_t tag)
{
    uint32_t count = len;
    // 使用16个flag
    uint32_t flagOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_SMALLDATA_GRAPH;
    uint32_t flagOffsetOut = flagOffset;
    uint32_t flagOffsetIn = rank_ * FLAG_INTERVAL + flagOffset;

    if (block_idx == rank_) {
        __gm__ T *inputGM = (__gm__ T *)input;
        __gm__ T *outputGM = (__gm__ T *)output;

        CpGM2GM(outputGM, inputGM, count);

        PipeBarrier<PIPE_MTE3>();
        
        // 卡内同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), (rankSize_ - 1) * tag);
    } else {
        __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx]);
        __gm__ T *outputGM = (__gm__ T *)output;
        // 告诉对端可以从本端拉走数据
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut + block_idx * FLAG_INTERVAL), tag);
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetOut + rank_ * FLAG_INTERVAL), tag);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        // 卡内同步
        CheckFlagGE((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), tag);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), -tag, true);

        PipeBarrier<PIPE_ALL>();

        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, count);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);

        PipeBarrier<PIPE_ALL>();

        // 本端告诉对端已经拉走数据
        SetFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetOut + rank_ * FLAG_INTERVAL + FLAG_SIZE), tag);
        
        // 确认对端已经将所有数据拉走
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut + block_idx * FLAG_INTERVAL + FLAG_SIZE), tag);
        
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut + block_idx * FLAG_INTERVAL), 0);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut + block_idx * FLAG_INTERVAL + FLAG_SIZE), 0);
    }
}

template <typename T>
__aicore__ inline void aiv_all_reduce_910b_smalldata_graph(KERNEL_ARGS_DEF)
{
    AivAllReduceSmallGraph910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag);
}
