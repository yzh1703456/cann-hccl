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

class AivAllReduceSmall910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceSmall910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivAllReduceSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint64_t count = len;

    // 用4个flag
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_SMALLDATA;
    uint32_t flagOffsetOut = ((tag % 2 == 0) ? FLAG_ONE_OFFSET : FLAG_THREE_OFFSET) + baseFlagOffset;
    uint32_t flagOffsetIn = ((tag % 2 == 0) ? FLAG_TWO_OFFSET : FLAG_FOUR_OFFSET) + baseFlagOffset;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    if (block_idx == rank_) {
        __gm__ T *inputGM = (__gm__ T *)input;
        __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
        __gm__ T *outputGM = (__gm__ T *)output;

        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, count);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf, localOut, count);

        PipeBarrier<PIPE_MTE3>();

        // 卡间同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag);

        DataCopyUB2GM(outputGT, localOut, count);
        inOutQue.FreeTensor(localOut);

        PipeBarrier<PIPE_MTE3>();
        
        // 卡内同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), tag);
    } else {
        __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx] + dataOffset);
        __gm__ T *outputGM = (__gm__ T *)output;

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        // 卡间同步
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetOut), tag);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther, count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        // 卡内同步
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), tag);
        PipeBarrier<PIPE_ALL>();

        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, count);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);
    }
}

template<typename T>
__aicore__ inline void aiv_all_reduce_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivAllReduceSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag);
}
