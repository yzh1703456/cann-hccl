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

class AivReduceScatterSmall910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterSmall910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivReduceScatterSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    // 共用16个flag
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_910B_SMALLDATA;
    uint32_t flagOffset = ((tag % 2 == 0) ? 0 : block_num * FLAG_SIZE) + flagOffsetBase;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx] + dataOffset);
    __gm__ T *outputGM = (__gm__ T *)output;

    uint64_t count = len;

    if (block_idx != rank_) {

        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);
        CpGM2GM(cclGMSelf + count * block_idx, inputGM + count * block_idx, count);
        // 卡间同步
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + block_idx * FLAG_SIZE), tag);

        // 对端到ub
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffset + rank_ * FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther[count * rank_], count);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();

        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + rank_ * FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, count);
        SetAtomicNone();

        inOutQue.FreeTensor(localOut);

    } else {
        CpGM2GM(outputGM, inputGM + rank_ * count, count);
        // 卡内同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + rank_ * FLAG_SIZE), tag);
    }
}


template<typename T>
__aicore__ inline void aiv_reduce_scatter_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivReduceScatterSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag);
}