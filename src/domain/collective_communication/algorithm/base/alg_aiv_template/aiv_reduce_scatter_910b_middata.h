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

class AivReduceScatterMid910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterMid910B() {}
    
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivReduceScatterMid910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint64_t count = len;

    // 用16个flagsize
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_910B_MIDDATA;
    uint32_t flagOffset = ((tag % 2 == 0) ? 0 : block_num * FLAG_SIZE) + flagOffsetBase;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[block_idx] + dataOffset);
    if (block_idx != rank_) {
        CpGM2GM(cclGmSelf + block_idx * count, inputGm + block_idx * count, count);
        // 卡内同步
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + rank_ * FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + block_idx * FLAG_SIZE), tag);
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffset + rank_ * FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGm, cclGmOther + rank_ * count, count, true, reduceOp_);
    } else {
        CpGM2GM(outputGm, inputGm + rank_ * count, count);
        // 卡内同步
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffset + rank_ * FLAG_SIZE), tag);
    }
}

template<typename T>
__aicore__ inline void aiv_reduce_scatter_910b_middata(KERNEL_ARGS_DEF)
{
    AivReduceScatterMid910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, len, tag);
}