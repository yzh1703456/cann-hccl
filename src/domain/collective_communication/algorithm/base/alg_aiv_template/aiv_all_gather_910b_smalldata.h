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

class AivAllGatherSmall910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherSmall910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivAllGatherSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    // 共用2个flag
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_ALL_GATHER_910B_SMALLDATA;
    uint32_t flagOffset = ((tag % 2 == 0) ? 0 : FLAG_SIZE) + flagOffsetBase;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx] + dataOffset);
    __gm__ T *outputGM = (__gm__ T *)output;

    uint64_t count = len;

    if (block_idx != rank_) {
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffset), tag);
        pipe_barrier(PIPE_ALL);

        CpGM2GM(outputGM + block_idx *count, cclGMOther, count);
        // 卡间同步
    } else {
        CpGM2GM(cclGMSelf, inputGM, count);
        // 卡间同步
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset), tag);
        CpGM2GM(outputGM + count * rank_, cclGMSelf, count);
    }
}

template<typename T>
__aicore__ inline void aiv_all_gather_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivAllGatherSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag);
}
