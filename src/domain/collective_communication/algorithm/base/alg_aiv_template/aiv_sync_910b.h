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

class AivSync910B : public AivCommBase {
public:
    __aicore__ inline AivSync910B() {}
    __aicore__ inline void Process(int32_t tag);
};

__aicore__ inline void AivSync910B::Process(int32_t tag)
{
    // 从0开始，用4个flag
    uint32_t flagOffset = 2 * 1024 * 1024;
    flagOffset += ((tag % AIV_PING_PONG_FACTOR_TWO == 0) ? 0 : rankSize_ * FLAG_SIZE);
    if (block_idx != rank_) {
        // 卡间同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffset + rank_ * FLAG_SIZE), 1);
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + block_idx * FLAG_SIZE), 1);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + block_idx * FLAG_SIZE), 0);
    } 
}

__aicore__ inline void aiv_sync_910b_inner(KERNEL_ARGS_DEF)
{
    AivSync910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process(tag);
}
