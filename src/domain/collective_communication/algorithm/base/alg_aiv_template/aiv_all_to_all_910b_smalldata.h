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

class AivAll2AllSmall910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllSmall910B() {}
 
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};
 
template<typename T>
__aicore__ inline void AivAll2AllSmall910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx]);
 
    // 使用24个flag
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_910B_SMALLDATA;

    GM_ADDR flagAddrSelf = GM_OUT[rank_] + baseFlagOffset;
    GM_ADDR flagAddrOther = GM_OUT[block_idx] + baseFlagOffset;
 
    // 共使用2组flag
    uint32_t initAckFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;
 
    uint64_t srcOffset = rank_ * len;
    uint64_t dstOffset = block_idx * len;
 
    if (block_idx != rank_) {
        CpGM2GM(cclGMSelf + dstOffset, inputGM + dstOffset, len);
 
        PipeBarrier<PIPE_ALL>();
 
        SetFlagNew((__gm__ int32_t *)(flagAddrOther + initAckFlagOffset + rank_ * FLAG_SIZE), tag);
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + initAckFlagOffset + block_idx * FLAG_SIZE), tag);
 
        PipeBarrier<PIPE_ALL>();
 
        CpGM2GM(outputGM + dstOffset, cclGMOther + srcOffset, len);
 
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(flagAddrOther + finalAckFlagOffset + rank_ * FLAG_SIZE), tag);
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + block_idx * FLAG_SIZE), tag);
    } else {
        CpGM2GM(outputGM + dstOffset, inputGM + srcOffset, len);
    }
}
 
template<typename T>
__aicore__ inline void aiv_all_to_all_910b_smalldata(KERNEL_ARGS_DEF)
{
    AivAll2AllSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag);
}