/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_GATHER_910B_RDMA_H
#define AIV_ALL_GATHER_910B_RDMA_H

#include "aiv_communication_base.h"
using namespace AscendC;

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__

template<typename T>
class AivAllGather910BRdma : public AivCommBase {
public:
    
    FORCE_INLINE_AICORE  AivAllGather910BRdma() {}

    /**
     *  8个核就够拉整个8个不同卡cclOut到userOut了
     */

    FORCE_INLINE_AICORE void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t bufferSize, uint64_t serverNum) 
    {
        if (block_idx >= rankSize_) {
            return;
        }

        if (block_idx == 0) {
            // 本卡该片数据已经可以被跨片读取
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + FLAG_SIZE), tag);
        }
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);

        // todo:1、serverNum需要赋值。 2、len是inputCount 还是inputSize还是 output相关？
        for (int i = 0; i < serverNum; i++) {
            int64_t sendSize = len * sizeof(T);
            int64_t sendSizeOffset = i * len * sizeof(T);
            int64_t receiveSizeOffset = (i * rankSize_ + block_idx) * len * sizeof(T);
            CpGM2GM<T>((__gm__ T*)((__gm__ char*)output + receiveSizeOffset), (__gm__ T*)((__gm__ char*)(GM_IN[block_idx]) + sendSizeOffset), len);
        }
        pipe_barrier(PIPE_ALL);
        //尾同步，每个卡搬完完后要进行标记。要确保所有卡都搬完再退出。
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_]) + FLAG_SIZE + block_idx*FLAG_SIZE, tag);
        pipe_barrier(PIPE_ALL);
        for (int j = 0; j< rankSize_; j++) {
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[j]) + FLAG_SIZE + block_idx*FLAG_SIZE, tag);
        }
    }
};

template <typename T>
FORCE_INLINE_AICORE void aiv_all_gather_910b_rdma(KERNEL_ARGS_DEF)
{
    AivAllGather910BRdma<T> op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process(input, output, len, tag, bufferSize, serverNum);
}
#endif // AIV_ALL_GATHER_910B_RDMA_H
