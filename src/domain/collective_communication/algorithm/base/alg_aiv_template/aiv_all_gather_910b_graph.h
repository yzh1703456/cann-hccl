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

class AivAllGatherBigGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherBigGraph910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template <typename T>
__aicore__ inline void AivAllGatherBigGraph910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t avgLengthPerSlice = len;
    uint32_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);
    uint32_t targetRank = block_idx; 

    // 共用16个flag
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_ALL_GATHER_910B_GRAPH;
    uint32_t flagOffsetStart = flagOffsetBase;
    uint32_t flagOffsetEnd = block_num * FLAG_SIZE + flagOffsetBase;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);
    
    int32_t inputOffset = targetRank * avgLengthPerSlice;
    int32_t cclGmSelfOffset = targetRank * avgLengthPerSlice;
    int32_t outputOffset = targetRank * avgLengthPerSlice;

    if (targetRank == rank_) {
        CpGM2GM(outputGm + rank_ * avgLengthPerSlice, inputGm, avgLengthPerSlice);
    } else if (targetRank != rank_) {
        __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetStart + rank_ * FLAG_SIZE);
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetStart + block_idx * FLAG_SIZE);
        //确定可以从对端拉数据
        SetFlagNew((__gm__ int32_t *)(ctrlFlagsGMX), tag);
        CheckFlagNew((__gm__ int32_t *)(ctrlFlagsGM), tag);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(ctrlFlagsGM), 0);
        //拉数据
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGm + targetRank * avgLengthPerSlice, cclGmOther, avgLengthPerSlice);
        pipe_barrier(PIPE_ALL);
        // 通知对端数据已经拉走(写对端add)
        SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetEnd + rank_ * FLAG_SIZE), tag);
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + block_idx * FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + block_idx * FLAG_SIZE), 0);
    }            
    return;
}

template<typename T>
__aicore__ inline void aiv_all_gather_910b_bigdata_graph(KERNEL_ARGS_DEF)
{
    AivAllGatherBigGraph910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, len, tag);
}