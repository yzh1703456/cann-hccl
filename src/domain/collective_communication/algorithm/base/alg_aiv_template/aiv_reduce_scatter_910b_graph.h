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
 
class AivReduceScatterBigGraph910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterBigGraph910B() {}

    template<typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        __gm__ int32_t* ctrlFlagsGM);
 
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivReduceScatterBigGraph910B::ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
    uint64_t count, __gm__ int32_t* ctrlFlagsGM)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    
    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

        GlobalTensor<int32_t> globalFlag;
        globalFlag.SetGlobalBuffer(ctrlFlagsGM, UB_FLAG_PAD_COUNT);
        LocalTensor<int32_t> localFlag = flagInQue.AllocTensor<int32_t>();

        DataCopy(localFlag, globalFlag, UB_FLAG_PAD_COUNT);

        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t localFlagValue = localFlag.GetValue(0);

        flagInQue.FreeTensor(localFlag);

        if (localFlagValue == 0) {
            continue;
        }

        uint64_t preparedBatchCount = localFlagValue;
        if (processedBatchCount >= preparedBatchCount) {
            continue;
        }

        uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
        if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) {
            curSize = avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T), true, reduceOp_);

        processedBatchCount = preparedBatchCount;
    }
}
template <typename T>
__aicore__ inline void AivReduceScatterBigGraph910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t avgLengthPerSlice = len;
    uint32_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx; // 0-7

    // 共使用16个flag
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_910B_GRAPH;
    uint32_t flagOffsetCount = flagOffsetBase;
    uint32_t flagOffsetEnd = rankSize_ * FLAG_SIZE + flagOffsetBase;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);

    int32_t inputOffset = targetRank * avgLengthPerSlice;
    int32_t cclGmSelfOffset = targetRank * avgLengthPerSlice;
    int32_t outputOffset = targetRank * avgLengthPerSlice;


    if (block_idx == rank_) {
        // 拷贝相应的数据到output
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + rank_ * FLAG_SIZE);
        uint64_t freq = avgSizePerSlice >= 2 * 1024 * 1024 ? 4 : 16;
        CpGM2GMWithFlagWrap(outputGm, inputGm + inputOffset, avgLengthPerSlice, ctrlFlagsGM, freq);
        // 确认本端全部reduce完成
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + rank_ * FLAG_SIZE), (rankSize_ - 1) * tag);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew(ctrlFlagsGM, 0);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + rank_ * FLAG_SIZE), 0);
    } else if (targetRank != rank_) {
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + rank_ * FLAG_SIZE);
        // ReduceWithFlag
        __gm__ int32_t *ctrlFlagsGMStartX = (__gm__ int32_t *)(GM_OUT[targetRank] +
            flagOffsetCount + rank_ * FLAG_SIZE);
        __gm__ int32_t *ctrlFlagsGMStart = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + targetRank * FLAG_SIZE);
        //确定可以从对端拉数据
        SetFlagNew((__gm__ int32_t *)(ctrlFlagsGMStartX), tag);
        CheckFlagNew((__gm__ int32_t *)(ctrlFlagsGMStart), tag);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(ctrlFlagsGMStart), 0);

        uint32_t cclGmOtherOffset = rank_ * avgLengthPerSlice;
        ReduceWithFlagWrap(outputGm, cclGmOther + cclGmOtherOffset, len, ctrlFlagsGM);
        
        // 通知对端数据已经拉走
        // 是否要加个check
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetEnd + rank_ * FLAG_SIZE), tag);
        // 确认对端已经拉走数据
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank * FLAG_SIZE), tag);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank * FLAG_SIZE), 0);
        // 通知本端reduce完成
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + rank_ * FLAG_SIZE), tag, true);
    }

    return;
}

template<typename T>
__aicore__ inline void aiv_reduce_scatter_910b_bigdata_graph(KERNEL_ARGS_DEF)
{
    AivReduceScatterBigGraph910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, len, tag);
}
