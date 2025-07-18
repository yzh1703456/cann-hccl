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

class AivReduceScatterBig910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterBig910B() {}
    template<typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        __gm__ int32_t* ctrlFlagsGM, __gm__ int32_t* ctrlFlagsGMX, int32_t tag);
    
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount, int32_t tag,
        uint64_t totallen, uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void ClearFlag(uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void EndSync(int32_t tag, uint32_t flagOffsetBase);
};

template<typename T>
__aicore__ inline void AivReduceScatterBig910B::ClearFlag(uint32_t flagOffsetBase)
{
    // 共用24个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    uint32_t flagOffsetEnd = rankSize_ * FLAG_SIZE + flagOffsetBase;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx;
    __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + targetRank * FLAG_SIZE);
    if (block_idx < rankSize_) {
        if (targetRank != rank_) {
            SetFlagNew(ctrlFlagsGM, 0);
        }
    } else if (targetRank == rank_) {
       SetFlagNew(ctrlFlagsGM, 0);
    }
}

template<typename T>
__aicore__ inline void AivReduceScatterBig910B::EndSync(int32_t tag, uint32_t flagOffsetBase)
{
    // 共用16个flag
    uint32_t flagOffset =  rankSize_ * FLAG_SIZE + rankSize_ * FLAG_SIZE + flagOffsetBase;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx;
    if (block_idx < rankSize_) {
        if (targetRank != rank_) {
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank ] + flagOffset + rank_  * FLAG_SIZE), tag);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank  * FLAG_SIZE), tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank  * FLAG_SIZE), 0);
        }
    }
}

template<typename T>
__aicore__ inline void AivReduceScatterBig910B::ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
    uint64_t count, __gm__ int32_t* ctrlFlagsGM, __gm__ int32_t* ctrlFlagsGMX, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    
    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

        GlobalTensor<int32_t> globalFlag;
        globalFlag.SetGlobalBuffer(ctrlFlagsGM, UB_FLAG_PAD_COUNT);
        GlobalTensor<int32_t> globalFlagX;
        globalFlagX.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        LocalTensor<int32_t> localFlag = flagInQue.AllocTensor<int32_t>();
        LocalTensor<int32_t> RemoteFlag = flagInQue.AllocTensor<int32_t>();

        DataCopy(localFlag, globalFlag, UB_FLAG_PAD_COUNT);
        DataCopy(RemoteFlag, globalFlagX, UB_FLAG_PAD_COUNT);

        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        int64_t localFlagValue = localFlag.GetValue(0) - tag;
        int64_t RemoteFlagValue = RemoteFlag.GetValue(0) - tag;

        flagInQue.FreeTensor(localFlag);
        flagInQue.FreeTensor(RemoteFlag);

        if (localFlagValue <= 0 || RemoteFlagValue <= 0) {
            continue;
        }

        uint64_t preparedBatchCount = (localFlagValue <= RemoteFlagValue) ? localFlagValue : RemoteFlagValue;
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


template<typename T>
__aicore__ inline void AivReduceScatterBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, uint64_t maxCount,
    int32_t tag, uint64_t totallen, uint32_t flagOffsetBase)
{
    uint64_t avgLengthPerSlice = len;
    uint64_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx;

    // 共用24个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    uint32_t flagOffsetEnd = rankSize_ * FLAG_SIZE + flagOffsetBase;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);
    __gm__ int32_t *flagLocal = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + rank_ * FLAG_SIZE);

    if (block_idx < blockNumPerGroup) {
        uint64_t inputOffset = targetRank * totallen;
        uint64_t cclGmSelfOffset = targetRank * maxCount;

        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + targetRank * FLAG_SIZE);

        if (targetRank != rank_) {
            CpGM2GMWithFlagWrap(cclGmSelf + cclGmSelfOffset, inputGm + inputOffset, avgLengthPerSlice, ctrlFlagsGM, 8, tag);
            //确定对端已经拉走数据
            pipe_barrier(PIPE_ALL);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank  * FLAG_SIZE), tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank  * FLAG_SIZE), 0);
        }
    } else if (targetRank != rank_) {
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + rank_* FLAG_SIZE);
        __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetCount + rank_ * FLAG_SIZE);

        uint64_t cclGmOtherOffset = rank_ * maxCount;

        ReduceWithFlagWrap(outputGm, cclGmOther + cclGmOtherOffset, len, ctrlFlagsGM, ctrlFlagsGMX, tag);

        pipe_barrier(PIPE_ALL);
        // 通知对端已把数据拉走
        SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetEnd + rank_ * FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        // 通知本端已相加
        SetFlagNew(flagLocal, tag, true);
    } else {
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + rank_ * FLAG_SIZE);
        uint64_t inputOffset = rank_ * totallen;

        CpGM2GMWithFlagWrap(outputGm, inputGm + inputOffset, avgLengthPerSlice, ctrlFlagsGM, 8, tag);
        // 确认已加完
        CheckFlagNew(flagLocal, (rankSize_ - 1) * tag);
        pipe_barrier(PIPE_ALL);
        SetFlagNew(flagLocal, 0);
    }

    return;
}

template<typename T>
__aicore__ inline void aiv_reduce_scatter_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivReduceScatterBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / rankSize / sizeof(T);
    uint64_t countLeft = len;

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << 15);
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_910B_BIGDATA;
    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        // 执行kernel
        op.Process<T>(curInput, curOutput, curCount, maxCountPerLoop, curTag, len, flagOffsetBase);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
    op.ClearFlag<T>(flagOffsetBase);
    if (tag == 1000) {
        op.EndSync<T>(tag, flagOffsetBase);
    }
}