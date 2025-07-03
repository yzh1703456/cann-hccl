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

class AivAllGatherBig910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherBig910B() {}

    template<typename T>
    __aicore__ inline void MemcpyWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        __gm__ int32_t* ctrlFlagsGMX, int32_t tag);

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, uint64_t totalLen,
        uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void ClearFlag(uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void EndSync(int32_t tag, uint32_t flagOffsetBase);
};

template<typename T>
__aicore__ inline void AivAllGatherBig910B::ClearFlag(uint32_t flagOffsetBase)
{
    // 用10个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount);
    if (block_idx < rankSize_ && block_idx == rank_) {
        SetFlagNew(ctrlFlagsGM, 0);
    }
}

template<typename T>
__aicore__ inline void AivAllGatherBig910B::EndSync(int32_t tag, uint32_t flagOffsetBase)
{
    // 用10个flag
    uint32_t flagOffset = FLAG_SIZE + FLAG_SIZE + rankSize_ * FLAG_SIZE + flagOffsetBase;
    __gm__ int32_t *ctrlFlagsGM;
    if (block_idx < rankSize_ && block_idx == rank_) {
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_; 
            ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[targetRank ] + flagOffset + rank_ * FLAG_SIZE);
            SetFlagNew(ctrlFlagsGM, tag);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_; 
            ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE);
            CheckFlagNew(ctrlFlagsGM, tag);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_; 
            ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE);
            SetFlagNew(ctrlFlagsGM, 0);
        }
    }
}

template<typename T>
__aicore__ inline void AivAllGatherBig910B::MemcpyWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
    uint64_t count, __gm__ int32_t* ctrlFlagsGMX, int32_t tag)
{
    uint64_t processedBatchCount = 0;
    uint64_t avgSizePerSlice = count * sizeof(T);
    
    while (true) {
        if (processedBatchCount >= CeilDiv(avgSizePerSlice, UB_DB_DATA_BATCH_SIZE)) {
            break;
        }

        GlobalTensor<int32_t> globalFlagX;
        globalFlagX.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
        LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();

        DataCopy(localFlagX, globalFlagX, UB_FLAG_PAD_COUNT);

        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t localFlagValueX = localFlagX.GetValue(0);

        flagInQue.FreeTensor(localFlagX);

        if (localFlagValueX <= tag) {
            continue;
        }

        uint64_t preparedBatchCount = localFlagValueX - tag;
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
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T));

        processedBatchCount = preparedBatchCount;
    }
}

template<typename T>
__aicore__ inline void AivAllGatherBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag,
    uint64_t totalLen, uint32_t flagOffsetBase)
{
    uint32_t avgLengthPerSlice = len;
    uint32_t avgSizePerSlice = avgLengthPerSlice * sizeof(T);

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx; 

    // 用10个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    uint32_t flagOffsetLocal = FLAG_SIZE + flagOffsetBase;
    uint32_t flagOffsetEnd = 2 * FLAG_SIZE + flagOffsetBase;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);

    if (block_idx < blockNumPerGroup) {
        int32_t outputOffset = targetRank * totalLen;
        if (block_idx == rank_) {
            __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount);
            CpGM2GMWithFlagWrap(cclGmSelf, inputGm, avgLengthPerSlice, ctrlFlagsGM, 8, tag);
            // 所有对端都取走数据
            pipe_barrier(PIPE_ALL);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetLocal), (rankSize_ - 1) * tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetLocal), 0);
        } else {
            __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetCount);
            MemcpyWithFlagWrap(outputGm + outputOffset, cclGmOther, len, ctrlFlagsGMX, tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetEnd + rank_ * FLAG_SIZE), tag);
            pipe_barrier(PIPE_ALL);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank * FLAG_SIZE), tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetLocal), tag, true);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank * FLAG_SIZE), 0);
            pipe_barrier(PIPE_ALL);
        }
    } else {
        CpGM2GM(outputGm + rank_ * totalLen, inputGm, avgLengthPerSlice);
    }    
    return;
}

template<typename T>
__aicore__ inline void aiv_all_gather_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivAllGatherBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft = len;

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << 13);
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_ALL_GATHER_910B_BIGDATA;
    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);
        
        // 执行kernel
        op.Process<T>(curInput, curOutput, curCount, curTag, len, flagOffsetBase);

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
