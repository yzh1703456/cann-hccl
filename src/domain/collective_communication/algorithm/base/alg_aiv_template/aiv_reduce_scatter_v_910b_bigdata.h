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

class AivReduceScatterVBig910B : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterVBig910B() {}
    template<typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        __gm__ int32_t* ctrlFlagsGM, __gm__ int32_t* ctrlFlagsGMX, int32_t tag);

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t selfOffset, uint64_t othersOffset, uint64_t len,
                                    uint64_t lenSelf, uint64_t maxCount, int32_t tagLeft, int32_t tagSelf, 
                                    ExtraArgs &extraArgs, uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void ClearFlag(uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void EndSync(int32_t tag, uint32_t flagOffsetBase);
};

template<typename T>
__aicore__ inline void AivReduceScatterVBig910B::ClearFlag(uint32_t flagOffsetBase)
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
__aicore__ inline void AivReduceScatterVBig910B::EndSync(int32_t tag, uint32_t flagOffsetBase)
{
    // 共用16个flag
    uint32_t flagOffset =  rankSize_ * FLAG_SIZE + rankSize_ * FLAG_SIZE + flagOffsetBase;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx;
    if (block_idx < rankSize_) {
        if (targetRank != rank_) {
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffset + rank_  * FLAG_SIZE), tag);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank  * FLAG_SIZE), tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank  * FLAG_SIZE), 0);
        }
    }
}

template<typename T>
__aicore__ inline void AivReduceScatterVBig910B::ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
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
__aicore__ inline void AivReduceScatterVBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t selfOffset,
    uint64_t othersOffset, uint64_t len, uint64_t lenSelf, uint64_t maxCount, int32_t tagLeft, int32_t tagSelf,
     ExtraArgs &extraArgs, uint32_t flagOffsetBase)
{
    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx;

    // 共用24个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    uint32_t flagOffsetEnd = rankSize_ * FLAG_SIZE + flagOffsetBase;

    __gm__ T *inputGm = (__gm__ T *)(input + othersOffset);
    __gm__ T *inputGmSelf = (__gm__ T *)(input + selfOffset);
    __gm__ T *outputGmSelf = (__gm__ T *)(output + selfOffset);

    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);
    __gm__ int32_t *flagLocal = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + rank_ * FLAG_SIZE);

    if (block_idx < blockNumPerGroup) {
        int32_t inputOffset = extraArgs.sendDispls[targetRank];
        int32_t cclGmSelfOffset = targetRank * maxCount;

        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + targetRank * FLAG_SIZE);

        if (targetRank != rank_) {
            CpGM2GMWithFlagWrap(cclGmSelf + cclGmSelfOffset, inputGm + inputOffset, len, ctrlFlagsGM, 8, tagLeft);
            //确定对端已经拉走数据
            pipe_barrier(PIPE_ALL);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank  * FLAG_SIZE), tagLeft);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd + targetRank  * FLAG_SIZE), 0);
        }
    } else if (targetRank != rank_) {
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + rank_* FLAG_SIZE);
        __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetCount + rank_ * FLAG_SIZE);

        uint32_t cclGmOtherOffset = rank_ * maxCount;
        ReduceWithFlagWrap(outputGmSelf, cclGmOther + cclGmOtherOffset, lenSelf, ctrlFlagsGM, ctrlFlagsGMX, tagSelf);

        pipe_barrier(PIPE_ALL);
        // 通知对端已把数据拉走
        SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetEnd + rank_ * FLAG_SIZE), tagSelf);
        pipe_barrier(PIPE_ALL);
        // 通知本端已相加
        SetFlagNew(flagLocal, tagSelf, true);
    } else {
        __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount + rank_ * FLAG_SIZE);
        int32_t inputOffset = extraArgs.sendDispls[rank_];
        CpGM2GMWithFlagWrap(outputGmSelf, inputGmSelf + inputOffset, lenSelf, ctrlFlagsGM, 8, tagSelf);
        // 确认已加完
        CheckFlagNew(flagLocal, (rankSize_ - 1) * tagSelf);
        pipe_barrier(PIPE_ALL);
        SetFlagNew(flagLocal, 0);
    }
    return;
}

template<typename T>
__aicore__ inline void aiv_reduce_scatter_v_910b_bigdata(EXTERN_KERNEL_ARGS_DEF)
{
    AivReduceScatterVBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    uint64_t countLeft;
    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / rankSize / sizeof(T);
    if (block_idx >= rankSize) {
        countLeft = extraArgs.sendCounts[block_idx - rankSize];
    } else {
        countLeft = extraArgs.sendCounts[block_idx];
    }
    uint64_t countSelf = extraArgs.sendCounts[rank];
    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;

    int32_t curTagLeft = (tag << 13);
    int32_t curTagSelf = curTagLeft;
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_REDUCE_SCATTER_V_910B_BIGDATA;

    uint64_t selfOffset = 0;
    uint64_t othersOffset = 0;
    while (countLeft > 0 || countSelf > 0) {
        if (block_idx == rank ||(block_idx >= rankSize && countSelf <= 0) ||
            (block_idx < rankSize && countLeft <= 0)) {
            break;
        }
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curCountSelf = (countSelf > maxCountPerLoop) ? maxCountPerLoop : countSelf;
        uint64_t curSize = curCount * sizeof(T);
        uint64_t curSizeSelf = curCountSelf * sizeof(T);
        // 执行kernel
        op.Process<T>(curInput, curOutput, selfOffset, othersOffset, curCount, curCountSelf, maxCountPerLoop,
            curTagLeft, curTagSelf, extraArgs, flagOffsetBase);
        countLeft -= curCount;
        countSelf -= curCountSelf;
        othersOffset += curSize;
        selfOffset += curSizeSelf;
        curTagLeft += curSize / UB_DB_DATA_BATCH_SIZE + 1;
        curTagSelf += curSizeSelf / UB_DB_DATA_BATCH_SIZE + 1;
    }
    op.ClearFlag<T>(flagOffsetBase);
    if (tag == 1000) {
        op.EndSync<T>(tag, flagOffsetBase);
    }
}