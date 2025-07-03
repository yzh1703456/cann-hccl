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

class AivAllReduceBig910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceBig910B() {}

    template<typename T>
    __aicore__ inline void ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        __gm__ int32_t* ctrlFlagsGM, __gm__ int32_t* ctrlFlagsGMX);

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivAllReduceBig910B::ReduceWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
    uint64_t count, __gm__ int32_t* ctrlFlagsGM, __gm__ int32_t* ctrlFlagsGMX)
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

        uint64_t localFlagValue = localFlag.GetValue(0);
        uint64_t RemoteFlagValue = RemoteFlag.GetValue(0);

        flagInQue.FreeTensor(localFlag);
        flagInQue.FreeTensor(RemoteFlag);

        if (localFlagValue == 0 || RemoteFlagValue == 0) {
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
__aicore__ inline void AivAllReduceBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = 0;

    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = (block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx); // 0-2*rankSize_
    
    // 共用32个flag
    uint32_t flagOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_BIGDATA;
    
    GM_ADDR flagAddrSelf = GM_OUT[rank_] + flagOffset;
    GM_ADDR flagAddrOther = GM_OUT[targetRank] + flagOffset;

    uint32_t pipeCtrlFlagOffset = 2 * MAX_RANK_SIZE * FLAG_SIZE;
    uint32_t midAckFlagOffset = MAX_RANK_SIZE * FLAG_SIZE;
    uint32_t finalAckFlagOffset = 3 * MAX_RANK_SIZE * FLAG_SIZE;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);

    if (block_idx < blockNumPerGroup) {
        uint64_t gmOffset = targetRank * avgLengthPerSlice;
        count = CalActualCount(targetRank, sliceCount, avgLengthPerSlice, tailLength);

        __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t *)(flagAddrSelf + pipeCtrlFlagOffset + targetRank * FLAG_SIZE);

        // 做localcopy, 写偏移16 FLAG_SIZE
        CpGM2GMWithFlagWrap(cclGmSelf + gmOffset, inputGm + gmOffset, count, ctrlFlagsGM);
    } else if (targetRank != rank_) {
        __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t *)(flagAddrSelf + pipeCtrlFlagOffset + rank_ * FLAG_SIZE);
        __gm__ int32_t* ctrlFlagsGMX = (__gm__ int32_t *)(flagAddrOther + pipeCtrlFlagOffset + rank_ * FLAG_SIZE);

        uint64_t gmOffset = rank_ * avgLengthPerSlice;

        count = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);

        // 做reduce, 检查偏移16 FLAG_SIZE
        ReduceWithFlagWrap(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, ctrlFlagsGM, ctrlFlagsGMX);
    }

    pipe_barrier(PIPE_ALL);

    if (block_idx >= blockNumPerGroup) {
        // reduce做完第1次全卡同步
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf), tag, true);
        
        // 全aiv退出前同步
        SetFlagNew((__gm__ int32_t*)(flagAddrSelf + FLAG_SIZE), tag, true);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);

        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + 2 * FLAG_SIZE), tag);
        return;
    }
    pipe_barrier(PIPE_ALL);
    
    if (block_idx == rank_) {
        // check 本端aiv 所有reduce结果是否完成
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf), rankSize_ * tag);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf), 0);

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);

        // 告诉别人自己已经加完所有卡了
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf + midAckFlagOffset + rank_ * FLAG_SIZE), tag);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    }

    // 每个aiv读相应对端的flag
    CheckFlagNew((__gm__ int32_t *)(flagAddrOther + midAckFlagOffset + block_idx * FLAG_SIZE), tag);

    // 清空计数
    __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t *)(flagAddrSelf + pipeCtrlFlagOffset + targetRank * FLAG_SIZE);
    SetFlagNew(ctrlFlagsGM, 0);

    pipe_barrier(PIPE_ALL);

    // 3. 每个aiv再把rankSize张卡上其他位置的数据搬运到本卡的对应位置
    uint64_t gmOffset = block_idx * avgLengthPerSlice;
    count = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);

    CpGM2GM(outputGm + gmOffset, cclGmOther + gmOffset, count);
    pipe_barrier(PIPE_ALL);

    // 通知对端，自己已经把对端的那片数据拉回来了
    SetFlagNew((__gm__ int32_t *)(flagAddrOther + finalAckFlagOffset + rank_ * FLAG_SIZE), tag);
    pipe_barrier(PIPE_ALL);
    
    // 确认对端已经将对应的数据拉走
    CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + block_idx * FLAG_SIZE), tag);
    pipe_barrier(PIPE_ALL);
    SetFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + block_idx * FLAG_SIZE), 0);
    pipe_barrier(PIPE_ALL);

    // 卡内所有aiv同步
    SetFlagNew((__gm__ int32_t*)(flagAddrSelf + FLAG_SIZE), tag, true);
    if (block_idx == rank_) {
        pipe_barrier(PIPE_ALL);
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + FLAG_SIZE), rankSize_ * 2 * tag);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t*)(flagAddrSelf + FLAG_SIZE), 0);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t*)(flagAddrSelf + 2 * FLAG_SIZE), tag);
    } else {
        pipe_barrier(PIPE_ALL);
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + 2 * FLAG_SIZE), tag);
    }
    
    return;
}

template<typename T>
__aicore__ inline void aiv_all_reduce_910b_bigdata(KERNEL_ARGS_DEF)
{
    AivAllReduceBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft = len;

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << 12);
    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);
        
        // 执行kernel
        op.Process<T>(curInput, curOutput, curCount, curTag);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += 1;
    }
}
