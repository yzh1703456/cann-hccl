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

class AivAllGatherVBig910B : public AivCommBase {
public:
    __aicore__ inline AivAllGatherVBig910B() {}

    template<typename T>
    __aicore__ inline void MemcpyWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther, uint64_t count,
        __gm__ int32_t* ctrlFlagsGMX, int32_t tag);

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t curCount,
                                   ExtraArgs &extraArgs, int32_t tag, uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void ClearFlag(uint32_t flagOffsetBase);

    template<typename T>
    __aicore__ inline void EndSync(int32_t tag, uint32_t flagOffsetBase);
};

template<typename T>
__aicore__ inline void AivAllGatherVBig910B::ClearFlag(uint32_t flagOffsetBase)
{
    //从576开始，用10个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount);
    if (block_idx < rankSize_ && block_idx == rank_) {
        SetFlagNew(ctrlFlagsGM, 0);
    }
}

template<typename T>
__aicore__ inline void AivAllGatherVBig910B::EndSync(int32_t tag, uint32_t flagOffsetBase)
{
    //从576开始，用10个flag
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
__aicore__ inline void AivAllGatherVBig910B::MemcpyWithFlagWrap(__gm__ T *cclGmSelf, __gm__ T *cclGmOther,
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

        uint64_t curSize = (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > avgSizePerSlice) ?
            avgSizePerSlice - processedBatchCount * UB_DB_DATA_BATCH_SIZE : 
            (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

        uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        CpGM2GM(cclGmSelf + curProcessedOffset, cclGmOther + curProcessedOffset, curSize / sizeof(T));

        processedBatchCount = preparedBatchCount;
    }
}

template<typename T>
__aicore__ inline void AivAllGatherVBig910B::Process(GM_ADDR input, GM_ADDR output, uint64_t curCount,
                                                     ExtraArgs &extraArgs, int32_t tag, uint32_t flagOffsetBase)
{
    uint32_t blockNumPerGroup = rankSize_;
    uint32_t targetRank = block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx;

    // 用10个flag
    uint32_t flagOffsetCount = flagOffsetBase;
    uint32_t flagOffsetLocal = FLAG_SIZE + flagOffsetBase;

    __gm__ T *inputGm = (__gm__ T *)input;
    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[targetRank]);

    if (block_idx < blockNumPerGroup) {
        if (block_idx == rank_) {   //把数据从UserIn 搬运到 CCLIn，同时检测有多少个核在搬运这个数据
            __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetCount);
            CpGM2GMWithFlagWrap(cclGmSelf, inputGm, curCount, ctrlFlagsGM, 8, tag);
            // 所有对端都取走数据
            pipe_barrier(PIPE_ALL);
            CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetLocal), (rankSize_ - 1) * tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetLocal), 0);
        } else {
            __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetCount);
            MemcpyWithFlagWrap(outputGm + extraArgs.recvDispls[targetRank], cclGmOther, curCount, ctrlFlagsGMX, tag);
            pipe_barrier(PIPE_ALL);
            SetFlagNew((__gm__ int32_t *)(GM_OUT[targetRank] + flagOffsetLocal), tag, true);
        }
    } else {
        CpGM2GM(outputGm + extraArgs.recvDispls[rank_], inputGm, curCount);
    }
    return;
}

template<typename T>
__aicore__ inline void aiv_all_gather_v_910b_bigdata(EXTERN_KERNEL_ARGS_DEF)
{
    AivAllGatherVBig910B op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint64_t maxCountPerLoop = bufferSize / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft;
    if (block_idx < rankSize) {
        countLeft = extraArgs.recvCounts[block_idx];
    } else {
        countLeft = extraArgs.recvCounts[rank];
    }

    GM_ADDR curInput = input;
    GM_ADDR curOutput = output;
    int32_t curTag = (tag << 15);
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET  * AIV_ALL_GATHER_V_910B_BIGDATA;

    while (countLeft > 0) {
        uint64_t curCount = countLeft > maxCountPerLoop ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        // 执行kernel
        op.Process<T>(curInput, curOutput, curCount, extraArgs, curTag, flagOffsetBase);

        countLeft -= curCount;
        curInput += curSize;
        curOutput += curSize;
        curTag += maxCountPerLoop * sizeof(T) / UB_DB_DATA_BATCH_SIZE + 1;  //确认按最大值增加tag的合理性
    }
    op.ClearFlag<T>(flagOffsetBase);
    if (tag == 1000) {
        op.EndSync<T>(tag, flagOffsetBase);
    }
}
