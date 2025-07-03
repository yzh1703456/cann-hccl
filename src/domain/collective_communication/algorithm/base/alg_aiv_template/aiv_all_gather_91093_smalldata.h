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

class AivAllGatherSmall91093 : public AivCommBase {
public:
    __aicore__ inline AivAllGatherSmall91093() {}

    __aicore__ inline void Init(GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4,
                                GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9,
                                GM_ADDR buffIn10, GM_ADDR buffIn11, GM_ADDR buffIn12, GM_ADDR buffIn13,
                                GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
                                GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5,
                                GM_ADDR buffOut6, GM_ADDR buffOut7, GM_ADDR buffOut8, GM_ADDR buffOut9,
                                GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
                                GM_ADDR buffOut14, GM_ADDR buffOut15, uint32_t rank, uint32_t rankSize,
                                uint32_t dataType, uint32_t reduceOp, uint32_t root)
    {
        InitBuffArray(buffIn0, buffIn1, buffIn2, buffIn3, buffIn4,
                buffIn5, buffIn6, buffIn7, buffIn8, buffIn9,
                buffIn10, buffIn11, buffIn12, buffIn13,
                buffIn14, buffIn15, buffOut0, buffOut1,
                buffOut2, buffOut3, buffOut4, buffOut5,
                buffOut6, buffOut7, buffOut8, buffOut9,
                buffOut10, buffOut11, buffOut12, buffOut13,
                buffOut14, buffOut15);

        rank_ = rank;
        rankSize_ = rankSize;

        useDoubleBuffer_ = true;

        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_4);
        localFlagTensor = localFlagBuf.Get<int32_t>();

        pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE); // double buffer
    }

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);

private:
    LocalTensor<int32_t> localFlagTensor;
};

template<typename T>
__aicore__ inline void AivAllGatherSmall91093::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t blockNumPerGroup = block_num / rankSize_; // block_num需要能被rankSize_整除
    uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;

    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    uint64_t blockOffset = blockIdxInGroup * avgLengthPerSlice;
    uint32_t dstRank = block_idx / blockNumPerGroup;

    // 共用2个flag
    uint32_t flagOffsetBase = BASE_FLAG_OFFSET * AIV_ALL_GATHER_91093_SMALLDATA;
    uint32_t flagOffset = (((tag % 2 == 0) ? 0 : blockNumPerGroup * FLAG_SIZE)) + flagOffsetBase;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[dstRank] + dataOffset);
    __gm__ T *outputGM = (__gm__ T *)output;

    if (dstRank != rank_) {
        GlobalTensor<int32_t> globalCheck;
        globalCheck.SetGlobalBuffer((__gm__ int32_t *)(GM_OUT[dstRank] + flagOffset + blockIdxInGroup * FLAG_SIZE), UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localFlagTensor[8], globalCheck, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localFlagTensor[8].GetValue(0) == tag) {
                break;
            }
        }
        SyncFunc<HardEvent::S_MTE2>();

        CpGM2GM(outputGM + dstRank * len + blockOffset, cclGMOther + blockOffset, count);
        // 卡间同步
    } else {
        CpGM2GM(cclGMSelf + blockOffset, inputGM + blockOffset, count);
        // 卡间同步
        GlobalTensor<int32_t> globalSet;
        globalSet.SetGlobalBuffer((__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + blockIdxInGroup * FLAG_SIZE), UB_FLAG_PAD_COUNT);
        localFlagTensor.SetValue(0, tag);
        PipeBarrier<PIPE_MTE3>();
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(globalSet, localFlagTensor, UB_FLAG_PAD_COUNT);

        CpGM2GM(outputGM + len * rank_ + blockOffset, inputGM + blockOffset, count); // 与上独立
    }
}

template<typename T>
__aicore__ inline void aiv_all_gather_91093_smalldata(KERNEL_ARGS_DEF)
{
    AivAllGatherSmall91093 op;
    op.Init(KERNEL_CLASS_INIT);
    op.Process<T>(input, output, len, tag);
}
