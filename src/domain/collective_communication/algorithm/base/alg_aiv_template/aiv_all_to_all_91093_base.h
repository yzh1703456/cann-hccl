/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALLTOALL_91093_BASE_H
#define AIV_ALLTOALL_91093_BASE_H

#include "aiv_communication_base.h"

using namespace AscendC;

class AivAll2All91093Base {
public:
    __aicore__ inline AivAll2All91093Base() {}

    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize, int32_t tag,
        uint32_t baseFlagOffset);

    template<typename T>
    __aicore__ inline void DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count);

    template<HardEvent event> 
    __aicore__ inline void SyncFunc();

    __aicore__ inline void BatchRecordWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag);

protected:
    uint32_t baseFlagOffset_ = 0;
    GM_ADDR flagAddrSelf_;
    uint32_t rank_;
    uint32_t rankSize_;

    TPipe pipe;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localClearTensor;
    TBuf<> bufferArgsBuf;
    LocalTensor<uint64_t> bufferArgsTensor; // buffer地址GM-UB
    TBuf<> offsetArgsBuf;
    LocalTensor<uint64_t> offsetArgsTensor; // count参数UB-GM，类似做allgather

    uint32_t numTargets = 0;
    uint32_t targetRanks[MAX_TARGET_NUM] = {}; // 最多768/48 = 16 轮
};

__aicore__ inline void AivAll2All91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize, int32_t tag,
    uint32_t baseFlagOffset)
{
    baseFlagOffset_ = baseFlagOffset;
    flagAddrSelf_ = buffOut0 + baseFlagOffset;

    rank_ = rank;
    rankSize_ = rankSize;

    pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE * FLAG_BUF_NUM);
    localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, 0);
    localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE);
    localClearTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE * IDX_2);
    localClearTensor.SetValue(0, 0);
    pipe.InitBuffer(bufferArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    bufferArgsTensor = bufferArgsBuf.Get<uint64_t>();
    pipe.InitBuffer(offsetArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    offsetArgsTensor = offsetArgsBuf.Get<uint64_t>();

    pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE);

    // 计算本core的numTargets和targetsList
    // 前concurrentSize/2个aiv负责与左边rank号的通信，后concurrentSize/2个负责与右边rank号的通信
    uint32_t halfConcurrent = block_num / 2; // block_num需要为偶数
    numTargets = (rankSize_ - 1) / block_num; // 除去本rank，可能需要补上一个
    uint32_t tailRankSize = (rankSize_ - 1) % block_num;
    uint32_t leftTailRankSize = 0;
    uint32_t rightTailRankSize = 0;
    if (tailRankSize > 0) {
        if (tailRankSize <= halfConcurrent) {
            leftTailRankSize = tailRankSize;
        } else {
            leftTailRankSize = halfConcurrent;
            rightTailRankSize = tailRankSize - halfConcurrent;
        }
        if (block_idx < halfConcurrent && (halfConcurrent - block_idx) <= leftTailRankSize) {
            numTargets += 1;
        }
        if (block_idx >= halfConcurrent && (block_idx - halfConcurrent + 1) <= rightTailRankSize) {
            numTargets += 1;
        }
    }

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank;
        if (block_idx < halfConcurrent) {
            targetRank = (rank_ + rankSize_ - (halfConcurrent - block_idx) - i * halfConcurrent) % rankSize_; // left
        } else {
            targetRank = (rank_ + (block_idx - halfConcurrent + 1) + i * halfConcurrent) % rankSize_; // right
        }
        targetRanks[i] = targetRank;
    }
}

template<typename T>
__aicore__ inline void AivAll2All91093Base::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, calCount);
    } else {
        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template<typename T>
__aicore__ inline void AivAll2All91093Base::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

template<typename T>
__aicore__ inline void AivAll2All91093Base::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }
    return;
}

template<HardEvent event> 
__aicore__ inline void AivAll2All91093Base::SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

__aicore__ inline void AivAll2All91093Base::BatchRecordWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag)
{
    // tx
    localSetTensor.SetValue(0, curTag);
    GlobalTensor<int32_t> globalTag;
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t i = 0; i < numTargets; i++) {
        GM_ADDR flagAddrOther = buffersOut[i] + baseFlagOffset_;
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrOther + flagOffset + rank_ * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }
    // rx and clear
    for (uint32_t i = 0; i < numTargets; i++) {
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrSelf_ + flagOffset + targetRanks[i] * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localCheckTensor, globalTag, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localCheckTensor.GetValue(0) == curTag) {
                break;
            }
        }
        DataCopy(globalTag, localClearTensor, UB_FLAG_PAD_COUNT); //清零
    }
}

#endif  /* AIV_ALLTOALL_91093_BASE_H */