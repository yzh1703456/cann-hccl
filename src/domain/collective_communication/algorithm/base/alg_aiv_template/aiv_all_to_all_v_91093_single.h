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

class AivAll2AllV91093Single : public AivCommBase {
public:
    __aicore__ inline AivAll2AllV91093Single() {}
 
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferSize,
        ExtraArgs* extraArgs);

    __aicore__ inline void CalBlockCountAndOffset(uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup,
        uint32_t padCount, uint64_t &count, uint64_t &blockOffset);
};

__aicore__ inline void AivAll2AllV91093Single::CalBlockCountAndOffset(uint64_t len, uint32_t blockNumPerGroup,
uint32_t blockIdxInGroup, uint32_t padCount, uint64_t &count, uint64_t &blockOffset)
{
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    blockOffset = blockIdxInGroup * avgLengthPerSlice;
}

template<typename T>
__aicore__ inline void AivAll2AllV91093Single::Process(GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t bufferSize,
    ExtraArgs* extraArgs)
{
    // 每张卡的CCLBuffer大小为bufferSize，平均分给ranksize块，每块的大小
    uint64_t avgBufferCount = bufferSize / block_num / sizeof(T); // block_num需要能被rankSize_整除

    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
 
    uint32_t blockNumPerGroup = block_num / rankSize_; 
    uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[dstRank]);

    // 使用96个flag
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_V_91093_SINGLE;
    GM_ADDR flagAddrSelf = GM_OUT[rank_] + baseFlagOffset;
    GM_ADDR flagAddrOther = GM_OUT[dstRank] + baseFlagOffset;
    uint32_t initAckFlagOffset = 0;
    uint32_t finalAckFlagOffset = block_num * FLAG_SIZE;

    uint32_t flagSetOffset = rank_ * blockNumPerGroup * FLAG_SIZE + blockIdxInGroup * FLAG_SIZE;
    uint32_t flagCheckOffset = block_idx * FLAG_SIZE; // dstRank * blockNumPerGroup * FLAG_SIZE

    uint64_t sendCount = extraArgs->sendCounts[dstRank];
    uint64_t recvCount = extraArgs->recvCounts[dstRank];
    uint64_t sendDispl = extraArgs->sendDispls[dstRank];
    uint64_t recvDispl = extraArgs->recvDispls[dstRank];

    uint64_t blockSendCount = 0;
    uint64_t blockSendOffset = 0;
    uint64_t blockRecvCount = 0;
    uint64_t blockRecvOffset = 0;
    CalBlockCountAndOffset(sendCount, blockNumPerGroup, blockIdxInGroup, padCount, blockSendCount, blockSendOffset);
    CalBlockCountAndOffset(recvCount, blockNumPerGroup, blockIdxInGroup, padCount, blockRecvCount, blockRecvOffset);

    if (dstRank == rank_) {
        CpGM2GM(outputGM + recvDispl + blockRecvOffset, inputGM + sendDispl + blockSendOffset, blockSendCount);
        return;
    }

    uint64_t maxCount = blockSendCount > blockRecvCount ? blockSendCount : blockRecvCount;
    uint32_t bufferLoopNum = (maxCount + avgBufferCount - 1) / avgBufferCount;

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        bool needSend = (blockSendCount > 0);
        bool needRead = (blockRecvCount > 0);

        PipeBarrier<PIPE_ALL>();

        // 每次最多处理avgBufferCount
        uint64_t localSendCount = blockSendCount > avgBufferCount ? avgBufferCount : blockSendCount;
        uint64_t localRecvOffset = avgBufferCount * (dstRank * blockNumPerGroup + blockIdxInGroup);
        CpGM2GM(cclGMSelf + localRecvOffset, inputGM + sendDispl + blockSendOffset, localSendCount);
        blockSendOffset += localSendCount;
        blockSendCount -= localSendCount;

        PipeBarrier<PIPE_ALL>();

        // localcopy后的同步
        if (needSend) {
            SetFlagNew((__gm__ int32_t *)(flagAddrOther + initAckFlagOffset + flagSetOffset), curTag);
        }
        if (needRead) {
            CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + initAckFlagOffset + flagCheckOffset), curTag);
        }

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        uint64_t remoteSendOffset = avgBufferCount * (rank_ * blockNumPerGroup + blockIdxInGroup);
        uint64_t remoteSendCount = blockRecvCount > avgBufferCount ? avgBufferCount : blockRecvCount;
        CpGM2GM(outputGM + recvDispl + blockRecvOffset, cclGMOther + remoteSendOffset, remoteSendCount);
        blockRecvOffset += remoteSendCount;
        blockRecvCount -= remoteSendCount;

        PipeBarrier<PIPE_ALL>();

        // read后的同步
        if (needRead) {
            SetFlagNew((__gm__ int32_t *)(flagAddrOther + finalAckFlagOffset + flagSetOffset), curTag);
        }
        if (needSend) {
            CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + flagCheckOffset), curTag);
        }

        curTag += 1;
    }
}
 
template<typename T>
__aicore__ inline void aiv_all_to_all_v_91093_single(KERNEL_ARGS_DEF, ExtraArgs* extraArgs)
{
    AivAll2AllV91093Single op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, tag, bufferSize, extraArgs);
}