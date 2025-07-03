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
#include "aiv_all_to_all_91093_base.h"

using namespace AscendC;

class AivAll2AllV91093 : public AivAll2All91093Base {
public:
    __aicore__ inline AivAll2AllV91093() {}

    __aicore__ inline void BatchRecordWaitV(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag,
        bool* needTx, bool* needRx);

    template<typename T>
    __aicore__ inline void Process(GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR input, GM_ADDR output, int32_t tag,
        uint64_t bufferSize, ExtraArgsV2* extraArgs);
};

__aicore__ inline void AivAll2AllV91093::BatchRecordWaitV(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag,
    bool* needTx, bool* needRx)
{
    // tx
    localSetTensor.SetValue(0, curTag);
    GlobalTensor<int32_t> globalTag;
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t i = 0; i < numTargets; i++) {
        if (!needTx[i]) {
            continue;
        }
        GM_ADDR flagAddrOther = buffersOut[i] + baseFlagOffset_;
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrOther + flagOffset + rank_ * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }
    // rx and clear
    for (uint32_t i = 0; i < numTargets; i++) {
        if (!needRx[i]) {
            continue;
        }
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

template<typename T>
__aicore__ inline void AivAll2AllV91093::Process(GM_ADDR buffIn0, GM_ADDR buffOut0, GM_ADDR input, GM_ADDR output,
    int32_t tag, uint64_t bufferSize, ExtraArgsV2* extraArgs)
{
    // 每张卡的CCLBuffer大小为bufferSize，平均分给ranksize块，每块的大小
    uint64_t avgBufferCount = bufferSize / rankSize_ / sizeof(T);

    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)buffIn0;

    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(buffOut0 + AIV_FLAG_BUFFER_SIZE - COMM_INFO_OFFSET);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    uint32_t cclReadyFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;

    // 准备参数，buffer地址和最大收发count
    GM_ADDR buffersIn[MAX_TARGET_NUM] = {};
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};
    uint64_t sendCounts[MAX_TARGET_NUM] = {};
    uint64_t recvCounts[MAX_TARGET_NUM] = {};
    uint64_t sendDispls[MAX_TARGET_NUM] = {};
    uint64_t recvDispls[MAX_TARGET_NUM] = {};
    uint64_t maxCount = 0;
    bool needSend[MAX_TARGET_NUM] = {0};
    bool needRead[MAX_TARGET_NUM] = {0};

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], 4); // buffersIn buffersOut
        sendCounts[i] = extraArgs->sendCounts[targetRank];
        recvCounts[i] = extraArgs->recvCounts[targetRank];
        sendDispls[i] = extraArgs->sendDispls[targetRank];
        recvDispls[i] = extraArgs->recvDispls[targetRank];

        maxCount = sendCounts[i] > maxCount ? sendCounts[i] : maxCount;
        maxCount = recvCounts[i] > maxCount ? recvCounts[i] : maxCount;
    }
    uint32_t bufferLoopNum = (maxCount + avgBufferCount - 1) / avgBufferCount;

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersIn[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx));
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    int32_t curTag = (tag << TAG_MOVE_LEFT_BITS);
    for (uint32_t loop = 0; loop < bufferLoopNum; loop++) {
        PipeBarrier<PIPE_ALL>();

        // 每次最多处理avgBufferCount
        for (uint32_t i = 0; i < numTargets; i++) {
            // 记录是否需要同步
            needSend[i] = (sendCounts[i] > 0);
            needRead[i] = (recvCounts[i] > 0);

            uint64_t localSendOffset = sendDispls[i];
            uint64_t localSendCount = sendCounts[i] > avgBufferCount ? avgBufferCount : sendCounts[i];
            uint64_t localRecvOffset = avgBufferCount * targetRanks[i];
            CpGM2GM(cclGMSelf + localRecvOffset, inputGM + localSendOffset, localSendCount);
            sendDispls[i] += localSendCount;
            sendCounts[i] -= localSendCount;
        }

        PipeBarrier<PIPE_ALL>();

        // localcopy后的同步
        BatchRecordWaitV(buffersOut, cclReadyFlagOffset, curTag, needSend, needRead);

        PipeBarrier<PIPE_ALL>();

        // 读对端ccl到usrout
        for (uint32_t i = 0; i < numTargets; i++) {
            __gm__ T *cclGMOther = (__gm__ T *)(buffersIn[i]);

            uint64_t remoteSendOffset = avgBufferCount * rank_;
            uint64_t localRecvOffset = recvDispls[i];
            uint64_t remoteSendCount = recvCounts[i] > avgBufferCount ? avgBufferCount : recvCounts[i];
            CpGM2GM(outputGM + localRecvOffset, cclGMOther + remoteSendOffset, remoteSendCount);
            recvDispls[i] += remoteSendCount;
            recvCounts[i] -= remoteSendCount;
        }

        PipeBarrier<PIPE_ALL>();

        // read后的同步
        BatchRecordWaitV(buffersOut, finalAckFlagOffset, curTag, needRead, needSend);

        curTag += 1;
    }

    // 最后一个核做localcopy
    if (block_idx == block_num - 1) {
        uint64_t sendCount = extraArgs->sendCounts[rank_];
        uint64_t sendOffset = extraArgs->sendDispls[rank_];
        uint64_t recvOffset = extraArgs->recvDispls[rank_];
        CpGM2GM(outputGM + recvOffset, inputGM + sendOffset, sendCount);
    }
}

template<typename T>
__aicore__ inline void aiv_all_to_all_v_91093(KERNEL_ARGS_DEF, ExtraArgsV2* extraArgs)
{
    AivAll2AllV91093 op;
    uint32_t baseFlagOffset = 0;
    op.Init(buffOut0, rank, rankSize, tag, baseFlagOffset);
    op.Process<T>(buffIn0, buffOut0, input, output, tag, bufferSize, extraArgs);
}