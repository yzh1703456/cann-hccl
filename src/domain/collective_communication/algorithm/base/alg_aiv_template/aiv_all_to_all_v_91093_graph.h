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

class AivAll2AllVGraph91093 : public AivAll2All91093Base {
public:
    __aicore__ inline AivAll2AllVGraph91093() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR buffOut0, GM_ADDR input, GM_ADDR output, int32_t tag,
        ExtraArgsV2* extraArgs);
};

template<typename T>
__aicore__ inline void AivAll2AllVGraph91093::Process(GM_ADDR buffOut0, GM_ADDR input, GM_ADDR output, int32_t tag,
    ExtraArgsV2* extraArgs)
{
    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;

    uint64_t argsCount = FLAG_SIZE * rankSize_ / sizeof(uint64_t);
    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(buffOut0 + AIV_FLAG_BUFFER_SIZE - COMM_INFO_OFFSET);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, argsCount);
    GlobalTensor<uint64_t> offsetArgsGT;
    __gm__ uint64_t *offsetsGmAddr = (__gm__ uint64_t *)(buffOut0 + AIV_FLAG_BUFFER_SIZE - GM_TMP_ARGS_OFFSET);
    offsetArgsGT.SetGlobalBuffer(offsetsGmAddr, argsCount);

    uint32_t paramReadyFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;

    // 准备参数
    GM_ADDR buffersIn[MAX_TARGET_NUM] = {};
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};
    uint64_t recvCounts[MAX_TARGET_NUM] = {};
    uint64_t sendDispls[MAX_TARGET_NUM] = {};
    uint64_t recvDispls[MAX_TARGET_NUM] = {};
    uint64_t remoteSendDispls[MAX_TARGET_NUM] = {}; // 经过GM交换得到

    // 把buffer地址搬到ub，把偏移参数搬到GM
    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], 4); // buffersIn buffersOut

        recvCounts[i] = extraArgs->recvCounts[targetRank];
        sendDispls[i] = extraArgs->sendDispls[targetRank];
        recvDispls[i] = extraArgs->recvDispls[targetRank];
        offsetArgsTensor.SetValue(i * 4, sendDispls[i]);
    }

    PipeBarrier<PIPE_ALL>();

    for (uint32_t i = 0; i < numTargets; i++) {
        DataCopy(offsetArgsGT[targetRanks[i] * 4], offsetArgsTensor[i * 4], 4);

        uint32_t curIdx = i * 4;
        buffersIn[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx));
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    PipeBarrier<PIPE_ALL>();

    // 偏移参数拷贝到自己GM后的同步
    BatchRecordWait(buffersOut, paramReadyFlagOffset, tag);

    PipeBarrier<PIPE_ALL>();

    for (uint32_t i = 0; i < numTargets; i++) {
        GlobalTensor<uint64_t> remoteOffsetArgsGT;
        __gm__ uint64_t *remoteOffsetsGmAddr =
            (__gm__ uint64_t *)(buffersOut[i] + AIV_FLAG_BUFFER_SIZE - GM_TMP_ARGS_OFFSET);
        remoteOffsetArgsGT.SetGlobalBuffer(remoteOffsetsGmAddr, argsCount);
        DataCopy(offsetArgsTensor[i * 4], remoteOffsetArgsGT[rank_ * 4], 4); // remote sendDispls
    }

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        remoteSendDispls[i] = offsetArgsTensor.GetValue(i * 4);
    }

    SyncFunc<HardEvent::S_MTE2>();

    // 读对端userin到usrout
    for (uint32_t i = 0; i < numTargets; i++) {
        __gm__ T *inputGMOther = (__gm__ T *)(buffersIn[i]);

        uint64_t remoteSendOffset = remoteSendDispls[i];
        uint64_t localRecvOffset = recvDispls[i];
        uint64_t remoteSendCount = recvCounts[i];
        CpGM2GM(outputGM + localRecvOffset, inputGMOther + remoteSendOffset, remoteSendCount);
    }

    PipeBarrier<PIPE_ALL>();

    // read后的同步
    BatchRecordWait(buffersOut, finalAckFlagOffset, tag);

    // 最后一个核做localcopy
    if (block_idx == block_num - 1) {
        uint64_t sendOffset = extraArgs->sendDispls[rank_];
        uint64_t recvOffset = extraArgs->recvDispls[rank_];
        uint64_t sendCount = extraArgs->recvCounts[rank_];
        CpGM2GM(outputGM + recvOffset, inputGM + sendOffset, sendCount);
    }
}

template<typename T>
__aicore__ inline void aiv_all_to_all_v_91093_graph(KERNEL_ARGS_DEF, ExtraArgsV2* extraArgs)
{
    AivAll2AllVGraph91093 op;
    uint32_t baseFlagOffset = DOUBLE * rankSize * FLAG_SIZE;
    op.Init(buffOut0, rank, rankSize, tag, baseFlagOffset);
    op.Process<T>(buffOut0, input, output, tag, extraArgs);
}