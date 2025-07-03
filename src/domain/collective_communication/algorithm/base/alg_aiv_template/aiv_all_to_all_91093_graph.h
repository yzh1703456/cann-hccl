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

class AivAll2AllGraph91093 : public AivAll2All91093Base {
public:
    __aicore__ inline AivAll2AllGraph91093() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR buffOut0, GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t len);
};

template<typename T>
__aicore__ inline void AivAll2AllGraph91093::Process(GM_ADDR buffOut0, GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t len)
{
    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;

    uint64_t argsCount = FLAG_SIZE * rankSize_ / sizeof(uint64_t);
    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(buffOut0 + AIV_FLAG_BUFFER_SIZE - COMM_INFO_OFFSET);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, argsCount);

    uint32_t initAckFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;

    // 准备参数
    GM_ADDR buffersIn[MAX_TARGET_NUM] = {};
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};

    // 把buffer地址搬到ub，把偏移参数搬到GM
    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], 4); // buffersIn buffersOut
    }

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersIn[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx));
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    PipeBarrier<PIPE_ALL>();

    // 首同步
    BatchRecordWait(buffersOut, initAckFlagOffset, tag);

    PipeBarrier<PIPE_ALL>();

    // 读对端userin到usrout
    for (uint32_t i = 0; i < numTargets; i++) {
        __gm__ T *inputGMOther = (__gm__ T *)(buffersIn[i]);
        CpGM2GM(outputGM + targetRanks[i] * len, inputGMOther + rank_ * len, len);
    }

    PipeBarrier<PIPE_ALL>();

    // read后的同步
    BatchRecordWait(buffersOut, finalAckFlagOffset, tag);

    // 最后一个核做localcopy
    if (block_idx == block_num - 1) {
        CpGM2GM(outputGM + rank_ * len, inputGM + rank_ * len, len);
    }
}

template<typename T>
__aicore__ inline void aiv_all_to_all_91093_graph(KERNEL_ARGS_DEF)
{
    AivAll2AllGraph91093 op;
    uint32_t baseFlagOffset = 6 * rankSize * FLAG_SIZE;
    op.Init(buffOut0, rank, rankSize, tag, baseFlagOffset);
    op.Process<T>(buffOut0, input, output, tag, len);
}