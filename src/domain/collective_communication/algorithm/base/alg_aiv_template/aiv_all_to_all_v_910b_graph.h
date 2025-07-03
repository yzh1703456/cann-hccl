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

class AivAll2AllVGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllVGraph910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs &extraArgs);
};

template<typename T>
__aicore__ inline void AivAll2AllVGraph910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag,
    ExtraArgs &extraArgs)
{
    uint32_t targetRank = block_idx; // 0-rankSize

    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[targetRank]);

    // 使用32个flag
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_V_910B_GRAPH;
    
    GM_ADDR flagAddrSelf = GM_OUT[rank_] + baseFlagOffset;
    GM_ADDR flagAddrOther = GM_OUT[targetRank] + baseFlagOffset;

    // 共使用4组flag
    uint32_t initAckFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;
    uint32_t paramDataFlagOffset = 2 * rankSize_ * FLAG_SIZE;
    uint32_t paramAckFlagOffset = 3 * rankSize_ * FLAG_SIZE;

    // 每张卡的rank号aiv先把sendcount和sendoffset搬运到自己的gm中
    if (targetRank == rank_) {
        LocalTensor<uint64_t> localOut = flagBatchSetQue.AllocTensor<uint64_t>();
        
        for (uint32_t i = 0; i < rankSize_; i++) {
            localOut.SetValue(i, extraArgs.sendCounts[i]);
            localOut.SetValue(i + rankSize_, extraArgs.sendDispls[i]);
        }

        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint64_t> outputGT;
        outputGT.SetGlobalBuffer((__gm__ uint64_t *)(flagAddrSelf + paramDataFlagOffset), 2 * rankSize_);
        DataCopyUB2GM(outputGT, localOut, 2 * rankSize_);
        flagBatchSetQue.FreeTensor(localOut);

        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < rankSize_; i++) {
            if (i == rank_) {
                continue;
            }
            GM_ADDR flagAddrX = GM_OUT[i] + baseFlagOffset;
            SetFlagNew((__gm__ int32_t *)(flagAddrX + paramAckFlagOffset + rank_ * FLAG_SIZE), tag);
        }
    }

    // 本卡已进入算子，通知其他卡可以搬运
    SetFlagNew((__gm__ int32_t *)(flagAddrSelf + initAckFlagOffset + targetRank * FLAG_SIZE), tag);
    PipeBarrier<PIPE_ALL>();
    // 确认对端已进入算子
    CheckFlagNew((__gm__ int32_t *)(flagAddrOther + initAckFlagOffset + rank_ * FLAG_SIZE), tag);
    PipeBarrier<PIPE_ALL>();
    
    uint64_t remoteSendOffset = 0; // 远端usrin发送给本端output的数据偏移，远端卡号为block_idx，可能为本rank
    uint64_t remoteSendCount = 0; // 远端ccl发送给本端output的数据量，远端可能为本rank
    if (targetRank != rank_) {
        // 确认对端targetRank号aiv已把sendcount数据搬运到GM
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + paramAckFlagOffset + targetRank * FLAG_SIZE), tag);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<uint64_t> inputGT;
        LocalTensor<uint64_t> localIn = flagBatchCheckQue.AllocTensor<uint64_t>();
        inputGT.SetGlobalBuffer((__gm__ uint64_t *)(flagAddrOther + paramDataFlagOffset), 2 * rankSize_);
        DataCopyGM2UB(localIn, inputGT, 2 * rankSize_);
        PipeBarrier<PIPE_ALL>();
        remoteSendCount = localIn.GetValue(rank_);
        remoteSendOffset = localIn.GetValue(rank_ + rankSize_);
        flagBatchCheckQue.FreeTensor(localIn);
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf + paramAckFlagOffset + targetRank * FLAG_SIZE), 0);
    } else { // targetRank == rank_不用check
        remoteSendCount = extraArgs.sendCounts[rank_];
        remoteSendOffset = extraArgs.sendDispls[rank_];
    }

    PipeBarrier<PIPE_ALL>();

    // 本端output接收远端usrin的数据偏移，目标远端卡号为block_idx，可能为本rank
    uint64_t localRecvOffset = extraArgs.recvDispls[targetRank];

    CpGM2GM(outputGM + localRecvOffset, cclGMOther + remoteSendOffset, remoteSendCount);
    PipeBarrier<PIPE_ALL>();

    // 通知对端，自己已经把对端的那片数据拉回来了
    SetFlagNew((__gm__ int32_t *)(flagAddrOther + finalAckFlagOffset + rank_ * FLAG_SIZE), tag);
    
    // 确认对端已经将对应的数据拉走
    CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + targetRank * FLAG_SIZE), tag);
    PipeBarrier<PIPE_ALL>();

    // 图模式最后清零flag
    SetFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + targetRank * FLAG_SIZE), 0);
    SetFlagNew((__gm__ int32_t *)(flagAddrSelf + initAckFlagOffset + targetRank * FLAG_SIZE), 0);
    return;
}

template<typename T>
__aicore__ inline void aiv_all_to_all_v_910b_graph(EXTERN_KERNEL_ARGS_DEF)
{
    AivAll2AllVGraph910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, tag, extraArgs);
}