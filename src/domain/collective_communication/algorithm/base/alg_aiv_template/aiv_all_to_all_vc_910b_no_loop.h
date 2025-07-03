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

class AivAll2AllVCNoLoop910B : public AivCommBase {
public:
    __aicore__ inline AivAll2AllVCNoLoop910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, ExtraArgs &extraArgs);
};

template<typename T>
__aicore__ inline void AivAll2AllVCNoLoop910B::Process(GM_ADDR input, GM_ADDR output, int32_t tag,
    ExtraArgs &extraArgs)
{
    uint32_t targetRank = (block_idx >= rankSize_ ? block_idx - rankSize_ : block_idx); // 0-2*rankSize

    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[targetRank]);

    // 使用24个flag
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_VC_910B_NO_LOOP;
    
    GM_ADDR flagAddrSelf = GM_OUT[rank_] + baseFlagOffset;
    GM_ADDR flagAddrOther = GM_OUT[targetRank] + baseFlagOffset;

    // 共使用3组flag
    uint32_t pipelineCtrlFlagOffset = 0;
    uint32_t finalAckFlagOffset = rankSize_ * FLAG_SIZE;
    uint32_t countResetFlagOffset = 2 * rankSize_ * FLAG_SIZE;
    uint32_t initAckFlagOffset = 3 * rankSize_ * FLAG_SIZE;

    if (block_idx < rankSize_) { // 前rankSize个aiv负责userin->cclin
        uint64_t localSendOffset = 0;
        for (uint32_t i = 0; i < block_idx; i++) {
            localSendOffset += extraArgs.sendCountMatrix[rank_ * rankSize_ + i];
        }
        uint64_t localSendCount = extraArgs.sendCountMatrix[rank_ * rankSize_ + block_idx];

        __gm__ int32_t* ctrlFlagsGM = (__gm__ int32_t *)(flagAddrSelf + pipelineCtrlFlagOffset + block_idx * FLAG_SIZE);
        CpGM2GMWithFlagWrap(cclGMSelf + localSendOffset, inputGM + localSendOffset, localSendCount, ctrlFlagsGM, 16);

        PipeBarrier<PIPE_ALL>();
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + countResetFlagOffset + block_idx * FLAG_SIZE), tag);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew(ctrlFlagsGM, 0);

    } else { // 后rankSize个aiv负责cclother->usrout
        // 读对端数据前确认对端已进入本算子
        SetFlagNew((__gm__ int32_t *)(flagAddrOther + initAckFlagOffset + rank_ * FLAG_SIZE), tag);
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + initAckFlagOffset + targetRank * FLAG_SIZE), tag);
        PipeBarrier<PIPE_ALL>();

        uint64_t remoteSendOffset = 0; // 远端ccl发送给本端output的数据偏移，远端卡号为block_idx，可能为本rank
        for (uint32_t i = 0; i < rank_; i++) {
            remoteSendOffset += extraArgs.sendCountMatrix[targetRank * rankSize_ + i];
        }

        uint64_t localRecvOffset = 0; // 本端output接收远端ccl的数据偏移，目标远端卡号为block_idx，可能为本rank
        for (uint32_t i = 0; i < targetRank; i++) {
            localRecvOffset += extraArgs.sendCountMatrix[i * rankSize_ + rank_];
        }

        // 远端ccl发送给本端output的数据量，远端可能为本rank
        uint64_t remoteSendCount = extraArgs.sendCountMatrix[targetRank * rankSize_ + rank_];
        uint64_t remoteSendSize = remoteSendCount * sizeof(T);

        __gm__ int32_t *ctrlFlagsGMX = (__gm__ int32_t *)(flagAddrOther + pipelineCtrlFlagOffset + rank_ * FLAG_SIZE);

        uint64_t processedBatchCount = 0;

        while (true) {
            if (processedBatchCount >= CeilDiv(remoteSendSize, UB_DB_DATA_BATCH_SIZE)) {
                break;
            }

            GlobalTensor<int32_t> globalFlagX;
            globalFlagX.SetGlobalBuffer(ctrlFlagsGMX, UB_FLAG_PAD_COUNT);
            LocalTensor<int32_t> localFlagX = flagInQue.AllocTensor<int32_t>();

            DataCopy(localFlagX, globalFlagX, UB_FLAG_PAD_COUNT);

            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

            uint64_t preparedBatchCount = localFlagX.GetValue(0);

            flagInQue.FreeTensor(localFlagX);

            if (preparedBatchCount == 0 || processedBatchCount >= preparedBatchCount) {
                continue;
            }

            uint64_t curSize = (preparedBatchCount - processedBatchCount) * UB_DB_DATA_BATCH_SIZE;
            if (preparedBatchCount * UB_DB_DATA_BATCH_SIZE > remoteSendSize) {
                curSize = remoteSendSize - processedBatchCount * UB_DB_DATA_BATCH_SIZE;
            }

            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

            uint64_t curProcessedOffset = processedBatchCount * UB_DB_DATA_BATCH_SIZE / sizeof(T);
            CpGM2GM(outputGM + localRecvOffset + curProcessedOffset, cclGMOther + remoteSendOffset + curProcessedOffset,
                curSize / sizeof(T));

            processedBatchCount = preparedBatchCount;
        }

        // 通知对端，自己已经把对端的那片数据拉回来了
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(flagAddrOther + finalAckFlagOffset + rank_ * FLAG_SIZE), tag);
        PipeBarrier<PIPE_ALL>();
        
        // 确认对端已经将对应的数据拉走
        CheckFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + targetRank * FLAG_SIZE), tag);
        PipeBarrier<PIPE_ALL>();
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + targetRank * FLAG_SIZE), 0);
        PipeBarrier<PIPE_ALL>();

        // 用于清零count flag
        SetFlagNew((__gm__ int32_t *)(flagAddrSelf + countResetFlagOffset + targetRank * FLAG_SIZE), tag);
    }
}

template<typename T>
__aicore__ inline void aiv_all_to_all_vc_910b_no_loop(EXTERN_KERNEL_ARGS_DEF)
{
    AivAll2AllVCNoLoop910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.Process<T>(input, output, tag, extraArgs);
}