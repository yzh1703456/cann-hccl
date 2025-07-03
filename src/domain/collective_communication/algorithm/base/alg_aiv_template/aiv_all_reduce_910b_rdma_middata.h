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
 
class AivAllReduceRdmaMid910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceRdmaMid910B() {}
 
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t aivRdmaStep);
 
    template<typename T>
    __aicore__ inline void ReduceScatter(__gm__ T *inputGM, __gm__ T *cclGMSelf,
        __gm__ T *cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength, int32_t tag);
    
    template<typename T>
    __aicore__ inline void AllGather(__gm__ T *outputGM, __gm__ T *cclGMSelf,
        __gm__ T *cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength, int32_t tag);
};
 
template<typename T>
__aicore__ inline void AivAllReduceRdmaMid910B::Process(GM_ADDR input, GM_ADDR output,
    uint64_t len, int32_t tag, int32_t aivRdmaStep)
{
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx]);
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;
 
    if (aivRdmaStep == 0) {
        ReduceScatter(inputGM, cclGMSelf, cclGMOther, sliceCount, avgLengthPerSlice, tailLength, tag);
    }
    if (aivRdmaStep == 2) {
        AllGather(outputGM, cclGMSelf, cclGMOther, sliceCount, avgLengthPerSlice, tailLength, tag);
    }
    return;
}
 
template<typename T>
__aicore__ inline void AivAllReduceRdmaMid910B::ReduceScatter(__gm__ T *inputGM, __gm__ T *cclGMSelf,
    __gm__ T *cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength, int32_t tag)
{
    // reduce scatter，数据从input输入，inputMem+0作为buffer，结果放在原位
    uint32_t flagBaseOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_RDMA_MIDDATA_STEP1;
    uint32_t flagOffsetOut = flagBaseOffset + block_idx * FLAG_INTERVAL;  // 给其他卡的标记，数据已在buffer中就绪
    uint32_t flagOffsetRemote = flagBaseOffset + rank_ * FLAG_INTERVAL;  // 本卡aiv需要读的其他卡的标记
    uint32_t flagOffsetIn = flagBaseOffset + rank_ * FLAG_INTERVAL;  // 给本卡其他aiv的标记，数据已在output中，可以累加
    if (block_idx == rank_) {
        int64_t curCount = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);
        
        // 本地拷贝 & 卡间同步
        CpGM2GM(cclGMSelf + avgLengthPerSlice * block_idx, inputGM + avgLengthPerSlice * block_idx, curCount);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag);  // 本卡该片数据已可以被跨片读取（也可累加）
    } else {
        int64_t curCount = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);
 
        // 本地拷贝 & 卡间同步
        CpGM2GM(cclGMSelf + avgLengthPerSlice * block_idx, inputGM + avgLengthPerSlice * block_idx, curCount);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag);  // 本卡该片数据已经可以被跨片读取
        
        // 检查对端数据就绪且本端就绪 & 跨片搬运
        curCount = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);
 
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetRemote), tag);
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), tag);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(cclGMSelf + avgLengthPerSlice * rank_, cclGMOther + avgLengthPerSlice * rank_, curCount,
            true, reduceOp_);
    }
    return;
}
 
template<typename T>
__aicore__ inline void AivAllReduceRdmaMid910B::AllGather(__gm__ T *outputGM, __gm__ T *cclGMSelf,
    __gm__ T *cclGMOther, uint64_t sliceCount, uint64_t avgLengthPerSlice, uint64_t tailLength, int32_t tag)
{
    // all gather, outputMem作为buffer且数据已经在对应位置，拷贝到output中
    uint32_t flagBaseOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_RDMA_MIDDATA_STEP2;
    uint32_t flagOffsetOut = flagBaseOffset + rank_ * FLAG_INTERVAL;  // 注意与RS不同，设置本卡的数据已经在buffer中就绪
    uint32_t flagOffsetRemote = flagBaseOffset + block_idx * FLAG_INTERVAL;  // 本卡aiv需要读的其他卡的标记
 
    if (block_idx == rank_) {
        int64_t curCount = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);
 
        // 本地拷贝 & 卡间同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag);  // 本卡该片数据已经可以被跨片读取
        CpGM2GM(outputGM + avgLengthPerSlice * block_idx, cclGMSelf + avgLengthPerSlice * block_idx, curCount);
    } else {
        int64_t curCount = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);
 
        // 检查对端就绪 & 跨片拷贝
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetRemote), tag);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGM + (block_idx * avgLengthPerSlice), cclGMOther + block_idx * avgLengthPerSlice, curCount);
        pipe_barrier(PIPE_ALL);
        
        // 末尾同步
        // 本卡已读完block_idx号对端上的rank号数据
        SetFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetOut + FLAG_SIZE), tag);
        pipe_barrier(PIPE_ALL);
        // 检查本卡上是否有block_idx号对端的读完标记
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetRemote + FLAG_SIZE), tag);
    }
    return;
}
 
template<typename T>
__aicore__ inline void aiv_all_reduce_910b_rdma_middata(KERNEL_ARGS_DEF)
{
    AivAllReduceRdmaMid910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag, aivRdmaStep);
}