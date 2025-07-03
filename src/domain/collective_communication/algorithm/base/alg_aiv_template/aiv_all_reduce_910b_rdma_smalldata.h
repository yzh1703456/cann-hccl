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
class AivAllReduceRdmaSmall910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceRdmaSmall910B() {}
 
    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag, int32_t aivRdmaStep);
 
    template <typename T>
    __aicore__ inline void ReduceScatter(__gm__ T *inputGM, __gm__ T *cclGMSelf,
    __gm__ T *cclGMOther, __gm__ T *outputGM, uint64_t sliceCount, uint64_t avgLengthPerRank, uint64_t tailLength,
    int32_t tag);
    
    template <typename T>
    __aicore__ inline void AllReduce(__gm__ T *inputGM, __gm__ T *cclGMSelf,
        __gm__ T *outputGM, int64_t len, int32_t tag);
    
    template<typename T>
    __aicore__ inline void AllGather(__gm__ T *inputGM, __gm__ T *cclGMSelf,
    __gm__ T *cclGMOther, __gm__ T *outputGM, uint64_t sliceCount, uint64_t avgLengthPerRank, uint64_t tailLength,
    int32_t tag);
};
 
template <typename T>
__aicore__ inline void AivAllReduceRdmaSmall910B::ReduceScatter(__gm__ T *inputGM, __gm__ T *cclGMSelf,
    __gm__ T *cclGMOther, __gm__ T *outputGM, uint64_t sliceCount, uint64_t avgLengthPerRank, uint64_t tailLength,
    int32_t tag)
{
    // reduce scatter，数据从input输入，inputMem+0作为buffer，结果放在inputMem+2M处
    uint32_t flagBaseOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_STEP1;
    uint32_t flagOffsetOut = flagBaseOffset + block_idx * FLAG_SIZE; // 给其他卡的标记，数据已在buffer中就绪
    uint32_t flagOffsetRemote = flagBaseOffset + rank_ * FLAG_SIZE;  // 本卡aiv需要读的其他卡的标记
    uint32_t flagOffsetIn = flagBaseOffset + block_num * FLAG_SIZE;  // 给本卡其他aiv的标记，数据已在output中，可以累加
 
    if (block_idx == rank_) {
        int64_t curCount = CalActualCount(rank_, rankSize_, avgLengthPerRank, tailLength);
 
        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, curCount);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf, curCount);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, curCount);
 
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[avgLengthPerRank * block_idx], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf[avgLengthPerRank * block_idx], localOut, curCount);
 
        // 卡间同步
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag); // 本卡该片数据已经可以被跨片读取
        DataCopyUB2GM(outputGT, localOut, curCount);
        inOutQue.FreeTensor(localOut);
 
        // 卡内同步
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), tag); // 本卡目的分片已经在output中
    } else {
        int64_t curCountBlk = CalActualCount(block_idx, rankSize_, avgLengthPerRank, tailLength);
 
        GlobalTensor<T> cclGTOther;
        cclGTOther.SetGlobalBuffer(cclGMOther, curCountBlk);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, curCountBlk);
 
        // 从input搬运到buffer
        CpGM2GM(cclGMSelf + avgLengthPerRank * block_idx, inputGM + avgLengthPerRank * block_idx, curCountBlk);
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag); // 本卡该片数据已经可以被跨片读取
 
        // 对端数据就绪后先搬到自己的UB，注意这里搬运的长度应当由rank_决定，而不是block_idx决定
        int64_t curCount = CalActualCount(rank_, rankSize_, avgLengthPerRank, tailLength);
        
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetRemote), tag);
        pipe_barrier(PIPE_ALL);
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, cclGTOther[avgLengthPerRank * rank_], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
 
        // 本端数据在output就绪后从UB中搬入
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetIn), tag);
        pipe_barrier(PIPE_ALL);
        SetAtomicOp<T>(reduceOp_);
        DataCopyUB2GM(outputGT, localOut, curCount);
        SetAtomicNone();
        inOutQue.FreeTensor(localOut);
    }
    return;
}
 
template <typename T>
__aicore__ inline void AivAllReduceRdmaSmall910B::AllReduce(__gm__ T *inputGM, __gm__ T *cclGMSelf,
    __gm__ T *outputGM, int64_t len, int32_t tag)
{
    // all reduce，仅适用于A + X单机跨aggregation场景，数据在buffer中，先本端数据拷贝到output中，再从对端拷贝到output中
    uint32_t flagBaseOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_STEP2;
    uint32_t flagOffsetStart = flagBaseOffset;           //  起始同步
    uint32_t flagOffsetEnd = flagBaseOffset + FLAG_SIZE; // 末尾同步
    uint32_t peerRank = 1 - rank_;
    int64_t count = len;
 
    __gm__ T *cclGMPeer = (__gm__ T *)(GM_IN[peerRank]);
 
    if (block_idx == 0) {
        // 本端数据已就绪
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetStart), tag);
        CpGM2GM(outputGM, cclGMSelf, count);
 
        // 起始同步，检查对端数据是否就绪
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[peerRank] + flagOffsetStart), tag);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGM, cclGMPeer, count, true, reduceOp_);
 
        // 末尾同步
        pipe_barrier(PIPE_ALL);
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetEnd), tag);
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[peerRank] + flagOffsetEnd), tag);
    }
    return;
}
 
template<typename T>
__aicore__ inline void AivAllReduceRdmaSmall910B::AllGather(__gm__ T *inputGM, __gm__ T *cclGMSelf,
    __gm__ T *cclGMOther, __gm__ T *outputGM, uint64_t sliceCount, uint64_t avgLengthPerRank, uint64_t tailLength,
    int32_t tag)
{
    // all gather, 数据从input输入（rdma结果位置），inputMem+8M作为buffer，结果放在output中
    uint32_t flagBaseOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_STEP3;
    uint32_t flagOffsetRemote = flagBaseOffset + block_idx * FLAG_SIZE;           // 本卡aiv需要读的其他卡的标记
    uint32_t flagOffsetOut = flagBaseOffset + rank_ * FLAG_SIZE;          // 注意与RS不同，设置本卡的数据已经在buffer中就绪
 
    if (block_idx == rank_) {
        int64_t curCount = CalActualCount(rank_, rankSize_, avgLengthPerRank, tailLength);
 
        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, curCount);
        GlobalTensor<T> cclGTSelf;
        cclGTSelf.SetGlobalBuffer(cclGMSelf, curCount);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, curCount);
 
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT, curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(cclGTSelf[avgLengthPerRank * block_idx], localOut, curCount);
        pipe_barrier(PIPE_ALL);
 
        // 卡间同步
        SetFlagNew((__gm__ int32_t *)(GM_OUT[rank_] + flagOffsetOut), tag); // 本卡该片数据已经可以被跨片读取
        DataCopyUB2GM(outputGT[avgLengthPerRank * block_idx], localOut, curCount);
        inOutQue.FreeTensor(localOut);
    } else {
        int64_t curCount = CalActualCount(block_idx, rankSize_, avgLengthPerRank, tailLength);
 
        CheckFlagNew((__gm__ int32_t *)(GM_OUT[block_idx] + flagOffsetRemote), tag);
        pipe_barrier(PIPE_ALL);
        CpGM2GM(outputGM + (block_idx * avgLengthPerRank), cclGMOther + block_idx * avgLengthPerRank, curCount);
    }
 
    return;
}
 
template<typename T>
__aicore__ inline void AivAllReduceRdmaSmall910B::Process(GM_ADDR input, GM_ADDR output,
    uint64_t len, int32_t tag, int32_t aivRdmaStep)
{
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[block_idx]);
    __gm__ T *outputGM = (__gm__ T *)output;
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t sliceCount = CeilDiv(len, avgLengthPerRank);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerRank;
 
    if (aivRdmaStep == 0) {
        ReduceScatter(inputGM, cclGMSelf, cclGMOther, outputGM, sliceCount, avgLengthPerRank, tailLength, tag);
    }
    if (aivRdmaStep == 1) {
        AllReduce(inputGM, cclGMSelf, outputGM, len, tag);
    }
    if (aivRdmaStep == 2) {
        AllGather(inputGM, cclGMSelf, cclGMOther, outputGM, sliceCount, avgLengthPerRank, tailLength, tag);
    }
    return;
}
 
template<typename T>
__aicore__ inline void aiv_all_reduce_910b_rdma_smalldata(KERNEL_ARGS_DEF)
{
    AivAllReduceRdmaSmall910B op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.Process<T>(input, output, len, tag, aivRdmaStep);
}