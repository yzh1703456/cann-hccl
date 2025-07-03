/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COMMUNICATION_BASE_H
#define AIV_COMMUNICATION_BASE_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t MAX_RANK_SIZE = 16; // server内最大卡数
constexpr uint32_t MAX_RANK_SIZE_A3 = 768; // 超节点内最大卡数
constexpr uint32_t MAX_TARGET_NUM = 20; // 最大轮数

struct ExtraArgs {
    uint64_t sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    uint64_t sendCounts[MAX_RANK_SIZE] = {};
    uint64_t sendDispls[MAX_RANK_SIZE] = {};
    uint64_t recvCounts[MAX_RANK_SIZE] = {};
    uint64_t recvDispls[MAX_RANK_SIZE] = {};
    uint64_t maxCount = 0;
};

struct ExtraArgsV2 {
    uint64_t sendCounts[MAX_RANK_SIZE_A3] = {};
    uint64_t sendDispls[MAX_RANK_SIZE_A3] = {};
    uint64_t recvCounts[MAX_RANK_SIZE_A3] = {};
    uint64_t recvDispls[MAX_RANK_SIZE_A3] = {};
};

#define KERNEL_ARGS_DEF \
GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, \
GM_ADDR buffIn4, GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, \
GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11, \
GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, \
GM_ADDR buffOut0, GM_ADDR buffOut1, GM_ADDR buffOut2, GM_ADDR buffOut3, \
GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7, \
GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, \
GM_ADDR buffOut12, GM_ADDR buffOut13, GM_ADDR buffOut14, GM_ADDR buffOut15, \
GM_ADDR input, GM_ADDR output, uint32_t rank, uint32_t rankSize, uint64_t len, \
uint32_t dataType, uint32_t reduceOp, uint32_t root, int32_t tag, bool isOpBase, uint64_t bufferSize, \
int32_t aivRdmaStep, bool useAivRdmaSmall, int32_t serverNum, uint32_t devType

#define KERNEL_ARGS_CALL \
buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, \
buffIn8, buffIn9, buffIn10, buffIn11, buffIn12, buffIn13, buffIn14, buffIn15, \
buffOut0, buffOut1, buffOut2, buffOut3, buffOut4, buffOut5, buffOut6, buffOut7, \
buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14, buffOut15, \
input, output, rank, rankSize, len, dataType, reduceOp, root, tag, isOpBase, bufferSize, aivRdmaStep, useAivRdmaSmall, \
serverNum, devType

#define KERNEL_CLASS_INIT \
buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, \
buffIn8, buffIn9, buffIn10, buffIn11, buffIn12, buffIn13, buffIn14, buffIn15, \
buffOut0, buffOut1, buffOut2, buffOut3, buffOut4, buffOut5, buffOut6, buffOut7, \
buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14, buffOut15, \
rank, rankSize, dataType, reduceOp, root

#define EXTERN_KERNEL_ARGS_DEF \
KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define EXTERN_KERNEL_ARGS_DEF_V2 \
KERNEL_ARGS_DEF, ExtraArgsV2 extraArgs

#define EXTERN_KERNEL_ARGS_CALL \
KERNEL_ARGS_CALL, extraArgs

constexpr uint64_t AIV_FLAG_BUFFER_SIZE = 3 * 1024 * 1024; // aiv算子的flag区域大小
constexpr uint64_t COMM_INFO_OFFSET = 32 * 1024; // 通信域内所有对端共享内存地址的信息距离aiv buffer末尾的偏移
constexpr uint64_t GM_TMP_ARGS_OFFSET = 64 * 1024;

constexpr uint64_t AIV_ALL_REDUCE_BIG_SIZE = 16 * 1024 * 1024;
constexpr uint64_t AIV_ALL_REDUCE_SMALL_SIZE = 64 * 1024;
constexpr uint64_t AIV_INIT_OFFSET = 0;
constexpr uint64_t AIV_PING_PONG_SIZE = 16 * 1024 * 1024;
constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;

constexpr uint64_t AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_MID_SIZE = 2 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_V_MID_SIZE = 2 * 1024 * 1024;
constexpr uint64_t AIV_ALL_TO_ALL_BIG_SIZE = 512 * 1024;

constexpr uint64_t AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint32_t BLOCK_DIM_THREE_PER_RANK_A3 = 3;
constexpr uint32_t BLOCK_DIM_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_BLOCK_DIM = 48;

constexpr uint32_t TAG_MOVE_LEFT_BITS = 12;

constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_FLAG_PAD_COUNT = 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;

constexpr uint64_t FLAG_SIZE = 32;
constexpr uint64_t FLAG_INTERVAL = FLAG_SIZE * 2;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;

constexpr uint64_t IDX_0 = 0;
constexpr uint64_t IDX_1 = 1;
constexpr uint64_t IDX_2 = 2;
constexpr uint64_t IDX_3 = 3;
constexpr uint64_t IDX_4 = 4;
constexpr uint64_t IDX_5 = 5;
constexpr uint64_t IDX_6 = 6;
constexpr uint64_t IDX_7 = 7;
constexpr uint64_t IDX_8 = 8;
constexpr uint64_t IDX_9 = 9;
constexpr uint64_t IDX_10 = 10;
constexpr uint64_t IDX_11 = 11;
constexpr uint64_t IDX_12 = 12;
constexpr uint64_t IDX_13 = 13;
constexpr uint64_t IDX_14 = 14;
constexpr uint64_t IDX_15 = 15;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;

// 当前每个kernel最多使用4组同步标记，这里预留6组
constexpr uint32_t MAX_FLAG_SIZE_PER_KERNEL = 6 * MAX_RANK_SIZE * FLAG_SIZE;

// 将__COUNTER__改为固定偏移，新执行器需添加新偏移
#define AIV_ALL_GATHER_91093_SMALLDATA_GRAPH 0
#define AIV_ALL_GATHER_910B_BIGDATA 1
#define AIV_ALL_GATHER_910B_GRAPH 2
#define AIV_ALL_GATHER_910B_RDMA_GRAPH 3
#define AIV_ALL_GATHER_910B_RDMA 4
#define AIV_ALL_GATHER_910B_SMALLDATA 5
#define AIV_ALL_GATHER_V_910B_BIGDATA 6
#define AIV_ALL_GATHER_V_910B_SMALLDATA 7
#define AIV_ALL_REDUCE_910B_BIGDATA_GRAPH 8
#define AIV_ALL_REDUCE_910B_BIGDATA 9
#define AIV_ALL_REDUCE_910B_MIDDATA 10
#define AIV_ALL_REDUCE_910B_RDMA_MIDDATA_GRAPH_STEP1 11
#define AIV_ALL_REDUCE_910B_RDMA_MIDDATA_STEP1 12
#define AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_GRAPH_STEP1 13
#define AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_STEP1 14
#define AIV_ALL_REDUCE_910B_SMALLDATA_GRAPH 15
#define AIV_ALL_REDUCE_910B_SMALLDATA 16
#define AIV_ALL_TO_ALL_91093_BASE 17
#define AIV_ALL_TO_ALL_91093_GRAPH 18
#define AIV_ALL_TO_ALL_91093 19
#define AIV_ALL_TO_ALL_910B_SMALLDATA 20
#define AIV_ALL_TO_ALL_RDMA_910B 21
#define AIV_ALL_TO_ALL_V_91093_GRAPH 22
#define AIV_ALL_TO_ALL_V_91093 23
#define AIV_ALL_TO_ALL_V_91093_SINGLE 24
#define AIV_ALL_TO_ALL_V_910B_GRAPH 25
#define AIV_ALL_TO_ALL_V_910B 26
#define AIV_ALL_TO_ALL_VC_910B_GRAPH 27
#define AIV_ALL_TO_ALL_VC_910B 28
#define AIV_ALL_TO_ALL_VC_910B_NO_LOOP 29
#define AIV_REDUCE_SCATTER_91093_SMALLDATA_GRAPH 30
#define AIV_REDUCE_SCATTER_910B_BIGDATA 31
#define AIV_REDUCE_SCATTER_910B_GRAPH 32
#define AIV_REDUCE_SCATTER_910B_MIDDATA 33
#define AIV_REDUCE_SCATTER_910B_RDMA_GRAPH 34
#define AIV_REDUCE_SCATTER_910B_RDMA 35
#define AIV_REDUCE_SCATTER_910B_SMALLDATA 36
#define AIV_REDUCE_SCATTER_V_910B_BIGDATA 37
#define AIV_REDUCE_SCATTER_V_910B_MIDDATA 38
#define AIV_REDUCE_SCATTER_V_910B_SMALLDATA 39
#define AIV_SYNC_910B 40
#define AIV_ALL_GATHER_91093_SMALLDATA 41
#define AIV_REDUCE_SCATTER_91093_SMALLDATA 42
#define AIV_ALL_REDUCE_910B_RDMA_MIDDATA_GRAPH_STEP2 43
#define AIV_ALL_REDUCE_910B_RDMA_MIDDATA_STEP2 44
#define AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_GRAPH_STEP2 45
#define AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_STEP2 46
#define AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_GRAPH_STEP3 47
#define AIV_ALL_REDUCE_910B_RDMA_SMALLDATA_STEP3 48

#define BASE_FLAG_OFFSET (MAX_FLAG_SIZE_PER_KERNEL)

#define DEV_TYPE_910_93 4

class AivCommBase {
public:
    __aicore__ inline AivCommBase() {}

    __aicore__ inline void Init(GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4,
                                GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9,
                                GM_ADDR buffIn10, GM_ADDR buffIn11, GM_ADDR buffIn12, GM_ADDR buffIn13,
                                GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
                                GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5,
                                GM_ADDR buffOut6, GM_ADDR buffOut7, GM_ADDR buffOut8, GM_ADDR buffOut9,
                                GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
                                GM_ADDR buffOut14, GM_ADDR buffOut15, uint32_t rank, uint32_t rankSize,
                                uint32_t dataType, uint32_t reduceOp, uint32_t root, bool useDoubleBuffer)
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
        reduceOp_ = reduceOp;

        useDoubleBuffer_ = useDoubleBuffer;

        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_4);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);

        pipe.InitBuffer(flagBatchSetQue, 1, UB_FLAG_SIZE_8); // 最多支持同时set8个flag值，256B可存放32个u64，最多2组16rank
        pipe.InitBuffer(flagBatchCheckQue, 1, UB_FLAG_SIZE_8); // 最多支持同时check8个flag值

        if (useDoubleBuffer) {
            pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE); // double buffer
        } else {
            pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }

        pipe.InitBuffer(flagInQue, AIV_PING_PONG_FACTOR_TWO, UB_FLAG_SIZE);
    }
    
    __aicore__ inline void InitBuffArray(GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4,
                                GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9,
                                GM_ADDR buffIn10, GM_ADDR buffIn11, GM_ADDR buffIn12, GM_ADDR buffIn13,
                                GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
                                GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5,
                                GM_ADDR buffOut6, GM_ADDR buffOut7, GM_ADDR buffOut8, GM_ADDR buffOut9,
                                GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
                                GM_ADDR buffOut14, GM_ADDR buffOut15)
    {
        GM_IN[IDX_0] = buffIn0;
        GM_IN[IDX_1] = buffIn1;
        GM_IN[IDX_2] = buffIn2;
        GM_IN[IDX_3] = buffIn3;
        GM_IN[IDX_4] = buffIn4;
        GM_IN[IDX_5] = buffIn5;
        GM_IN[IDX_6] = buffIn6;
        GM_IN[IDX_7] = buffIn7;
        GM_IN[IDX_8] = buffIn8;
        GM_IN[IDX_9] = buffIn9;
        GM_IN[IDX_10] = buffIn10;
        GM_IN[IDX_11] = buffIn11;
        GM_IN[IDX_12] = buffIn12;
        GM_IN[IDX_13] = buffIn13;
        GM_IN[IDX_14] = buffIn14;
        GM_IN[IDX_15] = buffIn15;

        GM_OUT[IDX_0] = buffOut0;
        GM_OUT[IDX_1] = buffOut1;
        GM_OUT[IDX_2] = buffOut2;
        GM_OUT[IDX_3] = buffOut3;
        GM_OUT[IDX_4] = buffOut4;
        GM_OUT[IDX_5] = buffOut5;
        GM_OUT[IDX_6] = buffOut6;
        GM_OUT[IDX_7] = buffOut7;
        GM_OUT[IDX_8] = buffOut8;
        GM_OUT[IDX_9] = buffOut9;
        GM_OUT[IDX_10] = buffOut10;
        GM_OUT[IDX_11] = buffOut11;
        GM_OUT[IDX_12] = buffOut12;
        GM_OUT[IDX_13] = buffOut13;
        GM_OUT[IDX_14] = buffOut14;
        GM_OUT[IDX_15] = buffOut15;
    }

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag) {}

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    __aicore__ inline uint64_t CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice,
        uint64_t tailLength);

    template<typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    template<typename T>
    __aicore__ inline void DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, bool atomic = false,
        uint32_t atomicOp = 0);

    template<typename T>
    __aicore__ inline void CpGM2GMWithFlagWrap(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count,
        __gm__ int32_t* ctrlFlagGM, uint64_t flushFrequency = 8, int32_t tag = 0);

    __aicore__ inline void SetFlagNew(__gm__ int32_t *ctrlFlagGM, int32_t setValue, bool atomic = false);

    __aicore__ inline void SetFlagBatch(__gm__ int32_t *ctrlFlagGM, int32_t setValue, int32_t count);

    __aicore__ inline void CheckFlagNew(__gm__ int32_t *ctrlFlagGM, int32_t checkValue);

    __aicore__ inline int32_t GetFlagNew(__gm__ int32_t *ctrlFlagGM);

    __aicore__ inline void CheckFlagGE(__gm__ int32_t *ctrlFlagGM, int32_t checkValue);

    template<HardEvent event> 
    __aicore__ inline void SyncFunc();

protected:
    GM_ADDR GM_IN[MAX_RANK_SIZE];
    GM_ADDR GM_OUT[MAX_RANK_SIZE];

    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t reduceOp_;

    bool useDoubleBuffer_;

    TPipe pipe;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localCheckGETensor;
    LocalTensor<int32_t> localGetTensor;

    TQue<QuePosition::VECOUT, 1> flagBatchSetQue;
    TQue<QuePosition::VECIN, 1> flagBatchCheckQue;

    TQue<QuePosition::VECIN, 1> flagInQue;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;
};

__aicore__ inline uint64_t AivCommBase::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t AivCommBase::CalActualCount(uint32_t sliceIdx, uint64_t sliceCount,
    uint64_t avgLengthPerSlice, uint64_t tailLength)
{
    if (sliceIdx == sliceCount - 1) {
        return tailLength;
    } else if (sliceIdx < sliceCount - 1) {
        return avgLengthPerSlice;
    } else {
        return 0;
    }
}

template<typename T>
__aicore__ inline void AivCommBase::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
            SetAtomicAdd<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
            SetAtomicMax<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
            SetAtomicMin<T>(); break;
        default:
            SetAtomicNone(); break;
    }
}

template<typename T>
__aicore__ inline void AivCommBase::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
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
__aicore__ inline void AivCommBase::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
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
__aicore__ inline void AivCommBase::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, bool atomic,
    uint32_t atomicOp)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);
    
    if (atomic) {
        SetAtomicOp<T>(atomicOp);
    }

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

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

    if (atomic) {
        SetAtomicNone();
    }
    return;
}

template<typename T>
__aicore__ inline void AivCommBase::CpGM2GMWithFlagWrap(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count,
    __gm__ int32_t* ctrlFlagGM, uint64_t flushFrequency, int32_t tag)
{
    uint64_t curBatchCount = 0;

    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }
    
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

        curBatchCount += 1;

        if (curBatchCount % flushFrequency == 0 || count == 0) {
            SyncFunc<HardEvent::MTE3_S>();

            SetFlagNew(ctrlFlagGM, curBatchCount + tag);
        }
    }
}

__aicore__ inline void AivCommBase::SetFlagNew(__gm__ int32_t *ctrlFlagGM, int32_t setValue, bool atomic)
{
    GlobalTensor<int32_t> globalSet;
    globalSet.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT);

    if (atomic) {
        Duplicate<int32_t>(localSetTensor, setValue, UB_FLAG_PAD_COUNT);

        SetAtomicAdd<int32_t>();
        PipeBarrier<PIPE_ALL>();
    } else {
        localSetTensor.SetValue(0, setValue);

        SyncFunc<HardEvent::S_MTE3>();
    }

    DataCopy(globalSet, localSetTensor, UB_FLAG_PAD_COUNT);

    if (atomic) {
        SetAtomicNone();
    }
}

__aicore__ inline void AivCommBase::SetFlagBatch(__gm__ int32_t *ctrlFlagGM, int32_t setValue, int32_t count)
{
    GlobalTensor<int32_t> globalBatchSet;
    globalBatchSet.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT * count);
    LocalTensor<int32_t> localBatchSet = flagBatchSetQue.AllocTensor<int32_t>();

    for (uint32_t i = 0; i < count; i++) {
        localBatchSet.SetValue(i * UB_FLAG_PAD_COUNT, setValue);
    }

    SyncFunc<HardEvent::S_MTE3>();

    DataCopy(globalBatchSet, localBatchSet, UB_FLAG_PAD_COUNT * count);

    flagBatchSetQue.FreeTensor(localBatchSet);
}

__aicore__ inline void AivCommBase::CheckFlagNew(__gm__ int32_t *ctrlFlagGM, int32_t checkValue)
{
    GlobalTensor<int32_t> globalCheck;
    globalCheck.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT);

    while (true) {
        DataCopy(localCheckTensor, globalCheck, UB_FLAG_PAD_COUNT);
        SyncFunc<HardEvent::MTE2_S>();

        if (localCheckTensor.GetValue(0) == checkValue) {
            break;
        }
    }
}

__aicore__ inline int32_t AivCommBase::GetFlagNew(__gm__ int32_t *ctrlFlagGM)
{
    GlobalTensor<int32_t> globalGet;
    globalGet.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT);

    DataCopy(localGetTensor, globalGet, UB_FLAG_PAD_COUNT);
    SyncFunc<HardEvent::MTE2_S>();

    int32_t val = localGetTensor.GetValue(0);

    return val + 1;
}

__aicore__ inline void AivCommBase::CheckFlagGE(__gm__ int32_t *ctrlFlagGM, int32_t checkValue)
{
    GlobalTensor<int32_t> globalCheck;
    globalCheck.SetGlobalBuffer(ctrlFlagGM, UB_FLAG_PAD_COUNT);

    while (true) {
        DataCopy(localCheckGETensor, globalCheck, UB_FLAG_PAD_COUNT);
        SyncFunc<HardEvent::MTE2_S>();

        int32_t flagValue = localCheckGETensor.GetValue(0);
        if (flagValue >= checkValue) {
            break;
        }
    }
}

template<HardEvent event> 
__aicore__ inline void AivCommBase::SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

#endif  /* AIV_COMMUNICATION_BASE_H */
