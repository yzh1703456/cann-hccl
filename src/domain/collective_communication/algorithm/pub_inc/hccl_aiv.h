/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_AIV_H
#define HCCL_AIV_H
 
#include <vector>
#include "string"
 
#include "hccl_types.h"
#include "runtime/kernel.h"
#include "hccl_common.h"
 
namespace hccl {
constexpr u64 AIV_ALL_REDUCE_BIG_SIZE = 16 * 1024 * 1024;
constexpr u64 AIV_ALL_REDUCE_A3_ENTRY_SIZE = 1 * 1024 * 1024; // AllReduce单张卡数据量A3
constexpr u64 AIV_REDUCE_SCATTER_BIG_SIZE = 190 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_MID_SIZE = 2 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_ALL_GATHER_BIG_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_ALL_TO_ALL_BIG_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_TO_ALL_A3_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_BIG_SIZE = 256 * 1024 * 1024;

constexpr u64 AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr u64 AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE = 760 * 1024;

constexpr u32 MAX_RANK_SIZE = 16; // server内最大卡数
constexpr u32 MAX_RANK_SIZE_A3 = 768; // 超节点内最大卡数

constexpr u32 BLOCK_DIM_THREE_PER_RANK_A3 = 3;
constexpr u32 BLOCK_DIM_FOUR_PER_RANK_A3 = 4;
constexpr u32 MAX_BLOCK_DIM = 48;

constexpr u64 COMM_INFO_OFFSET = 32 * 1024; // 通信域内所有对端共享内存地址的信息距离aiv buffer末尾的偏移

using AivTagArray = std::array<s32, MAX_RANK_SIZE_A3>;

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A2场景
using ExtraArgs = struct AlltoAllExtraArgs {
    u64 sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    u64 sendCounts[MAX_RANK_SIZE] = {};
    u64 sendDispls[MAX_RANK_SIZE] = {};
    u64 recvCounts[MAX_RANK_SIZE] = {};
    u64 recvDispls[MAX_RANK_SIZE] = {};
    u64 maxCount = 0;
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A3场景
struct ExtraArgsV2 {
    u64 sendCounts[MAX_RANK_SIZE_A3] = {};
    u64 sendDispls[MAX_RANK_SIZE_A3] = {};
    u64 recvCounts[MAX_RANK_SIZE_A3] = {};
    u64 recvDispls[MAX_RANK_SIZE_A3] = {};
};

// 表示算子属性的参数，相对固定
struct AivOpArgs {
    HcclCMDType cmdType;
    const void* input;
    const void* output; 
    u64 count;
    HcclDataType dataType;
    HcclReduceOp op;
    u32 root;
    bool isOpBase;
};

// 表示拓扑信息的参数
struct AivTopoArgs {
    u32 rank;
    u32 rankSize;
    u32 devId;
    u32 serverId;
    u32 serverNum;
    DevType devType;

    AivTopoArgs(u32 rank, u32 rankSize, u32 devId = MAX_RANK_SIZE, u32 serverId = 0, u32 serverNum = 1,
        DevType devType = DevType::DEV_TYPE_910B)
    : rank(rank), rankSize(rankSize), devId(devId), serverId(serverId), serverNum(serverNum), devType(devType)
    {
    }
};

// 表示AIV所需要的资源参数
struct AivResourceArgs {
    const std::string &commTag;
    rtStream_t stream;
    void** buffersIn; // 注册的CCLIN地址，所有卡可访问
    void** buffersOut; // 注册的CCLOUT地址，所有卡可访问
    u64 bufferSize;
};

// 表示AIV算法流程控制的参数
struct AivAlgArgs {
    s32 step;
    bool isSmallCount;

    explicit AivAlgArgs(s32 step = -1, bool isSmallCount = false)
    : step(step), isSmallCount(isSmallCount)
    {
    }
};

// 表示AIVProfiling所需要的参数
struct AivProfilingInfo{
    u32 tag = 0;
    u32 blockDim = 0;
    uint64_t beginTime = 0;
};

HcclResult RegisterKernel(DevType deviceType);

HcclResult ClearAivSyncBuf(void** cclBuffersOut, u32 rank, u32 rankSize, rtStream_t stream);

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, 
    AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgs &extraArgs, 
    AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgsV2 &extraArgs, 
    AivProfilingInfo& aivProfilingInfo);

u32 GetBlockDim(HcclCMDType cmdType, u32 rankSize, u64 dataSize, bool isOpBase, s32 aivRdmaStep, u32 serverNum,
    DevType devType);

HcclResult ReadBinFile(const std::string& fileName, std::string& buffer);

HcclResult GetTag(const std::string &tagKey, u32 rank, u32 devId, bool isOpBase, s32 &tag);

HcclResult ExtractIdentifier(const std::string &tagKey, bool isOpBase, std::string &res);
}


#endif // HCCL_AIV_H