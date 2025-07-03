/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <vector>
#include <iostream>
#include <fstream>
#include "mmpa_api.h"
#include "adapter_rts_common.h"
#include "hccl_aiv.h"
#include "../../framework/common/src/hashtable/universal_concurrent_map.h"
#include "workflow_pub.h"
#include "mem_device_pub.h"

using namespace std;
using BinHandle = void *;
extern HcclResult hrtDevBinaryRegister(const rtDevBinary_t *bin, BinHandle *handle);
extern HcclResult hrtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
    rtStream_t stream, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo);
extern HcclResult hrtFunctionRegister(BinHandle binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
    uint32_t funcMode);

namespace hccl {
constexpr u32 SIG_MOVE_LEFT_BITS = 20;
constexpr u32 BLOCK_DIM_FACTOR_TWO = 2;
constexpr u32 RANK_ZERO = 0;
constexpr u32 RANK_ONE = 1;
constexpr u32 RANK_TWO = 2;
constexpr u32 RANK_THREE = 3;
constexpr u32 RANK_FOUR = 4;
constexpr u32 RANK_FIVE = 5;
constexpr u32 RANK_SIX = 6;
constexpr u32 RANK_SEVEN = 7;

constexpr s32 TAG_INIT_VALUE = 1;
constexpr s32 TAG_RESET_COUNT = 1000;

constexpr u32 AIV_BUFFER_PING_PONG_FACTOR = 2;

constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024; // 最大读取100m的bin file到string中

constexpr s32 RESET_TAIL_SYNC_TAG = 2;
constexpr u32 AIV_FLAG_AREA_SIZE = 1024 * 1024;

enum class KernelArgsType {
    ARGS_TYPE_SERVER = 0, // kernel参数为单机内
    ARGS_TYPE_SUPERPOD = 1, // kernel参数包含多机，当前仅A3 AlltoAllV跨机场景
    ARGS_TYPE_DEFAULT
};

using AivKernelInfo = struct AivKernelInfoDef {
    const char* kernelName;
    HcclCMDType cmdType;
    HcclDataType dataType;
    KernelArgsType argsType;

    AivKernelInfoDef(const char* kernelName, HcclCMDType cmdType, HcclDataType dataType,
        KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
        : kernelName(kernelName), cmdType(cmdType), dataType(dataType), argsType(argsType)
    {
    }
};

static std::vector<AivKernelInfo> g_aivKernelInfoList = {
    // allreduce
    {"aiv_all_reduce_float", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_reduce_half", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_reduce_int16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_reduce_int32_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_reduce_int8_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_reduce_bfloat16_t", HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // alltoall alltoallvc
    {"aiv_all_to_all_vc_half", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_vc_int16_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_vc_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_vc_float", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_vc_int32_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_vc_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_vc_int8_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_vc_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_vc_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // alltoallv
    {"aiv_all_to_all_v_half", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_v_int16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_v_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_v_float", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_v_int32_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_v_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_v_int8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_v_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_v_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // alltoallv a3
    {"aiv_all_to_all_v_sp_half",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int16_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint16_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_float",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int32_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint32_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT32, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_int8_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_uint8_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_UINT8, KernelArgsType::ARGS_TYPE_SUPERPOD},
    {"aiv_all_to_all_v_sp_bfloat16_t",
        HcclCMDType::HCCL_CMD_ALLTOALLV, HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_SUPERPOD},
    // alltoall
    {"aiv_all_to_all_half", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_to_all_int16_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_to_all_uint16_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_to_all_float", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_to_all_int32_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_to_all_uint32_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_to_all_int8_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_to_all_uint8_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_to_all_bfloat16_t", HcclCMDType::HCCL_CMD_ALLTOALL, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // reducescatter
    {"aiv_reduce_scatter_float", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_half", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_int16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_int32_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_int8_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // reducescatterv
    {"aiv_reduce_scatter_v_float", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_v_half", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_v_int16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_v_int32_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_v_int8_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_v_bfloat16_t", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclDataType::HCCL_DATA_TYPE_BFP16},
     // allgather
    {"aiv_all_gather_half", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_int16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_uint16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_float", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_int32_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_uint32_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_int8_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_uint8_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_bfloat16_t", HcclCMDType::HCCL_CMD_ALLGATHER, HcclDataType::HCCL_DATA_TYPE_BFP16},
     // allgatherv
    {"aiv_all_gather_v_half", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_v_int16_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_v_uint16_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_v_float", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_v_int32_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_v_uint32_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_v_int8_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_v_uint8_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_v_bfloat16_t", HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclDataType::HCCL_DATA_TYPE_BFP16},
    // 同步
    {"hccl_aiv_sync", HcclCMDType::HCCL_CMD_INVALID, HcclDataType::HCCL_DATA_TYPE_RESERVED},
};

using AivKernelArgs = struct AivKernelArgsDef {
    const void* buffersIn[MAX_RANK_SIZE] = {}; // 注册的CCLIN地址，所有卡可访问
    const void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag; // 第几次调用，定时重置成1
    bool isOpBase;
    u64 bufferSize;
    s32 aivRdmaStep;  // 用于AIV与rdma组合的场景，在不同step中kernel完成多机通信的不同部分
    bool useAivRdmaSmall;  // 使用aivRdma小数据量kernel，否则使用中数据量kernel
    u32 serverNum;
    u32 devType;

    AivKernelArgsDef(void** buffIn, void** buffOut, const void* input, const void* output, u32 rank,
        u32 rankSize, u64 len, u32 dataType, u32 reduceOp, u32 root, s32 tag, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 aivRdmaStep = -1, bool useAivRdmaSmall = false, u32 serverNum = 1,
        u32 devType = 2)
        : input(input), output(output), rank(rank), rankSize(rankSize), len(len), dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag), isOpBase(isOpBase), bufferSize(bufferSize), aivRdmaStep(aivRdmaStep),
        useAivRdmaSmall(useAivRdmaSmall), serverNum(serverNum), devType(devType)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8 *) buffIn[i];
            buffersOut[i] = (u8 *) buffOut[i];
        }
    }
};

using AivExtraKernelArgs = struct AivExtraKernelArgsDef {
    const void* buffersIn[MAX_RANK_SIZE] = {}; // 注册的CCLIN地址，所有卡可访问
    const void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag; // 第几次调用，定时重置成1
    bool isOpBase;
    u64 bufferSize;
    s32 aivRdmaStep;  // 用于AIV与rdma组合的场景，在不同step中kernel完成多机通信的不同部分
    bool useAivRdmaSmall;  // 使用aivRdma小数据量kernel，否则使用中数据量kernel
    u32 serverNum;
    u32 devType;
    ExtraArgs extraArgs; // A2/A3单机

    AivExtraKernelArgsDef(void** buffIn, void** buffOut, const void* input, const void* output, u32 rank,
        u32 rankSize, u64 len, u32 dataType, u32 reduceOp, u32 root, s32 tag, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 aivRdmaStep = -1, bool useAivRdmaSmall = false, u32 serverNum = 1,
        u32 devType = 2, const ExtraArgs* extraArgsPtr = nullptr)
        : input(input), output(output), rank(rank), rankSize(rankSize), len(len), dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag), isOpBase(isOpBase), bufferSize(bufferSize), aivRdmaStep(aivRdmaStep),
        useAivRdmaSmall(useAivRdmaSmall), serverNum(serverNum), devType(devType)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8 *) buffIn[i];
            buffersOut[i] = (u8 *) buffOut[i];
        }
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

using AivExtraKernelArgsV2 = struct AivExtraKernelArgsV2Def {
    const void* buffersIn[MAX_RANK_SIZE] = {}; // 注册的CCLIN地址，所有卡可访问
    const void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    const void* input;
    const void* output;
    u32 rank;
    u32 rankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    s32 tag; // 第几次调用，定时重置成1
    bool isOpBase;
    u64 bufferSize;
    s32 aivRdmaStep;  // 用于AIV与rdma组合的场景，在不同step中kernel完成多机通信的不同部分
    bool useAivRdmaSmall;  // 使用aivRdma小数据量kernel，否则使用中数据量kernel
    u32 serverNum;
    u32 devType;
    ExtraArgsV2 extraArgs; // A3超节点内多机

    AivExtraKernelArgsV2Def(void** buffIn, void** buffOut, const void* input, const void* output, u32 rank,
        u32 rankSize, u64 len, u32 dataType, u32 reduceOp, u32 root, s32 tag, bool isOpBase = true,
        u64 bufferSize = 200 * 1024 * 1024, s32 aivRdmaStep = -1, bool useAivRdmaSmall = false, u32 serverNum = 1,
        u32 devType = 2, const ExtraArgsV2* extraArgsPtr = nullptr)
        : input(input), output(output), rank(rank), rankSize(rankSize), len(len), dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag), isOpBase(isOpBase), bufferSize(bufferSize), aivRdmaStep(aivRdmaStep),
        useAivRdmaSmall(useAivRdmaSmall), serverNum(serverNum), devType(devType)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8 *) buffIn[i];
            buffersOut[i] = (u8 *) buffOut[i];
        }
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

HcclResult GetAivOpBinaryPath(DevType deviceType, std::string &binaryPath)
{
    // 获取二进制文件路径
    std::string libPath;
    char *getPath = getenv("LD_LIBRARY_PATH");
    if (getPath != nullptr) {
        libPath = getPath;
    } else {
        HCCL_ERROR("[AIV][GetAivOpBinaryPath]ENV:LD_LIBRARY_PATH is not set");
        return HCCL_E_PARA;
    }

    size_t mid = libPath.find("fwkacllib/lib64");
    if (mid == libPath.npos) {
        HCCL_WARNING("[AIV][GetAivOpBinaryPath]ENV:LD_LIBRARY_PATH lack fwkacllib/lib64");

        mmDlInfo info;
        mmDladdr(reinterpret_cast<void *>(RegisterKernel), &info);

        CHK_PRT_RET(info.dli_fname == nullptr, HCCL_ERROR("[AIV][GetAivOpBinaryPath]get path of libhccl_alg.so failed"),
            HCCL_E_UNAVAIL);

        char resolvedPath[PATH_MAX];
        if (realpath(info.dli_fname, resolvedPath) == nullptr) {
            HCCL_ERROR("[AIV][GetAivOpBinaryPath]path %s is not a valid real path", info.dli_fname);
            return HCCL_E_INTERNAL;
        }
        binaryPath = resolvedPath;
        if (binaryPath.find("/libhccl_alg.so") != binaryPath.npos) {
            binaryPath.erase(binaryPath.find("/libhccl_alg.so"));
        } else {
            HCCL_ERROR("[AIV][GetAivOpBinaryPath]get binary path failed");
            return HCCL_E_PARA;
        }
        HCCL_DEBUG("[AIV][GetAivOpBinaryPath]op binary file path[%s]", binaryPath.c_str());
    } else {
        u32 diff;
        if (libPath.find(":", mid) == libPath.npos) {
            diff = libPath.length() - libPath.rfind(":", mid);
        } else {
            diff = libPath.find(":", mid) - libPath.rfind(":", mid);
        }
        binaryPath = libPath.substr(libPath.rfind(":", mid) + 1, diff - 1);
    }

    // 判断应该加载的文件
    switch (deviceType) {
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            binaryPath += "/hccl_aiv_op_ascend910B.o";
            break;
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_310P3:
        case DevType::DEV_TYPE_310P1:
        default:
            HCCL_ERROR("[AIV][GetAivOpBinaryPath]devType[%u] is not supported", deviceType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult ReadBinFile(const string& fileName, string& buffer)
{
    std::ifstream filestr;
    filestr.open(fileName.c_str(), std::ios::binary);
    if (!filestr) {
        HCCL_ERROR("[AIV][ReadBinFile]open file [%s] failed!", fileName.c_str());
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    filestr.seekg(0, std::ios::end);
    std::streampos fileSize = filestr.tellg();
    filestr.seekg(0, std::ios::beg);

    if (fileSize == 0 || fileSize >= MAX_BIN_FILE_SIZE) {
        HCCL_ERROR("[AIV][ReadBinFile] file [%s] size is invalid, is [%d]!", fileName.c_str(), fileSize);
        filestr.close();
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    buffer.resize(fileSize);
    filestr.read(&buffer[0], fileSize);

    filestr.close();
    return HCCL_SUCCESS;
}

s8* GetStubFunc(HcclCMDType cmdType, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
{
    return reinterpret_cast<s8*>(
        (((static_cast<s64>(cmdType) << SIG_MOVE_LEFT_BITS) + static_cast<s64>(dataType)) << SIG_MOVE_LEFT_BITS) +
        static_cast<s64>(argsType));
}

HcclResult RegisterBinaryKernel(const char* funcName, const string &binFile, s8* stubFunc)
{
    rtDevBinary_t binary;
    void* binHandle = nullptr;

    binary.data = binFile.c_str();
    binary.length = binFile.size();
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC; // AIV算子
    binary.version = 0;

    HcclResult ret = hrtDevBinaryRegister(&binary, &binHandle);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterBinaryKernel] errNo[0x%016llx] rtDevBinaryRegister aiv "
        "fail, return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);

    ret = hrtFunctionRegister(binHandle, stubFunc, funcName, funcName, 0);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterBinaryKernel] errNo[0x%016llx] rtFunctionRegister aiv "
        "fail, return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

// Kernel注册入口，全局只需要初始化一次
HcclResult RegisterKernel(DevType deviceType)
{
    static bool init = false;
    static mutex mut;
    lock_guard<mutex> guard(mut);
    if (init) {
        return HCCL_SUCCESS;
    }

    HcclResult ret;

    string binFilePath;
    ret = GetAivOpBinaryPath(deviceType, binFilePath);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] get aiv op binary path failed"), HCCL_E_RUNTIME);

    static string buffer;
    ret = ReadBinFile(binFilePath, buffer);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] read aiv kernel bin file failed"),
        HCCL_E_RUNTIME);

    for (auto &aivKernelInfo: g_aivKernelInfoList) {
        ret = RegisterBinaryKernel(aivKernelInfo.kernelName, buffer,
            GetStubFunc(aivKernelInfo.cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType));
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] register binary kernel for kernelName[%s] "
            "cmdType[%d] dataType[%s] argsType[%d] failed", aivKernelInfo.kernelName, aivKernelInfo.cmdType,
            GetDataTypeEnumStr(aivKernelInfo.dataType).c_str(), aivKernelInfo.argsType), HCCL_E_RUNTIME);
    }

    init = true;

    return HCCL_SUCCESS;
}

u32 GetBlockDim(HcclCMDType cmdType, u32 rankSize, u64 dataSize, bool isOpBase, s32 aivRdmaStep, u32 serverNum,
    DevType devType)
{
    u32 blockDim = rankSize; // 默认情况使用rankSize个AIV

    if (cmdType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        if (devType == DevType::DEV_TYPE_910_93 && serverNum > 1) { // block_num需要为偶数
            blockDim = (rankSize < MAX_BLOCK_DIM ? rankSize + rankSize % BLOCK_DIM_FACTOR_TWO : MAX_BLOCK_DIM);
        } else if (isOpBase && dataSize >= AIV_ALL_TO_ALL_BIG_SIZE) {
            blockDim = BLOCK_DIM_FACTOR_TWO * rankSize; // 单机场景，单算子AlltoAll使用2倍 rankSize个aiv
        }
    } else if (cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        if (devType == DevType::DEV_TYPE_910_93 && serverNum > 1) { // A3超节点内多机场景，block_num需要为偶数
            blockDim = (rankSize < MAX_BLOCK_DIM ? rankSize + rankSize % BLOCK_DIM_FACTOR_TWO : MAX_BLOCK_DIM);
        } else if (devType == DevType::DEV_TYPE_910_93 && isOpBase && cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            // A3单机单算子场景，block_num为3倍或者4倍的ranksize
            blockDim = rankSize * BLOCK_DIM_FOUR_PER_RANK_A3 > MAX_BLOCK_DIM ?
                rankSize * BLOCK_DIM_THREE_PER_RANK_A3 : rankSize * BLOCK_DIM_FOUR_PER_RANK_A3;
        } else if (isOpBase && aivRdmaStep == -1) { // 多机场景，AlltoAll使用rankSize个aiv
            blockDim = BLOCK_DIM_FACTOR_TWO * rankSize; // 单机场景，单算子AlltoAll使用2倍 rankSize个aiv
        }
    } else if (cmdType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        if (isOpBase && dataSize >= AIV_ALL_REDUCE_BIG_SIZE && aivRdmaStep < 0) {
            blockDim = BLOCK_DIM_FACTOR_TWO * rankSize; // 单机场景，单算子AllReduce大数据使用2倍 rankSize个aiv
        }
    } else if (cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        if (aivRdmaStep >= 0) {
            blockDim = rankSize; // 多机场景，单算子ReduceScatter使用rankSize个aiv
        } else if (devType == DevType::DEV_TYPE_910_93 && !isOpBase) {
            blockDim = rankSize * BLOCK_DIM_FOUR_PER_RANK_A3 > MAX_BLOCK_DIM ?
                rankSize * BLOCK_DIM_THREE_PER_RANK_A3 : rankSize * BLOCK_DIM_FOUR_PER_RANK_A3;
        } else if (isOpBase && dataSize > AIV_REDUCE_SCATTER_MID_SIZE) {
            blockDim = BLOCK_DIM_FACTOR_TWO * rankSize; // 单机场景，单算子ReduceScatter大数据使用2倍 rankSize个aiv
        }
    } else if (cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        if (isOpBase && dataSize > AIV_REDUCE_SCATTER_MID_SIZE) {
            blockDim = BLOCK_DIM_FACTOR_TWO * rankSize; // 单机场景，单算子ReduceScatter大数据使用2倍 rankSize个aiv
        }
    } else if (cmdType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        if (devType == DevType::DEV_TYPE_910_93 && !isOpBase) {
            blockDim = rankSize * BLOCK_DIM_FOUR_PER_RANK_A3 > MAX_BLOCK_DIM ?
                rankSize * BLOCK_DIM_THREE_PER_RANK_A3 : rankSize * BLOCK_DIM_FOUR_PER_RANK_A3;
        } else if (isOpBase && dataSize > AIV_ALL_GATHER_SMALL_SIZE) {
            blockDim += 1; // 单机场景，单算子AllGather大数据使用(rankSize + 1)个aiv
        } 
    } else if (cmdType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        if (isOpBase && dataSize > AIV_ALL_GATHER_SMALL_SIZE) {
            blockDim += 1; // 单机场景，单算子AllGather大数据使用(rankSize + 1)个aiv
        }
    }

    HCCL_INFO("[AIV][GetBlockDim] blockDim is set to [%u]", blockDim);
    return blockDim;
}

HcclResult ExtractIdentifier(const std::string &tagKey, bool isOpBase, std::string &res)
{
    HCCL_DEBUG("[AIV][ExtractIdentifier] tagKey is [%s]", tagKey.c_str());
    string tagKeyTmp = tagKey;
    size_t pos = tagKeyTmp.find("_loop");
    if (pos != tagKeyTmp.npos) {
        tagKeyTmp = tagKeyTmp.substr(0, pos); // 图模式先去除tag的loop部分
    }

    pos = tagKeyTmp.find('_');
    if (pos == tagKeyTmp.npos) {
        return HCCL_E_PARA;
    }
    tagKeyTmp = tagKeyTmp.substr(pos + 1, tagKeyTmp.size()); // 去除前面的算子名部分

    if (!isOpBase) {
        pos = tagKeyTmp.rfind('_');
        tagKeyTmp = tagKeyTmp.substr(0, pos); // 图模式去除末尾index
    }

    res = tagKeyTmp;
    HCCL_DEBUG("[AIV][ExtractIdentifier] processed res is [%s]", res.c_str());
    return HCCL_SUCCESS;
}

HcclResult Barrier(void** cclBuffersOut, u32 rank, u32 rankSize, rtStream_t stream, s32 step)
{
    AivKernelArgs aivKernelArgs {
        cclBuffersOut, cclBuffersOut, nullptr, nullptr, rank, rankSize, 0,
        HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED, 0, step,
        false, 0, 0, false
    };
    rtTaskCfgInfo_t taskCfgInfo = { 0, 0, 1 };
    rtArgsEx_t argsEx { &aivKernelArgs, nullptr, sizeof(aivKernelArgs), 0, 0, 0, 0, 0 };
    HcclResult ret = hrtKernelLaunchWithFlagV2(GetStubFunc(HcclCMDType::HCCL_CMD_INVALID,
        HcclDataType::HCCL_DATA_TYPE_RESERVED), rankSize, &argsEx, nullptr, stream, 0, &taskCfgInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][Barrier] errNo[0x%016llx] rtKernelLaunch aiv fail, "
        "return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult ClearAivSyncBuf(void** cclBuffersOut, u32 rank, u32 rankSize, rtStream_t stream)
{
    CHK_RET(Barrier(cclBuffersOut, rank, rankSize, stream, 1));
    DeviceMem zeroMem = DeviceMem::create(static_cast<u8 *>(cclBuffersOut[rank]) + AIV_FLAG_AREA_SIZE,
        AIV_FLAG_AREA_SIZE);
    DeviceMem flagMem = DeviceMem::create(static_cast<u8 *>(cclBuffersOut[rank]), AIV_FLAG_AREA_SIZE);
    CHK_RET(hrtMemAsyncCopy(flagMem.ptr(), AIV_FLAG_AREA_SIZE, zeroMem.ptr(), AIV_FLAG_AREA_SIZE,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream));
    CHK_RET(Barrier(cclBuffersOut, rank, rankSize, stream, RESET_TAIL_SYNC_TAG));
    HCCL_INFO("[AIV][ClearAivSyncBuf] clearaiv done");
    return HCCL_SUCCESS;
}

HcclResult GetTag(const std::string &tagKey, u32 rank, u32 devId, bool isOpBase, s32 &tag)
{
    static UniversalConcurrentMap<std::string, AivTagArray> tagMap;
    auto builder = []() -> AivTagArray {
        AivTagArray array;
        array.fill(0);
        return array;
    };

    std::string tagId;
    CHK_RET(ExtractIdentifier(tagKey, isOpBase, tagId));
    std::pair<UniversalConcurrentMap<std::string, AivTagArray>::Iterator, bool> tagIt;
    EXECEPTION_CATCH(tagIt = tagMap.EmplaceIfNotExist(tagId, builder), return HCCL_E_INTERNAL);
    if (tagIt.second) {
        HCCL_DEBUG("[AIV][GetTag]tagMap new insert, tagId[%s]", tagId.c_str());
    }

    AivTagArray &tags = tagMap[tagId];
    u32 tagRank = rank;
    if (devId != MAX_RANK_SIZE) {
        tagRank = devId;
    }
    HCCL_DEBUG("[AIV][GetTag]devId [%u] tagRank [%u]", devId, tagRank);
    tags[tagRank]++;
    if (tags[tagRank] > TAG_RESET_COUNT) {
        tags[tagRank] = TAG_INIT_VALUE;
    }
    tag = tags[tagRank];
    return HCCL_SUCCESS;
}

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, s32 tag, void* args, u32 argsSize, 
    AivProfilingInfo& aivProfilingInfo)
{
    HCCL_INFO("[AIV][ExecuteKernelLaunch] sendbuff [%p] recvbuff [%p] rank [%d] rankSize [%d] count [%llu] "
        "dataType [%s] reduceOp [%s] root [%d] tag [%d] isOpBase [%d] bufferSize [%llu] step [%d] "
        "isSmallCount [%d] serverNum [%d] devType[%d] extraArgsPtr [%p] argsSize [%d]", opArgs.input,
        opArgs.output, topoArgs.rank, topoArgs.rankSize, opArgs.count,
        GetDataTypeEnumStr(opArgs.dataType).c_str(), GetReduceOpEnumStr(opArgs.op).c_str(), opArgs.root,
        tag, opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum,
        topoArgs.devType, args, argsSize);
 
    HCCL_DEBUG("[AIV][ExecuteKernelLaunch] buffersIn [%p] [%p] [%p] [%p] [%p] [%p] [%p] [%p] "\
        "buffersOut [%p] [%p] [%p] [%p] [%p] [%p] [%p] [%p]", resourceArgs.buffersIn[RANK_ZERO],
        resourceArgs.buffersIn[RANK_ONE], resourceArgs.buffersIn[RANK_TWO], resourceArgs.buffersIn[RANK_THREE],
        resourceArgs.buffersIn[RANK_FOUR], resourceArgs.buffersIn[RANK_FIVE], resourceArgs.buffersIn[RANK_SIX],
        resourceArgs.buffersIn[RANK_SEVEN], resourceArgs.buffersOut[RANK_ZERO], resourceArgs.buffersOut[RANK_ONE],
        resourceArgs.buffersOut[RANK_TWO], resourceArgs.buffersOut[RANK_THREE], resourceArgs.buffersOut[RANK_FOUR],
        resourceArgs.buffersOut[RANK_FIVE], resourceArgs.buffersOut[RANK_SIX], resourceArgs.buffersOut[RANK_SEVEN]);

    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
    if (topoArgs.devType == DevType::DEV_TYPE_910_93 && opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV &&
        topoArgs.serverNum > 1) {
        argsType = KernelArgsType::ARGS_TYPE_SUPERPOD;
    }

    u32 perDataSize = SIZE_TABLE[opArgs.dataType];
    u64 dataSize = opArgs.count * perDataSize;
    u32 blockDim = GetBlockDim(opArgs.cmdType, topoArgs.rankSize, dataSize, opArgs.isOpBase, algArgs.step,
        topoArgs.serverNum, topoArgs.devType);

    rtTaskCfgInfo_t taskCfgInfo = { 0, 0, 1 };
    rtArgsEx_t argsEx { args, nullptr, argsSize, 0, 0, 0, 0, 0 };
    HcclResult ret = hrtKernelLaunchWithFlagV2(GetStubFunc(opArgs.cmdType, opArgs.dataType, argsType),
        blockDim, &argsEx, nullptr, resourceArgs.stream, 0, &taskCfgInfo);

    if (opArgs.isOpBase && (topoArgs.devType == DevType::DEV_TYPE_910B || topoArgs.serverNum == 1)) {
        if (tag == TAG_RESET_COUNT) { // 当前仅A2场景或者A3单机可以使用统一清零机制
            SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
            ClearAivSyncBuf(resourceArgs.buffersOut, topoArgs.rank, topoArgs.rankSize, resourceArgs.stream);
            SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        }
    }
    aivProfilingInfo.blockDim = blockDim;
    aivProfilingInfo.tag = tag;

    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][ExecuteKernelLaunch] errNo[0x%016llx] rtKernelLaunch aiv fail, "
        "return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);
    
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs,
    AivProfilingInfo& aivProfilingInfo)
{
    aivProfilingInfo.beginTime = MsprofSysCycleTime();
    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    s32 tag;
    CHK_RET(GetTag(resourceArgs.commTag, topoArgs.rank, topoArgs.devId, opArgs.isOpBase, tag));
    AivKernelArgs aivKernelArgs {
        resourceArgs.buffersIn,resourceArgs.buffersOut, opArgs.input, opArgs.output,
        topoArgs.rank, topoArgs.rankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, tag,
        opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum,
        static_cast<u32>(topoArgs.devType)
    };
    CHK_RET(ExecuteKernelLaunchInner(opArgs, topoArgs, resourceArgs, algArgs, tag, &aivKernelArgs,
        sizeof(aivKernelArgs), aivProfilingInfo));

    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgs &extraArgs, 
    AivProfilingInfo& aivProfilingInfo)
{
    aivProfilingInfo.beginTime = MsprofSysCycleTime();
    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    s32 tag;
    CHK_RET(GetTag(resourceArgs.commTag, topoArgs.rank, topoArgs.devId, opArgs.isOpBase, tag));

    AivExtraKernelArgs aivExtraKernelArgs {
        resourceArgs.buffersIn, resourceArgs.buffersOut, opArgs.input, opArgs.output,
        topoArgs.rank, topoArgs.rankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, tag,
        opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum,
        static_cast<u32>(topoArgs.devType), &extraArgs
    };
    CHK_RET(ExecuteKernelLaunchInner(opArgs, topoArgs, resourceArgs, algArgs, tag, &aivExtraKernelArgs,
        sizeof(aivExtraKernelArgs), aivProfilingInfo));

    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgsV2 &extraArgs, 
    AivProfilingInfo& aivProfilingInfo)
{
    aivProfilingInfo.beginTime = MsprofSysCycleTime();
    CHK_PTR_NULL(resourceArgs.buffersIn);
    CHK_PTR_NULL(resourceArgs.buffersOut);

    s32 tag;
    CHK_RET(GetTag(resourceArgs.commTag, topoArgs.rank, topoArgs.devId, opArgs.isOpBase, tag));

    AivExtraKernelArgsV2 aivExtraKernelArgs {
        resourceArgs.buffersIn, resourceArgs.buffersOut, opArgs.input, opArgs.output,
        topoArgs.rank, topoArgs.rankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, tag,
        opArgs.isOpBase, resourceArgs.bufferSize, algArgs.step, algArgs.isSmallCount, topoArgs.serverNum,
        static_cast<u32>(topoArgs.devType), &extraArgs
    };
    CHK_RET(ExecuteKernelLaunchInner(opArgs, topoArgs, resourceArgs, algArgs, tag, &aivExtraKernelArgs,
        sizeof(aivExtraKernelArgs), aivProfilingInfo));

    return HCCL_SUCCESS;
}

}   // ~~ namespace hccl