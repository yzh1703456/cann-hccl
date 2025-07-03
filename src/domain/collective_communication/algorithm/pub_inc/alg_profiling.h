/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once 
#include <thread>
#include "hccl_types.h"
#include "hccl_common.h"
#include "coll_alg_utils.h"

typedef void (*TaskCallBack)(void *userPtr, void *param, u32 length);

struct TaskParaAiv{
    HcclCMDType cmdType;
    u32 tag;
    u64 size;
    u32 blockDim;
    u32 rankSize;
    s32 aivRdmaStep;
    void* flagMem;
    TaskParaAiv():cmdType(HcclCMDType::HCCL_CMD_INVALID), tag(0), size(0), blockDim(0), rankSize(0), aivRdmaStep(0),flagMem(0)
    {}
    TaskParaAiv(HcclCMDType cmdType, u32 tag, u64 size, u32 blockDim, u32 rankSize, s32 aivRdmaStep, void* flagMem)
        : cmdType(cmdType),
          tag(tag),
          size(size),
          blockDim(blockDim),
          rankSize(rankSize),
          aivRdmaStep(aivRdmaStep),
          flagMem(flagMem)
    {}
};

struct AivTaskPara {
    void *stream{nullptr};
    bool isMainStream{false};
    u64 beginTime{0};
    union {
        struct TaskParaAiv aiv;
    };

    AivTaskPara() : stream(nullptr), isMainStream(false), beginTime(0)
    {}

    ~AivTaskPara() {}
};


HcclResult RegisterAlgCallBack(void* userPtr, TaskCallBack callback, s32 deviceLogicID);

HcclResult TaskAivProfiler(HcclCMDType cmdType, u32 tag, u64 size, u32 blockDim, u32 rankSize,
    void* flagMem, rtStream_t stream, s32 aivRdmaStep, uint64_t beginTime);

