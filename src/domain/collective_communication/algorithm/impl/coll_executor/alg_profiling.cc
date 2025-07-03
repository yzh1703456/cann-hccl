/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_profiling.h"

/*
 * GetTaskCallBack & GetAivCallBackUserPtr 类内静态数组均为解决thread_local core问题
 */

TaskCallBack* GetTaskCallBack(s32 deviceLogicID)
{
    static TaskCallBack aivCallBack[MAX_MODULE_DEVICE_NUM];
    if (deviceLogicID < 0 || static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM){
        HCCL_ERROR("[alg_profiling][GetTaskCallBack] deviceLogicID %d is invalid", deviceLogicID);
        return nullptr;
    }
    return &aivCallBack[deviceLogicID];
}

void** GetAivCallBackUserPtr(s32 deviceLogicID)
{
    static void* aivCallBackUserPtr[MAX_MODULE_DEVICE_NUM];
    if (deviceLogicID < 0 || static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM){
        HCCL_ERROR("[alg_profiling][GetAivCallBackUserPtr] deviceLogicID %d is invalid", deviceLogicID);
        return nullptr;
    }
    return &aivCallBackUserPtr[deviceLogicID];
}

HcclResult RegisterAlgCallBack(void* userPtr, TaskCallBack callback, s32 deviceLogicID)
{
    auto* aivCallBack = GetTaskCallBack(deviceLogicID);
    auto* aivCallBackUserPtr = GetAivCallBackUserPtr(deviceLogicID);
    if (aivCallBack == nullptr || aivCallBackUserPtr == nullptr){
        return HCCL_E_PTR;
    }
    *aivCallBack = callback;
    *aivCallBackUserPtr = userPtr;
    return HCCL_SUCCESS;
}

void SetupTaskParaAiv(AivTaskPara& taskPara, TaskParaAiv& para, HcclRtStream stream, u64 beginTime)
{
    taskPara.isMainStream = true;
    taskPara.stream = stream;
    taskPara.beginTime = beginTime;
    taskPara.aiv = para;
}

HcclResult TaskAivProfiler(HcclCMDType cmdType, u32 tag, u64 size, u32 blockDim, u32 rankSize,
                           void* flagMem, rtStream_t stream, s32 aivRdmaStep, uint64_t beginTime)
{
    s32 deviceLogicID = INVALID_INT;
    hrtGetDevice(&deviceLogicID);
    auto* aivCallBack = GetTaskCallBack(deviceLogicID);
    auto* aivCallBackUserPtr = GetAivCallBackUserPtr(deviceLogicID);

    if (aivCallBack==nullptr || aivCallBackUserPtr==nullptr || (*aivCallBack) == nullptr || (*aivCallBackUserPtr) == nullptr){
        return HCCL_E_PTR;
    }

    TaskParaAiv para(cmdType, tag, size, blockDim, rankSize, aivRdmaStep, flagMem);
    AivTaskPara taskPara;

    SetupTaskParaAiv(taskPara, para, stream, beginTime);

    (*aivCallBack)((*aivCallBackUserPtr), static_cast<void *>(&taskPara), sizeof(struct AivTaskPara));
    return HCCL_SUCCESS;
}