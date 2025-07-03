/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_runner.h"
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "runtime/rt_error_codes.h"

using namespace hccl;
PluginRunner::PluginRunner(ProfilerBase *profiler) : profiler_(profiler) {}

PluginRunner::~PluginRunner() {}

HcclResult isStreamCapture(rtStream_t stream, bool& isCapture)
{   
    isCapture = false;
    DevType devType;   
    CHK_RET(hrtGetDeviceType(devType));
    if(GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
       HCCL_WARNING("[PluginRunner][isStreamCapture]Stream capture only support opbase mode!");
       return HCCL_SUCCESS;
    }
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    rtModel_t rtModel = nullptr;
    rtError_t ret = rtStreamGetCaptureInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[PluginRunner][isStreamCapture]Stream capture not support!");
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != RT_ERROR_NONE,
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet stream get capture status fail. return[%d]", ret), HCCL_E_RUNTIME);
    }
    
    switch (captureStatus) {
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_ACTIVE: {
            isCapture = true;
            break;
        }
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE: {
            isCapture = false;
            break;
        }
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_MAX: {
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet stream capture status MAX.");
            break;
        }
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_INVALIDATED: {
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet stream capture status invalidated.");
            break;
        }
        default: {
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet not support stream capture status.");
            break;
        }
    }
    return HCCL_SUCCESS;
}

template <typename T> 
void PluginRunner::operator () (rtStream_t stream, TaskType taskType, const T &para) const
{   
    //capture模式下hrtGetStreamId获取的是原来的流对应ID，与实际执行流不是同一个
    //capture模式下hrtGetTaskIdAndStreamID获取实际执行的streamID和taskID
    u32 threadLastTaskID = 0;
    u32 threadLastStreamID = 0;
    s32 streamID = 0;
    HcclResult ret;
    bool isCapture = false;
    CHK_PRT(isStreamCapture(stream, isCapture));

    if (profiler_ == nullptr) return;

    if (GetExternalInputHcclEnableFfts() &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        ret = hrtGetStreamId(stream, streamID);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[PluginRunner][Operator]rtGet stream id fail. return[%d]", ret),);

        u32 castStreamID = static_cast<u32>(streamID);
        if (isCapture) {
            ret = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret),);
            profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID, taskType, para);
        } else {
            profiler_->Save(castStreamID, threadLastTaskID, taskType, para);
        }
    } else {
        ret = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret),);

        if (isCapture) {
            ret = hrtGetStreamId(stream, streamID);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret),);
            u32 castStreamID = static_cast<u32>(streamID);
            profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID, taskType, para);
        } else {
            profiler_->Save(threadLastStreamID, threadLastTaskID, taskType, para);
        }
    }
}

template void PluginRunner::operator ()<TaskParaDMA>(rtStream_t, TaskType, const TaskParaDMA&) const;
template void PluginRunner::operator ()<TaskParaReduce>(rtStream_t, TaskType, const TaskParaReduce&) const;
template void PluginRunner::operator ()<TaskParaNotify>(rtStream_t, TaskType, const TaskParaNotify&) const;

void PluginRunner::operator () (rtStream_t stream) const
{
    u32 threadLastTaskID = 0;
    u32 threadLastStreamID = 0;
    s32 streamID = 0;
    HcclResult ret;
    bool isCapture = false;
    CHK_PRT(isStreamCapture(stream, isCapture));

    if (profiler_ == nullptr) return;

    ret = hrtGetTaskIdAndStreamID(threadLastTaskID, threadLastStreamID);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
    HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret),);

    if (isCapture) {
        ret = hrtGetStreamId(stream, streamID);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[PluginRunner][Operator]rtGet stream id fail. return[%d]", ret),);
        u32 castStreamID = static_cast<u32>(streamID);
        profiler_->Save(castStreamID, threadLastStreamID, threadLastTaskID);
    } else {
        profiler_->Save(threadLastStreamID, threadLastTaskID);
    }
}

void PluginRunner::operator () (const TaskParaHost &paraHost) const
{
    if (profiler_ != nullptr) {
        profiler_->SaveToLog(paraHost);
    }
}

void PluginRunner::operator () (rtStream_t stream, const TaskParaAiv &paraAiv) const
{
    u32 taskID = 0;
    u32 streamID = 0;
    HcclResult ret;

    ret = hrtGetTaskIdAndStreamID(taskID, streamID);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[PluginRunner][Operator]rtGet task id and stream id fail. return[%d]", ret),);

    if (profiler_ != nullptr) {
        profiler_->Save(streamID, taskID, paraAiv);
    }
}