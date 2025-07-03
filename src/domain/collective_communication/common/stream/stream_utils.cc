/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_utils.h"
#include "log.h"
#include "runtime/stream.h"
#include "runtime/rt_model.h"
#include "workflow_pub.h"

HcclResult GetStreamCaptureInfo(rtStream_t stream, rtModel_t &rtModel, bool &isCapture)
{
    isCapture = false;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_WARNING("[%s]Stream capture only support opbase mode.", __func__);
        return HCCL_SUCCESS;
    }
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    rtError_t ret = rtStreamGetCaptureInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamGetCaptureInfo fail.  return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    switch (captureStatus) {
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_ACTIVE: {
            isCapture = true;
            break;
        }
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE: {
            HCCL_DEBUG("[%s]Stream capture status NONE.", __func__);
            break;
        }
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_MAX: {
            HCCL_ERROR("[%s]Stream capture status MAX.", __func__);
            break;
        }
        case rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_INVALIDATED: {
            HCCL_ERROR("[%s]Stream capture status invalidated.", __func__);
            break;
        }
        default: {
            HCCL_ERROR("[%s]Unsupported stream capture status.", __func__);
            break;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AddStreamToModel(rtStream_t stream, rtModel_t &rtModel)
{
    rtError_t ret = rtStreamAddToModel(stream, rtModel);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s]rtStreamAddToModel failed. ret[%d].", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult GetModelId(rtModel_t &rtModel, u32 &modelId)
{
    rtError_t ret = rtModelGetId(rtModel, &modelId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s]rtModelGetId failed. ret[%d].", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}