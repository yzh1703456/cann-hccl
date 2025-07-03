/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef STREAM_UTILS_H
 #define STREAM_UTILS_H

 #include "hccl/base.h"
 #include "external/runtime/rt_error_codes.h"

 HcclResult GetStreamCaptureInfo(rtStream_t stream, rtModel_t &rtModel, bool &isCapture);
 HcclResult AddStreamToModel(rtStream_t stream, rtModel_t &rtModel);
 HcclResult GetModelId(rtModel_t &rtModel, u32 &modelId);

 #endif