 /*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_PROF_H
#define HCCL_INC_ADAPTER_PROF_H

#include <hccl/hccl_types.h>

#include "runtime/rt.h"
#include "dlprof_function.h"
#include "hccl/base.h"

uint64_t hrtMsprofGetHashId(const char *hashInfo, uint32_t length);
uint64_t hrtMsprofSysCycleTime(void);

HcclResult hrtMsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName);
HcclResult hrtMsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);
HcclResult hrtMsprofReportApi(uint32_t agingFlag, const MsprofApi *api);
HcclResult hrtMsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);
HcclResult hrtMsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

HcclResult hrtProfRegisterCtrlCallback(uint32_t logId, rtProfCtrlHandle callback);

#endif
