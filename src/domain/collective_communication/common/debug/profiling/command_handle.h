/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_PROFILING_COMMAND_HANDLE_H
#define COMMON_PROFILING_COMMAND_HANDLE_H

#include "runtime/base.h"

namespace hccl {
rtError_t CommandHandle(uint32_t rtType, void *data, uint32_t len);

rtError_t EsCommandHandle(uint32_t rtType, void *data, uint32_t len);
}
#endif // COMMON_PROFILING_COMMAND_HANDLE_H
