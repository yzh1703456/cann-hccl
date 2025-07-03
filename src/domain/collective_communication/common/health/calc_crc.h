/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_CRC_H
#define CALC_CRC_H

#include <string>
#include <hccl/hccl_types.h>
#include "hccl/base.h"


namespace hccl {
class CalcCrc {
public:
    CalcCrc() = delete;
    ~CalcCrc() = delete;
    CalcCrc(CalcCrc const&) = delete;
    CalcCrc(CalcCrc&&) = delete;
    CalcCrc& operator=(CalcCrc const&) = delete;
    CalcCrc& operator=(CalcCrc&&) = delete;

    static HcclResult HcclCalcCrc(const char *data, u64 length, u32 &crcValue);
};
}
#endif  // CALC_CRC_H
