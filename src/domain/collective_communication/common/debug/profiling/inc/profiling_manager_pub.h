/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_PROFILING_PROFILING_MANAGER_PUB_H
#define COMMON_PROFILING_PROFILING_MANAGER_PUB_H

#include <string>
#include <cstdio>
#include "hccl_common.h"
#include "profiler_base_pub.h"
#include "adapter_prof.h"

namespace hccl {

class ProfilingManagerPub {
public:
    static HcclResult CallMsprofReportMultiThreadInfo(const std::vector<uint32_t> &tidInfo);
    static HcclResult GetAddtionInfoState();
    static HcclResult GetTaskApiState();
    static HcclResult CallMsprofReportHostApi(HcclCMDType cmdType, uint64_t beginTime, u64 count, HcclDataType dataType,
        AlgType algType, uint64_t groupName, u32 blockDim=0);
    static HcclResult CallMsprofReportMc2CommInfo(uint64_t timeStamp, const void *data, int len);
    static HcclResult CallMsprofReportHostNodeApi(uint64_t beginTime, uint64_t endTime, const std::string profName,
        uint32_t threadId);
    static HcclResult CallMsprofReportHostNodeBasicInfo(uint64_t endTime, const std::string profName,
        uint32_t threadId);
    static HcclResult CallMsprofReportNodeInfo(uint64_t beginTime, uint64_t endTime,
        const std::string profName, uint32_t threadId);
    static bool GetAllState();
    static HcclResult ClearStoragedProfilingInfo();
    static void SetCaptureStatus(bool isCapture);
};
} // namespace hccl
#endif // COMMON_PROFILING_PROFILING_MANAGER_H
