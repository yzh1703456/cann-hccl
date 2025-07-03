/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "profiling_manager.h"
#include "adapter_prof.h"
#include "adapter_rts_common.h"
#include "runtime/base.h"
#include "profiler_base_pub.h"
#include "workflow_pub.h"
#include "profiling_manager_pub.h"

namespace hccl {
HcclResult ProfilingManagerPub::CallMsprofReportMultiThreadInfo(const std::vector<uint32_t> &tidInfo)
{
    return ProfilingManager::Instance().CallMsprofReportMultiThreadInfo(tidInfo);
}

HcclResult ProfilingManagerPub::GetAddtionInfoState()
{
    return ProfilingManager::Instance().GetAddtionInfoState();
}

HcclResult ProfilingManagerPub::GetTaskApiState()
{
    return ProfilingManager::Instance().GetTaskApiState();
}
HcclResult ProfilingManagerPub::CallMsprofReportHostApi(HcclCMDType cmdType, uint64_t beginTime, u64 count,
    HcclDataType dataType, AlgType algType, uint64_t groupName, u32 blockDim)
{
    return ProfilingManager::Instance().CallMsprofReportHostApi(cmdType, beginTime, count,
        dataType, algType, groupName, blockDim);
}

HcclResult ProfilingManagerPub::CallMsprofReportMc2CommInfo(uint64_t timeStamp, const void *data, int len)
{
    CHK_PTR_NULL(data);
    return ProfilingManager::Instance().CallMsprofReportMc2CommInfo(timeStamp, data, len);
}
 
HcclResult ProfilingManagerPub::CallMsprofReportHostNodeApi(uint64_t beginTime, uint64_t endTime,
    const std::string profName, uint32_t threadId)
{
    uint64_t itemId = hrtMsprofGetHashId(profName.c_str(), profName.length());

    return ProfilingManager::Instance().CallMsprofReportHostNodeApi(beginTime, endTime, itemId, threadId);
}

HcclResult ProfilingManagerPub::CallMsprofReportHostNodeBasicInfo(uint64_t endTime, const std::string profName,
    uint32_t threadId)
{
    uint64_t itemId = hrtMsprofGetHashId(profName.c_str(), profName.length());

    return ProfilingManager::Instance().CallMsprofReportHostNodeBasicInfo(endTime, itemId, threadId);
}

HcclResult ProfilingManagerPub::CallMsprofReportNodeInfo(uint64_t beginTime, uint64_t endTime,
        const std::string profName, uint32_t threadId)
{
    return ProfilingManager::Instance().CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId);
}

bool ProfilingManagerPub::GetAllState()
{
    return ProfilingManager::Instance().GetAllState();
}

HcclResult ProfilingManagerPub::ClearStoragedProfilingInfo()
{
    return ProfilingManager::Instance().ClearStoragedProfilingInfo();
}

void ProfilingManagerPub::SetCaptureStatus(bool isCapture)
{
    ProfilingManager::Instance().SetCaptureStatus(isCapture);
}

}