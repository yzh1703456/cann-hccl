/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_OVERFLOW_PUB_H
#define TASK_OVERFLOW_PUB_H
#include <mutex>
#include <vector>
#include "runtime/rt.h"
#include "profiler_base_pub.h"

namespace hccl {

class TaskOverflow : public ProfilerBase {
public:
    explicit TaskOverflow(u32 deviceLogicId);
    ~TaskOverflow() override;
    HcclResult Save(u32 &streamID, u32 &taskID, TaskType &taskType, const TaskParaDMA &paraDMA) override;
    HcclResult Save(u32 &streamID, u32 &taskID, TaskType &taskType, const TaskParaReduce &paraReduce) override;
    HcclResult Save(u32 &streamID, u32 &taskID, TaskType &taskType, const TaskParaNotify &paraNotify) override;
    HcclResult Save(u32 &streamID, u32 &taskID, const TaskParaAiv &paraAiv) override;
    HcclResult Save(u32 &streamID, u32 &taskID) override;
    HcclResult Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType &taskType, const TaskParaDMA &paraDMA) override;
    HcclResult Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType &taskType, const TaskParaReduce &paraReduce) override;
    HcclResult Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType &taskType, const TaskParaNotify &paraNotify) override;
    HcclResult Save(u32 captureStreamID, u32 streamID, u32 taskID) override;
    uint32_t GetTaskName(TaskType taskType) const;
    HcclResult Run(const StepData &stepData) override;
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfoVector);
    HcclResult Flush() override;
    HcclResult SaveToLog(const TaskParaHost &paraHost) override;
protected:
private:
    std::vector<HcclDumpInfo> dumpInfoVetcor_;
    std::mutex dumpInfoVetcorMutex_;
};
}
#endif
