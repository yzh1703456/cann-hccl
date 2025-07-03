/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROFILER_MANAGER_IMPL_H
#define PROFILER_MANAGER_IMPL_H
#include <mutex>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "dlprof_function.h"
#include "dlrt_function.h"
#include "task_profiling_pub.h"
#include "task_overflow_pub.h"
#include "task_exception_handler_pub.h"
#include "externalinput_pub.h"
#include "plugin_runner_pub.h"
#include "command_handle.h"
#include "profiler_base_pub.h"
#include "profiling_manager.h"
#include "alg_profiling.h"

namespace hccl {
class ProfilerManagerImpl {
public:
    ProfilerManagerImpl(s32 devicePhyId, s32 deviceLogicId, u32 realUserRank);
    ~ProfilerManagerImpl();
    HcclResult InitProfiler();
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo);
    void TaskSdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA &para);
    void TaskRdmaProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaDMA &para);
    void TaskReduceInlineProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce &para);
    void TaskReduceTbeProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaReduce &para);
    void TaskRecordProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify &para);
    void TaskWaitProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaNotify &para);
    void TaskAivProfiler(ProfilerType profilerType, HcclRtStream stream, TaskParaAiv &para);
    void TaskProfiler(ProfilerType profilerType, HcclRtStream stream);
    void TaskProfiler(ProfilerType profilerType, TaskParaHost &para);
    void TaskProfilerHandle(void *param, u32 length);
    void TaskAivProfilerHandle(void *param, u32 length);

private:
    s32 devicePhyId_;
    s32 deviceLogicId_;
    u32 realUserRank_;
    // profiling 相关资源
    std::unique_ptr<TaskProfiling> profiler_;
    std::shared_ptr<TaskExceptionHandler> taskExceptionHandler_;
    std::shared_ptr<TaskOverflow> taskOverflowHandler_;
    std::map<ProfilerType, hccl::PluginRunner> callbacks_;
    std::mutex mutex_;

    void RegisterCallBack(ProfilerType name, hccl::PluginRunner &callback);
    void HandleTask(struct TaskPara *taskPara, u32 &ctxId, ProfTaskType &profTaskType);
    void HandleGraphLaunchTask(struct TaskPara *taskPara);
};

} // namespace hccl
#endif