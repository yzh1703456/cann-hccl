/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PLUGIN_RUNNER_PUB_H
#define PLUGIN_RUNNER_PUB_H

#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "profiler_base_pub.h"
#include "task_exception_handler_pub.h"
#include "task_profiling_pub.h"
namespace hccl {
class PluginRunner {
public:
    explicit PluginRunner(ProfilerBase *profiler);
    ~PluginRunner();
    template <typename T>
    void operator () (rtStream_t stream, TaskType taskType, const T &para) const;
    void operator () (rtStream_t stream, const TaskParaAiv &paraAiv) const;
    void operator () (rtStream_t stream) const; // FFTS+ launch
    void operator () (const TaskParaHost &paraHost) const;
    void operator () (const StepData &stepData);
protected:
private:
    ProfilerBase *profiler_;
};
} // namespace hccl

#endif /* __PROFILER_BASE_PUB_H__ */
