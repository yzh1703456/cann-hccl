/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_TASK_ABORT_HANDLER_PUB_H
#define HCCL_TASK_ABORT_HANDLER_PUB_H
#include "hccl_common.h"

namespace hccl {
class hcclComm;
class HcclCommunicator;
class TaskAbortHandler {
public:
    TaskAbortHandler();
    ~TaskAbortHandler();
    static HcclResult Init(HcclCommunicator *communicator);
    static HcclResult DeInit(HcclCommunicator *communicator);
private:
};
}

#endif
