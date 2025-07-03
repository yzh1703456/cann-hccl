/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlrt_function.h"
#include "log.h"

namespace hccl {
DlRtFunction &DlRtFunction::GetInstance()
{
    static DlRtFunction hcclDlRtFunction;
    return hcclDlRtFunction;
}

DlRtFunction::DlRtFunction()
{
}

DlRtFunction::~DlRtFunction()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

rtError_t rtProfRegisterCtrlCallbackStub(uint32_t modelid, rtProfCtrlHandle callback)
{
    HCCL_WARNING("Entry rtProfRegisterCtrlCallbackStub");
    return RT_ERROR_NONE;
}

HcclResult DlRtFunction::DlRtFunctionStubInit()
{
    dlrtProfRegisterCtrlCallback = (rtError_t(*)(uint32_t, rtProfCtrlHandle))rtProfRegisterCtrlCallbackStub;
    return HCCL_SUCCESS;
}

HcclResult DlRtFunction::DlRtFunctionInterInit()
{
    dlrtProfRegisterCtrlCallback = (rtError_t(*)(uint32_t, rtProfCtrlHandle))dlsym(handle_,
        "rtProfRegisterCtrlCallback");
    return HCCL_SUCCESS;
}

HcclResult DlRtFunction::DlRtFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = dlopen("libruntime.so", RTLD_NOW);
    }
    if (handle_ != nullptr) {
        CHK_RET(DlRtFunctionInterInit());
    } else {
        CHK_RET(DlRtFunctionStubInit());
        HCCL_WARNING("dlopen libruntime.so failed");
    }
    return HCCL_SUCCESS;
}
}
