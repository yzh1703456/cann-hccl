/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlprof_function.h"
#include "log.h"

namespace hccl {
DlProfFunction &DlProfFunction::GetInstance()
{
    static DlProfFunction hcclDlProfFunction;
    return hcclDlProfFunction;
}

DlProfFunction::DlProfFunction()
{
    DlProfFunctionStubInit();
}

DlProfFunction::~DlProfFunction()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

int32_t MsprofRegisterCallbackStub(uint32_t moduleId, ProfCommandHandle handle)
{
    HCCL_WARNING("Entry MsprofRegisterCallbackStub");
    return 0;
}

int32_t MsprofRegTypeInfoStub(uint16_t level, uint32_t typeId, const char *typeName)
{
    HCCL_WARNING("Entry MsprofRegTypeInfoStub");
    return 0;
}

int32_t MsprofReportApiStub(uint32_t agingFlag, const MsprofApi *api)
{
    HCCL_WARNING("Entry MsprofReportApiStub");
    return 0;
}

int32_t MsprofReportCompactInfoStub(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    HCCL_WARNING("Entry MsprofReportCompactInfoStub");
    return 0;
}

int32_t MsprofReportAdditionalInfoStub(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
{
    HCCL_WARNING("Entry MsprofReportAdditionalInfoStub");
    return 0;
}

uint64_t MsprofGetHashIdStub(const char *hashInfo, size_t length)
{
    HCCL_WARNING("Entry MsprofGetHashIdStub");
    return 0;
}

uint64_t MsprofSysCycleTimeStub()
{
    HCCL_WARNING("Entry MsprofSysCycleTimeStub");
    return 0;
}

void DlProfFunction::DlProfFunctionStubInit()
{
    dlMsprofRegisterCallback = (s32(*)(uint32_t, ProfCommandHandle))MsprofRegisterCallbackStub;
    dlMsprofRegTypeInfo = (s32(*)(uint16_t, uint32_t, const char *))MsprofRegTypeInfoStub;
    dlMsprofReportApi = (s32(*)(uint32_t, const MsprofApi *))MsprofReportApiStub;
    dlMsprofReportCompactInfo = (s32(*)(uint32_t, const VOID_PTR, uint32_t))MsprofReportCompactInfoStub;
    dlMsprofReportAdditionalInfo = (s32(*)(uint32_t, const VOID_PTR, uint32_t))MsprofReportAdditionalInfoStub;
    dlMsprofGetHashId = (uint64_t(*)(const char *, uint32_t))MsprofGetHashIdStub;
    dlMsprofSysCycleTime = (uint64_t(*)(void))MsprofSysCycleTimeStub;
}

HcclResult DlProfFunction::DlProfFunctionInterInit()
{
    dlMsprofRegisterCallback = (s32(*)(uint32_t, ProfCommandHandle))dlsym(handle_,
        "MsprofRegisterCallback");
    CHK_SMART_PTR_NULL(dlMsprofRegisterCallback);

    dlMsprofRegTypeInfo = (s32(*)(uint16_t, uint32_t, const char *))dlsym(handle_,
        "MsprofRegTypeInfo");
    CHK_SMART_PTR_NULL(dlMsprofRegTypeInfo);

    dlMsprofReportApi = (s32(*)(uint32_t, const MsprofApi *))dlsym(handle_,
        "MsprofReportApi");
    CHK_SMART_PTR_NULL(dlMsprofReportApi);

    dlMsprofReportCompactInfo = (s32(*)(uint32_t, const VOID_PTR, uint32_t))dlsym(handle_,
        "MsprofReportCompactInfo");
    CHK_SMART_PTR_NULL(dlMsprofReportCompactInfo);

    dlMsprofReportAdditionalInfo = (s32(*)(uint32_t, const VOID_PTR, uint32_t))dlsym(handle_,
        "MsprofReportAdditionalInfo");
    CHK_SMART_PTR_NULL(dlMsprofReportAdditionalInfo);

    dlMsprofGetHashId = (uint64_t(*)(const char *, uint32_t))dlsym(handle_,
        "MsprofGetHashId");
    CHK_SMART_PTR_NULL(dlMsprofGetHashId);

    dlMsprofSysCycleTime = (uint64_t(*)(void))dlsym(handle_,
        "MsprofSysCycleTime");
    CHK_SMART_PTR_NULL(dlMsprofSysCycleTime);

    return HCCL_SUCCESS;
}

HcclResult DlProfFunction::DlProfFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = dlopen("libprofapi.so", RTLD_NOW);
    }
    if (handle_ != nullptr) {
        CHK_RET(DlProfFunctionInterInit());
    }
    return HCCL_SUCCESS;
}
}
