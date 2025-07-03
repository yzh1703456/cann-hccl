/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCEPTION_HANDLER_H
#define EXCEPTION_HANDLER_H

#include <exception>
#include <stdexcept>
#include <string>
#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "log.h"

namespace hccl {

// 宏定义，用于包装 C 接口函数的异常处理
#define EXCEPTION_HANDLE_BEGIN try {
#define EXCEPTION_HANDLE_END_INFO(func_name) } catch (...) { \
    return ExceptionHandler::HandleException(func_name); }

#define EXCEPTION_HANDLE_END EXCEPTION_HANDLE_END_INFO(__func__)

#define EXCEPTION_THROW_IF_ERR(call, errString) \
    do { \
        HcclResult ret = call; \
        ExceptionHandler::ThrowIfErrorCode(ret, errString, __FILE__, __LINE__, __func__); \
    } while (0)

#define EXCEPTION_THROW_IF_COND_ERR(condition, errString) \
    do { \
        if (condition) { \
            HcclResult ret = HCCL_E_INTERNAL; \
            ExceptionHandler::ThrowIfErrorCode(ret, errString, __FILE__, __LINE__, __func__); \
        } \
    } while (0)

// 异常处理器类
class ExceptionHandler {
public:
    static HcclResult HandleException(const char* functionName);

    static void ThrowIfErrorCode(HcclResult errorCode, const std::string &errString, const char* fileName,
        s32 lineNum, const char* functionName);
};
}

#endif
