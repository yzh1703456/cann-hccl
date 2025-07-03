/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exception_handler.h"

namespace hccl {
using namespace std;

HcclResult ExceptionHandler::HandleException(const char* functionName)
{
    try {
        // 重新抛出当前捕获的异常
        throw;
    } catch (const out_of_range& e) {
        HCCL_ERROR("%s: Out of range error, what: %s", functionName, e.what());
        return HCCL_E_NOT_FOUND;
    } catch (const runtime_error& e) {
        HCCL_ERROR("%s: Runtime error, what: %s", functionName, e.what());
        return HCCL_E_RUNTIME;
    } catch (const logic_error& e) {
        HCCL_ERROR("%s: Logic error, what: %s", functionName, e.what());
        return HCCL_E_INTERNAL;
    } catch (const exception& e) {
        HCCL_ERROR("%s: Standard exception, what: %s", functionName, e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("%s: Unknown exception", functionName);
        return HCCL_E_INTERNAL;
    }
}

void ExceptionHandler::ThrowIfErrorCode(HcclResult errorCode, const string &errString, const char* fileName,
    s32 lineNum, const char* functionName)
{
    if (LIKELY(errorCode == HCCL_SUCCESS)) {
        return;
    }

    string prefix = string(fileName) + ":" + to_string(lineNum) + "," + string(functionName);
    string suffix = "ret=" + to_string(errorCode) + " " + errString;

    switch (errorCode) {
        case HCCL_E_NOT_FOUND:
            throw out_of_range(prefix + ", Error: Out of range, " + suffix);
            break;
        case HCCL_E_INTERNAL:
            throw logic_error(prefix + ", Error: Logic error, " + suffix);
            break;
        case HCCL_E_RUNTIME:
            throw runtime_error(prefix + ", Error: Runtime error, " + suffix);
            break;
        default:
            throw runtime_error(prefix + ", Error: Default error code, " + suffix);
            break;
    }
}

}