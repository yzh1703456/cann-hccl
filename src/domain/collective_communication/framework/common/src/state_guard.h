/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STATE_GUARD_H
#define STATE_GUARD_H

namespace hccl {

template <typename Resource, typename State>
class StateGuard {
public:
    StateGuard(Resource* resource, State targetState) : resource_(resource) {
        if (resource_ != nullptr) {
            initialState_ = resource_->GetState();  // 直接调用GetState
            resource_->SetState(targetState);   // 设置目标状态
        }
    }

    ~StateGuard() {
        if (resource_ != nullptr) {
            resource_->SetState(initialState_); // 恢复初始状态
        }
    }

private:
    Resource* resource_ = nullptr;
    State initialState_ = {};
};

} // namespace hccl

#endif  // STATE_GUARD_H
