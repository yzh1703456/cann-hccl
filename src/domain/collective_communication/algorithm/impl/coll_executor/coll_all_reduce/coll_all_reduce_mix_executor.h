/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALL_REDUCE_MIX_EXECUTOR_H
#define COLL_ALL_REDUCE_MIX_EXECUTOR_H

#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceMixExecutor : public CollAllReduceExecutor {

public:
    CollAllReduceMixExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllReduceMixExecutor() = default;

private:
    /* *************** 资源计算 *************** */
    void ParseParam(const OpParam& param) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    bool IsHugeData(const u64 curSize) override;
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

    bool meshSinglePlane_ = false;
};

} // namespace hccl

#endif