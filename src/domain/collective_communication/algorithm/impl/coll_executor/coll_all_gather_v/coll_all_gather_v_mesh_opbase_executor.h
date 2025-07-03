/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHERV_MESH_OPBASE_EXECUTOR_H
#define COLL_ALLGATHERV_MESH_OPBASE_EXECUTOR_H
#include "coll_all_gather_v_executor.h"
namespace hccl {
class CollAllGatherVMeshOpbaseExecutor : public CollAllGatherVExecutor {
public:
    explicit CollAllGatherVMeshOpbaseExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllGatherVMeshOpbaseExecutor() = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    HcclResult CalcCurCountsAndCurDispls(const u64 maxTotalCount, std::vector<u64> &countsLeft,
        std::vector<u64> &displs, std::vector<u64> &curCounts, std::vector<u64> &curDispls, bool &finished) override;
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    bool IsHugeData(const u64 curSize) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem);
};

} // namespace hccl

#endif