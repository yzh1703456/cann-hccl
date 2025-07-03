/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_MIX_EXECUTOR_H
#define COLL_REDUCESCATTER_MIX_EXECUTOR_H
#include "coll_reduce_scatter_executor.h"
namespace hccl {
class CollReduceScatterMixExecutor : public CollReduceScatterExecutor {
public:
    explicit CollReduceScatterMixExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollReduceScatterMixExecutor() = default;

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u32 unitSize) override;
    bool IsHugeData(const u64 curSize, OpParam *param = nullptr) override;
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    /* **************** 数据准备*************** */
    void FillMultiRingSlice(const ExecMem &execMem, const std::vector<std::vector<Slice>> &multiStreamSlice,
        u32 sliceNum, u32 level1RankSize, const u32 ringIndex, std::vector<Slice> &dataSlice);
    void CalLevel0DataSegsSlice(const ExecMem &execMem, const std::vector<std::vector<Slice>> &multiStreamSlice,
        u32 sliceNum, u32 level1RankSize, std::vector<std::vector<Slice>> &level0DataSegsSlice);
    HcclResult CalLevel1DataSegsSlice(const ExecMem &execMem, const u32 &commIndex,
        u32 sliceNum, u32 level1RankSize, std::vector<Slice> &level1DataSegsSlice);

    bool meshSinglePlane_ = false;
};

} // namespace hccl

#endif
