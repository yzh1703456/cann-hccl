/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_MESH_MIX_PUB_H
#define REDUCE_SCATTER_MESH_MIX_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class ReduceScatterMeshMix : public AlgTemplateBase {
public:
    explicit ReduceScatterMeshMix(const HcclDispatcher dispatcher);
    ~ReduceScatterMeshMix() override;
    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, 
        const u64 count, const HcclDataType dataType, const Stream &stream, 
        const HcclReduceOp reductionOp, const u32 root, 
        const std::vector<Slice> &slices, const u64 baseOffset, 
        const u64 reduceAttrBitMap, std::vector<Stream> &meshStreams, 
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignal, 
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, 
        u32 interRank, u32 interRankSize, HcomCollOpInfo *opInfo) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();

    u64 reduceAttr_ = 0;
    std::vector<Stream> meshStreams_;         /* * 多steam* */
    const std::vector<std::shared_ptr<LocalNotify>> *meshSignalPtr_{nullptr};    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>> *meshSignalAuxPtr_{nullptr}; /* 从stream wait，主steam record */
    u32 interRank_ = 0;
    u32 interRankSize_ = 0;
    HcomCollOpInfo *opInfo_{nullptr};
    std::vector<Slice> scratchSlices_;
};
} // namespace hccl

#endif /* REDUCE_SCATTER_MESH_MIX_PUB_H */
