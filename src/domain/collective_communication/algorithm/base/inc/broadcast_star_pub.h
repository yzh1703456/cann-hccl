/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_STAR_PUB_H
#define BROADCAST_STAR_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class BroadcastStar : public AlgTemplateBase {
public:
    explicit BroadcastStar(const HcclDispatcher dispatcher);
    ~BroadcastStar() override;

    HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, 
        const u64 count, const HcclDataType dataType, const Stream &stream, 
        const HcclReduceOp reductionOp, const u32 root, 
        const std::vector<Slice> &slices, const u64 baseOffset, 
        std::vector<u32> nicRankList, u32 userRank) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport>> &links) override;
protected:
private:
    HcclResult RunRecvBroadcast(const u32 srcRank, const u32 dstRank, const Slice &slice,
        const std::vector<std::shared_ptr<Transport>> &links);
    HcclResult RunSendBroadcast(const u32 dstRank, const Slice &slice,
        const std::vector<std::shared_ptr<Transport>> &links);
    HcclResult ExecuteBarrierSrcRank(std::shared_ptr<Transport> link, Stream &stream) const;

    u32 userRank_ = 0;
};
} // namespace hccl

#endif /* BROADCAST_STAR_H */