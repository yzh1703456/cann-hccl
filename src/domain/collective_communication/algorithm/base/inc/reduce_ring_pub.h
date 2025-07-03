/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_RING_PUB_H
#define REDUCE_RING_PUB_H

#include "alg_template_base_pub.h"
#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class ReduceRing : public AlgTemplateBase {
public:
    using AlgTemplateBase::Prepare;
    explicit ReduceRing(const HcclDispatcher dispatcher);

    ~ReduceRing() override;

    /* 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能,后续还要调用其它的基类Prepare函数实现其它成员变量初始化 */
    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo *opInfo = nullptr) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
        const std::vector<std::shared_ptr<Transport> > &links) override;

protected:
private:
    std::unique_ptr<Sender> senderInfo_;
    std::unique_ptr<Reducer> reducerInfo_;
    std::shared_ptr<Transport> linkLeft_;
    std::shared_ptr<Transport> linkRight_;

    DeviceMem scratch_; /* * 临时Device/Host memory(本层output或下层操作的结果) */

    u64 reduceAttr_;       /* 0x1:表示data_type + reduce_type支持inlinereduce  */
};
}  // namespace hccl

#endif /* REDUCE_RING_PUB_H */
