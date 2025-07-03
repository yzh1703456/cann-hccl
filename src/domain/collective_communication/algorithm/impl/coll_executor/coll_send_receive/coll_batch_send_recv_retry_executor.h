/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_SEND_RECV_RETRY_EXECUTOR_H
#define COLL_BATCH_SEND_RECV_RETRY_EXECUTOR_H

#include "coll_comm_executor.h"
#include "coll_batch_send_recv_executor.h"

namespace hccl {
class CollBatchSendRecvRetryExecutor : public CollBatchSendRecvExecutor {
public:
    CollBatchSendRecvRetryExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBatchSendRecvRetryExecutor() = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    HcclResult CreatePairWiseList(HcclSendRecvItem *sendRecvInfo, u32 itemNum);
    virtual HcclResult GetPairWiseList(std::vector<std::vector<HcclSendRecvItem*>> &sendRecvPairList);
private:
    HcclResult CalcSendSlices(AlgResourceResponse& algRes, HcclSendRecvItem* sendItem);
    HcclResult CalcRecvSlices(AlgResourceResponse& algRes, HcclSendRecvItem* recvItem);
    HcclResult CheckSendRecvPair(const std::vector<HcclSendRecvItem*> &sendRecvPair);
    HcclResult RunLoop(OpParam &param, AlgResourceResponse &algRes, const std::vector<HcclSendRecvItem*> &sendRecvPair);
    HcclResult CalcStreamNum(u32& streamNum) override;
private:
    std::vector<HcclSendRecvItem*> sendDeque_;
    std::vector<HcclSendRecvItem*> recvDeque_;
    std::vector<std::vector<HcclSendRecvItem*>> sendRecvPairList_;
    HcclSendRecvType sendRecvType_;
};
} // namespace hccl

#endif