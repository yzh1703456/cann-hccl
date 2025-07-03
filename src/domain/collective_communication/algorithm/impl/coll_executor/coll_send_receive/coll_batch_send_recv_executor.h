/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_SEND_RECV_EXECUTOR_H
#define COLL_BATCH_SEND_RECV_EXECUTOR_H

#include "coll_comm_executor.h"

namespace hccl {
class CollBatchSendRecvExecutor : public CollCommExecutor {
public:
    CollBatchSendRecvExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBatchSendRecvExecutor() = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    // 增量建链资源计算接口
    HcclResult CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest& resourceRequest) override;
protected:
    /* *************** 资源计算 *************** */
    void ParseParam(const OpParam& param) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */

    u64 CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize);
    u64 CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize);
    HcclResult ProcessSendDataSlice(Stream& stream, bool needStreamSync, bool retryEnable);
    HcclResult ProcessRecvDataSlice(Stream& stream, bool retryEnable);
    HcclResult CalcSendSlices(AlgResourceResponse& algRes);
    HcclResult CalcRecvSlices(AlgResourceResponse& algRes);
    struct SendRecvSlice {
        u8* addr;
        u64 size;
        u32 remoteRank;
        SendRecvSlice(u8* addr, u64 size, u32 remoteRank) : addr(addr), size(size), remoteRank(remoteRank) {}
    };

    u32 remoteUserRank_ = 0;
    const u32 MAX_LOOP_IN_ONCE_LAUNCH = 200;
    std::deque<SendRecvSlice> sendDataSilces_;
    std::deque<SendRecvSlice> recvDataSilces_;
private:
    HcclResult RunLoopInHostUnfoldMode(OpParam& param);
    HcclResult RunLoopInAicpuUnfoldMode(OpParam& param);
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult GetPairWiseList(HcclSendRecvItem *sendRecvInfo, u32 itemNum);
    HcclResult ProcessSelfSendRecvTasks(Stream& stream);
    
    HcclResult MainPostSubWait(Stream& mainStream, Stream& subStream);
    HcclResult SubPostMainWait(Stream& mainStream, Stream& subStream);
    HcclResult SendKernelRun(Stream& stream, ExecMem &execMem, u32 remoteUserRank, bool retryEnable);
    HcclResult RecvKernelRun(Stream& stream, ExecMem &execMem, u32 remoteUserRank, bool retryEnable);
    HcclResult GetTransport(u32 commIndex, u32 remoteUserRank, LINK &targetLink);

private:

    std::set<u32> commTargetUserRankSet_;
    std::deque<HcclSendRecvItem*> sendToSelfDeque_;
    std::deque<HcclSendRecvItem*> recvFromSelfDeque_;
    std::deque<HcclSendRecvItem*> sendDeque_;
    std::deque<HcclSendRecvItem*> recvDeque_;
};
} // namespace hccl

#endif