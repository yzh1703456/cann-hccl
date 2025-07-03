/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RUN_ALLTOALL_STAGED_AIV_RDMA_EXECUTOR_H
#define COLL_RUN_ALLTOALL_STAGED_AIV_RDMA_EXECUTOR_H
#include "coll_all_to_all_executor.h"
#include "hccl_aiv.h"

namespace hccl {
class CollRunAlltoAllStagedAivRdmaExecutor : public CollAlltoAllExecutor {

public:
    CollRunAlltoAllStagedAivRdmaExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollRunAlltoAllStagedAivRdmaExecutor() = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;

private:
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    HcclResult GetIfNeedAivBuffer(bool &needAivBuffer) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

    HcclResult RunAlltoAllStaged1InAIV(const OpParam &param, ExecMem &execMem);
    HcclResult PrepareAivBuffers(DeviceMem &inputMem, DeviceMem &outputMem, void **dataBuffers, void **flagBuffers);    
    HcclResult RunAlltoAllStaged2(const OpParam &param, ExecMem &execMem);
    void CalcInterMeshAggregationAlltoAllMemInfo(const OpParam &param, 
        std::map<u32, std::list<OneSendRecvAddrInfo>> &sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>> &recvAddrInfosInter);

    /* *************** 算法参数 *************** */
    u32 sendDataSize_ = 0;
    u32 recvDataSize_ = 0;
    SubCommInfo innerCommInfo_;
    SubCommInfo outerCommInfo_;
};

} // namespace hccl

#endif