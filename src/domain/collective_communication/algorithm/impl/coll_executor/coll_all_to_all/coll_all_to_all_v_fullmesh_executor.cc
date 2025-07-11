/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_all_to_all_v_fullmesh_executor.h"

namespace hccl {

CollRunAlltoAllVFullMesh::CollRunAlltoAllVFullMesh(const HcclDispatcher dispatcher,
                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollRunAlltoAllVFullMesh::CalcStreamNum(u32& streamNum)
{
    if (SatisfyIntraSuperPod(topoAttr_.deviceType, topoAttr_.userRankSize, topoAttr_.useSuperPodMode,
                             topoAttr_.superPodNum)) {
        streamNum = topoAttr_.userRankSize - 1;
    } else {
        streamNum = 0;
    }
    HCCL_INFO("[CollRunAlltoAllVFullMesh][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh
HcclResult CollRunAlltoAllVFullMesh::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVFullMesh::CalcLevel2CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}


HcclResult CollRunAlltoAllVFullMesh::CalAlltoAllFullMeshCommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
     // A+X单机双module启用下，未使能RDMA不能进行一层pairWise。
    bool isDifModule = topoAttr_.serverNum == 1 && topoAttr_.isDiffDeviceModule &&
        topoAttr_.userRankSize > HCCL_ALLTOALLV_P2P_SIZE;
    CHK_PRT_RET(isDifModule && !algoAttr_.isUsedRdmaLevel0,
        HCCL_ERROR("[CalAlltoAllFullMeshCommInfo] not support dual modules in a single server" \
                   " when RDMA disabled "), HCCL_E_NOT_SUPPORT);

    // 将网卡初始化判断，提到上层调用，减少无必要的循环依赖。
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        // level0 - level1 全连接通信域
        CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
        // level2 层通信域
        CHK_RET(CalcLevel2CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));
    } else {
        // level0 - level1 全连接通信域
        CHK_RET(CalcLevel0CommInfo(TransportMemType::PARAM_INPUT, TransportMemType::PARAM_OUTPUT, opTransport));
        // level2 层通信域
        CHK_RET(CalcLevel2CommInfo(TransportMemType::PARAM_INPUT, TransportMemType::PARAM_OUTPUT, opTransport));
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVFullMesh::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalAlltoAllFullMeshCommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllVFullMesh::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollRunAlltoAllVFullMesh][KernelRun] alltoall fullmesh start");
    bool opbaseCopyMode = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_;

    // 构造入参
    AlltoAllVBufferInfo sendInfo;
    sendInfo.mem = opbaseCopyMode ? execMem.inputMem : algResResp_->paramInputMem;
    sendInfo.counts = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].sendCounts[0];
    sendInfo.displs = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].sendDispls[0];
    sendInfo.dataType = param.All2AllDataDes.sendType;

    AlltoAllVBufferInfo recvInfo;
    recvInfo.mem = opbaseCopyMode ? execMem.outputMem : algResResp_->paramOutputMem;
    recvInfo.counts = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].recvCounts[0];
    recvInfo.displs = &allMeshAggregationSendRecvInfo_[topoAttr_.userRank].recvDispls[0];
    recvInfo.dataType = param.All2AllDataDes.recvType;

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    bool usePairwiseAlgorithm =
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_NA &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE;
    if (SatisfyIntraSuperPod(topoAttr_.deviceType, topoAttr_.userRankSize,
        topoAttr_.useSuperPodMode, topoAttr_.superPodNum) && !usePairwiseAlgorithm) {
        HCCL_INFO("[CollRunAlltoAllVFullMesh] one level read only algo");

        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
            CHK_RET(ActiveSlaveStreams(param.stream));
        }
        CHK_RET(AddSubStreamToProfiling());

        std::unique_ptr<AlgTemplateBase> alltoallReadOnly = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_MESH_READ_ONLY, dispatcher_);
        CHK_SMART_PTR_NULL(alltoallReadOnly);

        AlltoAllUserRankInfo userRankInfo;
        userRankInfo.userRank = topoAttr_.userRank;
        userRankInfo.userRankSize = topoAttr_.userRankSize;
        std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosIntra;
        std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosIntra;
        CalcIntraMeshAggregationAlltoAllMemInfo(userRankInfo, allMeshAggregationSendRecvInfo_, sendAddrInfosIntra,
            recvAddrInfosIntra, topoAttr_.userRankSize, true);
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(alltoallReadOnly->Prepare(algResResp_->paramInputMem, algResResp_->paramOutputMem,
                execMem.inputMem, execMem.outputMem, sendAddrInfosIntra, recvAddrInfosIntra, workflowMode_,
                const_cast<Stream&>(param.stream), algResResp_->slaveStreams,
                algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank,
                topoAttr_.userRankSize, level0CommInfo.links, allMeshAggregationSendRecvInfo_));
        } else {
            CHK_RET(alltoallReadOnly->Prepare(algResResp_->paramInputMem, algResResp_->paramOutputMem,
                algResResp_->paramInputMem, algResResp_->paramOutputMem, sendAddrInfosIntra,
                recvAddrInfosIntra, workflowMode_, const_cast<Stream&>(param.stream), algResResp_->slaveStreams,
                algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank,
                topoAttr_.userRankSize, level0CommInfo.links, allMeshAggregationSendRecvInfo_));
        }
        CHK_RET(alltoallReadOnly->RunAsync());
        return HCCL_SUCCESS;
    }

    // 执行算法
    std::map<u32, std::vector<u64>> rankSendDisplsMap;
    std::map<u32, std::vector<u64>> rankRecvDisplsMap;
    if (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || isAlltoAllZCopyMode_) {
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            rankSendDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo_[i].sendOffset));
            rankRecvDisplsMap.insert(std::pair<u32, std::vector<u64>>(i, allMeshAggregationSendRecvInfo_[i].recvOffset));
        }
    }

    std::unique_ptr<AlgTemplateBase> pairWisePtr = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_PAIRWISE, dispatcher_);
    CHK_SMART_PTR_NULL(pairWisePtr);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !isAlltoAllZCopyMode_) { // 单算子 && Buffer Copy模式
        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, execMem.inputMem, execMem.outputMem, isAlltoAllZCopyMode_,
            const_cast<Stream&>(param.stream), workflowMode_, rankSendDisplsMap, rankRecvDisplsMap));
        CHK_RET(RunAlltoAllTemplate(pairWisePtr, level0CommInfo));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        isAlltoAllZCopyMode_) {
        DeviceMem dstMem = execMem.inputMem.range(0, algResResp_->paramInputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, algResResp_->paramInputMem, const_cast<Stream&>(param.stream)));

        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, execMem.inputMem, execMem.outputMem,
            isAlltoAllZCopyMode_, const_cast<Stream&>(param.stream),
            workflowMode_, rankSendDisplsMap, rankRecvDisplsMap));
        CHK_RET(RunAlltoAllTemplate(pairWisePtr, level0CommInfo)); // inputMem_ -> outputMem_

        DeviceMem srcMem = execMem.outputMem.range(0, algResResp_->paramOutputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, algResResp_->paramOutputMem, srcMem, const_cast<Stream&>(param.stream)));
    } else if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        CHK_RET(pairWisePtr->Prepare(sendInfo, recvInfo, isAlltoAllZCopyMode_, const_cast<Stream&>(param.stream),
        workflowMode_, rankSendDisplsMap, rankRecvDisplsMap));
        // 保证最新的commMesh是为该次alltoallv创建（不支持多线程）
        CHK_RET(RunAlltoAllTemplate(pairWisePtr, level0CommInfo));
    } else {
        HCCL_ERROR("[hcclImpl][RunAlltoAllVFullMesh]work flow mode is invalid");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[CollRunAlltoAllVFullMesh] excutor run success");

    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllVFullMesh", AlltoAllVFullMesh, CollRunAlltoAllVFullMesh);
} // namespace hccl