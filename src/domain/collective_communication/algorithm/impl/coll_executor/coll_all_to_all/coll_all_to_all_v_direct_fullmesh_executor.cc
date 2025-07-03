/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "coll_all_to_all_v_direct_fullmesh_executor.h"

namespace hccl {

CollRunAlltoAllDirectFullmesh::CollRunAlltoAllDirectFullmesh(const HcclDispatcher dispatcher,
                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollRunAlltoAllDirectFullmesh::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;

    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType_);

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollRunAlltoAllDirectFullmesh][Orchestrate]errNo[0x%016llx]excutor run failed",
            HCCL_ERROR_CODE(ret)), ret);

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());

    HCCL_INFO("tag[%s], AlltoAllDirectFullmesh tempAlg orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclOpMetaInfo CollRunAlltoAllDirectFullmesh::GetOpMeta(HcclCMDType opType, const u64 size)
{
    (void)opType;
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, size, true);
    return opMeta;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSDMAGroupInfo(const u32 userRank,
    u32& devNumInlocalPod, u32& rankIdxInPod)
{
    (void) userRank;
    if (topoMatcher_->GetExternalInputInterHccsDisable()) {
        CHK_RET(topoMatcher_->GetLocalServerRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    } else {
        CHK_RET(topoMatcher_->GetLocalSuperPodRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    }
    CHK_PRT_RET(devNumInlocalPod == INVALID_VALUE_RANKSIZE,
        HCCL_ERROR("[CollRunAlltoAllDirectFullmesh][GetLocalSDMAGroupInfo]get local superPod total ranksize failed."),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcStreamNum(u32& streamNum)
{
    // 每个超节点内的卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    // 单超节点场景需要的从流数量
    streamNum = (devNumInlocalPod > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
        (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE * RANK_SET_COMPUTE_CONST) : (devNumInlocalPod * RANK_SET_COMPUTE_CONST);

    // 多超节点场景下，RDMA会设置独立的并发度
    if ((topoAttr_.userRankSize - devNumInlocalPod) > 0) {
        streamNum += 1; // 一条从流专门用来管理超节点间的RDMA通信
        u32 totalRdmaRankNum = topoAttr_.userRankSize - devNumInlocalPod;
        streamNum += (totalRdmaRankNum > ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) ?
            (ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) : (totalRdmaRankNum);
    }

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh
// 超节点内建SDMA链路；超节点间建RDMA链路
HcclResult CollRunAlltoAllDirectFullmesh::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));

    LevelNSubCommTransport &commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto &transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 - level1 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallV(const OpParam &param)
{
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64 *>(param.All2AllDataDes.sendCounts) + j);
        u64 curSendDispls = *(static_cast<const u64 *>(param.All2AllDataDes.sdispls) + j);
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendOffset[j] = curSendDispls * SIZE_TABLE[param.All2AllDataDes.sendType];

        u64 curRecvCounts = *(static_cast<const u64 *>(param.All2AllDataDes.recvCounts) + j);
        u64 curRecvDispls = *(static_cast<const u64 *>(param.All2AllDataDes.rdispls) + j);
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvOffset[j] = curRecvDispls * SIZE_TABLE[param.All2AllDataDes.recvType];

        HCCL_DEBUG("GetLocalSendRecvInfoforAlltoallV rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu]", topoAttr_.userRank, localSendRecvInfo_.sendCounts[j],
            localSendRecvInfo_.sendDispls[j], localSendRecvInfo_.recvCounts[j],
            localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoall(const OpParam &param)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = param.All2AllDataDes.sendCount;
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendLength;
        localSendRecvInfo_.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = param.All2AllDataDes.sendCount;
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvLength;
        localSendRecvInfo_.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG("GetLocalSendRecvInfoforAlltoall rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu]", topoAttr_.userRank, localSendRecvInfo_.sendCounts[j],
            localSendRecvInfo_.sendDispls[j], localSendRecvInfo_.recvCounts[j],
            localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallVC(const OpParam &param)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    u64 rankSize = topoAttr_.userRankSize;
    u64 usrRank = topoAttr_.userRank;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + usrRank * rankSize + j);
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendLength;
        localSendRecvInfo_.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + usrRank + rankSize * j);
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvLength;
        localSendRecvInfo_.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG("GetLocalSendRecvInfoforAlltoallVC rank[%u], sendCounts[%llu], sendDispls[%llu] "\
            "recvCounts[%llu], recvDispls[%llu]", topoAttr_.userRank, localSendRecvInfo_.sendCounts[j],
            localSendRecvInfo_.sendDispls[j], localSendRecvInfo_.recvCounts[j],
            localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetAlltoAllvTmpRankSendRecvInfo(const OpParam &param)
{
    localSendRecvInfo_.sendCounts.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendDispls.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendLength.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendOffset.resize(topoAttr_.userRankSize, 0);

    localSendRecvInfo_.recvCounts.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvDispls.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvLength.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvOffset.resize(topoAttr_.userRankSize, 0);
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallV(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(GetLocalSendRecvInfoforAlltoall(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallVC(param));
    } else {
        HCCL_ERROR("Only support optype alltoall , alltoallv and alltoallvc !");
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][KernelRun] alltoall fullmesh start.");

    // 准备数据
    CHK_RET(GetAlltoAllvTmpRankSendRecvInfo(param));

    // 获取当前超节点内总卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);

    CHK_RET(AddSubStreamToProfiling());

    bool isSuPodAsym = false;
    if (topoAttr_.superPodNum > 1) {
        isSuPodAsym = (topoAttr_.multiModuleDiffDeviceNumMode || topoAttr_.multiSuperPodDiffServerNumMode);
    } else {
        isSuPodAsym = topoMatcher_->GetExternalInputInterHccsDisable() && topoAttr_.multiModuleDiffDeviceNumMode;
    }

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);

    CHK_SMART_PTR_NULL(tempAlg);

    PrepareData prepareData;
    prepareData.stream = param.stream;
    prepareData.userRank = topoAttr_.userRank;
    prepareData.userRankSize = topoAttr_.userRankSize;
    prepareData.linksPtr = &level0CommInfo.links;
    prepareData.localSendRecvInfoPtr = &localSendRecvInfo_;
    prepareData.devNumInlocalPod = devNumInlocalPod;
    prepareData.rankIdxInPod = rankIdxInPod;

    prepareData.inputMem = algResResp_->paramInputMem;
    prepareData.outputMem = algResResp_->paramOutputMem;
    prepareData.cclInMem = execMem.inputMem;
    prepareData.cclOutMem = execMem.outputMem;
    prepareData.workMode = workflowMode_;
    prepareData.subStreamsPtr = &algResResp_->slaveStreams;
    prepareData.signalPtr = &algResResp_->notifiesMain;
    prepareData.signalAuxPtr = &algResResp_->notifiesAux;
    prepareData.isSuPodAsym = isSuPodAsym;
    prepareData.opType = param.opType;
    prepareData.algOpContext = algOpContext_;

    CHK_RET(tempAlg->Prepare(prepareData));

    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] excutor run success.");
    if (algOpContext_.opRetryHandler.isPostSync == true) {
        OpParam postSyncParam = param;
        CHK_RET(InplaceOpSync(postSyncParam, execMem));
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("RunAlltoAllDirectFullmesh", AlltoAllVDirectFullMesh, CollRunAlltoAllDirectFullmesh);
} // namespace hccl