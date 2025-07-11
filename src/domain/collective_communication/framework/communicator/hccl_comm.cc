/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <algorithm>
#include <arpa/inet.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <hccl/hccl_types.h>
#include "device_capacity.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "task_abort_handler_pub.h"
#include "coll_alg_utils.h"
#include "env_config.h"
#include "i_hccl_one_sided_service.h"

namespace hccl {
RankTable_t g_hcclDefaultRankTable;

hcclComm::hcclComm(u64 inCCLbufferSize, u64 outCCLbufferSize, std::string identifier)
    : barrierSendBuf(nullptr), barrierRecvBuf(nullptr),
      inCCLbufferSize_(inCCLbufferSize), outCCLbufferSize_(outCCLbufferSize),
      deviceType_(DevType::DEV_TYPE_COUNT), isFirstBarrier_(true), identifier_(identifier), isHeterogComm_(false),
      isResetDevice_(false), isSpecialType_(false), communicator_(nullptr)
{
    indirectInCCLbuffer_ = DeviceMem();
    indirectOutCCLbuffer_ = DeviceMem();
    barrierInMemory_ = DeviceMem();
    barrierOutMemory_ = DeviceMem();
}

hcclComm::~hcclComm()
{
    RealeaseBarrierMemory();
    (void)UnRegistTaskAbortHandler();
    communicator_ = nullptr;
}

HcclResult hcclComm::ReleaseSubComms() const
{
    CHK_SMART_PTR_NULL(communicator_);

    CHK_RET(communicator_->ReleaseCommInfos());

    return HCCL_SUCCESS;
}

void hcclComm::ReleaseCommCCLbuffer() const
{
    if (!communicator_) {
        return;
    }
    communicator_->ReleaseCommCCLbuffer();
}

void hcclComm::RealeaseBarrierMemory()
{
    barrierInMemory_.free();
    barrierOutMemory_.free();
}

void hcclComm::UpdateIsHaveCpuRank(const RankTable_t &rankTable)
{
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankTable.rankList[i].deviceInfo.devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
    }
}

void hcclComm::UpdateIsHaveCpuRank(const std::vector<RankInfo> &rankList)
{
    for (u32 i = 0; i < rankList.size(); i++) {
        // 同一server的标识IP 是一样的，所以可以以此推算出平均dev个数
        if (rankList[i].devicePhyId == HOST_DEVICE_ID) {
            isHaveCpuRank_ = true;
        }
    }
}

HcclResult hcclComm::init(HcclCommParams &params, const RankTable_t &rankTable)
{
    UpdateIsHaveCpuRank(rankTable);
    isHeterogComm_ = params.isHeterogComm;

    HCCL_INFO("hcclComm init workmode [%d]", params.commWorkMode);
    if (params.commWorkMode == WorkMode::HCCL_MODE_AI_CPU) {
        isSpecialType_ = true;
    }
    CHK_RET(InitImpl(params.deviceType));

    /* 强行将最后一个字符置0, 确保其可以做字符串操作 */
    params.id.internal[HCCL_ROOT_INFO_BYTES - 1] = '\0';

    /* 入参判断 */
    if (params.rank >= params.totalRanks) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] rank[%u] out of range[0, %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            params.rank, params.totalRanks - 1);
        return HCCL_E_PARA;
    }
    params.identifier = identifier_;

    CHK_RET(communicator_->AtomicInitSet());                  /* 初始化竞争, 只允许被初始化一次 */
    HcclResult ret = communicator_->Init(params, rankTable);  /* 初始化实例, 失败则重新开放初始化竞争 */
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] hccl initialize failed", HCCL_ERROR_CODE(ret));
        communicator_->AtomicInitClear();
        return ret;
    }

    if (params.totalRanks != 1 ) {
        CHK_RET(communicator_->InitCCLbuffer(inCCLbufferSize_, outCCLbufferSize_));
    }

    HCCL_USER_CRITICAL_LOG("hcclCommInitInfo:commId[%s], rank[%u], totalRanks[%u], serverId[%s], deviceType[%d]," \
        "logicDevId[%d], identifier[%s]", params.id.internal, params.rank, params.totalRanks, params.serverId.c_str(),
        params.deviceType, params.logicDevId, params.identifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
                          WorldGroupInfo &groupCommonData)
{
    UpdateIsHaveCpuRank(rankList);
    /* 强行将最后一个字符置0, 确保其可以做字符串操作 */
    params.id.internal[HCCL_ROOT_INFO_BYTES - 1] = '\0';

    HCCL_USER_CRITICAL_LOG("rootInfo[%s], rank[%u], totalRanks[%u], chip[%d], logicDevId[%d]", params.id.internal,
        params.rank, params.totalRanks, params.deviceType, params.logicDevId);

    HCCL_INFO("rootInfo init group workmode[%d]", params.commWorkMode);
    if (params.commWorkMode == WorkMode::HCCL_MODE_AI_CPU) {
        isSpecialType_ = true;
    }
    CHK_RET(InitImpl(params.deviceType));

    /* 入参判断 */
    if (params.rank >= params.totalRanks) {
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] rank[%u] out of range[0, %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            params.rank, params.totalRanks - 1);
        return HCCL_E_PARA;
    }

    params.identifier = identifier_;

    CHK_RET(communicator_->CheckDeviceType(params.deviceType));                /* 芯片类型检查 */
    CHK_RET(communicator_->AtomicInitSet());                                 /* 初始化竞争, 只允许被初始化一次 */
    HcclResult ret = communicator_->Init(params, rankList, groupCommonData); /* 初始化实例, 失败则重新开放初始化竞争 */
    if (ret != HCCL_SUCCESS) {
        communicator_->AtomicInitClear();
        HCCL_ERROR("[HcclComm][Init]errNo[0x%016llx] hccl initialize failed", HCCL_ERROR_CODE(ret));
        return ret;
    }
    return ret;
}

HcclResult hcclComm::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
{
    // 校验config中TC的合法性
    if (trafficClass == HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET && serviceLevel == HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET) {
        HCCL_INFO("[SetQpQosAttr]The TC and SL do not use the config configuration. " \
            "It will use environment variables to configure. TC[%u], SL[%u]",
            EnvConfig::GetExternalInputRdmaTrafficClass(), EnvConfig::GetExternalInputRdmaServerLevel());
        return HCCL_SUCCESS;
    } else if (trafficClass != HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET && serviceLevel == HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET) {
        serviceLevel = EnvConfig::GetExternalInputRdmaServerLevel();
        HCCL_INFO("[SetQpQosAttr]The SL is not configured. It will use the environment value[%u]", serviceLevel);
    } else if (trafficClass == HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET && serviceLevel != HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET) {
        trafficClass = EnvConfig::GetExternalInputRdmaTrafficClass();
        HCCL_INFO("[SetQpQosAttr]The TC is not configured. It will use the environment value[%u]", trafficClass);
    }

    // 若转换出错或者设置的RDMATrafficClass不在有效范围内，则报错
    if (trafficClass < EnvConfig::HCCL_RDMA_TC_MIN || trafficClass > EnvConfig::HCCL_RDMA_TC_MAX) {
        HCCL_ERROR("[SetQpQosAttr]rdmaTrafficClass is invalid. except:[%u, %u], actual:[%u]",
            EnvConfig::HCCL_RDMA_TC_MIN, EnvConfig::HCCL_RDMA_TC_MAX, trafficClass);
        return HCCL_E_PARA;
    }
    // 若设置的RDMATrafficClass不是4的整数倍，则报错
    if (trafficClass % EnvConfig::HCCL_RDMA_TC_BASE != 0) {
        HCCL_ERROR("[SetQpQosAttr]rdmaTrafficClass[%u] is not a multiple of [%u]",
            trafficClass, EnvConfig::HCCL_RDMA_TC_BASE);
        return HCCL_E_PARA;
    }

    // 校验config中SL是否合法
    if (serviceLevel < EnvConfig::HCCL_RDMA_SL_MIN || serviceLevel > EnvConfig::HCCL_RDMA_SL_MAX) {
        HCCL_ERROR("[SetQpQosAttr]rdmaServiceLevel is invalid. except:[%u, %u], actual:[%u]",
            EnvConfig::HCCL_RDMA_SL_MIN, EnvConfig::HCCL_RDMA_SL_MAX, serviceLevel);
        return HCCL_E_PARA;
    }

    HCCL_INFO("[SetQpQosAttr] rdmaTrafficClass[%u], rdmaServiceLevel[%u]", trafficClass, serviceLevel);
    communicator_->SetQpQosAttr(trafficClass, serviceLevel);

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateGroup(const std::string &group, const u32 &groupRank, const u32 &userRank,
                                 const std::vector<u32> &groupRanks, std::shared_ptr<hcclComm> &groupComm)
{
    // 增加输出日志关键字
    HCCL_INFO("HCCL_KEY_INFO: group[%s], groupRank[%u], userRank[%u], groupComm[%p]", group.c_str(),
        groupRank, userRank, groupComm.get());

    // 入参有消息校验
    if (group.length() == 0) {
        HCCL_ERROR("[Create][Group]errNo[0x%016llx] group name lenth is 0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (groupRank >= groupRanks.size()) {
        HCCL_ERROR("[Create][Group]errNo[0x%016llx] group rank[%u] out of range [0,%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), groupRank, groupRanks.size() - 1);
        return HCCL_E_PARA;
    }

    HcclRootInfo id;
    CHK_RET(GetUniqueId(&id));

    HcclCommParams params;
    params.rank = groupRank;
    params.userRank = userRank;
    params.totalRanks = groupRanks.size();
    params.isHeterogComm = isHeterogComm_;
    s32 iret = snprintf_s(params.id.internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s%s",
                          id.internal, "-", group.c_str());

    CHK_PRT_RET((iret == -1), HCCL_ERROR("[Create][Group]errNo[0x%016llx] get group unique id falied",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);

    WorldGroupInfo groupCommonData;

    CHK_RET(communicator_->GetGroupCommonData(groupCommonData));
    params.logicDevId = groupCommonData.deviceLogicId;
    params.profilingInitiated = groupCommonData.profilingInitiated;
    params.deviceType = deviceType_;
    params.hcomGroupNicInit = communicator_->GetNicInitialized();
    std::vector<RankInfo> rankList;

    CHK_RET(communicator_->GetGroupRanksInfo(groupRanks, rankList));

    groupComm.reset(new (std::nothrow) hccl::hcclComm(0, 0, group));
    CHK_SMART_PTR_NULL(groupComm);
    CHK_RET(groupComm->init(params, rankList, groupCommonData));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::DestroyGroup(const std::string &group) const
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("start destroy group: group[%s]", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("algType[%s]", AlgTypeToStr(algType).c_str());
    return communicator_->GetAlgType(algType, opType);
}

void hcclComm::PrintSubmittedOpCnt(const std::string &tag, HcclResult ret)
{
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);
    HCCL_ERROR("[HcclComm][%s]errNo[0x%016llx] index[%u]", tag.c_str(), ret, index);
}

HcclResult hcclComm::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
                               HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], count[%llu], data_type[%s]", tag.c_str(), inputPtr,
        GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AllGather]errNo[0x%016llx] all gather tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(inputCount));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    HcclResult ret = communicator_->AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream, opInfo);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AllGatherV(const std::string &tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], count[%llu], data_type[%s]", tag.c_str(), sendCount,
        GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AllGatherV]errNo[0x%016llx] all gather tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(dataType, false));
    HcclResult ret = communicator_->AllGatherV(tag, sendBuf, sendCount, recvBuf, recvCounts, rdispls, dataType, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
        tag.c_str(), inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    HcclResult ret = communicator_->AllGatherOutPlace(tag, inputPtr, outputPtr, inputCount, dataType, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AllGatherVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, 
    u64 inputCount, const void *outputCounts, const void *outputDispls, HcclDataType dataType, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], counts[%llu], data_type[%d]",
        tag.c_str(), inputPtr, outputPtr, outputCounts, dataType);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->AllGatherVOutPlace(tag, inputPtr, outputPtr, inputCount, outputCounts, outputDispls,
        dataType, stream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, SyncMode syncMode)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
               tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
               GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AllReduce]errNo[0x%016llx] all reduce tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    HcclResult ret = communicator_->AllReduce(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}


HcclResult hcclComm::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, SyncMode syncMode)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]", tag.c_str(),
        inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    HcclResult ret = communicator_->AllReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream, syncMode);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                               const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                               rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendCounts);
    CHK_PTR_NULL(sdispls);
    CHK_PTR_NULL(recvCounts);
    CHK_PTR_NULL(rdispls);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AlltoAllV]errNo[0x%016llx] alltoallv tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(sendType, false));

    CHK_RET(communicator_->CheckDataType(recvType, false));

    HcclResult ret = communicator_->AlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType,
        stream, tag);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(sendType, false));
    CHK_RET(communicator_->CheckDataType(recvType, false));

    HcclResult ret = communicator_->AlltoAllVOutPlace(
        sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, stream, tag);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
                                const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(sendCountMatrix);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AlltoAllVC]errNo[0x%016llx] alltoallvc tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(sendType, false));

    CHK_RET(communicator_->CheckDataType(recvType, false));

    HcclResult ret = communicator_->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(sendType, false));
    CHK_RET(communicator_->CheckDataType(recvType, false));

    HcclResult ret = communicator_->AlltoAllVCOutPlace(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType, const void *recvBuf,
    u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    /* * 入参检查 */
    CHK_PTR_NULL(communicator_);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][AlltoAll]errNo[0x%016llx] alltoall tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckDataType(sendType, false));

    HcclResult ret = communicator_->AlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream, tag);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
                               HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO:tag[%s], ptr[%p], count[%llu], data_type[%s], root[%u]",
               tag.c_str(), ptr, count, GetDataTypeEnumStr(dataType).c_str(), root);

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(ptr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Broadcast]errNo[0x%016llx] broadcast tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    /* * 初始化检查 */
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(root));
    HcclResult ret = communicator_->Broadcast(tag, ptr, count, dataType, root, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO:tag[%s], ptr[%p], count[%llu], data_type[%s], root[%u]", tag.c_str(), ptr, count,
        GetDataTypeEnumStr(dataType).c_str(), root);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(root));
    HcclResult ret = communicator_->BroadcastOutPlace(tag, ptr, count, dataType, root, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], recvCount[%llu], data_type[%s], root[%u]",
        tag.c_str(), inputPtr, outputPtr, recvCount, GetDataTypeEnumStr(dataType).c_str(), root);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Scatter]errNo[0x%016llx] scatter tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(recvCount));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(root));
    HcclResult ret = communicator_->ScatterOutPlace(tag, inputPtr, outputPtr, recvCount, dataType, root, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                                   HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], "
               "op[%s]", tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
               GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][ReduceScatter]errNo[0x%016llx] reduceScatter tag length is"\
            "0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    HcclResult ret = communicator_->ReduceScatter(tag, inputPtr, outputPtr, count, dataType, op, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
        tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    HcclResult ret = communicator_->ReduceScatterOutPlace(tag, inputPtr, outputPtr, count, dataType, op, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceScatterV(const std::string &tag, void *inputPtr,
    const void *inputCounts, const void *inputDispls, void *outputPtr, u64 outputCount,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], " \
        "input_counts[%llu], input_displs[%llu], output_count[%llu], data_type[%s], op[%s]",
        tag.c_str(), inputPtr, outputPtr, inputCounts, inputDispls, outputCount, 
        GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][ReduceScatterV]errNo[0x%016llx] reduceScatterV tag length is"\
            "0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    // ReduceScatterV只支持inlinereduce，因此不支持int64类型
    if(dataType == HCCL_DATA_TYPE_INT64) {
        HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported.",
            HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    HcclResult ret = communicator_->ReduceScatterV(tag, inputPtr, inputCounts, inputDispls, 
                        outputPtr, outputCount, dataType, op, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceScatterVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, 
    const void *inputCounts, const void *inputDispls, u64 outputCount, 
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], " \
        "input_counts[%llu], input_displs[%llu], output_count[%llu], data_type[%s], op[%s]",
        tag.c_str(), inputPtr, outputPtr, inputCounts, inputDispls, outputCount, 
        GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    HcclResult ret = communicator_->ReduceScatterVOutPlace(tag, inputPtr, outputPtr, 
                        inputCounts, inputDispls, outputCount, dataType, op, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
                            HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",\
               tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(),
               GetReduceOpEnumStr(op).c_str(), root);

    /* * 入参检查 */
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Reduce]errNo[0x%016llx] reduce tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckReductionOp(op));
    CHK_RET(communicator_->CheckUserRank(root));
    HcclResult ret = communicator_->Reduce(tag, inputPtr, outputPtr, count, dataType, op, root, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
        tag.c_str(), inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        root);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, true));
    CHK_RET(communicator_->CheckReduceDataType(dataType, op));
    CHK_RET(communicator_->CheckUserRank(root));
    HcclResult ret = communicator_->ReduceOutPlace(tag, inputPtr, outputPtr, count, dataType, op, root, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::BatchSendRecv(const std::string &tag, struct HcclSendRecvItemDef* sendRecvItemsPtr,
    u32 itemNum, rtStream_t stream)
{
    HcclResult ret = communicator_->BatchSendRecv(tag, sendRecvItemsPtr, itemNum, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
                          u32 destRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], count[%llu], data_type[%s], destRank[%u]",
        tag.c_str(), inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank);

    /* 入参检查 */
    CHK_PTR_NULL(inputPtr);

    if (tag.empty()) {
        HCCL_ERROR("[HcclComm][Send]errNo[0x%016llx] send tag length is 0",
            HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(destRank));
    HcclResult ret = communicator_->Send(tag, inputPtr, count, dataType, destRank, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], count[%llu], data_type[%s], destRank[%u],",
        tag.c_str(), inputPtr, count, GetDataTypeEnumStr(dataType).c_str(), destRank);

    /* 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(destRank));
    HcclResult ret = communicator_->SendOutPlace(tag, inputPtr, count, dataType, destRank, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], output_ptr[%p], count[%llu], data_type[%s], srcRank[%u]",
        tag.c_str(), outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank);

    /* * 入参检查 */
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(srcRank));
    HcclResult ret = communicator_->ReceiveOutPlace(tag, outputPtr, count, dataType, srcRank, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult hcclComm::receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
                             u32 srcRank, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], output_ptr[%p], count[%llu], data_type[%s], srcRank[%u]",
               tag.c_str(), outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), srcRank);

    /* * 入参检查 */
    CHK_PTR_NULL(outputPtr);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][Receive]errNo[0x%016llx] receive tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(count));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    CHK_RET(communicator_->CheckUserRank(srcRank));
    HcclResult ret = communicator_->Receive(tag, outputPtr, count, dataType, srcRank, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

// 目前支持按tag对资源释放、解绑定
HcclResult hcclComm::ClearOpResource(const std::string &tag)
{
    CHK_RET(communicator_->ClearOpResource(tag));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ClearAivSyncBuf(bool aivClearEnable)
{
    CHK_RET(communicator_->ClearAivSyncBuf(aivClearEnable));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetUniqueId(HcclRootInfo *uniqueId)
{
    CHK_PTR_NULL(uniqueId);

    std::string uniqueIdGot = HcclCommunicator::GetUniqueId();
    s32 ret = snprintf_s(uniqueId->internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1,
                         "%s%s", "hccl-", uniqueIdGot.c_str());
    CHK_PRT_RET((ret == -1), HCCL_ERROR("[Get][UniqueId]errNo[0x%016llx] get unique id failed,uniqueId[%p]",
        HCCL_ERROR_CODE(ret), uniqueId), HCCL_E_MEMORY);

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateCommCCLbuffer() const
{
    CHK_RET(communicator_->CreateCommCCLbuffer());

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateIndirectCCLbuf()
{
    indirectInCCLbuffer_ = DeviceMem::alloc(sizeof(uintptr_t), true);
    CHK_SMART_PTR_NULL(indirectInCCLbuffer_);
    indirectOutCCLbuffer_ = DeviceMem::alloc(sizeof(uintptr_t), true);
    CHK_SMART_PTR_NULL(indirectOutCCLbuffer_);

    return HCCL_SUCCESS;
}

void hcclComm::ReleaseIndirectCCLbuf()
{
    indirectInCCLbuffer_.free();
    indirectOutCCLbuffer_.free();
}

HcclResult hcclComm::GetIndirectInCCLbuf(void* &ptr, u64 &size)
{
    ptr = indirectInCCLbuffer_.ptr();
    size = sizeof(uintptr_t);
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetIndirectOutCCLbuf(void* &ptr, u64 &size)
{
    ptr = indirectOutCCLbuffer_.ptr();
    size = sizeof(uintptr_t);
    return HCCL_SUCCESS;
}
std::string hcclComm::GetIdentifier()
{
    return identifier_;
}

HcclResult hcclComm::CommCheckErrorCqe(HcclResult &result)
{
    CHK_RET(communicator_->GetCqeError(result));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitImpl(DevType deviceType)
{
    HCCL_INFO("InitImpl Implementation isHeterogComm_[%d] isHaveCpuRank_[%d] deviceType[%d] isSpecialType_[%d]",
        isHeterogComm_,
        isHaveCpuRank_,
        deviceType,
        isSpecialType_);

    communicator_.reset(new (std::nothrow) HcclCommunicator());
    CHK_SMART_PTR_NULL(communicator_);
    deviceType_ = deviceType;
    CHK_RET(RegistTaskAbortHandler());

    return HCCL_SUCCESS;
}

HcclResult hcclComm::CreateBarrierMemory()
{
    if (isFirstBarrier_) {
        // 申请device内存
        barrierInMemory_ = DeviceMem::alloc(HCCL_BARRIER_DEFAULT_COUNT * sizeof(float));
        barrierOutMemory_ = DeviceMem::alloc(HCCL_BARRIER_DEFAULT_COUNT * sizeof(float));
        CHK_PRT_RET(!barrierInMemory_, HCCL_ERROR("[Create][BarrierMemory]create barrier input memory fail"),
            HCCL_E_PTR);
        CHK_PRT_RET(!barrierOutMemory_, HCCL_ERROR("[Create][BarrierMemory]create barrier output memory fail"),
            HCCL_E_PTR);

        barrierSendBuf = static_cast<void *>(barrierInMemory_.ptr());
        barrierRecvBuf = static_cast<void *>(barrierOutMemory_.ptr());

        // device内存清0
        // 申请host内存，并将初始值设置为0
        HostMem barrierHostMem = HostMem::alloc(HCCL_BARRIER_DEFAULT_COUNT * sizeof(float));
        CHK_SMART_PTR_NULL(barrierHostMem);
        s32 sRet = memset_s(barrierHostMem.ptr(), barrierHostMem.size(), 0, barrierHostMem.size());
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Create][BarrierMemory]mem set failed.errorno[%d]", sRet), HCCL_E_MEMORY);

        CHK_RET(hrtMemSyncCopy(barrierSendBuf, barrierInMemory_.size(), barrierHostMem.ptr(), barrierHostMem.size(),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        CHK_RET(hrtMemSyncCopy(barrierRecvBuf, barrierOutMemory_.size(), barrierHostMem.ptr(), barrierHostMem.size(),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        isFirstBarrier_ = false;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetOneSidedService(IHcclOneSidedService** service)
{
    CHK_RET(communicator_->GetOneSidedService(service));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
{
    CHK_RET(communicator_->InitOneSidedServiceNetDevCtx(remoteRankId));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetInCCLbuffer(void* &buffer, u64 &size)
{
    CHK_RET(communicator_->GetInCCLbuffer(buffer, size));

    return HCCL_SUCCESS;
}
HcclResult hcclComm::GetOutCCLbuffer(void* &buffer, u64 &size)
{
    CHK_RET(communicator_->GetOutCCLbuffer(buffer, size));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetUserRank(u32 &userRank)
{
    userRank = communicator_->GetUserRank();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetGroupRank(u32 &userRank)
{
    userRank = communicator_->GetGroupRank();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetRankSize(u32 &rankSize)
{
    rankSize = communicator_->GetRankSize();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType optype) const
{
    return communicator_->GetWorkspaceSubStreamNum(streamNum, dataSize, optype);
}
HcclResult hcclComm::GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
    u32 &rankSize, u64 &size)
{
    return communicator_->GetWorkspaceMemSize(opType, count, dataType, rankSize, size, deviceType_);
}

HcclResult hcclComm::GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize) const
{
    return communicator_->GetAllReduceScratchSize(count, dataType, scratchSize);
}

HcclResult hcclComm::SetQosCfg(const u32 qosCfg)
{
    return communicator_->SetQosCfg(qosCfg);
}

HcclResult hcclComm::ResetQosCfg()
{
    return communicator_->ResetQosCfg();
}

HcclResult hcclComm::GetQosCfg(u32& qosCfg)
{
    return communicator_->GetQosCfg(qosCfg);
}

// 设定 workspace 资源
HcclResult hcclComm::SetWorkspaceResource(const std::string &tag, void *memPtr, u64 maxSize,
                                          std::vector<rtStream_t> &stream)
{
    return communicator_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
}

HcclResult hcclComm::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    return communicator_->CreateOpBasedResources(opType, tag, opInfo);
}

HcclResult hcclComm::GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation)
{
    return communicator_->GetDeviceNumPerAggregation(deviceNumPerAggregation);
}

HcclResult hcclComm::GetBandWidthPerNPU(u32 level, float &bandWidth)
{
    return communicator_->GetBandWidthPerNPU(level, bandWidth);
}

HcclResult hcclComm::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize) const
{
    CHK_RET(communicator_->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
    u64 &memSize) const
{
    CHK_RET(communicator_->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
{
    CHK_RET(communicator_->SetGlobalWorkSpace(globalWorkSpaceAddr));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetAttachedStream(const std::vector<rtStream_t> &streams)
{
    CHK_RET(communicator_->SetAttachedStream(streams));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo)
{
    CHK_RET(communicator_->GetandClearOverFlowTasks(hcclDumpInfo));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::SupportDeterministicOptim(bool &isDeterministicOptim)
{
    CHK_RET(communicator_->SupportDeterministicOptim(isDeterministicOptim));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetHccsLinkNum(u32 &numHccsLink)
{
    return communicator_->GetHccsLinkNum(numHccsLink);
}

HcclResult hcclComm::GetDeviceId(s32 &deviceId)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->GetDeviceId(deviceId));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetDevType(DevType &devType)
{
    devType = deviceType_;
    return HCCL_SUCCESS;
}

HcclResult hcclComm::IsStandardCard(bool &isStandardCard)
{
        isStandardCard = communicator_->IsStandardCard();

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Is310PDuoCard(bool &is310PDuoCard)
{
    is310PDuoCard = communicator_->Is310PDuoCard();
    return HCCL_SUCCESS;
}

HcclResult hcclComm::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
    HcclDataType dataType, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_DEBUG("HCCL_KEY_INFO: tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s]",
               tag.c_str(), inputPtr, outputPtr, inputCount, GetDataTypeEnumStr(dataType).c_str());

    /* * 入参检查 */
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(tag.empty(), HCCL_ERROR("[HcclComm][Gather]errNo[0x%016llx] gather tag length is 0",
        HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);

    CHK_RET(communicator_->CheckCount(inputCount));
    CHK_RET(communicator_->CheckDataType(dataType, false));
    HcclResult ret = communicator_->Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, stream);
    if (ret != HCCL_SUCCESS) {
        PrintSubmittedOpCnt(tag, ret);
        return ret;
    }

    return HCCL_SUCCESS;
}

bool hcclComm::IsNeedResetDevice()
{
    return isResetDevice_;
}

HcclResult hcclComm::ResetDeviceEnable()
{
    isResetDevice_ = true;
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SaveTraceInfo(std::string &logInfo)
{
    CHK_PRT(communicator_->SaveTraceInfo(logInfo));

    return HCCL_SUCCESS;
}

bool hcclComm::GetCommResource(const std::string &tag, void **commContext)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: GetCommResource commContext[%p]", commContext);

    return (communicator_->GetCommResource(tag, commContext));
}

bool hcclComm::GetCommResource(void *&commContext)
{
    HCCL_INFO("HCCL_KEY_INFO: GetCommResource commContext[%p]", commContext);
    return communicator_->GetCommResource(commContext);
}

HcclResult hcclComm::SetStopFlag(bool value)
{
    if (communicator_ != nullptr) {
        return communicator_->SetStopFlag(value);
    }
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetState(HcclCommState state)
{
    if (communicator_ != nullptr) {
        return communicator_->SetState(state);
    }
    return HCCL_SUCCESS;
}

HcclCommState hcclComm::GetState()
{
    if (communicator_ != nullptr) {
        return communicator_->GetState();
    }
    return HcclCommState::IDLE;
}

HcclResult hcclComm::AllocComResourceByTiling(const std::string &algConfig,
    const std::string &tag, uint32_t opType, uint32_t reduceType, rtStream_t stream)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: AllocComResourceByTiling algConfig[%s], tag[%s], opType[%u], reduceType[%u]",
        algConfig.c_str(), tag.c_str(), opType, reduceType);

    return communicator_->AllocComResourceByTiling(algConfig, tag, opType, reduceType, stream);
}

HcclResult hcclComm::CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
    void **commContext)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: CreateCommResource commContext[%p], isOpbaseMode[%u]", commContext, isOpbaseMode);

    CHK_RET(communicator_->CreateCommResource(tag, aiCpuStream, isOpbaseMode, commContext));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetAicpuOpStreamNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: GetAicpuOpStreamNotify commContext[%p]", opStream);

    CHK_RET(communicator_->GetAicpuOpStreamNotify(opStream, aicpuNotifyNum, aicpuNotify));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
{
    /* 增加输出日志关键字 */
    HCCL_INFO("HCCL_KEY_INFO: Mc2AiCpuStreamAllocAndGet streamMode[%u]", streamMode);

    CHK_RET(communicator_->Mc2AiCpuStreamAllocAndGet(streamMode, aiCpuStream));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize)
{
    HCCL_INFO("HCCL_KEY_INFO: GetTopoDesc topoDescs[%p] topoSize[%u]", topoDescs, topoSize);

    CHK_RET(communicator_->GetTopoDesc(topoDescs, topoSize));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable)
{
    CHK_RET(communicator_->ReStartVnic(params, rankTable));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetDeterministicConfig(const u8 deterministic)
{
    CHK_RET(communicator_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetAivModeConfig(const bool aivMode)
{
    CHK_RET(communicator_->SetAivModeConfig(aivMode));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetAicpuUnfoldConfig(const bool aicpuUnfold)
{
    CHK_RET(communicator_->SetAicpuUnfoldConfig(aicpuUnfold));
    return HCCL_SUCCESS;
}

u64 hcclComm::GetConfigInCCLbufferSize()
{
    return inCCLbufferSize_;
}

u64 hcclComm::GetConfigOutCCLbufferSize()
{
    return outCCLbufferSize_;
}

u32 hcclComm::GetRankTableCrc()
{
    return communicator_->GetRankTableCrc();
}

u32 hcclComm::GetServerNum()
{
    return communicator_->GetServerNum();
}

u32 hcclComm::GetModuleNum()
{
    return communicator_->GetModuleNum();
}

HcclResult hcclComm::GetCommParams(HcclCommParams &params)
{
    CHK_RET(communicator_->GetCommParams(params));
    params.deviceType = deviceType_;
    params.isHeterogComm = isHeterogComm_;
    params.identifier = identifier_;
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetCommRankTable(RankTable_t &rankTable)
{
    CHK_RET(communicator_->GetCommRankTable(rankTable));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::RegistTaskAbortHandler() const
{
    HCCL_INFO("RegistTaskAbortHandler begin");
    CHK_RET(TaskAbortHandler::Init(communicator_.get()));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::UnRegistTaskAbortHandler() const
{
    HCCL_INFO("UnRegistTaskAbortHandler begin");
    CHK_RET(TaskAbortHandler::DeInit(communicator_.get()));

    return HCCL_SUCCESS;
}

HcclResult hcclComm::Suspend()
{
    communicator_->Suspend();
    return HCCL_SUCCESS;
}

HcclResult hcclComm::Resume()
{
    communicator_->Resume();
    return HCCL_SUCCESS;
}

HcclResult hcclComm::InitZeroCopyMemoryAgent()
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->InitZeroCopyMemoryAgent());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::DeinitZeroCopyMemoryAgent()
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->DeinitZeroCopyMemoryAgent());
    return HCCL_SUCCESS;
}

HcclResult hcclComm::SetMemoryRange(void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->SetMemoryRange(baseVirPtr, size, alignment, flags));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::UnsetMemoryRange(void *baseVirPtr)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->UnsetMemoryRange(baseVirPtr));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::ActivateCommMemory(void *virPtr, size_t size, size_t offset, void* handle, uint64_t flags)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->ActivateCommMemory(virPtr, size, offset, handle, flags));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::DeactivateCommMemory(void *virPtr)
{
    CHK_SMART_PTR_NULL(communicator_);
    CHK_RET(communicator_->DeactivateCommMemory(virPtr));
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetBlockDim(u32& blockDim)
{
    return communicator_->GetBlockDim(blockDim);
}
}  // namespace hccl
