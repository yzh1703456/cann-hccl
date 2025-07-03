/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/time.h> /* 获取时间 */

#include "profiler_base_pub.h"
#include "adapter_rts_common.h"
#include "runtime/base.h"
#include "externalinput_pub.h"

using namespace hccl;
std::array<std::map<s32, s32>, MAX_MODULE_DEVICE_NUM> ProfilerBase::streamPlaneMap_;
std::array<std::map<s32, const std::string>, MAX_MODULE_DEVICE_NUM> ProfilerBase::streamTagMap_;
std::array<std::map<s32, AlgType>, MAX_MODULE_DEVICE_NUM> ProfilerBase::streamAlgTypeMap_;
std::array<std::map<const std::string, const std::string>, MAX_MODULE_DEVICE_NUM> ProfilerBase::tagGroupMap_;
std::array<std::map<const std::string, const HcclWorkflowMode>, MAX_MODULE_DEVICE_NUM> ProfilerBase::tagModeMap_;
std::array<std::map<const std::string, GroupRankInfo>, MAX_MODULE_DEVICE_NUM> ProfilerBase::groupRankMap_;
std::array<std::map<const std::string, OpDataInfo>, MAX_MODULE_DEVICE_NUM> ProfilerBase::tagOpDataMap_;
std::array<std::map<const std::string, u32>, MAX_MODULE_DEVICE_NUM> ProfilerBase::groupIndexMap_;
std::array<std::map<const std::string, u32>, MAX_MODULE_DEVICE_NUM> ProfilerBase::sendRecvGroupIndexMap_;
std::array<std::map<const std::string, std::string>, MAX_MODULE_DEVICE_NUM> ProfilerBase::groupUdiMap_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> ProfilerBase::streamMutex_;
bool ProfilerBase::isSendRecv_[MAX_MODULE_DEVICE_NUM];
u32 ProfilerBase::index_[MAX_MODULE_DEVICE_NUM];

const std::array<uint32_t, HCCL_REDUCE_RESERVED> ProfilerBase::opString = {static_cast<u32>(OpDict::SUM),
    static_cast<u32>(OpDict::PROD), static_cast<u32>(OpDict::MAX), static_cast<u32>(OpDict::MIN)};

const std::array<uint32_t, HCCL_DATA_TYPE_RESERVED> ProfilerBase::dataTypeString = {
    static_cast<u32>(DataType::DINT8), static_cast<u32>(DataType::DINT16), static_cast<u32>(DataType::DINT32),
    static_cast<u32>(DataType::DFP16), static_cast<u32>(DataType::DFP32), static_cast<u32>(DataType::DINT64),
    static_cast<u32>(DataType::DUINT64)
};
// 16位浮点在CPU中找不到对应的数据类型, 故直接写立即数 : 2
const std::array<s32, HCCL_DATA_TYPE_RESERVED> ProfilerBase::sizeOf = { sizeof(s8),    sizeof(short), sizeof(s32), 2,
                                                                        sizeof(float), sizeof(s64),   sizeof(u64) };
ProfilerBase::ProfilerBase(u32 deviceLogicId) : deviceLogicId_(deviceLogicId) {}

ProfilerBase::~ProfilerBase() {}

HcclResult ProfilerBase::AddStream(s32 streamID, const std::string &tag, s32 planeID, AlgType algType)
{
    s32 deviceLogicId = -1;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("[Add][Stream]deviceLogicId_[%d]"
        "is bigger than HCCL_AISERVER_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("AddStream: streamID[%d], tag[%s], planeId[%d], algType[%s], deviceLogicId[%d]", streamID, tag.c_str(),
        planeID, AlgTypeToStr(algType).c_str(), deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);

        if (streamTagMap_[deviceLogicId].find(streamID) != streamTagMap_[deviceLogicId].end() ||
            streamPlaneMap_[deviceLogicId].find(streamID) != streamPlaneMap_[deviceLogicId].end() ||
            streamAlgTypeMap_[deviceLogicId].find(streamID) != streamAlgTypeMap_[deviceLogicId].end()) {
            HCCL_DEBUG("streamID[%d] already exist, oldTag[%s], oldPlaneId[%d]", streamID,
                streamTagMap_[deviceLogicId][streamID].c_str(), streamPlaneMap_[deviceLogicId][streamID]);
            streamTagMap_[deviceLogicId].erase(streamID);
            streamPlaneMap_[deviceLogicId].erase(streamID);
            streamAlgTypeMap_[deviceLogicId].erase(streamID);
            streamTagMap_[deviceLogicId].insert(std::make_pair<s32 &, const std::string &>(streamID, tag));
            streamPlaneMap_[deviceLogicId][streamID] = planeID;
            streamAlgTypeMap_[deviceLogicId][streamID] = algType;
            return HCCL_SUCCESS;
        }
        streamTagMap_[deviceLogicId].insert(std::make_pair<s32 &, const std::string &>(streamID, tag));
        streamPlaneMap_[deviceLogicId][streamID] = planeID;
        streamAlgTypeMap_[deviceLogicId][streamID] = algType;
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::DelStream(s32 streamID)
{
    s32 deviceLogicId = -1;
    CHK_RET(hrtGetDevice(&deviceLogicId));

    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("deviceLogicId_[%d] is bigger"
        "than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);

    HCCL_DEBUG("DelStream: streamID[%d] tag[%s], planeId[%d], algType[%s], deviceLogicId[%d]", streamID,
        streamTagMap_[deviceLogicId][streamID].c_str(), streamPlaneMap_[deviceLogicId][streamID],
        AlgTypeToStr(streamAlgTypeMap_[deviceLogicId][streamID]).c_str(), deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        streamTagMap_[deviceLogicId].erase(streamID);
        streamPlaneMap_[deviceLogicId].erase(streamID);
        streamAlgTypeMap_[deviceLogicId].erase(streamID);
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::AddTag(const std::string &tag, const std::string &group, const HcclWorkflowMode &workFlowMode,
    bool isSendRecv)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("deviceLogicId_[%d] is bigger than"
        " MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("AddTag: tag[%s] group[%s] deviceLogicId[%d]", tag.c_str(), group.c_str(), deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        tagGroupMap_[deviceLogicId].insert(std::make_pair<const std::string &, const std::string &>(tag, group));
        tagModeMap_[deviceLogicId].insert(std::make_pair<const std::string &,
            const HcclWorkflowMode &>(tag, workFlowMode));
        if (isSendRecv == false) {
            groupIndexMap_[deviceLogicId][group]++;
        } else {
            sendRecvGroupIndexMap_[deviceLogicId][group]++;
        }
        isSendRecv_[deviceLogicId] = isSendRecv;
        index_[deviceLogicId] = isSendRecv ? sendRecvGroupIndexMap_[deviceLogicId][group] : groupIndexMap_[deviceLogicId][group];
        HCCL_DEBUG("IndexMap: tag[%s] group[%s] groupIndexMap_[%d]:%u sendRecvGroupIndexMap_[%d]:%u", tag.c_str(), group.c_str(),
            deviceLogicId, groupIndexMap_[deviceLogicId][group], deviceLogicId, sendRecvGroupIndexMap_[deviceLogicId][group]);
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::DelTag(const std::string &tag)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("deviceLogicId_[%d] is bigger"
        "than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("DelTag: tag[%s] group[%s] deviceLogicId[%d]", tag.c_str(), tagGroupMap_[deviceLogicId][tag].c_str(),
        deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        tagGroupMap_[deviceLogicId].erase(tag);
        tagModeMap_[deviceLogicId].erase(tag);
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::AddOpData(const std::string &tag, u64 count, const void *src, const void *dst,
    HcclDataType dataType, u32 rootId, const std::string &group, HcclReduceOp reduceType)
{
    if (GetExternalInputHcclDftLevel() == false) {
        return HCCL_SUCCESS;
    }
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("deviceLogicId_[%d] is bigger than"
        " MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("AddOpData: tag[%s] count[%u] src[%p] dst[%p] dataType[%s] deviceLogicId[%d] group[%s]",
        tag.c_str(), count, src, dst, GetDataTypeEnumStr(dataType).c_str(), deviceLogicId, group.c_str());
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        OpDataInfo opData;
        opData.count = count;
        opData.deviceId = deviceLogicId;
        opData.src = src;
        opData.dst = dst;
        opData.dataType = dataType;
        opData.index = index_[deviceLogicId];
        opData.rootId = rootId;
        (void)gettimeofday(&opData.tv, nullptr);
        opData.reduceType = reduceType;
        tagOpDataMap_[deviceLogicId][tag] = opData;
    }
    return HCCL_SUCCESS;
}
 
HcclResult ProfilerBase::DelOpData(const std::string &tag)
{
    if (GetExternalInputHcclDftLevel() == false) {
        return HCCL_SUCCESS;
    }
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("deviceLogicId_[%d] is bigger"
        "than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("DelOpData: tag[%s] deviceLogicId[%d]", tag.c_str(), deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        tagOpDataMap_[deviceLogicId].erase(tag);
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::AddGroupRankInfo(const std::string &group, u32 rankSize, u32 rankId, bool isSendRecv,
    u32 remoteRankId)
{
    if (GetExternalInputHcclDftLevel() == false) {
        return HCCL_SUCCESS;
    }
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("deviceLogicId_[%d] is bigger than"
        " MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("AddGroupRankInfo: group[%s] rankSize[%u] rankId[%u] deviceLogicId[%d]", group.c_str(), rankSize,
        rankId, deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        GroupRankInfo groupRankInfo;
        groupRankInfo.rankSize = rankSize;
        groupRankInfo.rankId = rankId;
        groupRankInfo.remoteRankId = remoteRankId;
        groupRankMap_[deviceLogicId][group] = groupRankInfo;
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::DelGroupRankInfo(const std::string &group)
{
    if (GetExternalInputHcclDftLevel() == false) {
        return HCCL_SUCCESS;
    }
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("deviceLogicId_[%d] is bigger"
        "than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("DelGroupRankInfo: group[%s] deviceLogicId[%d]", group.c_str(), deviceLogicId);
    {
        std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
        groupRankMap_[deviceLogicId].erase(group);
    }
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::GetTagByStream(u32 &streamID, std::string &tag)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("deviceLogicId_[%d] is bigger"
        "than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);

    std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
    CHK_PRT_RET(streamTagMap_[deviceLogicId].find(streamID) == streamTagMap_[deviceLogicId].end(),
        HCCL_DEBUG("stream id[%u] not found in profiler.", streamID), HCCL_SUCCESS);
    tag = streamTagMap_[deviceLogicId][streamID];
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::GetAlgTypeByStream(u32 &streamID, AlgType &algType)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetAlgTypeByStream]rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("[GetAlgTypeByStream] "
        "deviceLogicId_[%d] is bigger than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
        HCCL_E_INTERNAL);

    std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
    CHK_PRT_RET(streamAlgTypeMap_[deviceLogicId].find(streamID) == streamAlgTypeMap_[deviceLogicId].end(),
        HCCL_DEBUG("[GetAlgTypeByStream] stream id[%u] not found in profiler.", streamID), HCCL_SUCCESS);
    algType = streamAlgTypeMap_[deviceLogicId][streamID];
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::GetGroupNameByTag(const std::string &tag, std::string &group)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetGroupNameByTag]rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("[GetGroupNameByTag] "
        "deviceLogicId_[%d] is bigger than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
        HCCL_E_INTERNAL);

    std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
    CHK_PRT_RET(tagGroupMap_[deviceLogicId].find(tag) == tagGroupMap_[deviceLogicId].end(),
        HCCL_DEBUG("[GetGroupNameByTag] tag[%s] not found in profiler.", tag.c_str()), HCCL_SUCCESS);
    group = tagGroupMap_[deviceLogicId][tag];
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::GetRankInfoByGroup(const std::string &group, GroupRankInfo &groupRankInfo)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetRankInfoByGroup]rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("[GetRankInfoByGroup] "
        "deviceLogicId_[%d] is bigger than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
        HCCL_E_INTERNAL);

    std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
    CHK_PRT_RET(groupRankMap_[deviceLogicId].find(group) == groupRankMap_[deviceLogicId].end(),
        HCCL_DEBUG("[GetRankInfoByGroup] group[%s] not found in profiler.", group.c_str()), HCCL_SUCCESS);
    groupRankInfo = groupRankMap_[deviceLogicId][group];
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::GetOpDataInfoByTag(const std::string &tag, OpDataInfo &opDataInfo)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetOpDataInfoByTag]rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_WARNING("[GetOpDataInfoByTag] "
        "deviceLogicId_[%d] is bigger than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM),
        HCCL_E_INTERNAL);

    std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
    CHK_PRT_RET(tagOpDataMap_[deviceLogicId].find(tag) == tagOpDataMap_[deviceLogicId].end(),
        HCCL_DEBUG("[GetOpDataInfoByTag] tag[%u] not found in profiler.", tag.c_str()), HCCL_SUCCESS);
    opDataInfo = tagOpDataMap_[deviceLogicId][tag];
    return HCCL_SUCCESS;
}

void ProfilerBase::GetSubmittedOpCnt(u32 &index)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    index = 0;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetSubmittedOpCnt]rts get device error");
        return;
    }
    if (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[GetSubmittedOpCnt] "
        "deviceLogicId_[%d] is bigger than MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM);
        return;
    }

    std::unique_lock<std::mutex> lock(streamMutex_[deviceLogicId]);
    HCCL_DEBUG("GetSubmittedOpCnt: index_[%d][%u]", deviceLogicId, index_[deviceLogicId]);
    index = index_[deviceLogicId];
    return;
}

HcclResult ProfilerBase::AddGroupUdi(const std::string &group, const std::string &udi)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("deviceLogicId_[%d] is bigger than"
        " MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_RUN_INFO("AddGroupUdi: group[%s] udi[%s] deviceLogicId[%d]", group.c_str(), udi.c_str(),
        deviceLogicId);
    std::lock_guard<std::mutex> lock(streamMutex_[deviceLogicId]);
    groupUdiMap_[deviceLogicId].insert(
        std::make_pair<const std::string &, const std::string &>(group, udi));
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::DelGroupUdi(const std::string &group)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("deviceLogicId_[%d] is bigger than"
        " MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("DelGroupUdi: group[%s] deviceLogicId[%d]", group.c_str(), deviceLogicId);
    std::lock_guard<std::mutex> lock(streamMutex_[deviceLogicId]);
    groupUdiMap_[deviceLogicId].erase(group);
    return HCCL_SUCCESS;
}

HcclResult ProfilerBase::GetUdiByGroup(const std::string &group, std::string &udi)
{
    s32 deviceLogicId = -1;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("rts get device error"), ret);
    CHK_PRT_RET(static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM, HCCL_ERROR("deviceLogicId_[%d] is bigger than"
        " MAX_MODULE_DEVICE_NUM[%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_DEBUG("GetUdiByGroup: group[%s] deviceLogicId[%d]", group.c_str(), deviceLogicId);
    std::lock_guard<std::mutex> lock(streamMutex_[deviceLogicId]);
    CHK_PRT_RET(groupUdiMap_[deviceLogicId].find(group) == groupUdiMap_[deviceLogicId].end(),
    HCCL_DEBUG("[GetUdiByGroup] group[%s] not found in profiler.", group.c_str()), HCCL_SUCCESS);
    udi = groupUdiMap_[deviceLogicId].find(group)->second;
    return HCCL_SUCCESS;
}