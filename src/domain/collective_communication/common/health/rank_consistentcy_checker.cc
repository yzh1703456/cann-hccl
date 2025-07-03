/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_pub.h"
#include "rank_consistentcy_checker.h"
#include "calc_crc.h"

namespace hccl {

RankConsistentcyChecker::RankConsistentcyChecker() : cannVersion_{0}, cannVerCheckSwitch_(false),
    infoFlagVer_(false), configFileExist_(false)
{
}

RankConsistentcyChecker::~RankConsistentcyChecker() = default;

RankConsistentcyChecker& RankConsistentcyChecker::GetInstance(s32 deviceLogicId)
{
    static RankConsistentcyChecker instance[MAX_MODULE_DEVICE_NUM];
    if (deviceLogicId == HOST_DEVICE_ID) {
        HCCL_INFO("[GetInstance] deviceLogicId[-1] is HOST_DEVICE_ID");
        return instance[0];
    }
    hrtGetDevice(&deviceLogicId);
    HCCL_INFO("[GetInstance] get deviceLogicId[%d]", deviceLogicId);
    CHK_PRT_RET((static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM || deviceLogicId < 0),
        HCCL_WARNING("[R]deviceLogicId[%d] is invalid", deviceLogicId), instance[0]);

    return instance[deviceLogicId];
}

// all gather
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag, u64 count,
    HcclDataType dataType, u64 inCclBufferSize, u64 outCclBufferSize, const char *group, u32 crc)
{
    return RecordOpPara(opCMD, tag, count, dataType, HCCL_REDUCE_RESERVED, 0, 0, 0, 0, inCclBufferSize,
        outCclBufferSize, group, crc);
}

// all reduce
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag, u64 count,
    HcclDataType dataType, HcclReduceOp op, u64 inCclBufferSize, u64 outCclBufferSize, const char *group, u32 crc)
{
    return RecordOpPara(opCMD, tag, count, dataType, op, 0, 0, 0, 0, inCclBufferSize,
        outCclBufferSize, group, crc);
}

// broadcast
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag, u64 count,
    HcclDataType dataType, u32 root, u64 inCclBufferSize, u64 outCclBufferSize, const char *group, u32 crc)
{
    return RecordOpPara(opCMD, tag, count, dataType, HCCL_REDUCE_RESERVED, root, 0, 0, 0, inCclBufferSize,
        outCclBufferSize, group, crc);
}

// reduce
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, u64 inCclBufferSize, u64 outCclBufferSize,
    const char *group, u32 crc)
{
    return RecordOpPara(opCMD, tag, count, dataType, op, root, 0, 0, 0, inCclBufferSize, outCclBufferSize, group, crc);
}

// send && receive
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag, u64 count,
    HcclDataType dataType, u32 rank, u32 srTag, u32 selfRank, u64 inCclBufferSize, u64 outCclBufferSize,
    const char *group, u32 crc)
{
    return RecordOpPara(opCMD, tag, count, dataType, HCCL_REDUCE_RESERVED, 0, rank, srTag, selfRank,
        inCclBufferSize, outCclBufferSize, group, crc);
}

// batchsendrecv
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag,
    u64 inCclBufferSize, u64 outCclBufferSize, const char *group, u32 crc)
{
    return RecordOpPara(opCMD, tag, 0, HCCL_DATA_TYPE_RESERVED, HCCL_REDUCE_RESERVED, 0, 0, 0, 0, inCclBufferSize,
        outCclBufferSize, group, crc);
}

// all gather v
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag,
    const void *counts, const void *displs, const u32 rankSize,
    HcclDataType dataType, u64 inCclBufferSize, u64 outCclBufferSize, const char *group, u32 crc)
{
    CHK_RET(RecordOpPara(opCMD, tag, 0, dataType, HCCL_REDUCE_RESERVED, 0, 0, 0, 0, inCclBufferSize,
        outCclBufferSize, group, crc));
    CHK_RET(RecordVaringOpPara(tag, counts, displs, rankSize));
    return HCCL_SUCCESS;
}

// reduce scatter v
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag,
    const void *counts, const void *displs, const u32 rankSize,
    HcclDataType dataType, HcclReduceOp op, u64 inCclBufferSize, u64 outCclBufferSize, const char *group, u32 crc)
{
    CHK_RET(RecordOpPara(opCMD, tag, 0, dataType, op, 0, 0, 0, 0, inCclBufferSize,
        outCclBufferSize, group, crc));
    CHK_RET(RecordVaringOpPara(tag, counts, displs, rankSize));
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::RecordVaringOpPara(const std::string &tag, const void *counts, const void *displs,
    const u32 rankSize)
{
    u32 countsCrc;
    CHK_RET(CalcRawDataCrc(static_cast<const char_t*>(counts), rankSize * sizeof(u64), countsCrc));
    crcRecords_[tag][HcclCrcRecordType::HCCL_CRC_RECORD_VARING_COUNTS] = countsCrc;

    u32 displsCrc;
    CHK_RET(CalcRawDataCrc(static_cast<const char_t*>(displs), rankSize * sizeof(u64), displsCrc));
    crcRecords_[tag][HcclCrcRecordType::HCCL_CRC_RECORD_VARING_DISPLACEMENTS] = displsCrc;
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::DelOpPara(const std::string &tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    CHK_PRT_RET(!cmdInfoMap_.erase(tag),
        HCCL_ERROR("[RankConsistentcyChecker][DelOpPara]CMD info for tag[%s] does not exist, delete fail.",
        tag.c_str()), HCCL_E_INTERNAL);
    CHK_PRT_RET(!infoFlagCmdMap_.erase(tag),
        HCCL_ERROR("[RankConsistentcyChecker][DelOpPara]CMD info flag cmd for tag[%s] does not exist, delete fail.",
        tag.c_str()), HCCL_E_INTERNAL);
    crcRecords_.erase(tag);
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::RecordVerInfo(const std::string &versionInfo)
{
    u32 strLen = versionInfo.length();
    s32 sRet = memset_s(cannVersion_, MAX_CANN_VERSION_LEN + 1, 0, MAX_CANN_VERSION_LEN + 1);
    CHK_PRT_RET(sRet != EOK, HCCL_WARNING("[RankConsistentcyChecker][RecordVerInfo]memory set 0 fail for version str "
        "array. return[%d].", sRet), HCCL_SUCCESS);

    CHK_PRT_RET(strLen == 0, HCCL_WARNING("[Record][CannVersion] version information str is empty."),
        HCCL_SUCCESS);

    CHK_PRT_RET(strLen >= MAX_CANN_VERSION_LEN, HCCL_WARNING("[Record][CannVersion]"
        "length of version information str is too long."), HCCL_SUCCESS);
    sRet = strncpy_s(cannVersion_, MAX_CANN_VERSION_LEN + 1, versionInfo.c_str(), strLen);
    CHK_PRT_RET(sRet != EOK, HCCL_WARNING("[Record][CannVersion] length of version information str is too long."),
        HCCL_SUCCESS);

    infoFlagVer_ = true;
    return HCCL_SUCCESS;
}

u64 RankConsistentcyChecker::GetRankConsistentDataLength()
{
    return sizeof(HcclCheckInfo);
}

void RankConsistentcyChecker::RecordProtocolType(ProtocolType protocolType)
{
    HCCL_INFO("[RankConsistentcyChecker][RecordProtocolType]protocolType set to [%d].",
        static_cast<s32>(protocolType));
    protocolType_ = protocolType;
    return;
}

HcclResult RankConsistentcyChecker::GetCheckFrame(u8 *destBuf, u64 maxDestBuf, const std::string &tag)
{
    CHK_PTR_NULL(destBuf);
    // 要发送的校验帧
    HcclCheckInfo checkInfo;
    u64 checkInfoLen = sizeof(checkInfo);
    HcclResult ret = GenerateCheckFrame(checkInfo, tag);
    checkInfo.cmdInfo.selfRank = 0; // 自身的group rank 不做校验,置0
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[RankConsistentcyChecker][GetCheckFrame]generate check frame fail. "
        "return[%d]", ret), ret);

    s32 sret = memcpy_s(destBuf, maxDestBuf, &checkInfo, checkInfoLen);
    CHK_PRT_RET(sret != EOK, HCCL_ERROR("[RankConsistentcyChecker][GetCheckFrame]frame len[%llu] is bigger than "
        "dest buffer len[%llu].", checkInfoLen, maxDestBuf), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::CheckFrameRecv(const u8 *recvBuf, u32 recvBufLen, const std::string &tag)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(recvBufLen == 0 || recvBufLen > MAX_FRAME_LEN,
        HCCL_ERROR("[RankConsistentcyChecker][CheckFrameRecv] errNo[0x%016llx] recvBufLen is wrong.",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);

    CHK_PRT_RET(recvBufLen < sizeof(HcclCheckInfo),
        HCCL_ERROR("[RankConsistentcyChecker][CheckFrameRecv] errNo[0x%016llx] recvBufLen[%u]is less than "
        "check info[%zu].", HCCL_ERROR_CODE(HCCL_E_INTERNAL), recvBufLen, sizeof(HcclCheckInfo)), HCCL_E_PARA);

    HcclCheckInfo checkInfoRecv;
    // 对固定长度的全局数组变量，结构体变量进行初始化和拷贝，可以不用检查初始化安全函数返回值
    (void)memset_s(&checkInfoRecv, sizeof(HcclCheckInfo), 0, sizeof(HcclCheckInfo));
    (void)memcpy_s(&checkInfoRecv, sizeof(HcclCheckInfo), recvBuf, sizeof(HcclCheckInfo));

    HcclCheckInfo checkInfo;
    CHK_RET(GenerateCheckFrame(checkInfo, tag));
    if (checkInfo.cmdInfo.cmdType == HcclCMDType::HCCL_CMD_SEND) {
        checkInfo.cmdInfo.cmdType = HcclCMDType::HCCL_CMD_RECEIVE;
        checkInfo.cmdInfo.rank = checkInfo.cmdInfo.selfRank;
    } else if (checkInfo.cmdInfo.cmdType == HcclCMDType::HCCL_CMD_RECEIVE) {
        checkInfo.cmdInfo.cmdType = HcclCMDType::HCCL_CMD_SEND;
        checkInfo.cmdInfo.rank = checkInfo.cmdInfo.selfRank;
    }

    checkInfo.cmdInfo.selfRank = 0; // 自身的子group rank 不做校验
    if (CompareFrame(checkInfo, checkInfoRecv)) {
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[RankConsistentcyChecker][CheckFrameRecv] check success, len of frame[%u], len of check data[%zu].",
        recvBufLen, sizeof(checkInfo));
    return HCCL_SUCCESS;
}

void RankConsistentcyChecker::ClearCheckInfo()
{
    configFileExist_ = false;
    infoFlagVer_ = false;
    ClearCrcInfo();
    cmdInfoMap_.clear();
    infoFlagCmdMap_.clear();
    // 相关规范的例外场景，对固定数组的memset_s可以不判断返回值
    (void)memset_s(cannVersion_, MAX_CANN_VERSION_LEN + 1, 0, MAX_CANN_VERSION_LEN + 1);
    protocolType_ = ProtocolType::RESERVED;
    return;
}

HcclResult RankConsistentcyChecker::CalcStringCrc(const char *str, u32 &crc)
{
    // 计算字符串CRC
    HcclResult ret = CalcCrc::HcclCalcCrc(str, strlen(str), crc);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankConsistentcyChecker][CalcStringCrc] errNo[0x%016llx] calc string crc error",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);

    HCCL_DEBUG("[RankConsistentcyChecker][CalcStringCrc] result crc[%u].", crc);
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::CalcRawDataCrc(const void *ptr, u64 length, u32 &crc)
{
    // 计算内存数据块CRC
    HcclResult ret = CalcCrc::HcclCalcCrc(static_cast<const char*>(ptr), length, crc);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[RankConsistentcyChecker][CalcRawDataCrc] errNo[0x%016llx] calc string crc error",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);

    HCCL_DEBUG("[RankConsistentcyChecker][CalcRawDataCrc] result crc[%u].", crc);
    return HCCL_SUCCESS;
}

void RankConsistentcyChecker::SetCheckCannVersionSwitch(const bool cannVerCheckSwitch)
{
    cannVerCheckSwitch_ = cannVerCheckSwitch;
    return;
}

// private
HcclResult RankConsistentcyChecker::RecordOpPara(HcclCMDType opCMD, const std::string &tag, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, u32 rank, u32 srTag, u32 selfRank, u64 inCclBufferSize,
    u64 outCclBufferSize, const char *group, u32 crc)
{
    HcclCMDInfo cmdInfo;
    // 相关规范的例外场景，对固定数组的memset_s可以不判断返回值
    (void)memset_s(&cmdInfo, sizeof(cmdInfo), 0, sizeof(cmdInfo));

    cmdInfo.cmdType = opCMD;
    s32 sRet = strncpy_s(cmdInfo.tag, TAG_MAX_LEN + 1, tag.c_str(), tag.length());
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[RankConsistentcyChecker][RecordOpPara]errNo[0x%016llx] strlen[%u] of tag is "
        "longer than buffer[%u].", HCCL_ERROR_CODE(HCCL_E_PARA), tag.length(), TAG_MAX_LEN), HCCL_E_PARA);

    cmdInfo.count = count;
    cmdInfo.dataType = dataType;

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    sRet = strncpy_s(cmdInfo.group, GROUP_NAME_MAX_LEN + 1, strGroup.c_str(), strGroup.length());
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[RankConsistentcyChecker][RecordOpPara]errNo[0x%016llx] strlen[%u] group is "
        "longer than buffer[%u].", HCCL_ERROR_CODE(HCCL_E_PARA), strGroup.length(), GROUP_NAME_MAX_LEN), HCCL_E_PARA);

    cmdInfo.op = op;
    cmdInfo.root = root;
    cmdInfo.rank = rank;
    cmdInfo.srTag = srTag;
    cmdInfo.selfRank = selfRank;
    cmdInfo.inCclBufferSize = inCclBufferSize;
    cmdInfo.outCclBufferSize = outCclBufferSize;

    std::lock_guard<std::mutex> lock(mutex_);
    cmdInfoMap_[tag] = cmdInfo;
    infoFlagCmdMap_[tag] = true;
    crcRecords_[tag][HcclCrcRecordType::HCCL_CRC_RECORD_RANKTABLE] = crc;

    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::GetOpParaByTag(const std::string &tag, HcclCMDInfo &CMDInfoOutput)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto getResult = cmdInfoMap_.find(tag);
    CHK_PRT_RET(getResult == cmdInfoMap_.end(),
        HCCL_ERROR("[RankConsistentcyChecker][GetOpParaByTag]There is not any CMD infomation for tag[%s]",
        tag.c_str()), HCCL_E_INTERNAL);
    CMDInfoOutput = getResult->second;
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::GetCrcByTag(const std::string &tag, HcclCRCInfo &crcInfo)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto recordsIter = crcRecords_.find(tag);
    CHK_PRT_RET(recordsIter == crcRecords_.end(),
        HCCL_ERROR("[RankConsistentcyChecker][GetCrcByTag]There is not any CRC infomation for tag[%s]",
        tag.c_str()), HCCL_E_INTERNAL);
    crcInfo.crcNum = 0;
    const auto &tagRecords = recordsIter->second;
    for (const auto &record : tagRecords) {
        crcInfo.crcArray[crcInfo.crcNum++] = record.second;
        HCCL_DEBUG("[RankConsistentcyChecker][GetCrcByTag]Append crc[%u] for tag[%s].", record.second, tag.c_str());
    }

    HCCL_INFO("[RankConsistentcyChecker][GetCrcByTag]After adding crc for tag[%s], crcNum set to [%u].",
        tag.c_str(), crcInfo.crcNum);
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::GenerateCheckFrame(HcclCheckInfo &checkInfo, const std::string &tag)
{
    // 初始化用于发送的BUFFER
    // 对入参的结构体变量指向的内存进行初始化时，使用了变量的结构体类型大小进行初始化，
    // 如果指针不为空，可以不检查初始化安全函数的返回值
    u64 checkInfoLen = sizeof(HcclCheckInfo);
    (void)memset_s(&checkInfo, checkInfoLen, 0, checkInfoLen);

    // 添加CRC字段到校验帧
    u32 crcLen = crcTable_.size();
    checkInfo.crcInfoGlobal.configFileExist_ = configFileExist_;
    if (crcLen != 0) {
        CHK_PRT_RET(crcLen > MAX_CRC_LEN,
            HCCL_ERROR("[RankConsistentcyChecker][GenerateCheckFrame]crc num[%u] is too big.", crcLen),
            HCCL_E_INTERNAL);
        checkInfo.crcInfoGlobal.crcNum = crcLen;
        CHK_RET(GetCrc(crcLen, &checkInfo.crcInfoGlobal.crcArray[0]));
    }
    // 添加CMD参数信息到校验帧
    auto getResult = infoFlagCmdMap_.find(tag);
    if (getResult != infoFlagCmdMap_.end()) {
        CHK_PRT_RET(GetOpParaByTag(tag, checkInfo.cmdInfo) != HCCL_SUCCESS,
            HCCL_ERROR("[RankConsistentcyChecker][GenerateCheckFrame]get Op para by tag[%s] fail.", tag.c_str()),
            HCCL_E_INTERNAL);
        checkInfo.crcInfoOp.configFileExist_ = configFileExist_;
        // 添加CRC字段到校验帧
        CHK_PRT_RET(GetCrcByTag(tag, checkInfo.crcInfoOp) != HCCL_SUCCESS,
            HCCL_ERROR("[RankConsistentcyChecker][GenerateCheckFrame]get ranktable crc by tag[%s] fail.", tag.c_str()),
            HCCL_E_INTERNAL);
    }
    // 添加HCCL版本信息到校验帧
    if (infoFlagVer_) {
        HCCL_DEBUG("version information is [%s].", cannVersion_);
        s32 sret = memcpy_s(checkInfo.version, MAX_CANN_VERSION_LEN + 1, cannVersion_, strlen(cannVersion_));
        CHK_PRT_RET(sret != EOK,
            HCCL_ERROR("[RankConsistentcyChecker][GenerateCheckFrame] memcpy failed. errorno [%d].",
            sret), HCCL_E_MEMORY);
    }
    // 添加拉远通信传输类型校验
    // 910* 不会配置isTcpMode，因此910*在此处的待校验值是一致的
    checkInfo.protocolType = protocolType_;
    HCCL_INFO("loc protocolType is [%d].", checkInfo.protocolType);

    return HCCL_SUCCESS;
}

bool RankConsistentcyChecker::CompareSection(const char *pRawData, const char *recvBuf, u32 len)
{
    for (u32 i = 0; i < len; i++) {
        if (*(pRawData + i) != *(recvBuf + i)) {
            return false;
        }
    }
    return true;
}

bool RankConsistentcyChecker::CompareCrcInfo(const std::string &tag, HcclCRCInfo &crcInfo, HcclCRCInfo &crcInfoRecv)
{
    bool bIsDiff = false;
    // 检校验整体是否一致
    if (!CompareSection(reinterpret_cast<char_t *>(&crcInfo), reinterpret_cast<char_t *>(&crcInfoRecv), sizeof(crcInfo))) {
        bIsDiff = true;
        // 检查每种CRC类型是否一致
        for (auto i = 0U; i < crcInfo.crcNum; ++i) {
            if (crcInfo.crcArray[i] != crcInfoRecv.crcArray[i]) {
                ReportCrcCheckFailed(tag, static_cast<HcclCrcRecordType>(i), crcInfo.crcArray[i],
                    crcInfoRecv.crcArray[i]);
            }
        }
    }
    return bIsDiff;
}

void RankConsistentcyChecker::ReportCmdInfoCheckFailed(const std::string &tag, const std::string &paraName,
    const std::string &localPara, const std::string &remotePara)
{
    RPT_INPUT_ERR(true, "EI0005", std::vector<std::string>({ "tag", "para_name", "local_para", "remote_para" }),
        std::vector<std::string>({ tag, paraName, localPara, remotePara }));
    HCCL_ERROR(
        "[RankConsistentcyChecker][ReportCmdInfoCheckFailed]CMD information %s check fail. local[%s], remote[%s]",
        paraName.c_str(), localPara.c_str(), remotePara.c_str());
}

void RankConsistentcyChecker::ReportCmdInfoCheckFailed(const std::string &tag, const std::string &paraName,
    uint32_t localPara, uint32_t remotePara)
{
    RPT_INPUT_ERR(true, "EI0005", std::vector<std::string>({ "tag", "para_name", "local_para", "remote_para" }),
        std::vector<std::string>({ tag, paraName, std::to_string(localPara), std::to_string(remotePara) }));
    HCCL_ERROR(
        "[RankConsistentcyChecker][ReportCmdInfoCheckFailed]CMD information %s check fail. local[%u], remote[%u]",
        paraName.c_str(), localPara, remotePara);
}

void RankConsistentcyChecker::ReportCrcCheckFailed(const std::string &tag, HcclCrcRecordType crcType,
    const uint32_t localCrc, const uint32_t remoteCrc)
{
    const auto crcTypeStr = GetCRCTypeEnumStr(crcType);
    RPT_INPUT_ERR(true, "EI0005", std::vector<std::string>({ "tag", "para_name", "local_para", "remote_para" }),
        std::vector<std::string>({ tag, crcTypeStr, std::to_string(localCrc), std::to_string(remoteCrc) }));
    HCCL_ERROR(
        "[RankConsistentcyChecker][ReportCrcCheckFailed]CRC for %s check fail. local[%u], remote[%u]",
        crcTypeStr.c_str(), localCrc, remoteCrc);
}

void RankConsistentcyChecker::CompareCmdInfo(HcclCheckInfo &checkInfo, HcclCheckInfo &checkInfoRecv)
{
    auto localInfo = &checkInfo.cmdInfo;
    auto remoteInfo = &checkInfoRecv.cmdInfo;

    if (!CompareSection(localInfo->tag, remoteInfo->tag, TAG_MAX_LEN + 1)) {
        ReportCmdInfoCheckFailed(localInfo->tag, "tag", localInfo->tag, remoteInfo->tag);
    }

    if (localInfo->cmdType != remoteInfo->cmdType) {
        ReportCmdInfoCheckFailed(localInfo->tag, "cmdType",
            (uint32_t)localInfo->cmdType, (uint32_t)remoteInfo->cmdType);
    }

    if (localInfo->count != remoteInfo->count) {
        ReportCmdInfoCheckFailed(localInfo->tag, "count", localInfo->count, remoteInfo->count);
    }

    if (localInfo->dataType != remoteInfo->dataType) {
        ReportCmdInfoCheckFailed(localInfo->tag, "dataType",
            (uint32_t)localInfo->dataType, (uint32_t)remoteInfo->dataType);
    }

    if (localInfo->op != remoteInfo->op) {
        ReportCmdInfoCheckFailed(localInfo->tag, "op", (uint32_t)localInfo->op, (uint32_t)remoteInfo->op);
    }

    if (!CompareSection(localInfo->group, remoteInfo->group, GROUP_NAME_MAX_LEN + 1)) {
        ReportCmdInfoCheckFailed(localInfo->tag, "group", localInfo->group, remoteInfo->group);
    }

    if (localInfo->root != remoteInfo->root) {
        ReportCmdInfoCheckFailed(localInfo->tag, "root", localInfo->root, remoteInfo->root);
    }

    if (localInfo->rank != remoteInfo->rank) {
        ReportCmdInfoCheckFailed(localInfo->tag, "rank", localInfo->rank, remoteInfo->rank);
    }

    if (localInfo->srTag != remoteInfo->srTag) {
        ReportCmdInfoCheckFailed(localInfo->tag, "srTag", localInfo->srTag, remoteInfo->srTag);
    }

    if (localInfo->inCclBufferSize != remoteInfo->inCclBufferSize) {
        ReportCmdInfoCheckFailed(localInfo->tag, "inCclBufferSize", localInfo->inCclBufferSize,
            remoteInfo->inCclBufferSize);
    }

    if (localInfo->outCclBufferSize != remoteInfo->outCclBufferSize) {
        ReportCmdInfoCheckFailed(localInfo->tag, "outCclBufferSize", localInfo->outCclBufferSize,
            remoteInfo->outCclBufferSize);
    }

    return;
}

bool RankConsistentcyChecker::CompareFrame(HcclCheckInfo &checkInfo, HcclCheckInfo &checkInfoRecv)
{
    bool bIsDiff = false;
    if (CompareCrcInfo(checkInfo.cmdInfo.tag, checkInfo.crcInfoGlobal, checkInfoRecv.crcInfoGlobal)) {
        HCCL_ERROR("[RankConsistentcyChecker][CompareFrame]errNo[0x%016llx] CRC check fail, please check the "
            "rankTable file and hccl_config file.", HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        bIsDiff = true;
    }
    if (CompareCrcInfo(checkInfo.cmdInfo.tag, checkInfo.crcInfoOp, checkInfoRecv.crcInfoOp)) {
        HCCL_ERROR("[RankConsistentcyChecker][CompareFrame]errNo[0x%016llx] Op CRC check fail, please check the op"
            " paramaters, rankTable file and hccl_config file.", HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        bIsDiff = true;
    }
    if (!CompareSection(reinterpret_cast<char_t *>(&checkInfo.cmdInfo),
        reinterpret_cast<char_t *>(&checkInfoRecv.cmdInfo), sizeof(checkInfo.cmdInfo))) {
        CompareCmdInfo(checkInfo, checkInfoRecv);
        HCCL_ERROR("[RankConsistentcyChecker][CompareFrame]errNo[0x%016llx] CMD check fail",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        bIsDiff = true;
    }
    HCCL_INFO("loc protocolType is [%d], rem protocolType is [%d].",
        checkInfo.protocolType, checkInfoRecv.protocolType);
    if (checkInfo.protocolType != checkInfoRecv.protocolType) {
        HCCL_ERROR("[RankConsistentcyChecker][CompareFrame]errNo[0x%016llx] ProtocolType check fail",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        bIsDiff = true;
    }

    // Cann版本校验，只在集合通信场景校验CANN版本
    if (cannVerCheckSwitch_) {
        std::string localCannVersion = checkInfo.version;
        std::string remoteCannVersion = checkInfoRecv.version;
        if (localCannVersion.empty() || remoteCannVersion.empty()) { // cann版本信息读取失败，返回告警
            HCCL_WARNING("[RankConsistentcyChecker][CompareFrame] Cann version str is empty. local_version %s,"
                "remote_version %s.", checkInfo.version, checkInfoRecv.version);
        } else if (localCannVersion != remoteCannVersion) { // cann版本信息读取成功，且版本不一致
            RPT_INPUT_ERR(true, "EI0008", std::vector<std::string>({"tag", "local_version", "remote_version"}),
                std::vector<std::string>({checkInfo.cmdInfo.tag, localCannVersion, remoteCannVersion}));
            HCCL_ERROR("[RankConsistentcyChecker][CompareFrame] errNo[0x%016llx] Inconsistent CANN Versions."
                "local_version %s, remote_version %s.", HCCL_ERROR_CODE(HCCL_E_INTERNAL),
                checkInfo.version, checkInfoRecv.version);
            bIsDiff = true;
        }
    }
    return bIsDiff;
}

HcclResult RankConsistentcyChecker::AddCrc(const u32 crcValue)
{
    HCCL_DEBUG("crcValue[%u].", crcValue);
    crcTable_.push_back(crcValue);
    HCCL_DEBUG("num[%llu].", crcTable_.size());
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::ClearCrcInfo(void)
{
    this->crcTable_.clear();
    if (this->crcTable_.size() != 0) {
        HCCL_ERROR("[Clear][CrcInfo]errNo[0x%016llx] clear crcTable_ is failed", HCCL_ERROR_CODE(HCCL_E_INTERNAL));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult RankConsistentcyChecker::GetCrc(u32 num, u32 *crcAddr)
{
    CHK_PTR_NULL(crcAddr);
    HCCL_DEBUG("num[%u], crc[%u].", num, *crcAddr);

    if (num == 0) {
        HCCL_ERROR("[Get][Crc]errNo[0x%016llx] In get crc the value of num is 0", HCCL_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (num != crcTable_.size()) {
        HCCL_ERROR("[Get][Crc]errNo[0x%016llx] num error inputNum[%u], localNum[%llu]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), num, crcTable_.size());
        return HCCL_E_INTERNAL;
    }

    for (u32 i = 0; i < num; i++) {
        crcAddr[i] = crcTable_[i];
    }
    return HCCL_SUCCESS;
}
}