/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_base.h"
#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "workflow_pub.h"
#include "param_check_pub.h"
#include "rank_consistentcy_checker.h"
#include "externalinput_pub.h"
#include "env_config.h"
#include "../common/src/topo/topoinfo_detect.h"
#include "../common/src/topo/topoinfo_ranktable_partition.h"
#include "../common/src/state_guard.h"
#include "sal_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_prof.h"
#include "adapter_qos_pub.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "mem_host_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "external/runtime/rt_error_codes.h"
#include "mmpa_api.h"

#define DOUBLE_SIZE 2

using namespace std;
using namespace hccl;

const int32_t MC2_TILING_VERSION_DEFAULT = 1;
const int32_t MC2_TILING_VERSION_V2 = 2;
const int64_t MC2_TILING_OFFSET = 1;
const std::string HCCL_ALLTOALL = "ALLTOALL";
const std::string HCCL_ALLTOALLV = "ALLTOALLV";
const std::string HCCL_ALLTOALLVC = "ALLTOALLVC";

thread_local map<std::string, shared_ptr<TopoInfoDetect>> g_topoDetectServerPtrMap;

HcclResult GetCaptureInfo(aclrtStream stream, std::string& captureInfo, bool& isCapture)
{   
    isCapture = false;
    DevType devType;   
    CHK_RET(hrtGetDeviceType(devType));
    if(GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_WARNING("[%s]Stream capture only support opbase mode!", __func__);
        return HCCL_SUCCESS;
    }
    rtStreamCaptureStatus captureStatus = rtStreamCaptureStatus::RT_STREAM_CAPTURE_STATUS_NONE;
    rtModel_t rtModel = nullptr;
    rtError_t ret = rtStreamGetCaptureInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support!", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != RT_ERROR_NONE,
            HCCL_ERROR("[%s]rtGet stream get capture status fail. return[%d]", __func__, ret), HCCL_E_RUNTIME);
    }
    std::string modelstr;
    if (captureStatus == RT_STREAM_CAPTURE_STATUS_ACTIVE) {
        isCapture = true;
        uint32_t modelId = 0;
        ret = rtModelGetId(rtModel, &modelId);
        CHK_PRT_RET(ret != RT_ERROR_NONE,
            HCCL_ERROR("[%s]rtGet stream get capture model id fail. return[%d]", __func__, ret), HCCL_E_RUNTIME);
        modelstr = to_string(modelId);
    } else {
        modelstr = "none";
    }
    captureInfo = ", capture status[" + to_string(captureStatus) + "], model id[" + modelstr + "].";
    ProfilingManagerPub::SetCaptureStatus(isCapture);
    return HCCL_SUCCESS;
}

HcclResult CallMsprofReportHostApi(hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count,
    HcclDataType dataType, std::string tag)
{
    if (GetIfProfile()) {
        AlgType algType;
        if(cmdType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV){
            algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_PAIRWISE;
            algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
        } else if (cmdType == HcclCMDType::HCCL_CMD_SEND || cmdType == HcclCMDType::HCCL_CMD_RECEIVE){
            algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
            algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
        } else {
            CHK_RET(hcclComm->GetAlgType(algType, cmdType));
        }

        u32 blockDim = 0;
        hcclComm->GetBlockDim(blockDim);

        uint64_t groupName = hrtMsprofGetHashId(hcclComm->GetIdentifier().c_str(), hcclComm->GetIdentifier().length());
        CHK_RET_AND_PRINT_IDE(ProfilingManagerPub::CallMsprofReportHostApi(cmdType, beginTime, count, dataType, algType,
            groupName, blockDim), tag.c_str());
    }
    return HCCL_SUCCESS;
}

thread_local s32 g_hcclDeviceId = INVALID_INT;
std::mutex g_opHcomInfosMutex{};
HcclOpInfoCtx g_opHcomInfos[MAX_MODULE_DEVICE_NUM + 1];

HcclResult HcclGetDeviceId(void)
{
    if (g_hcclDeviceId == INVALID_INT) {
        CHK_PRT_RET(hrtGetDevice(&g_hcclDeviceId) != HCCL_SUCCESS,
            HCCL_WARNING("[HcclGetDeviceId] get fail deviceLogicId[%d]", g_hcclDeviceId), HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(static_cast<u32>(g_hcclDeviceId) >= MAX_MODULE_DEVICE_NUM,
        HCCL_WARNING("[HcclGetDeviceId]deviceLogicId[%d] is bigger than HCCL_AISERVER_DEVICE_NUM_MAX:[%u]",
        g_hcclDeviceId, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_INFO("[HcclGetDeviceId] deviceLogicId[%d] ", g_hcclDeviceId);
    return HCCL_SUCCESS;
}

s32 HcclGetThreadDeviceId()
{
    CHK_PRT_RET(HcclGetDeviceId() != HCCL_SUCCESS, HCCL_WARNING("[HcclGetThreadDeviceId] get fail deviceLogicId[%d]",
        g_hcclDeviceId), INVALID_INT);
    return g_hcclDeviceId;
}

HcclOpInfoCtx &GetHcclOpInfoCtx(void)
{
    if (HcclGetDeviceId() == HCCL_SUCCESS) {
        std::lock_guard<std::mutex> lock(g_opHcomInfosMutex);
        if (!g_opHcomInfos[g_hcclDeviceId].isUsed) {
            HCCL_INFO("[GetHcclOpInfoCtx] Set device, use g_hcclDeviceId[%d] ", g_hcclDeviceId);
            if (g_opHcomInfos[MAX_MODULE_DEVICE_NUM].isUsed) {
                g_hcclDeviceId = MAX_MODULE_DEVICE_NUM;
                HCCL_INFO("[GetHcclOpInfoCtx] Used cover bottom g_hcclDeviceId[%d]", g_hcclDeviceId);
                return g_opHcomInfos[g_hcclDeviceId];
            }
        }
        g_opHcomInfos[g_hcclDeviceId].isUsed = true;
        return g_opHcomInfos[g_hcclDeviceId];
    }

    std::lock_guard<std::mutex> lock(g_opHcomInfosMutex);
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        if (g_opHcomInfos[i].isUsed) {
            g_hcclDeviceId = i;
            HCCL_INFO("[GetHcclOpInfoCtx] Not set device, Used g_hcclDeviceId[%u] ", i);
            return g_opHcomInfos[g_hcclDeviceId];
        }
    }
    g_hcclDeviceId = MAX_MODULE_DEVICE_NUM;
    g_opHcomInfos[MAX_MODULE_DEVICE_NUM].isUsed = true;
    HCCL_INFO("[GetHcclOpInfoCtx] Used cover bottom g_hcclDeviceId[%d]", g_hcclDeviceId);
    return g_opHcomInfos[MAX_MODULE_DEVICE_NUM];
}

HcclResult GetDeviceComm(uint32_t ndev, const HcclRootInfo &rootHandle, const s32 rank, const s32 logicDeviceId,
    HcclComm &comm)
{
    //给当前线程添加名字
    SetThreadName("Hccl_GetDevComm");

    CHK_PRT_RET(hrtSetDevice(logicDeviceId) != HCCL_SUCCESS,
        HCCL_ERROR("[GetDeviceComm] set fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
    HcclResult ret = HcclCommInitRootInfo(ndev, &rootHandle, rank, &comm);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        comm = nullptr;
        HCCL_ERROR("[GetDeviceComm] rank[%d] Get device comm failed!", rank);
        CHK_PRT_RET(hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[GetDeviceComm] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
        return ret;
    }
    hcclComm *pComm = static_cast<hcclComm *>(comm);
    pComm->ResetDeviceEnable();
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommAll(uint32_t ndev, int32_t *devices, HcclComm *comms)
{
    // 入参校验
    CHK_PRT_RET(ndev == 0, HCCL_ERROR("[HcclGetCommAll] ndev is unvalid ndev[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    //给当前线程添加名字
    SetThreadName("Hccl_GetCommAll");

    CHK_PRT_RET(hrtSetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCommAll] set fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);

    // 获取通信域之前, 先把所有通信域设置为空
    for (uint32_t i = 0; i < ndev; i++) {
        comms[i] = nullptr;
    }

    HcclRootInfo rootHandle;
    CHK_RET(HcclGetRootInfo(&rootHandle));
    
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(new (std::nothrow) std::thread(&GetDeviceComm, ndev, std::ref(rootHandle), rankId,
            devices[rankId], std::ref(comms[rankId])));
        CHK_PRT_RET(!threads[rankId], HCCL_ERROR("[HcclGetCommAll]threads[%u] reset failed ", rankId), HCCL_E_INTERNAL);
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域初始化失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_ERROR("[HcclGetCommAll] rank[%u] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(hrtResetDevice(devices[0]) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclGetCommAll] reset fail devices[0][%d]", devices[0]), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitAll(uint32_t ndev, int32_t *devices, HcclComm *comms)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参校验
    CHK_PRT_RET(ndev <= 0, HCCL_ERROR("[HcclCommInitAll] ndev is unvalid ndev:[%u]", ndev), HCCL_E_PARA);
    CHK_PTR_NULL(comms);
    CHK_PTR_NULL(devices);

    // 判断设备List中是否有重复id,报错退出
    set<int32_t> devSet(devices, devices + ndev);
    uint32_t devSetSize = devSet.size();
    CHK_PRT_RET((devSetSize != ndev),
        HCCL_ERROR("[HcclCommInitAll] Duplicate device id exist in the device list. devSetSize:[%u], ndev:[%u]",
        devSetSize, ndev),
        HCCL_E_PARA);

    std::future<HcclResult> threadResult;
    std::unique_ptr<std::thread> getCommThread;
    getCommThread.reset(new (std::nothrow) std::thread(
        [=, &threadResult]() { threadResult = std::async(std::launch::async, HcclGetCommAll, ndev, devices, comms); }));
    CHK_PRT_RET(!getCommThread, HCCL_ERROR("[HcclCommInitAll]thread reset failed "), HCCL_E_INTERNAL);
    getCommThread->join();

    HcclResult ret = threadResult.get();
    if (ret != HCCL_SUCCESS) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
                comms[i] = nullptr;
            }
        }
        HCCL_ERROR("HcclCommInitAll failed! threadResult[%d]", ret);
        return ret;
    }
    HCCL_RUN_INFO("HcclCommInitAll success, take time [%lld]us, deviceLogicId[%d]", DURATION_US(TIME_NOW() - startut),
        deviceLogicId);
    return HCCL_SUCCESS;
}

std::map<s32, std::unordered_map<std::string, std::shared_ptr<HcclOpInfoCtx>>> g_oneSidedCommHcomInfos;
std::set<HcclComm> g_oneSidedCommSet;

/* 仅提供判断功能, 调用前需校验参数有效性*/
bool IsOneSidedComm(HcclComm comm)
{
    return g_oneSidedCommSet.find(comm) != g_oneSidedCommSet.end();
}

/* 仅提供判断功能, 调用前需校验参数有效性*/
bool IsCommNameExistInOneSidedComms(s32 deviceLogicId, const std::string &commName)
{
    bool exist = g_oneSidedCommHcomInfos.count(deviceLogicId) != 0 &&
                g_oneSidedCommHcomInfos[deviceLogicId].count(commName) != 0;
    if (exist && g_oneSidedCommHcomInfos[deviceLogicId][commName]->isUsed) {
        return true;
    }
    return false;
}

HcclResult DeInitOneSidedHcomInfo(s32 deviceLogicId, const std::string &commName)
{
    CHK_PRT_RET(deviceLogicId == INVALID_INT,
                HCCL_ERROR("[HcclCommDestroy][DeInitOneSidedHcomInfo] deviceLogicId is error."),
                HCCL_E_PARA);
    CHK_PRT_RET(commName.empty(),
                HCCL_ERROR("[HcclCommDestroy][DeInitOneSidedHcomInfo] commName is error."),
                HCCL_E_PARA);
    g_oneSidedCommHcomInfos[deviceLogicId].erase(commName);
    return HCCL_SUCCESS;
}

/*
 * g_oneSidedCommHcomInfos 初始化
 * s32 deviceLogicId
 * const string &commName : 通信域名，用户确保全局唯一
 */
HcclResult InitOneSidedHcomInfo(s32 deviceLogicId, const std::string &commName)
{
    CHK_PRT_RET(deviceLogicId == INVALID_INT,
                HCCL_ERROR("[InitOneSidedHcomInfo] deviceLogicId is error."),
                HCCL_E_PARA);
    CHK_PRT_RET(commName.empty(),
                HCCL_ERROR("[InitOneSidedHcomInfo] commName is error."),
                HCCL_E_PARA);
    // comm name exit && isUsed = true
    bool isCommNameExist = IsCommNameExistInOneSidedComms(deviceLogicId, commName);
    CHK_PRT_RET(isCommNameExist, HCCL_ERROR("[Init][InitOneSidedHcomInfo] comm Name exist."), HCCL_E_PARA);
    // 确保 deviceLogicId 和 commName 的 map 已经被初始化
    if (g_oneSidedCommHcomInfos.find(deviceLogicId) == g_oneSidedCommHcomInfos.end()) {
        g_oneSidedCommHcomInfos[deviceLogicId] = {};
    }
    // comm name not exit
    if (g_oneSidedCommHcomInfos[deviceLogicId].count(commName) == 0) {
        std::shared_ptr<HcclOpInfoCtx> opBaseHcomPtr;
        EXECEPTION_CATCH((opBaseHcomPtr = std::make_shared<HcclOpInfoCtx>()), return HCCL_E_PARA);
        g_oneSidedCommHcomInfos[deviceLogicId][commName] = opBaseHcomPtr;
    }
    // comm name exit && isUsed = False
    g_oneSidedCommHcomInfos[deviceLogicId][commName]->isUsed = true;
    return HCCL_SUCCESS;
}

HcclOpInfoCtx &GetOneSidedOpInfoCtx(s32 deviceLogicId, const std::string &commName)
{
    std::shared_ptr<HcclOpInfoCtx> oneSidedHComPtr = g_oneSidedCommHcomInfos[deviceLogicId][commName];
    return *oneSidedHComPtr;
}

HcclResult CheckOpBasedHcom(HcclOpInfoCtx &opBaseHcom, const uint32_t rank, const CommConfig &commConfig)
{
    /* 防止重复调用初始化 */
    CHK_PRT_RET((opBaseHcom.pComm != nullptr), HCCL_ERROR("[Init][CheckOpBasedHcom]errNo[0x%016llx] rank[%u] "\
        "op_base hccl multiple initialization", HCCL_ERROR_CODE(HCCL_E_UNAVAIL), rank), HCCL_E_UNAVAIL);
    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR("[Init][CheckOpBasedHcom]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
                HCCL_ERROR_CODE(HCCL_E_PARA),
                commIdentifier.c_str()),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult InitCommClusterInfo(std::string &rankTableM, const uint32_t rank, const CommConfig &commConfig,
    HcclOpInfoCtx& opBaseHcom, HcclComm *comm)
{
    u32 rankTableSize = 0;
    HcclResult ret = HcomCheckRankTable(rankTableM.c_str(), rankTableSize);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommClusterInfo]check rankTable string error, rankTableSize [%u].",
            rankTableSize), HCCL_E_PARA);

    const std::string commIdentifier = commConfig.GetConfigCommName();
    opBaseHcom.pComm.reset(
        new (std::nothrow) hccl::hcclComm(
            commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier));
    CHK_PTR_NULL(opBaseHcom.pComm);

    /* --------------初始化------------------------- */
    bool errorFlag = false;
    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        ret = InitOtherInfo(opBaseHcom.params, rankTableM.c_str());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init other Info.",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        HCCL_INFO("rootInfo[%s]", opBaseHcom.params.id.internal);

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] init work flow mode error.",
                HCCL_ERROR_CODE(ret)), errorFlag = true);

        ret = CfgGetClusterInfo(rankTableM, to_string(rank), opBaseHcom.params, opBaseHcom.rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx]"\
                "info error:rank[%u]", HCCL_ERROR_CODE(ret), rank), errorFlag = true);

        ret = opBaseHcom.pComm->init(opBaseHcom.params, opBaseHcom.rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] hcclComm init error.",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 设置确定性计算配置 */
        ret = opBaseHcom.pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set deterministic error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = opBaseHcom.pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set TC and SL error or Invalid configuration parameter.",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 设置AIV模式 */
        ret = opBaseHcom.pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = opBaseHcom.pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set aicpu error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = ShowRanktableConfigInfo(opBaseHcom.cloudFlag, opBaseHcom.params,
            opBaseHcom.rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] put ranktable info error.",
                HCCL_ERROR_CODE(ret)), errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *comm = opBaseHcom.pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[opBaseHcom.pComm->GetIdentifier()] = opBaseHcom.pComm;

	// 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将pComm赋值到hcomInfo.pComm
	if (opBaseHcom.pComm->GetIdentifier() == HCCL_WORLD_GROUP) {
	    HcomGetCtxHomInfo().pComm = opBaseHcom.pComm;
	}
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[Init][CommClusterInfo]HcclCommInitClusterInfo failed, rankNum[%u], rank[%u], server[%s],"\
            "device[%d], return[0x%016llx]", opBaseHcom.rankTable.rankNum, rank,
            opBaseHcom.params.serverId.c_str(), opBaseHcom.params.logicDevId, HCCL_ERROR_CODE(ret));
        (void)HcclCommDestroy(opBaseHcom.pComm.get());
        return ret;
    }

    /* 关键状态记录 */
    HCCL_INFO("%s success, rankNum[%u], rank[%u], server[%s], device[%d].",
        __func__, opBaseHcom.rankTable.rankNum, rank, opBaseHcom.params.serverId.c_str(),
        opBaseHcom.params.logicDevId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    HCCL_RUN_INFO("Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].",
        __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.",
        __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.",
        __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][HcclCommInitClusterInfo]errNo[0x%016llx], clusterInfo[%s], rank[%u], "
        "load rankTable error.", HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank), HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx &opBaseHcom = GetHcclOpInfoCtx();
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].",
        __func__, DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoMemConfig(const char *rankTableString, uint32_t rank,
    HcclCommConfig *config, HcclComm *comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参合法性校验
    CHK_PTR_NULL(rankTableString);
    CHK_PTR_NULL(config);
    CHK_PTR_NULL(config->hcclCommName);
    CHK_PTR_NULL(comm);

    HCCL_RUN_INFO("Entry-%s: rankTableString[%s], rank[%u], deviceLogicId[%d].",
        __func__, rankTableString, rank, deviceLogicId);

    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error", __func__, HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
    HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)),
        HCCL_E_PARA);

    CHK_PRT_RET(strlen(config->hcclCommName) == 0,
        HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] hcclCommName is error."),
        HCCL_E_PARA);

    std::string rankTableM(rankTableString);
    std::string identifier = config->hcclCommName;
    CommConfig commConfig(identifier);
    HCCL_RUN_INFO("Entry-%s: %s", "hcclCommName", identifier.c_str());

    /* 读取用户配置 */
    ret = commConfig.Load(config);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig]errNo[0x%016llx] load comm config failed.",
        HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    CHK_PRT_RET(deviceLogicId == INVALID_INT,
        HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] deviceLogicId is error."),
        HCCL_E_PARA);

    bool isCommNameExist = IsCommNameExistInOneSidedComms(deviceLogicId, identifier);
    CHK_PRT_RET(isCommNameExist, HCCL_ERROR("[Init][HcclCommInitClusterInfoMemConfig] comm Name exist."), HCCL_E_PARA);

    CHK_RET(InitOneSidedHcomInfo(deviceLogicId, identifier));

    const std::string commIdentifier = commConfig.GetConfigCommName();
    HcclOpInfoCtx &oneSidedHCom = GetOneSidedOpInfoCtx(deviceLogicId, commIdentifier);

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, oneSidedHCom, comm));

    g_oneSidedCommSet.insert(*comm);

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]%s success, take time [%lld]us, rankTableString[%s], rank[%u], deviceLogicId[%d].",
        __func__, DURATION_US(TIME_NOW() - startut), rankTableString, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config,
    HcclComm *comm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    HCCL_RUN_INFO("Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].",
        __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    // 检查配置参数是否为空
    RPT_INPUT_ERR(config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),
        std::vector<std::string>({"HcclCommInitClusterInfoConfig", "config", "nullptr", "please check comm"}));
    CHK_SMART_PTR_NULL(config);

    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.",
        __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.",
        __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.",
        __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][HcclCommInitClusterInfoConfig]errNo[0x%016llx] clusterInfo[%s] rank[%u] "
        "load rankTable error.", HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank), HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx &opBaseHcom = GetHcclOpInfoCtx();
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].",
        __func__, DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfigInner(hccl::hcclComm *globalComm, uint32_t rankNum, uint32_t *rankIds,
    uint32_t subCommRankId, CommConfig &commConfig, HcclComm *subComm)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams globalParams{};
    RankTable_t globalRankTable{};
    CHK_RET(globalComm->GetCommParams(globalParams));
    CHK_RET(globalComm->GetCommRankTable(globalRankTable));

    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();

    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR("[%s]errNo[0x%016llx]The comm name[%s] already exists in Group2Comm map.",
            __func__, HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA);

    std::shared_ptr<hccl::hcclComm> pComm;
    pComm.reset(new (std::nothrow) hccl::hcclComm(
            commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier));
    CHK_PTR_NULL(pComm);

    bool errorFlag = false;
    hccl::HcclCommParams subParams{};
    hccl::RankTable_t subRankTable{};
    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关

        std::unique_ptr<TopoinfoRanktablePartition> pTopoPartition;
        pTopoPartition.reset(new (std::nothrow) hccl::TopoinfoRanktablePartition(globalParams, globalRankTable));
        CHK_RET(pTopoPartition->GenerateSubRankTable(rankNum, rankIds, subRankTable));
        CHK_RET(pTopoPartition->GenerateSubParams(subRankTable, subCommRankId, subParams));

        std::string rankTableM = "";
        CHK_RET(pTopoPartition->GetRankTableStr(subRankTable, rankTableM));

        ret = InitOtherInfo(subParams, rankTableM.c_str());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init other Info.",
            __func__, HCCL_ERROR_CODE(ret)), errorFlag = true);
        ret = pComm->init(subParams, subRankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] hcclComm init error.",
            __func__, HCCL_ERROR_CODE(ret)), errorFlag = true);
        HCCL_INFO("[HcclCreateSubCommConfigInner]comm id[%s]", subParams.id.internal);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set deterministic error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set TC and SL error",
            __func__, HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 设置AIV模式 */
        ret = pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set aivMode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        
        /* 设置AICPU */
        ret = pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set aicpu error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        
        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] init workflow mode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = DisplayRanktableInfo(subRankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] print ranktable info error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *subComm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
    } while(0);

    if (errorFlag) {
        HCCL_ERROR("[%s]Create sub communication failed, return[0x%016llx], " \
            "rankNum[%u], subCommRankId[%u], sub commm identifier[%s], server[%s], logicDevId[%d]",
            __func__, HCCL_ERROR_CODE(ret), rankNum, subCommRankId, commIdentifier.c_str(),
            GetLocalServerId(subParams.serverId).c_str(), subParams.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }

    HCCL_RUN_INFO("%s success, sub commm identifier[%s], rankNum[%u], rank[%u], server[%s], device[%d].",
        __func__, commIdentifier.c_str(), subRankTable.rankNum, subCommRankId,
        subParams.serverId.c_str(), subParams.logicDevId);
    return HCCL_SUCCESS;
}

HcclResult SubCommIsOneSidedComm(HcclComm *comm)
{
    if (IsOneSidedComm(*comm)) {
        HCCL_ERROR("[%s]errNo[0x%016llx] oneSidedComm does not support create sub comm.",
            __func__, HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT));
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum, uint32_t *rankIds,
    uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig *config, HcclComm *subComm)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    HCCL_RUN_INFO("Entry-%s: rankNum[%u], rank[%u], deviceLogicId[%d]",
        __func__, rankNum, subCommRankId, deviceLogicId);
    CHK_SMART_PTR_NULL(subComm);
    if (*subComm != nullptr) {
        HCCL_WARNING("[%s]The value pointed by output param subComm is not nullptr. " \
            "Please be ware of possible memory leak.", __func__);
    }
    CHK_PRT_RET(rankIds == nullptr && subCommId == INVALID_SUBCOMM_ID,
        HCCL_RUN_INFO("[HCCL_TRACE]HcclCreateSubCommConfig return, rankIds is nullptr and subCommId is 0xFFFFFFFF, " \
            "this device is not in the sub comm, deviceLogicId[%u].", deviceLogicId), HCCL_SUCCESS);
    CHK_PRT_RET(rankIds == nullptr || subCommId == INVALID_SUBCOMM_ID,
        HCCL_ERROR("[%s]errNo[0x%016llx] " \
            "rankIds[%p] is nullptr xor subCommId[%llu] is invalid. " \
            "The two parameters should only be both valid or both invalid.",
            __func__, HCCL_ERROR_CODE(HCCL_E_PARA), rankIds, subCommId), HCCL_E_PARA);
    CHK_RET(SubCommIsOneSidedComm(comm));

    HcclResult ret = HCCL_SUCCESS;
    // 入参合法性校验
    CHK_PRT_RET((rankNum == 0), HCCL_ERROR("[%s]errNo[0x%016llx] Rank num cannot be zero.",
        __func__, HCCL_ERROR_CODE(HCCL_E_PARA)), HCCL_E_PARA);
    CHK_PRT_RET((subCommRankId >= rankNum), HCCL_ERROR("[%s]errNo[0x%016llx] subCommRankId[%u] should be less " \
        "than rankNum[%u].", __func__, HCCL_ERROR_CODE(HCCL_E_PARA), subCommRankId, rankNum), HCCL_E_PARA);

    RPT_INPUT_ERR(config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateSubCommConfig", "config", "nullptr", "please check comm"}));
    CHK_SMART_PTR_NULL(config);
    CHK_SMART_PTR_NULL(comm);

    ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    hccl::hcclComm *globalComm = static_cast<hccl::hcclComm*>(*comm);
    CHK_PTR_NULL(globalComm);

    std::string identifier = globalComm->GetIdentifier() + "_sub_" + to_string(subCommId);
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    CHK_RET(HcclCreateSubCommConfigInner(globalComm, rankNum, rankIds, subCommRankId, commConfig, subComm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]%s success, take time [%lld]us, " \
        "sub commm identifier[%s], rankNum[%u], rank[%u], deviceLogicId[%d]",
        __func__, DURATION_US(TIME_NOW() - startut), commConfig.GetConfigCommName().c_str(),
        rankNum, subCommRankId, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // input check
    CHK_PTR_NULL(rootInfo);
    HCCL_RUN_INFO("Entry-HcclGetRootInfo:rootInfo[%p], deviceLogicId[%d]", rootInfo, deviceLogicId);
    // get commId from env
    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXECEPTION_CATCH((topoDetectServer = std::make_shared<TopoInfoDetect>()),
        return HCCL_E_MEMORY);
    CHK_RET(topoDetectServer->SetupServer(rootHandle));

    if (sizeof(HcclRootHandle) > HCCL_ROOT_INFO_BYTES) {
        HCCL_ERROR("[Get][RootInfo]hccl root info overflow. max length: %u, actual:%zu, identifier[%s]",
            HCCL_ROOT_INFO_BYTES, sizeof(HcclRootHandle), rootHandle.identifier);
        return HCCL_E_INTERNAL;
    } else {
        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &rootHandle, sizeof(HcclRootHandle));
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Get][RootInfo]memcpy root info fail. errorno[%d] "\
            "params:destMaxSize[%u], count[%u]", sRet, HCCL_ROOT_INFO_BYTES,
            sizeof(HcclRootHandle)), HCCL_E_MEMORY);
    }

    HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
    EXECEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectServer.insert({rootHandle.identifier, topoDetectServer}),
        return HCCL_E_MEMORY);
    /* 首节点诊断信息记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]HcclGetRootInfo success, take time [%lld]us, identifier[%s]",
        DURATION_US(TIME_NOW() - startut), rootHandle.identifier);
    return HCCL_SUCCESS;
}

HcclResult GetSelfClusterInfo(const HcclBasicRankInfo &rankInfo, HcclCommParams &params)
{
    params.deviceType = rankInfo.deviceType;
    params.rank = rankInfo.rank;
    params.userRank = rankInfo.rank;
    params.logicDevId = rankInfo.deviceLogicID;
    params.totalRanks = rankInfo.rankSize;
    params.serverId = rankInfo.hostIP.GetReadableAddress();

    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm commHandle, char *commName)
{
    CHK_PTR_NULL(commHandle);
    CHK_PTR_NULL(commName);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(commHandle);
    s32 ret = strncpy_s(commName, ROOTINFO_INDENTIFIER_MAX_LENGTH, hcclComm->GetIdentifier().c_str(),
        hcclComm->GetIdentifier().size());
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("HcclGetCommName str copy fail. return[%d]", ret), HCCL_E_INTERNAL);
    HCCL_INFO("HcclGetCommName input handle=%p commName=%s", commHandle, commName);
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommHandle(const char *commName, std::shared_ptr<hccl::hcclComm> &comm)
{
    CHK_PTR_NULL(commName);
    std::string group(commName);

    s32 deviceLogicId = 0;
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && IsCommNameExistInOneSidedComms(deviceLogicId, commName)) {
        HcclOpInfoCtx &oneSidedHcom = GetOneSidedOpInfoCtx(deviceLogicId, commName);
        comm = oneSidedHcom.pComm;
        return HCCL_SUCCESS;
    }

    HcclOpInfoCtx &opBaseHcom = GetHcclOpInfoCtx();
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter == opBaseHcom.opGroup2CommMap.end()) {
        HCCL_WARNING("please check the group name is correct, group=%s", commName);
        return HCCL_E_PARA;
    } else {
        comm = iter->second;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommConnections(const HcclRootHandle &rootHandle, HcclCommConnections &commConnections)
{
    HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
    auto iterServer = opBaseInfo.hcclCommTopoInfoDetectServer.find(rootHandle.identifier);
    if (iterServer == opBaseInfo.hcclCommTopoInfoDetectServer.end()) {
        commConnections.isRoot = false;
    } else {
        commConnections.isRoot = true;
        CHK_RET(iterServer->second->GetServerConnections(commConnections.serverConnections));
    }

    auto iterAgent = opBaseInfo.hcclCommTopoInfoDetectAgent.find(rootHandle.identifier);
    if (iterAgent == opBaseInfo.hcclCommTopoInfoDetectAgent.end()) {
        HCCL_ERROR("hccl get agent connections failed, rootHandle.identifier=%s", rootHandle.identifier);
        return HCCL_E_PARA;
    } else {
        CHK_RET(iterAgent->second->GetAgentConnection(commConnections.agentConnection));
    }
    return HCCL_SUCCESS;
}

void HcclCloseCommConnections(const std::string &identifier)
{
    HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
    EXECEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectServer.erase(identifier), return);
    EXECEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectAgent.erase(identifier), return);
    return;
}

HcclResult InitCommRootInfo(const u32 nRanks, const u32 rank, const HcclRootHandle &rootHandle,
    const CommConfig &commConfig, HcclComm *comm)
{
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    std::shared_ptr<hccl::hcclComm> pComm;
    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();

    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(
        iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR("[Init][InitCommRootInfo]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
        HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA
    );

    hccl::HcclCommParams params;
    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        pComm.reset(new hccl::hcclComm(commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(),
            commIdentifier));
        CHK_SMART_PTR_NULL(pComm);
        std::shared_ptr<TopoInfoDetect> topoDetectAgent;
        EXECEPTION_CATCH((topoDetectAgent = std::make_shared<TopoInfoDetect>()),
            return HCCL_E_MEMORY);
        ret = topoDetectAgent->SetupAgent(nRanks, rank, rootHandle);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo]errNo[0x%016llx] setup topo detect error",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        RankTable_t rankTable;
        ret = topoDetectAgent->GetCluterInfo(rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo]errNo[0x%016llx] GetCluterInfo error",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 初始化hccl comm */
        HcclBasicRankInfo localRankInfo;
        ret = topoDetectAgent->GetLocalRankInfo(localRankInfo);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfo]errNo[0x%016llx] GetLocalRankInfo error.",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        ret = GetSelfClusterInfo(localRankInfo, params);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] GetRankInfo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = topoDetectAgent->WaitComplete(rootHandle);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] wait complete topo detect error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        CHK_RET(DisplayRanktableInfo(rankTable));

        ret = topoDetectAgent->GetAgentListenSocket(params.commPortConfig);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][RootInfo]HcclGetCommListenSockets failed."),
            errorFlag = true);

        bool retryEnable = GetExternalInputIntraServerRetryEnable() || GetExternalInputInterServerRetryEnable() ||
            GetExternalInputInterSuperPodRetryEnable();
        if (retryEnable) {
            EXECEPTION_CATCH(opBaseHcom.hcclCommTopoInfoDetectAgent.insert({ rootHandle.identifier, topoDetectAgent }),
                return HCCL_E_MEMORY);
            ret = HcclGetCommConnections(rootHandle, params.commConnections);
            CHK_PRT_BREAK(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][RootInfo]HcclGetCommConnections failed."),
                errorFlag = true);
        } else {
            ret = topoDetectAgent->Teardown();
            CHK_PRT_BREAK(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][RootInfo]errNo[0x%016llx] Teardown topo detect error", HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init work flow mode error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitOtherInfo(params, nullptr);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init other Info", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        HCCL_INFO("rootInfo[%s], params.logiceDevice[%d]", params.id.internal, params.logicDevId);
        ret = pComm->init(params, rankTable);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] hcclComm init error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set deterministic error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set TC and SL error or Invalid configuration parameter.",
            HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 设置AIV模式 */
        ret = pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set aicpu error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *comm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
        lock.unlock();

        // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将pComm赋值到hcomInfo.pComm
        if (pComm->GetIdentifier() == HCCL_WORLD_GROUP) {
            HcomGetCtxHomInfo().pComm = pComm;
        }

        ret = HcomSetGroupTopoInfo(pComm->GetIdentifier().c_str(), nRanks);
        CHK_PRT_BREAK(ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] setGroupTopoInfo error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);
    } while (0);

    if (errorFlag) {
        HCCL_ERROR("[InitCommRootInfo]Init failed, return[0x%016llx], rankNum[%u], rank[%u], "\
            "rootInfo identifier[%s], server[%s], logicDevId[%d]", HCCL_ERROR_CODE(ret), nRanks, rank,
            commIdentifier.c_str(), GetLocalServerId(params.serverId).c_str(), params.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }

    HCCL_INFO("[InitCommRootInfo]Init success, rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
              "logicDevId[%d]",
        nRanks, rank, commIdentifier.c_str(), params.serverId.c_str(), params.logicDevId);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoInner(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
                                     HcclComm *comm, string &identifier)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclUs startut = TIME_NOW();

    CHK_SMART_PTR_NULL(rootInfo);
    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), rootInfo->internal, sizeof(HcclRootHandle));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Init][RootInfoInner]memcpy root info fail. errorno[%d] "\
        "params:destMaxSize[%u], count[%u]", sRet, sizeof(HcclRootHandle),
        sizeof(HcclRootHandle)), HCCL_E_MEMORY);
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    identifier = rootHandle.identifier;

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET((nRanks == 0), HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] nRanks[%u] should "\
        "be greater than 0.", HCCL_ERROR_CODE(HCCL_E_PARA), nRanks), HCCL_E_PARA);

    CHK_PRT_RET((rank >= nRanks), HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] rank[%u] should "\
        "be less than nRanks[%u].", HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks), HCCL_E_PARA);
    CHK_SMART_PTR_NULL(comm);

    ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] init "\
        "external input error", HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoInner]errNo[0x%016llx] init environment config error.",
        HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcclCommInitRootInfoInner:ranks[%u], rank[%u], rootinfo: host ip[%s] port[%u] "\
        "nicDeploy[%d] identifier[%s], deviceLogicId[%d]", nRanks, rank, rootHandle.ip, rootHandle.port,
        rootHandle.nicDeploy, rootHandle.identifier, deviceLogicId);

    CommConfig commConfig(rootHandle.identifier);

    /* --------------初始化------------------------- */
    ret = InitCommRootInfo(nRanks, rank, rootHandle, commConfig, comm);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfig]errNo[0x%016llx]HcclCommInitRootInfo failed.",
        HCCL_ERROR_CODE(ret)),
        ret);

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO("[HCCL_TRACE]HcclCommInitRootInfoInner success, take time [%lld]us, rankNum[%u], rank[%u]",
        DURATION_US(TIME_NOW() - startut), nRanks, rank);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm)
{
    HcclResult ret = HCCL_SUCCESS;
    string identifier;
    ret = HcclCommInitRootInfoInner(nRanks, rootInfo, rank, comm, identifier);
    if (g_topoDetectServerPtrMap.find(identifier) != g_topoDetectServerPtrMap.end()) {
        g_topoDetectServerPtrMap[identifier] = nullptr;
    }
    return ret;
}

HcclResult HcclCommInitRootInfoConfigInner(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm, string &identifier)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclUs startut = TIME_NOW();

    CHK_SMART_PTR_NULL(rootInfo);

    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), rootInfo->internal, sizeof(HcclRootHandle));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[Init][RootInfo]memcpy root info fail. errorno[%d] "\
        "params:destMaxSize[%u], count[%u]", sRet, sizeof(HcclRootHandle),
        sizeof(HcclRootHandle)), HCCL_E_MEMORY);
    rootHandle.identifier[ROOTINFO_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    identifier = rootHandle.identifier;

    // 检查配置参数是否为空
    RPT_INPUT_ERR(config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCommInitRootInfoConfigInner", "config", "nullptr", "please check comm"}));
    CHK_SMART_PTR_NULL(config);

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET((nRanks == 0),
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] nRanks[%u] should be greater than 0.",
        HCCL_ERROR_CODE(HCCL_E_PARA), nRanks),
        HCCL_E_PARA);

    CHK_PRT_RET((rank >= nRanks),
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] rank[%u] should be less than nRanks[%u].",
        HCCL_ERROR_CODE(HCCL_E_PARA), rank, nRanks),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(comm);

    ret = InitExternalInput();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] init "\
        "external input error", HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] init "\
        "environment config error", HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    /* 读取用户配置 */
    CommConfig commConfig(rootHandle.identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx] load comm config failed.",
        HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    /* 接口交互信息日志 */
    HCCL_RUN_INFO("Entry-HcclCommInitRootInfoConfigInner:ranks[%u], rank[%u], rootinfo: host ip[%s] "\
        "port[%u] nicDeploy[%d] identifier[%s], deviceLogicId[%d]", nRanks, rank, rootHandle.ip,
        rootHandle.port, rootHandle.nicDeploy, commConfig.GetConfigCommName().c_str(), deviceLogicId);

    /* --------------初始化------------------------- */
    ret = InitCommRootInfo(nRanks, rank, rootHandle, commConfig, comm);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][CommRootInfoConfigInner]errNo[0x%016llx]HcclCommInitRootInfoConfigInner failed.",
        HCCL_ERROR_CODE(ret)),
        ret);

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    HCCL_RUN_INFO("[HCCL_TRACE]HcclCommInitRootInfoConfigInner success, take time [%lld]us, "\
        "rankNum[%u], rank[%u]", DURATION_US(TIME_NOW() - startut), nRanks, rank);

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank,
    const HcclCommConfig *config, HcclComm *comm)
{
    HcclResult ret = HCCL_SUCCESS;
    string identifier;
    ret = HcclCommInitRootInfoConfigInner(nRanks, rootInfo, rank, config, comm, identifier);
    if (g_topoDetectServerPtrMap.find(identifier) != g_topoDetectServerPtrMap.end()) {
        g_topoDetectServerPtrMap[identifier] = nullptr;
    }
    return ret;
}

HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue)
{
    if (config == HCCL_DETERMINISTIC) {
        char* mmSysGetEnvValue = nullptr;
        MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
        std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
        if (hcclDeterministicEnv == "EmptyString") {
            if (configValue.value == 1) {
                CHK_RET(SetDeterministic(true));
                HCCL_INFO("[HcclSetConfig] Set HCCL_DETERMINISTIC is true");
            } else if (configValue.value == 0) {
                CHK_RET(SetDeterministic(false));
                HCCL_INFO("[HcclSetConfig] Set HCCL_DETERMINISTIC is false");
            } else {
                HCCL_ERROR("[HcclSetConfig] HCCL_DETERMINISTIC is only support 0 or 1");
                return HCCL_E_PARA;
            }
        } else {
            HCCL_WARNING("[HcclSetConfig] HCCL_DETERMINISTIC has been setted by Env, so will not be reseted again");
            return HCCL_SUCCESS;
        }
        HcclOpInfoCtx& opBaseInfo = GetHcclOpInfoCtx();
        // 遍历所有的通信域设置其确定性计算配置参数
        for (auto it = opBaseInfo.opGroup2CommMap.begin(); it != opBaseInfo.opGroup2CommMap.end(); it++) {
            CHK_RET(it->second->SetDeterministicConfig(configValue.value));
        }
        // 遍历 OneSided 通信域，设置 second->SetDeterministicConfig(configValue.value)
        for (auto &oneSidedCommHcomInfoMap : g_oneSidedCommHcomInfos) {
            for (auto &oneSidedCommHcomInfoPtrMap : oneSidedCommHcomInfoMap.second) {
                CHK_RET(oneSidedCommHcomInfoPtrMap.second->pComm->SetDeterministicConfig(configValue.value));
                for (auto it = oneSidedCommHcomInfoPtrMap.second->opGroup2CommMap.begin();
                    it != oneSidedCommHcomInfoPtrMap.second->opGroup2CommMap.end();
                    it++) {
                    CHK_RET(it->second->SetDeterministicConfig(configValue.value));
                }
            }
        }
        }
    return HCCL_SUCCESS;
}

HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue *configValue)
{
    CHK_PTR_NULL(configValue);
    if (config == HCCL_DETERMINISTIC) {
        configValue->value = GetExternalInputHcclDeterministic() ? 1 : 0;
        HCCL_INFO("[HcclGetConfig] HCCL_DETERMINISTIC is [%d]", configValue->value);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclSetIfProfile()
{
    bool ifOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool state = ProfilingManagerPub::GetAllState();
    SetIfProfile((!ifOpbase) || (!state));
    return HCCL_SUCCESS;
}

void HcclResetIfProfile()
{
    SetIfProfile(true);
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();

    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }

    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return all reduce success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllReduce", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllReduce", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllReduce", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], localRank[%u], streamId[%d],"
        "comm[%p], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        localRank, streamId, comm, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

    std::string logInfo = "Entry-HcclAllReduce: " + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, count, dataType, tag));

    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAllReduce:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}


HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    // Allreduce入参定义
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());
    HCCL_PROFILER_ADD_OPDATA_OP(tag, HCCL_BARRIER_DEFAULT_COUNT, hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, \
        dataType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier(), HcclReduceOp::HCCL_REDUCE_RESERVED);
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetGroupRank(rankId), tag.c_str());
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, rankId);
    
    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], count[%d], dataType[%s], op[%s], streamId[%d],"
        "deviceLogicId[%d]",
        tag.c_str(), hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, HCCL_BARRIER_DEFAULT_COUNT,
        GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclBarrier:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateBarrierMemory(), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierSendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierRecvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf,
        HCCL_BARRIER_DEFAULT_COUNT, dataType, op, stream, SyncMode::UNLIMITED_TIMEWAITSYNCMODE), tag.c_str());

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, HCCL_BARRIER_DEFAULT_COUNT,
        dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclBarrier:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return broadcast success"), HCCL_SUCCESS);

    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBroadcast", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(buf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclBroadcast", "buf", "nullptr", "please check buf"}));
    CHK_PTR_NULL(buf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "Broadcast_" + hcclComm->GetIdentifier();

    CHK_RET(HcomCheckOpParam(tag.c_str(), count, dataType, stream));

    HcomCollOpInfo opInfo = {"", buf, buf, count, dataType, root, HCCL_REDUCE_RESERVED};

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], buf[%p], count[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), buf, count, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclBroadcast:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_BROADCAST, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(buf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->BroadcastOutPlace(tag, buf, count, dataType, root, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BROADCAST, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclBroadcast:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input recvCount is 0, return reduce scatter success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatter", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "ReduceScatter_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], op[%s],"
        "streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclReduceScatter:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReduceScatterOutPlace(tag, sendBuf, recvBuf, recvCount, dataType, op, stream),
                          tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, beginTime, recvCount,
        dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclReduceScatter:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
    void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参合法性校验
    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "sendCounts", "nullptr", "please check sendCounts"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(sendDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "sendDispls", "nullptr", "please check sendDispls"}));
    CHK_PTR_NULL(sendDispls);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    if (UNLIKELY(recvCount > 0 && recvBuf == nullptr)) {
        RPT_INPUT_ERR(true, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "recvBuf", "nullptr", "please check recvBuf"}));
        CHK_PTR_NULL(recvBuf);
    }
    
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const string tag = "ReduceScatterV_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(userRank), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());

    u64 maxCount = 0;
    u64 inputCount = 0;
    u64* counts = static_cast<u64 *>(const_cast<void*>(sendCounts));
    for (u32 i = 0; i < rankSize; i++) {
        CHK_PRT_RET(counts[i] > SYS_MAX_COUNT,
            HCCL_ERROR("HcclReduceScatterV sendCounts[%u][%llu] is invalid.(bigger than MAX count[%llu])",
                i, counts[i], SYS_MAX_COUNT),
            HCCL_E_PARA);
        inputCount += counts[i];
        maxCount = std::max(maxCount, counts[i]);
    }
    CHK_PRT_RET(inputCount == 0, HCCL_WARNING("The inputCount is 0, this reduce scatter v has no task to execute, "
        "returning success."), HCCL_SUCCESS);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduceScatterV", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], sendDispls[%p], recvCount[%llu], dataType[%s], op[%s],"
        "streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, sendCounts, sendDispls, recvCount,
        GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclReduceScatterV:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    PrintCountsAndDispls(rankSize, sendCounts, sendDispls, tag.c_str());

    CheckCountsAndDispls(rankSize, sendCounts, sendDispls, tag.c_str());

    const u64 countOfThisRank = static_cast<const u64 *>(sendCounts)[userRank];
    CHK_PRT_RET(recvCount != countOfThisRank,
        HCCL_ERROR("[HcclReduceScatterV] input recvCount[%llu] is not equal to sendCounts[%u][%llu]", recvCount,
        userRank, countOfThisRank),
        HCCL_E_PARA);

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReduceScatterVOutPlace(tag, sendBuf, recvBuf, sendCounts, sendDispls, recvCount,
                        dataType, op, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, beginTime, maxCount,
        dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();

    /* 关键状态记录 */
    std::string endInfo = "HcclReduceScatterV:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult CheckScatterInputPara(uint64_t recvCount, HcclComm comm, void *recvBuf)
{
    CHK_PRT_RET(recvCount == 0, HCCL_WARNING("input count is 0, return scatter success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclScatter", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclScatter", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);

    return HCCL_SUCCESS;
}

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_RET(CheckScatterInputPara(recvCount, comm, recvBuf));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    u32 commRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(commRank));
    if (commRank == root) { // 本rank为root节点，send_buff不为空
        RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcclScatter", "sendBuf", "nullptr", "please check sendBuf"}));
        CHK_PTR_NULL(sendBuf);
    }

    // 同通信域同算子复用tag
    const string tag = "Scatter_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), recvCount, dataType, stream), tag.c_str());

    HcomCollOpInfo opInfo = {"", sendBuf, recvBuf, recvCount, dataType, root};

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], recvCount[%llu], dataType[%s], root[%u], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, recvCount, GetDataTypeEnumStr(dataType).c_str(), root, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclScatter:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_SCATTER, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ScatterOutPlace(tag, sendBuf, recvBuf, recvCount, dataType, root, stream),
                          tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SCATTER, beginTime, recvCount, dataType,
        tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclScatter:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(sendCount == 0, HCCL_WARNING("input sendCount is 0, return HcclAllGather success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGather", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGather", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGather", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const std::string tag = "AllGather_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], dataType[%s], streamId[%d],"
        "deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, sendCount, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAllGather:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllGatherOutPlace(tag, sendBuf, recvBuf, sendCount, dataType, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLGATHER, beginTime, sendCount, dataType,
        tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAllGather:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf,
    const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参合法性校验
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGatherV", "recvCounts", "nullptr", "please check recvCounts"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(recvDispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGatherV", "recvDispls", "nullptr", "please check recvDispls"}));
    CHK_PTR_NULL(recvDispls);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGatherV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    if (UNLIKELY(sendCount > 0 && sendBuf == nullptr)) {
            RPT_INPUT_ERR(true, "EI0003",\
            std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
            std::vector<std::string>({"HcclAllGatherV", "sendBuf", "nullptr", "please check sendBuf"}));
            CHK_PTR_NULL(sendBuf);
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    // 同通信域同算子复用tag
    const std::string tag = "AllGatherV_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), sendCount, dataType, stream), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(userRank), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, userRank), tag.c_str());

    u64 maxCount = 0;
    u64 outputCount = 0;
    u64* counts = static_cast<u64 *>(const_cast<void*>(recvCounts));
    for (u32 i = 0; i < rankSize; i++) {
        CHK_PRT_RET(counts[i] > SYS_MAX_COUNT,
            HCCL_ERROR("HcclAllGatherV recvCounts[%u][%llu] is invalid.(bigger than MAX count[%llu])",
                i, counts[i], SYS_MAX_COUNT),
            HCCL_E_PARA);
        outputCount += counts[i];
        maxCount = std::max(maxCount, counts[i]);
    }
    CHK_PRT_RET(outputCount == 0, HCCL_WARNING("The outputCount is 0, this all gather v has no task to execute, "
        "returning success."), HCCL_SUCCESS);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAllGatherV", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    
    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], recvCounts[%llu], recvDispls[%llu], "
        "dataType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, sendCount, recvCounts, recvDispls,
        GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAllGatherV:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    PrintCountsAndDispls(rankSize, recvCounts, recvDispls, tag.c_str());

    CheckCountsAndDispls(rankSize, recvCounts, recvDispls, tag.c_str());

    const u64 countOfThisRank = static_cast<const u64*>(recvCounts)[userRank];
    CHK_PRT_RET(sendCount != countOfThisRank,
        HCCL_ERROR("[HcclAllGatherV] input sendCount[%llu] is not equal to recvCounts[%u][%llu]", sendCount,
        userRank, countOfThisRank),
        HCCL_E_PARA);

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if(sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AllGatherVOutPlace(tag, sendBuf, recvBuf, sendCount, recvCounts, recvDispls,
        dataType, stream), tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLGATHER_V, beginTime, maxCount, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();

    /* 关键状态记录 */
    std::string endInfo = "HcclAllGatherV:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return HcclSend success"), HCCL_SUCCESS);
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(stream);

    CHK_RET(HcomCheckCount(count));
    CHK_RET(HcomCheckDataType(dataType));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);

    // 同算子复用tag，为实现通信域复用，根据srRank和dstRank构造Tag
    u32 localRank = INVALID_VALUE_RANKID, rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const string tag = "worldCommSendRecv_" + std::to_string(localRank) + "_" + std::to_string(destRank) + "_" +
        hcclComm->GetIdentifier();

    HcomCollOpInfo opInfo = {"", sendBuf, sendBuf, count, dataType, 0, HCCL_REDUCE_RESERVED};

    HCCL_PROFILER_ADD_TAG_SENDRECV(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());

    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    HCCL_PROFILER_ADD_OPDATA_OP(tag, count, sendBuf, sendBuf, dataType, INVALID_VALUE_RANKID, \
        hcclComm->GetIdentifier(), HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(hcclComm->GetIdentifier(), rankSize, localRank, destRank);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], count[%llu], dataType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, count, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclSend:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_SEND, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->SendOutPlace(tag, sendBuf, count, dataType, destRank, stream), tag.c_str());

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_SEND, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclSend:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                    HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return HcclRecv success"), HCCL_SUCCESS);
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);

    CHK_RET(HcomCheckCount(count));
    CHK_RET(HcomCheckDataType(dataType));

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);

    // 同算子复用tag，为实现通信域复用，根据srRank和dstRank构造Tag
    u32 localRank = INVALID_VALUE_RANKID, rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const string tag = "worldCommSendRecv_" + std::to_string(srcRank) + "_" + std::to_string(localRank) + "_" +
        hcclComm->GetIdentifier();

    HcomCollOpInfo opInfo = {"", recvBuf, recvBuf, count, dataType, 0, HCCL_REDUCE_RESERVED};

    HCCL_PROFILER_ADD_TAG_SENDRECV(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());

    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    HCCL_PROFILER_ADD_OPDATA_OP(tag, count, recvBuf, recvBuf, dataType, INVALID_VALUE_RANKID, \
        hcclComm->GetIdentifier(), HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(hcclComm->GetIdentifier(), rankSize, localRank, srcRank);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], recvBuf[%p], count[%llu], dataType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclRecv:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateOpBasedResources(HcclCMDType::HCCL_CMD_RECEIVE, tag, opInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReceiveOutPlace(tag, recvBuf, count, dataType, srcRank, stream), tag.c_str());

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_RECEIVE, beginTime, count, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclRecv:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedCommDestroy(HcclComm comm, s32 deviceLogicId, HcclUs startut)
{
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    std::string group = hcclComm->GetIdentifier();
    CHK_PRT_RET(group.empty(),
                HCCL_ERROR("[HcclCommDestroy][HcclOneSidedCommDestroy] commName is error."),
                HCCL_E_PARA);
    HCCL_RUN_INFO("Entry-%s: deviceLogicId[%d], commName[%s]",
                  __func__, deviceLogicId, group.c_str());

    HcclOpInfoCtx &opBaseHcom = GetOneSidedOpInfoCtx(deviceLogicId, hcclComm->GetIdentifier());

    g_oneSidedCommSet.erase(comm);

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXECEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR("[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    opBaseHcom.isUsed = false;
    CHK_RET(DeInitOneSidedHcomInfo(deviceLogicId, group));

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    HCCL_USER_CRITICAL_LOG("op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d]",
        DURATION_US(endut - startut), group.c_str(), deviceLogicId);

    return HCCL_SUCCESS;
}

static HcclResult ResetDevice(hccl::hcclComm* hcclComm)
{
    s32 logicDeviceId = 0;
    hcclComm->GetDeviceId(logicDeviceId);
    g_hcclDeviceId = logicDeviceId;
    if (hcclComm->IsNeedResetDevice()) {
        HCCL_RUN_INFO("op_base com destroy, com is not global com");
        HCCL_RUN_INFO("[HcclCommDestroy] reset logicDeviceId[%d]", logicDeviceId);
        CHK_PRT_RET(hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommDestroy] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm)
{
    HCCL_RUN_INFO("Entry-%s: op_base comm destroy begin", __func__);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroy] Get device fail, comm=%p", comm), ret);
    CHK_PRT_RET(comm == nullptr, HCCL_WARNING("[Destroy][HcclComm]An empty comm given, skip destroy."), HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    HcclCommState state = hcclComm->GetState();
    if (state == HcclCommState::INUSE) {
        HCCL_WARNING("[HcclCommDestroy] comm is in use, please try again later");
        return HCCL_E_AGAIN;
    }
    hcclComm->DeinitZeroCopyMemoryAgent();
    HCCL_RUN_INFO("[HcclCommDestroy] comm state is %s", HcclCommStateToString(state));

    CHK_RET(hcclComm->SetStopFlag(true));
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(ResetDevice(hcclComm));

    if (IsOneSidedComm(comm)) {
        return HcclOneSidedCommDestroy(comm, deviceLogicId, startut);
    }

    HcclOpInfoCtx& opBaseHcom = GetHcclOpInfoCtx();
    string group;
    if (comm == opBaseHcom.pComm.get()) {
        group = opBaseHcom.pComm->GetIdentifier();
        opBaseHcom.pComm = nullptr;
        HcclCloseCommConnections(group);
    } else {
        HCCL_RUN_INFO("com is not global com");
        group = hcclComm->GetIdentifier();
    }

    // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将hcomInfo.pComm设为nullptr
    if (hcclComm->GetIdentifier() == HCCL_WORLD_GROUP) {
        HcomGetCtxHomInfo().pComm = nullptr;
    }

    HcomUnSetGroupTopoInfo(group.c_str());

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXECEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR("[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(), deviceLogicId);
        return HCCL_E_PARA;
    }
    ProfilingManagerPub::ClearStoragedProfilingInfo();

    HcclUs endut = TIME_NOW();

    // 删除groupName和UDI的映射
    HCCL_PROFILER_DEL_GROUP_UDI(group);

    /* 关键状态记录 */
    HCCL_RUN_INFO("op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d].",
        DURATION_US(endut - startut), group.c_str(), deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclGenerateCommId(hccl::HcclCommParams &params)
{
    s32 sRet = memset_s(params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(params.id.internal));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GenerateCommId]memory set error. return[%d].", sRet), HCCL_E_PARA);

    HcclRootInfo uniqueId;
    std::string group;
    CHK_RET(hcclComm::GetUniqueId(&uniqueId));

    if (!params.isHeterogComm) {
        group = "hccl_world_group";
    } else {
        group = "hccl_heterog_group";
    }

    sRet = snprintf_s(params.id.internal, HCCL_ROOT_INFO_BYTES, HCCL_ROOT_INFO_BYTES - 1, "%s%s%s",
        uniqueId.internal, "-", group.c_str());
    CHK_PRT_RET(sRet == -1, HCCL_ERROR("[GenerateCommId]errNo[0x%016llx] sal snprintf_s error",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);
    HCCL_INFO("params.id.internal [%s]", params.id.internal);
    return HCCL_SUCCESS;
}

HcclResult InitOtherInfo(hccl::HcclCommParams &params, const char *rankTable)
{
    // 记录版本信息
    std::string curVersion = GetExternalInputCannVersion();
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordVerInfo(curVersion));

    // ranktableCRC计算
    if (rankTable == nullptr) {
        HCCL_INFO("rank table is null, rankTableCrc is 0.");
    } else {
        HcclResult ret = HcomCalcCRC(params, rankTable);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] calc ranktable crc error",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)), HCCL_E_INTERNAL);
    }

    // 生成通信域标识符
    HcclResult ret = HcclGenerateCommId(params);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Init][OtherInfo]errNo[0x%016llx] generate CommId error, params: dest[%p]",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL), params.id.internal), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterLoop(const std::string &tag, void *inputPtr, void *outputPtr, const u64 &count,
                             HcclDataType dataType, HcclReduceOp op, hccl::hcclComm *hcclComm, rtStream_t stream)
{
    HcclResult ret;
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;
    u32 ranktableCrc = hcclComm->GetRankTableCrc();

    CHK_RET(hcclComm->GetInCCLbuffer(commInputPtr, commInputSize));

    CHK_RET(hcclComm->GetOutCCLbuffer(commOutputPtr, commOutputSize));

    u32 unitSize;
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    char *curInputPtr = static_cast<char *>(inputPtr);
    char *curOutputPtr = static_cast<char *>(outputPtr);
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
    CHK_RET(hcclComm->GetQosCfg(qosCfg));

    CHK_PRT_RET(rankSize * unitSize == 0, HCCL_ERROR("The result of rankSize * unitSize is 0"), HCCL_E_PARA);
    u64 maxCountPerLoop = commInputSize / (rankSize * unitSize); // 中转内存单次最多能够接受的output count
    u64 curCount = 0;

    for (u64 countLeft = count, inputOffset = 0, outputOffset = 0; countLeft > 0; countLeft -= curCount) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_INFO("-OP_BASE-ReduceScatterLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        // 判断剩余数据量对应的input size是否大于中转input size
        curCount = ((countLeft * unitSize * rankSize) > commInputSize) ? maxCountPerLoop : countLeft;
        u64 curSize = curCount * unitSize; // 单位：字节

        for (u32 i = 0; i < rankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            ret = hrtMemAsyncCopyByQos(static_cast<char *>(commInputPtr) + curSize * i, curSize,
                curInputPtr + count * unitSize * i, curSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
                stream, qosCfg);
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE inputbuffer transit,[%u]slice memcopy "\
                    "failed", i), HCCL_E_MEMORY);
        }
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
            tag, curCount, dataType, op, commInputSize, commOutputSize, nullptr, ranktableCrc));

        ret = hcclComm->ReduceScatter(tag, commInputPtr, commOutputPtr, curCount, dataType, op, stream);

        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][ReduceScatter]errNo[0x%016llx] op_base hcclComm reduce_scatter error, "\
            "tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s]",
            HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, commOutputPtr, curCount,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str()), ret);
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

        CHK_RET(hrtMemAsyncCopyByQos(curOutputPtr, curSize, commOutputPtr, curSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg));

        CHK_PRT_RET((curCount == 0), HCCL_ERROR("[Loop][ReduceScatter]In OP_BASE curCount is zero"), HCCL_E_PARA);
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

// 获取算子所需workspace memory大小[byte]
HcclResult HcclGetOpBasedMemSize(const HcclCMDType &opType, u64 &size,
    const HcomCollOpInfo &opInfo)
{
    u64 opMemSize = 0;

    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        // ReduceScatter 算子所需memory大小为 GetExternalInputCCLBuffSize()
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        if (IsSupportSDMAReduce(opInfo.inputAddr, opInfo.outputAddr, opInfo.dataType, opInfo.reduceOp) &&
            IsSupportRDMAReduce(opInfo.dataType, opInfo.reduceOp) && devType == DevType::DEV_TYPE_910B) {
                opMemSize = 0;
            } else {
                opMemSize = GetExternalInputCCLBuffSize();
            }
    } else {
        opMemSize = 0;
    }
    size = HCCL_WORKSPACE_MEM_32_KB + opMemSize;
    HCCL_INFO("workspace memory size: op[%d], memory size[%llu]", opType, size);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rankSize);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 tmpRankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(tmpRankSize));
    *rankSize = tmpRankSize;
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRankSize success, rankSizePtr[%p], rankSize[%u]", rankSize, tmpRankSize);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(rank);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 tmpRankId = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(tmpRankId));
    *rank = tmpRankId;
    /* 关键状态记录 */
    HCCL_INFO("HcclGetRankId success, rankIdPtr[%p], rankId[%u]", rank, tmpRankId);
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    CHK_PRT_RET(sendCount == 0 && recvCount == 0,
        HCCL_WARNING("sendCount and recvCount are both 0, return alltoall success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(sendCount != recvCount,
        HCCL_ERROR("sendCount[%lu] and recvCount[%lu] are not equal, please check params",
            sendCount, recvCount), HCCL_E_PARA);
    CHK_PRT_RET(sendType != recvType,
        HCCL_ERROR("sendType[%s] and recvType[%s] are not equal, please check params",
            GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str()), HCCL_E_PARA);

    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAll", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    CHK_PRT_RET(sendBuf == recvBuf,
        HCCL_ERROR("[HcclAlltoAll] sendBuf and recvBuf cannot be same."), HCCL_E_PARA);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);

    const std::string tag = HCCL_ALLTOALL + "_" + hcclComm->GetIdentifier();
    CHK_RET(HcomCheckOpParam(tag.c_str(), 0, sendType, stream));
    CHK_RET(HcomCheckDataType(recvType));
    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));
    HCCL_PROFILER_ADD_OPDATA_OP(tag, sendCount, sendBuf, recvBuf, sendType, INVALID_VALUE_RANKID, \
        hcclComm->GetIdentifier(), HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);

    // 接口交互信息日志
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendCount[%llu], recvCount[%llu], sendType[%s], recvType[%s], streamId[%d],"
        "deviceLogicId[%d]",
        tag.c_str(), sendCount, recvCount, GetDataTypeEnumStr(sendType).c_str(), GetDataTypeEnumStr(recvType).c_str(),
        streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAlltoAll:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    /* 记录cclBufferSize用于一致性校验 */
    CHK_RET_AND_PRINT_IDE(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLTOALL,
        tag.c_str(), sendCount, sendType, hcclComm->GetConfigInCCLbufferSize(), 0, nullptr,
        hcclComm->GetRankTableCrc()), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, stream, tag),
                          tag.c_str());

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALL, beginTime, sendCount, sendType,
        tag));

    HcclUs endut = TIME_NOW();
    std::string endInfo = "HcclAlltoAll:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

// sendBuf & recvBuf为device mem, 其它为host mem
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream)
{
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    RPT_INPUT_ERR(sendCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sendCounts", "nullptr", "please check sendCounts"}));
    CHK_PTR_NULL(sendCounts);
    RPT_INPUT_ERR(sdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "sdispls", "nullptr", "please check sdispls"}));
    CHK_PTR_NULL(sdispls);
    RPT_INPUT_ERR(recvCounts == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "recvCounts", "nullptr", "please check recvCounts"}));
    CHK_PTR_NULL(recvCounts);
    RPT_INPUT_ERR(rdispls == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "rdispls", "nullptr", "please check rdispls"}));
    CHK_PTR_NULL(rdispls);
    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllV", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);

    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    CHK_RET(HcomCheckAlltoAllVExternalMem(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));

    const std::string tag = HCCL_ALLTOALLV + "_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataType(recvType), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_TAG(HCCL_ALLTOALL_PARA_ALLGATHER, hcclComm->GetIdentifier(),
        HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));
    HCCL_PROFILER_ADD_OPDATA_OP(tag, 0, sendBuf, recvBuf, sendType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier(), \
        HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, localRank);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p], recvCounts[%p], sendType[%s],"
        "recvType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, sendCounts, recvCounts, GetDataTypeEnumStr(sendType).c_str(),
        GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAlltoAllV:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }
    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    /* 记录cclBufferSize用于一致性校验 */
    CHK_RET_AND_PRINT_IDE(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLTOALLV,
        tag.c_str(), 0, HCCL_DATA_TYPE_RESERVED, hcclComm->GetConfigInCCLbufferSize(), 0, nullptr,
        hcclComm->GetRankTableCrc()), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    if (!GetExternalInputHcclEnableFfts()) {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf,
            recvCounts, rdispls, recvType, stream, tag), tag.c_str());
    } else {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllVOutPlace(sendBuf, sendCounts, sdispls, sendType, recvBuf,
            recvCounts, rdispls, recvType, stream, tag), tag.c_str());
    }

    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64 *>(sendCounts) + i);
    }
    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(HCCL_ALLTOALL_PARA_ALLGATHER);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLV, beginTime, sendCount, sendType, tag));

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAlltoAllV:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix,
    HcclDataType sendType, const void *recvBuf, HcclDataType recvType,
    HcclComm comm, rtStream_t stream)
{
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    RPT_INPUT_ERR(sendCountMatrix == nullptr, "EI0003",\
        std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "sendCountMatrix", "nullptr", "please check sendCountMatrix"}));
    CHK_PTR_NULL(sendCountMatrix);
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclAlltoAllVC", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);

    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));
    u32 rank = 0;
    hcclComm->GetUserRank(rank);
    u32 userRank = 0;
    hcclComm->GetGroupRank(userRank);

    CHK_RET(HcomCheckAlltoAllVCExternalMem(sendBuf, sendCountMatrix, recvBuf, rankSize, rank));
    const std::string tag = HCCL_ALLTOALLVC + "_" + hcclComm->GetIdentifier();
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), 0, sendType, stream), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckDataType(recvType), tag.c_str());

    HCCL_PROFILER_ADD_TAG(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());

    HCCL_PROFILER_ADD_OPDATA_OP(tag, 0, sendBuf, recvBuf, sendType, INVALID_VALUE_RANKID, hcclComm->GetIdentifier(), \
        HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK(hcclComm->GetIdentifier(), rankSize, userRank);

    u64 sendCountMatrixHash;
    HcomGetHashFromSendCountMatrix(sendCountMatrixHash, sendCountMatrix, rankSize, tag);

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], recvBuf[%p],"
        "recvType[%s], streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, sendCountMatrixHash, GetDataTypeEnumStr(sendType).c_str(), recvBuf,
        GetDataTypeEnumStr(recvType).c_str(), streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclAlltoAllVC:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    if (sendBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());
    }
    if (recvBuf != nullptr) {
        CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());
    }

    /* 记录cclBufferSize用于一致性校验 */
    CHK_RET_AND_PRINT_IDE(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLTOALLVC,
        tag.c_str(), 0, HCCL_DATA_TYPE_RESERVED, hcclComm->GetConfigInCCLbufferSize(), 0, nullptr,
        hcclComm->GetRankTableCrc()), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    if (!GetExternalInputHcclEnableFfts()) {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllVC(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, stream, tag),
            tag.c_str());
    } else {
        CHK_RET_AND_PRINT_IDE(hcclComm->AlltoAllVCOutPlace(sendBuf, sendCountMatrix, sendType, recvBuf,
            recvType, stream, tag), tag.c_str());
    }

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());
    u64 sendCount = 0;
    for (u32 i = 0; i < rankSize; i++) {
        sendCount += *(static_cast<const u64 *>(sendCountMatrix) + userRank * rankSize + i);
    }
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLTOALLVC, beginTime, sendCount, sendType,
        tag));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclAlltoAllVC:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream)
{
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    HcclUs startut = TIME_NOW();
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return reduce success"), HCCL_SUCCESS);
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduce", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduce", "sendBuf", "nullptr", "please check sendBuf"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclReduce", "recvBuf", "nullptr", "please check recvBuf"}));
    CHK_PTR_NULL(recvBuf);
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "Reduce_" + hcclComm->GetIdentifier();
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp(op), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckUserRank(rankSize, root), tag.c_str());

    s32 streamId = 0;
    CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], root[%u],"
        "streamId[%d], deviceLogicId[%d]",
        tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
        root, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclReduce:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetDefaultQosConfig(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());

    CHK_RET_AND_PRINT_IDE(hcclComm->ReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, root, stream),
                              tag.c_str());

    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_REDUCE, beginTime, count, dataType, tag));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclReduce:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceLoop(const std::string &tag, void *inputPtr, void *outputPtr, const u64 count,
    HcclDataType dataType, HcclReduceOp op, const u32 root, hccl::hcclComm *hcclComm, rtStream_t stream)
{
    HcclSetIfProfile();
    
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;
    u32 ranktableCrc = hcclComm->GetRankTableCrc();

    HcclResult ret;
    CHK_RET(hcclComm->GetInCCLbuffer(commInputPtr, commInputSize));

    CHK_RET(hcclComm->GetOutCCLbuffer(commOutputPtr, commOutputSize));

    u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
    CHK_RET(hcclComm->GetQosCfg(qosCfg));

    u32 unitSize;
    CHK_RET(SalGetDataTypeSize(dataType, unitSize));

    char *curInputPtr = static_cast<char *>(inputPtr);
    char *curOutputPtr = static_cast<char *>(outputPtr);
    u64 inputOffset = 0;
    u64 outputOffset = 0;
    u64 countLeft = count;

    while (countLeft > 0) {
        curInputPtr += inputOffset;
        curOutputPtr += outputOffset;
        HCCL_DEBUG("-OP_BASE-ReduceLoop:inputOffset[%llu], outputOffset[%llu]", inputOffset, outputOffset);
        u64 curCount = ((countLeft * unitSize) > commInputSize) ? (commInputSize / unitSize) : countLeft; // 单次执行操作的数据量
        u64 curSize = curCount * unitSize; // 单位 byte

        HCCL_DEBUG("-OP_BASE-ReduceLoop:curInputPtr[%p], curOutputPtr[%p], curCount[%llu], curSize[%llu]",
            curInputPtr, curOutputPtr, curCount, curSize);

        u32 commRank = INVALID_VALUE_RANKID;
        CHK_RET(hcclComm->GetUserRank(commRank));

        CHK_RET(hrtMemAsyncCopyByQos(commInputPtr, curSize, curInputPtr, curSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg));

        /* 记录指令信息用于一致性校验 */
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE, tag,
            curCount, dataType, op, root, commInputSize, commOutputSize, nullptr, ranktableCrc));

        /* 入参的正确性由HCCL确保 */
        ret = hcclComm->Reduce(tag, commInputPtr, commOutputPtr, curCount, dataType, op, root, stream);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Reduce]errNo[0x%016llx] op_base hcclComm reduce error, tag[%s], "\
            "input_ptr[%p], output_ptr[%p], count[%llu], data_type[%s], op[%s], root[%u]",
            HCCL_ERROR_CODE(ret), tag.c_str(), commInputPtr, commOutputPtr, curCount,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), root), ret);
        ret = RankConsistentcyChecker::GetInstance().DelOpPara(tag);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Loop][Reduce]errNo[0x%016llx] delete CMD with parameters error. tag[%s]",
                HCCL_ERROR_CODE(ret), tag.c_str()), ret);

        if (commRank == root) { // 只root rank需要把数据从中转内存拷贝出去
            CHK_RET(hrtMemAsyncCopyByQos(curOutputPtr, curSize, commOutputPtr, curSize,
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, stream, qosCfg));
        }

        countLeft -= curCount;
        inputOffset = curSize;
        outputOffset = curSize;
    }

    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV的函数接口，目前不对外开放，仅图模式动态shape使用
 * **********************************************************************
 */
HcclResult HcclGatherAlltoAllV(HcomGatherAllToAllVParams params, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(params.addrInfoCountPerRank);
    CHK_PTR_NULL(params.recvcounts);
    CHK_PTR_NULL(params.gatheredbuf);
    CHK_PTR_NULL(params.rdispls);

    const u32 NUM_TWO = 2;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(rankSize));

    CHK_RET(SetDefaultQosConfig(hcclComm));

    // 同通信域同算子复用tag
    const string tag = "Reduce_" + hcclComm->GetIdentifier();

    std::vector<u64> addrInfoCountPerRank(rankSize, 0);
    CHK_RET_AND_PRINT_IDE(hrtMemSyncCopy(addrInfoCountPerRank.data(), rankSize * sizeof(u64),
        params.addrInfoCountPerRank, rankSize * sizeof(u64),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST), tag.c_str());
    u64 blockNum = 0;
    for (u32 index = 0; index < rankSize; index++) {
        blockNum += addrInfoCountPerRank[index];
    }
    if (blockNum != 0) {
        CHK_PTR_NULL(params.addrInfo);
    }
    std::vector<u64> addrInfo(blockNum * NUM_TWO, 0);
    CHK_RET_AND_PRINT_IDE(hrtMemSyncCopy(addrInfo.data(), addrInfo.size() * sizeof(u64), params.addrInfo,
        addrInfo.size() * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST), tag.c_str());

    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s]", tag.c_str());

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclGatherAlltoAllV:" + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());

    // 执行gather
    u64 sendCounts = static_cast<u64>(rankSize);
    u64 sdispls = static_cast<u64>(rankSize);

    // step1 gather
    GatherPara gatherPara;
    gatherPara.addrInfo = addrInfo;
    gatherPara.rankSize = rankSize;
    gatherPara.addrInfoCountPerRank = addrInfoCountPerRank;
    gatherPara.addrLength = params.addrLength;
    CHK_RET_AND_PRINT_IDE(RunGather(&sendCounts, &sdispls, params.gatheredbuf, gatherPara), tag.c_str());

    // step2 alltoallv
    CHK_RET_AND_PRINT_IDE(HcclAlltoAllV(params.gatheredbuf, &sendCounts, &sdispls, params.recvtype, params.recvbuf,
        params.recvcounts, params.rdispls, params.recvtype, comm, stream), tag.c_str());

    HcclUs endut = TIME_NOW();
    std::string endInfo = "HcclGatherAlltoAllV:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us," + std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV step1 执行gather，出参作为step2的入参
 * **********************************************************************
 */
HcclResult RunGather(u64 *sendCounts, u64 *sdispls, void *sendDevBuf, GatherPara &gatherPara)
{
    u64 memSize = 0;
    const u32 GATHER_THREAD_NUM = 16;
    const u32 NUM_TWO = 2;
    u64 perThreadCount = gatherPara.addrInfo.size() / NUM_TWO / GATHER_THREAD_NUM;
    std::vector<u64> perThreadCounts(GATHER_THREAD_NUM, perThreadCount);
    perThreadCounts[GATHER_THREAD_NUM - 1] =
        gatherPara.addrInfo.size() / NUM_TWO - perThreadCount * (GATHER_THREAD_NUM - 1);
    std::vector<u64> offset(GATHER_THREAD_NUM, 0);
    if (gatherPara.addrLength == -1) { // 数据包长度不一样的情况
        u32 offsetIndex = 0;
        for (u32 index = 1; index < gatherPara.addrInfo.size(); index += NUM_TWO) { // 由于是二元组，单数为数据包的长度，每个循环+2
            /* 如果数据包数量小于线程数量则offset全置为0 */
            if (perThreadCount != 0 && index / NUM_TWO % perThreadCount == 0 && offsetIndex < GATHER_THREAD_NUM) {
                /* 条件1：当累加的数量达到perThreadCount时往offset中填入累加值，即可计算出前面thread产生的offset值 */
                /* 条件2：由于第0个thread的offset为0，后面的线程的offset为前面线程处理数据量的累加，因此对最后一个值弃之不用 */
                offset[offsetIndex] = memSize;
                offsetIndex++;
            }
            memSize += gatherPara.addrInfo[index];
        }
    } else {
        memSize = gatherPara.addrInfo.size() / NUM_TWO * gatherPara.addrInfo[1];
        for (u32 index = 0; index < GATHER_THREAD_NUM; index++) {
            offset[index] = index * perThreadCount * gatherPara.addrInfo[1];
        }
    }

    // 多线程拷贝
    HostMem tmpHostMem = HostMem::alloc(memSize);
    std::vector<std::unique_ptr<std::thread>> threads(GATHER_THREAD_NUM);
    for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
        OpBaseMemPara memPara;
        memPara.beginIndex = num * perThreadCount * NUM_TWO;
        memPara.count = perThreadCounts[num];
        memPara.tmpMemSize = memSize;
        threads[num].reset(new (std::nothrow) std::thread(&GatherMemCopyThread, tmpHostMem.ptr(),
            offset[num], std::ref(gatherPara.addrInfo), memPara));
        CHK_PRT_RET(!threads[num], HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV]threads[%u] reset "\
            "failed ", num), HCCL_E_INTERNAL);
    }

    // 构造入参
    auto ret = memset_s(sendCounts, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]",
        gatherPara.rankSize * sizeof(u64)), HCCL_E_SYSCALL);
    u64 prevNum = 0;
    u64 nextNum = 0;
    for (u32 index = 0; index < gatherPara.addrInfoCountPerRank.size(); index++) {
        nextNum += gatherPara.addrInfoCountPerRank[index];
        for (u64 i = NUM_TWO * prevNum; i < NUM_TWO * nextNum; i += NUM_TWO) {
            *(sendCounts + index) += gatherPara.addrInfo[i + 1];
        }
        prevNum = nextNum;
    }

    ret = memset_s(sdispls, gatherPara.rankSize * sizeof(u64), 0, gatherPara.rankSize * sizeof(u64));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[Exec][EnqueueGatherAlltoAllV] mem set failed, count[%lld]",
        gatherPara.rankSize * sizeof(u64)), HCCL_E_SYSCALL);
    u64 displ = 0;
    for (u32 i = 0; i < gatherPara.rankSize; i++) {
        *(sdispls + i) = displ;
        displ += *(sendCounts + i);
    }

    // 等待线程执行完毕
    for (u32 num = 0; num < GATHER_THREAD_NUM; num++) {
        threads[num]->join();
    }

    CHK_RET(hrtMemSyncCopy(sendDevBuf, memSize, tmpHostMem.ptr(), memSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 单算子GatherAllToAllV gather多线程拷贝
 * **********************************************************************
 */
void GatherMemCopyThread(void *baseAddr, u64 offset, std::vector<u64> &addrInfo, OpBaseMemPara memCpyPara)
{
    //给当前线程添加名字
    SetThreadName("Hccl_GatherCopy");

    void *addr = nullptr;
    const u32 NUM_TWO = 2;
    u64 length = 0;
    auto destMax = [&]()-> u64 {
        return memCpyPara.tmpMemSize < offset ? 0 : memCpyPara.tmpMemSize - offset;
    };

    for (u32 index = 0; index < memCpyPara.count; index++) {
        addr = reinterpret_cast<void *>(addrInfo[memCpyPara.beginIndex + NUM_TWO * index]);
        length = addrInfo[memCpyPara.beginIndex + index * NUM_TWO + 1];
        if (memcpy_s(static_cast<s8 *>(baseAddr) + offset, destMax(), addr, length) != EOK) {
            HCCL_ERROR("[MemCopy][GatherAlltoAllV] mem copy failed, destMax[%llu], count[%llu]",
                memCpyPara.tmpMemSize - offset, length);
            return;
        }
        offset += length;
    }
}

HcclResult SetDefaultQosConfig(hccl::hcclComm *hcclComm)
{
    u32 qosCfg = INVALID_QOSCFG; // qos不使能的情况下为全F
    CHK_RET(hcclComm->GetQosCfg(qosCfg));
    // 防止Lowering下Qos值被覆盖
    if (qosCfg == INVALID_QOSCFG) {
        CHK_RET(hrtGetQosConfig(HCCL_STREAM_DEFAULT_GROUP_ID, qosCfg));
        HCCL_DEBUG("Call SetDefaultQosConfig, qosCfg[%x]", qosCfg);
        CHK_RET(hcclComm->SetQosCfg(qosCfg));
    }
    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * 获取HCCL错误
 * **********************************************************************
 */
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCommGetAsyncError", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(asyncError);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->CommCheckErrorCqe(*asyncError));

    return HCCL_SUCCESS;
}

/*
 * **********************************************************************
 * HCCL提供错误码到字符串的转换
 * **********************************************************************
 */
const char *HcclGetErrorString(HcclResult code)
{
    if (code < HcclResult::HCCL_SUCCESS || code >= HcclResult::HCCL_E_RESERVED) {
        return "unknow error";
    }
    static const std::map<HcclResult, std::string> errorMap = {{HCCL_SUCCESS, "no error"},
        {HCCL_E_PARA, "parameter error"}, {HCCL_E_PTR, "empty pointer"},
        {HCCL_E_MEMORY, "memory error"}, {HCCL_E_INTERNAL, "internal error"},
        {HCCL_E_NOT_SUPPORT, "not support feature"}, {HCCL_E_NOT_FOUND, "not found specific resource"},
        {HCCL_E_UNAVAIL, "resource unavailable"}, {HCCL_E_SYSCALL, "call system interface error"},
        {HCCL_E_TIMEOUT, "timeout"}, {HCCL_E_OPEN_FILE_FAILURE, "open file fail"},
        {HCCL_E_TCP_CONNECT, "tcp connect fail"}, {HCCL_E_ROCE_CONNECT, "roce connect fail"},
        {HCCL_E_TCP_TRANSFER, "tcp transfer fail"}, {HCCL_E_ROCE_TRANSFER, "roce transfer fail"},
        {HCCL_E_RUNTIME, "call runtime api fail"}, {HCCL_E_DRV, "call driver api fail"},
        {HCCL_E_PROFILING, "call profiling api fail"}, {HCCL_E_CCE, "call cce api fail"},
        {HCCL_E_NETWORK, "call network api fail"}, {HCCL_E_AGAIN, "try again"},
        {HCCL_E_REMOTE, "error cqe"}};

    return errorMap.at(code).data();
}

/*
 * 配置溢出检测地址
 */
HcclResult SetOverFlowAddr(hccl::hcclComm *hcclComm)
{
    std::vector<void *> globalWorkSpaceAddr;
    CHK_RET(hcclComm->SetGlobalWorkSpace(globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclCreateComResource(const char *commName, u32 streamMode, void** commContext)
{
    RPT_INPUT_ERR(commName == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commName", "nullptr", "please check commName"}));

    RPT_INPUT_ERR(commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commContext", "nullptr", "please check commContext"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    CHK_RET(hrtGetDeviceRefresh(&g_hcclDeviceId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(commName, hcclComm));
    HcclComm comm = hcclComm.get();
    CHK_RET(HcclCreateComResourceByComm(comm, streamMode, true, commContext));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResource(HcclComm comm, u32 streamMode, void** commContext, bool isMC2 = false)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commContext", "nullptr", "please check commContext"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid

    CHK_RET(HcclCreateComResourceByComm(comm, streamMode, true, commContext, isMC2));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* Mc2Tiling, void** commContext)
{
    // 校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(Mc2Tiling);
    CHK_PTR_NULL(commContext);

    HcclUs startut = TIME_NOW();
    // 获取streamMode
    uint64_t streamMode = 0;
    CHK_RET(hrtStreamGetMode(stream, &streamMode));

    // 兼容老版本
    uint32_t *pVersion = reinterpret_cast<uint32_t *>(Mc2Tiling);
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    HCCL_INFO("[%s]version ptr[%p] val[%u] devType[%u] streamMode[%u]", __func__, pVersion, *pVersion, devType,
        streamMode);
    if (*pVersion <= MC2_TILING_VERSION_DEFAULT || devType != DevType::DEV_TYPE_910_93) {
        return HcclAllocComResource(comm, streamMode, commContext, true);
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    string commIdentifier = hcclComm->GetIdentifier();
    HCCL_INFO("[%s]commIdentifier[%s]", __func__, commIdentifier.c_str());

    // 根据streamMode创建aicpuStream
    rtStream_t aicpuStream{};
    CHK_RET(hcclComm->Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream));

    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "commIdentifier[%s], version[%u]",
        commIdentifier.c_str(), *pVersion);
    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, commIdentifier[%s].", commIdentifier.c_str()));

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(localRank));

    /* 接口交互信息日志 */
    std::string logInfo = "Entry-HcclMc2ComResourceByTiling:localRank[" + std::to_string(localRank)
        + "]" + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveTraceInfo(logInfo));

    CHK_RET(HcclMc2ComResourceByTiling(comm, pVersion, Mc2Tiling, aicpuStream));

    // 获取 commContext
    hcclComm->GetCommResource(*commContext);

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclMc2ComResourceByTiling success, HcclMc2ComResourceByTiling take time ["
        + std::to_string(DURATION_US(endut - startut).count()) + "]us, localRank["
        + std::to_string(localRank) + "] " + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveTraceInfo(endInfo));
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcclMc2ComResourceByTiling(HcclComm comm, uint32_t *pVersion, void *mc2Tiling, rtStream_t &aicpuStream)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    string commIdentifier = hcclComm->GetIdentifier();

    // 遍历MC2Tiling中的Mc2HcommCfg, 将其中hcomId于commHandle->indetifier_进行对比, 如一致则根据该 Mc2HcommCfg 创建通信资源
    uint32_t *pMc2HcommCnt = pVersion + MC2_TILING_OFFSET;
    HCCL_INFO("[%s]mc2HcommCnt ptr[%p] val[%u] commIdentifier[%s]", __func__, pMc2HcommCnt, *pMc2HcommCnt,
        commIdentifier.c_str());

    if (*pVersion == MC2_TILING_VERSION_V2) {
        Mc2ServerCfg *pServerCfg = reinterpret_cast<Mc2ServerCfg *>(pMc2HcommCnt + MC2_TILING_OFFSET);
        HCCL_INFO("[%s]serverCfg ptr[%p] size[%u]", __func__, pServerCfg, sizeof(Mc2ServerCfg));
        Mc2HcommCfg *pCfg = reinterpret_cast<Mc2HcommCfg *>(pServerCfg + MC2_TILING_OFFSET);

        for (uint64_t i = 0; i < *pMc2HcommCnt; i++) {
            Mc2HcommCfg &cfg = pCfg[i];
            HCCL_INFO("[%s]cfg[%u] ptr[%p] size[%u] groupName[%s]",
                __func__, i, &cfg, sizeof(Mc2HcommCfg), cfg.groupName);
            if (string(cfg.groupName) == commIdentifier) { // 创建通信资源
                HCCL_INFO("[%s] cfg[%u] match commIdentifier", __func__, i);
                string algConfig(cfg.algConfig);
                string tag = string(cfg.groupName) + to_string(cfg.opType);
                tag.append("_mc2");
                HcclResult ret = hcclComm->AllocComResourceByTiling(algConfig, tag, cfg.opType, cfg.reduceType, aicpuStream);
                CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("HcclMc2ComResourceByTiling version[%u] tag[%s] algConfig[%s]",
                    *pVersion, tag.c_str(), algConfig.c_str()), ret);
            }
        }
    } else {
        uint32_t *offset = pMc2HcommCnt + MC2_TILING_OFFSET;
        uint32_t offsetSize = 8;
        for (uint64_t i = 0; i < *pMc2HcommCnt; i++) {
            if (i >= offsetSize) {
                HCCL_WARNING("struct offset out of range, i[%u] offsetSize[%u] cnt[%u]", i, offsetSize, *pMc2HcommCnt);
                break;
            }
            uint32_t offsetNum = offset[i];
            Mc2HcommCfg *cfg = reinterpret_cast<Mc2HcommCfg *>(reinterpret_cast<uint8_t *>(mc2Tiling) + offsetNum);
            HCCL_INFO("[%s]cfg[%u] ptr[%p] size[%u] groupName[%s] offset[%p] offsetNum[%u]",
                __func__, i, cfg, sizeof(Mc2HcommCfg), cfg->groupName, offset, offsetNum);
            if (string(cfg->groupName) == commIdentifier) { // 创建通信资源
                HCCL_INFO("[%s]cfg[%u] match commIdentifier", __func__, i);
                string algConfig(cfg->algConfig);
                string tag = string(cfg->groupName) + to_string(cfg->opType);
                tag.append("_mc2");
                HcclResult ret = hcclComm->AllocComResourceByTiling(algConfig, tag, cfg->opType, cfg->reduceType, aicpuStream);
                CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("HcclMc2ComResourceByTiling version[%u] tag[%s] algConfig[%s]",
                    *pVersion, tag.c_str(), algConfig.c_str()), ret);
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode,
    void** commContext, bool isMC2)
{
    HcclUs startut = TIME_NOW();
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(commContext == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclCreateComResource", "commContext", "nullptr", "please check commContext"}));

    // 同通信域同算子复用tag
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    u32 moduleNum = hcclComm->GetModuleNum();
    // mc2算子更改tag 
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    string tag = "CreatecomResource_" + hcclComm->GetIdentifier();
    if (isMC2 && devType == DevType::DEV_TYPE_910B && 
        (moduleNum > HCCL_DEVICE_NUM_ONE)) {
        tag += HCCL_MC2_MULTISERVER_SUFFIX;
    }
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag[%s], commContext[%p]", tag.c_str(),
        commContext);
    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

    if (LIKELY(hcclComm->GetCommResource(tag, commContext))) {
        /* 接口交互信息日志 */
        HcclUs endGetResourcetut = TIME_NOW();
        std::string getComReslogInfo = "HcclCreateComResource get ComResource success:take time" +
            std::to_string(DURATION_US(endGetResourcetut - startut).count()) + "us, "
            + std::string(stackLogBuffer);
        CHK_RET(hcclComm->SaveTraceInfo(getComReslogInfo));
        return HCCL_SUCCESS;
    }
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetUserRank(localRank));

    /* 接口交互信息日志 */
    std::string logInfo = "Entry-HcclCreateComResource:localRank[" + std::to_string(localRank)
        + "]" + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveTraceInfo(logInfo));
    // SetWorkflowMode性能开销hrtGetDevice，0.11us
    HcclUs middleut0 = TIME_NOW();
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    HcclUs middleut1 = TIME_NOW();
    rtStream_t stream;
    CHK_RET(hcclComm->Mc2AiCpuStreamAllocAndGet(streamMode, stream));
    CHK_RET(hcclComm->CreateCommResource(tag, stream, isOpbaseMode, commContext));

    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclCreateComResource success, HcclCreateComResource take time ["
        + std::to_string(DURATION_US(endut - startut).count()) + "]us, CreateComResource take time ["
        + std::to_string(DURATION_US(endut - middleut1).count()) + "]us, SetWorkflowMode take time ["
        + std::to_string(DURATION_US(middleut1 - middleut0).count()) + "]us, localRank["
        + std::to_string(localRank) + "] " + std::string(stackLogBuffer);
    CHK_RET(hcclComm->SaveTraceInfo(endInfo));
    return HCCL_SUCCESS;
}

void PrintCountsAndDispls(const u32 length, const void *counts, const void *displs, const std::string &tag)
{
    // 打印counts和displs
    const u64 *countsPtr = static_cast<const u64 *>(counts);
    const u64 *displsPtr = static_cast<const u64 *>(displs);
    if (HcclCheckLogLevel(DLOG_DEBUG)) {
        std::ostringstream countsStream;
        std::ostringstream displsStream;
        countsStream << "[ ";
        displsStream << "[ ";
        for (u32 i = 0; i < length; ++i) {
            countsStream << countsPtr[i] << " ";
            displsStream << displsPtr[i] << " ";
        }
        countsStream << "]";
        displsStream << "]";
        HCCL_DEBUG("[PrintCountsAndDispls]tag[%s], counts%s", tag.c_str(), countsStream.str().c_str());
        HCCL_DEBUG("[PrintCountsAndDispls]tag[%s], displs%s", tag.c_str(), displsStream.str().c_str());
    }
}

void CheckCountsAndDispls(const u32 length, const void *counts, const void *displs, const std::string &tag)
{
    // 校验counts和displs是否匹配
    const u64 *countsPtr = static_cast<const u64 *>(counts);
    const u64 *displsPtr = static_cast<const u64 *>(displs);
    u64 displsCal = 0;

    for (u32 i = 0; i < length; i++) {
        if (displsCal != displsPtr[i]) {
            HCCL_WARNING("[CheckCountsAndDispls]tag[%s], displs[%u]: [%llu] memory is discontinuous.",
                tag.c_str(), i, displsPtr[i]);
        }

        displsCal = displsCal + countsPtr[i];
    }
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclGetAicpuOpStreamNotify(const char *commName, rtStream_t* opstream, void** aicpuNotify)
{
    RPT_INPUT_ERR(commName == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "commName", "nullptr", "please check commName"}));

    RPT_INPUT_ERR(opstream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "opstream", "nullptr", "please check opstream"}));

    RPT_INPUT_ERR(aicpuNotify == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "aicpuNotify", "nullptr", "please check aicpuNotify"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    CHK_RET(hrtGetDeviceRefresh(&g_hcclDeviceId));
    std::shared_ptr<hccl::hcclComm> hcclComm;
    CHK_RET(HcomGetCommByGroup(commName, hcclComm));

    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opstream, 1, aicpuNotify));
    return HCCL_SUCCESS;
}

HcclResult HcclGetAicpuOpStreamAndNotify(HcclComm comm, rtStream_t* opstream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(opstream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "opstream", "nullptr", "please check opstream"}));

    RPT_INPUT_ERR(aicpuNotify == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetAicpuOpStream", "aicpuNotify", "nullptr", "please check aicpuNotify"}));
    // 切换线程后获取不到hcom上下文，需重新刷新一次线程操作的deviceid
    CHK_RET(hrtGetDeviceRefresh(&g_hcclDeviceId));
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

    CHK_RET(hcclComm->GetAicpuOpStreamNotify(opstream, aicpuNotifyNum, aicpuNotify));
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    HcclUs startut = TIME_NOW();
    std::string captureInfo;
    bool isCapture;
    CHK_PRT(GetCaptureInfo(stream, captureInfo, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PTR_NULL(sendRecvInfo);
    CHK_PRT_RET((itemNum == 0), HCCL_WARNING("[BatchSendRecv] taskList itemNum is zero."), HCCL_SUCCESS);

    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 若任务不同，也复用tag
    const string tag = "worldBatchSendRecv_" + hcclComm->GetIdentifier();
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetGroupRank(rankId), tag.c_str());

    /* 记录接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "tag[%s], itemNum[%u], streamId[%d], deviceLogicId[%d]", tag.c_str(), itemNum, streamId, deviceLogicId);

    CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
    std::string logInfo = "Entry-HcclBatchSendRecv:" + std::string(stackLogBuffer) + captureInfo;
    CHK_RET(hcclComm->SaveTraceInfo(logInfo));

    for (u32 i = 0; i < itemNum; i++) {
        CHK_PTR_NULL((sendRecvInfo + i)->buf);
        CHK_RET(HcomCheckDataType((sendRecvInfo + i)->dataType));
        CHK_RET(HcomCheckCount((sendRecvInfo + i)->count));
        CHK_RET(HcomCheckUserRank(rankSize, (sendRecvInfo + i)->remoteRank));
        char stackLogBuffer[LOG_TMPBUF_SIZE];
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "SendRecvItem : SendRecvType[%d], remoteRank[%d], count[%llu], dataType[%d], buf[%p].",
            (sendRecvInfo + i)->sendRecvType, (sendRecvInfo + i)->remoteRank, (sendRecvInfo + i)->count,
            (sendRecvInfo + i)->dataType, (sendRecvInfo + i)->buf);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "[HcclBatchSendRecv]" + std::string(stackLogBuffer);
        CHK_RET(hcclComm->SaveTraceInfo(logInfo));
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    HCCL_PROFILER_ADD_TAG_SENDRECV(tag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(streamId, tag, 0, AlgType::Reserved());

    HCCL_PROFILER_ADD_OPDATA_OP(tag, 0, nullptr, nullptr, HcclDataType::HCCL_DATA_TYPE_RESERVED, INVALID_VALUE_RANKID, \
        hcclComm->GetIdentifier(), HcclReduceOp::HCCL_REDUCE_RESERVED);
    HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(hcclComm->GetIdentifier(), rankSize, rankId, sendRecvInfo->remoteRank);
    CHK_RET_AND_PRINT_IDE(hcclComm->BatchSendRecv(tag, sendRecvInfo, itemNum, stream), tag.c_str());

    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(streamId);
    HCCL_PROFILER_DEL_TAG(tag);
    HCCL_PROFILER_DEL_OPDATA(tag);
    HCCL_PROFILER_DEL_GROUPRANK(hcclComm->GetIdentifier());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, beginTime, sendRecvInfo->count,
        sendRecvInfo->dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "HcclBatchSendRecv:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
    CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclDeviceRefresh(void)
{
    HcclResult ret = hrtGetDeviceRefresh(&g_hcclDeviceId);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Get][DeviceRefresh]errNo[0x%016llx] g_hcclDeviceId[%d]"
        "get device refresh error.", ret, g_hcclDeviceId), ret);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclGetTopoDesc(HcclComm comm, HcclTopoDescs *topoDescs, uint32_t topoSize)
{
    // 入参合法性校验
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetTopoDesc", "comm", "nullptr", "please check comm"}));

    RPT_INPUT_ERR(topoDescs == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclGetTopoDesc", "topoDescs", "nullptr", "please check topoDescs"}));

    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->GetTopoDesc(topoDescs, topoSize));

    return HCCL_SUCCESS;
}

HcclResult HcclCommSuspend(HcclComm comm)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    HcclUs startut = TIME_NOW();
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->Suspend());
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommSuspend:success, take time:[%lld]us, comm[%s]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommResume(HcclComm comm)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    HcclUs startut = TIME_NOW();
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->Resume());
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommResume:success, take time:[%lld]us, comm[%s]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

uint32_t HcclGetCommConfigCapability()
{
    // RESERVED在枚举中是最后一个，返回RESERVED说明它前面所有的配置项都支持
    return static_cast<uint32_t>(HCCL_COMM_CONFIG_RESERVED);
}

HcclResult HcclCommSetMemoryRange(HcclComm comm, void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(baseVirPtr);

    HcclUs startut = TIME_NOW();
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->SetMemoryRange(baseVirPtr, size, alignment, flags));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommSetMemoryRange:success, take time:[%lld]us, comm[%s] basePtr[%p] size[%lu] alignment[%lu] flags[%lu]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), baseVirPtr, size, alignment, flags);
    return HCCL_SUCCESS;
}

HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void *baseVirPtr)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(baseVirPtr);

    HcclUs startut = TIME_NOW();
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->UnsetMemoryRange(baseVirPtr));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommUnsetMemoryRange:success, take time:[%lld]us, comm[%s] basePtr[%p]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), baseVirPtr);
    return HCCL_SUCCESS;
}

HcclResult HcclCommActivateCommMemory(HcclComm comm, void *virPtr, size_t size, size_t offset, void* handle, uint64_t flags)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(virPtr);
    CHK_PTR_NULL(handle);

    HcclUs startut = TIME_NOW();
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->ActivateCommMemory(virPtr, size, offset, handle, flags));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommActivateCommMemory:success, take time:[%lld]us, comm[%s] virPtr[%p] size[%lu] offset[%lu] "
        "handle[%p] flags[%lu]", DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), virPtr, size,
        offset, handle, flags);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void *virPtr)
{
    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(virPtr);

    HcclUs startut = TIME_NOW();
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    CHK_RET(hcclComm->DeactivateCommMemory(virPtr));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommDeactivateCommMemory:success, take time:[%lld]us, comm[%s] virPtr[%p]",
        DURATION_US(endut - startut).count(), hcclComm->GetIdentifier().c_str(), virPtr);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
