/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <sys/time.h>
#include "externalinput_pub.h"
#include "env_config.h"
#include <memory>
#include "p2p_mgmt_pub.h"
#include "opexecounter_pub.h"
#include "config.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "profiling_manager_pub.h"
#include "task_exception_handler_pub.h"
#include "rank_consistentcy_checker.h"
#include "hccl_aiv.h"
#include "task_abort_handler_pub.h"
#include "adapter_rts_common.h"
#include "coll_alg_utils.h"
#include "../common/src/state_guard.h"
#include "alg_profiling.h"
#include "preempt_port_manager.h"
#include "stream_utils.h"

using namespace std;

typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);

namespace hccl {
static HcclOneSideServiceCallBack g_hcclOneSidedServiceCallback = nullptr;
static std::mutex g_hcomInitMutex;
std::mutex HcclCommunicator::linkResMapMutex_;
std::unordered_map<Transport*, std::pair<std::string, RankId>> HcclCommunicator::linkResMap_;
constexpr u32 MEMORY_CAPACITY = 256 * 1024;
constexpr u32 WAIT_PREPARE_SLEEP_TIME = 5000;
constexpr u32 SINGLE_SERVER_NUM = 1;
constexpr u32 CONN_LIMIT = 4096;
constexpr u32 COMM_DEV_TYPE_DIGIT_NUM = 8;
constexpr u32 TILINGDATA_BUF_SIZE = 32 * 1024; //单位：字节
constexpr u32 ALLTOALL_INFO_MATRIX_SIZE = 4;
constexpr u32 AICPU_RETRY_LINKROCE_DEFAULT = 0;
constexpr u32 AICPU_RETRY_LINKROCE_BACKUP = 1;
constexpr u32 SINGLE_PROCESS_MIN_PORT = 1024;
constexpr u32 SINGLE_PROCESS_MAX_PORT = 65535;
enum TransferMemInfoIdx {
    TRANSFER_MEM_INFO_KEY_IDX = 0,
    TRANSFER_MEM_INFO_VALUE_IDX = 1,
    TRANSFER_MEM_INFO_RDMA_ENVELOPE_IDX = 2,
    TRANSFER_MEM_INFO_IDX_NUM = 3
};

unordered_map<std::string, std::string> ALGCFG_TO_NAME = {
    {"AllGather=level0:ring", "AllGatherRingFor91093Executor"},
    {"AllGather=level0:fullmesh", "AllGatherMeshOpbaseExecutor"},
    {"AllGather=level0:doublering", "AllGatherRingFor91093Executor"},
    {"ReduceScatter=level0:ring", "ReduceScatterRingFor91093Executor"},
    {"ReduceScatter=level0:fullmesh", "ReduceScatterMeshDmaEliminationExecutor"},
    {"ReduceScatter=level0:doublering", "ReduceScatterRingFor91093Executor"},
    {"AllReduce=level0:ring", "AllReduceRingForRingFor91093Executor"},
    {"AllReduce=level0:fullmesh", "AllReduceMeshOpbaseLoopExecutor"},
    {"AllReduce=level0:doublering", "AllReduceRingForRingFor91093Executor"},
    {"AlltoAll=level0:fullmesh;level1:pairwise", "RunAlltoAllDirectFullmesh"}
};

struct HcclCMDTypeHash
{
    size_t operator()(HcclCMDType t) const
    {
        return static_cast<size_t>(t);
    }
};

unordered_map<HcclCMDType, std::string, HcclCMDTypeHash> CMDTYPE_TO_KEYWORD = {
    {HcclCMDType::HCCL_CMD_ALLGATHER, "AllGather"},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "ReduceScatter"},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, "AllReduce"},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, "AlltoAll"},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, "AlltoAll"},
    {HcclCMDType::HCCL_CMD_ALLTOALL, "AlltoAll"}
};

HcclCommunicator::HcclCommunicator()
    : dispatcher_(nullptr), vDispatcher_(nullptr), notifyPool_(nullptr),
      initializedFlag_(ATOMIC_FLAG_INIT), userRank_(INVALID_VALUE_RANKID), realUserRank_(INVALID_VALUE_RANKID),
      userRankSize_(INVALID_VALUE_RANKSIZE), drvInit_(false), inlineReduceSwitchOn_(true),
      nicDeployment_(NICDeployment::NIC_DEPLOYMENT_DEVICE), devicePhyId_(INVALID_UINT),
      deviceLogicId_(-1), localRank_(INVALID_VALUE_RANKID), hostSocketHandle_(nullptr),
      isUsedRdmaLevel0_(false), nicInitialized_(0), hcomGroupNicInit_(false),
      profilingMode_(HcomProfilingMode::PROFILING_CLOSE), raResourceInit_(false),
      interServer_(false), isSingleMeshAggregation_(false), cclBufferManager_(CCLBufferManager()),
      isExecuteProfilingInit_(false), deviceType_(DevType::DEV_TYPE_COUNT),
      commHandle_(nullptr),
      commWorkMode_(WorkMode::HCCL_MODE_NORMAL), meshAggregationRankSize_(0), isHaveCpuRank_(false), ranktableCrc_(0),
      pMsgInfosMem_(nullptr), pReqInfosMem_(nullptr), memBlocksManager_(nullptr), pRecvWrInfosMem_(nullptr),
      transportResInfo_(mrManager_, pMsgInfosMem_, pReqInfosMem_, memBlocksManager_, pRecvWrInfosMem_),
      multiModuleDiffDeviceNumMode_(false), multiSuperPodDiffServerNumMode_(false),
      isStandardCard_(false), is310PDuoCard_(false), hccsPortNum_(-1),
      loopBackIp_(HcclIpAddress(COMM_LOOPBACK_IP)), profilingInitiated_(false), callbackThreadId_(INVALID_U64),
      role_(SERVER_ROLE_SOCKET), mrManagerInit_(false),
      isHostUseDevNic_(false),
      isAllRankSamePlane_(false), serverNum_(0), moduleNum_(0)
{
    mrManager_.reset(new (std::nothrow) MrManager());
    if (mrManager_ == nullptr) {
        HCCL_ERROR("new MrManager failed!");
    }
}

HcclCommunicator::~HcclCommunicator()
{
    HCCL_DEBUG("Enter ~HcclCommunicator.");

    DeinitZeroCopyMemoryAgent(true);
    (void)DestroyAicpuComm();
    (void)UnRegisterBackGroundThread();

    UnRegisterToHeartBeat();

    if (implAlg_ != nullptr) {
        delete implAlg_;
        implAlg_ = nullptr;
    }

    for (auto &res :resMap_) {
        DestroyAlgResource(res.second);
    }
    if (opRetryManager_ != nullptr) {
        OpRetryManager::DeleteLinkInfoByIdentifier(deviceLogicId_, identifier_);
        opRetryManager_->UnRegisterOpRetryManager(identifier_);
        opRetryManager_ = nullptr;
    }
    resMap_.clear();
    deviceResOrigMem_.clear();
    hostResMap_.clear();
    tagCommInfo_.clear();
    tagWorkSpaceMem_.clear();
    tagStreamInfo_.clear();
    if (opRetryStreamPtr_ != nullptr) {
        opRetryStreamPtr_->clear();
        opRetryStreamPtr_ = nullptr;
    }
    (void)UnRegistTaskExceptionHandler();
    kfcControlTransferH2D_ = nullptr;
    kfcStatusTransferD2H_ = nullptr;

    oneSideService_ = nullptr;
    if (isOneSidedServiceNetDevCtxInited) {
        DeInitOneSidedServiceNetDevCtx();
    }

    MrManagerDeInit();

    /* 网络资源销毁 */
    DestroyNetworkResources();
    notifyPool_ = nullptr;
    /* driver关联资源释放 */
    if (drvInit_) {
        if (DisablePreResource() != HCCL_SUCCESS) {
            HCCL_WARNING("driver resource is not released successfully");
        }
    }

    if (isExecuteProfilingInit_) {
        (void)DeinitProfiling();
    }

    if (OpExeCounter::GetInstance(deviceLogicId_).DeInitCounter() != HCCL_SUCCESS) {
        HCCL_WARNING("op exec counter resource free failed");
    }

    /* 销毁当前trace句柄 */
    if (opBaseAtraceInfo_ != nullptr) {
        opBaseAtraceInfo_->DeInit();
        opBaseAtraceInfo_ = nullptr;
    }

    ReleaseWorkSpacebuffer();
    ReleaseCommContextbuffer();

    for (u32 i = 0; i < sizeof(aicpuOpNotify_) / sizeof(aicpuOpNotify_[0]); i++) {
        if (localAiCpuOpNotify_[i]) {
            HcclResult ret = localAiCpuOpNotify_[i]->Destroy();
            localAiCpuOpNotify_[i] = nullptr;
            if (ret != RT_ERROR_NONE) {
                HCCL_ERROR("[Destroy][AicpuNotify]errNo[0x%016llx] rt notify destroy fail, "\
                    "aicpuOpNotify[%u] return[%d].", HCCL_ERROR_CODE(HCCL_E_RUNTIME), i, ret);
            }
        }
    }

    while (!aiCpuNoIpcEvnet_.empty()) {
        rtEvent_t eventInfo = aiCpuNoIpcEvnet_.back();
        HcclResult ret = hrtEventDestroy(eventInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Destroy][AicpuNoIpcEvnet]errNo[0x%016llx] rt event destroy fail, "\
                "return[%d].", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        }
        aiCpuNoIpcEvnet_.pop_back();
    }

    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    if (vDispatcher_ != nullptr) {
        HcclDispatcherDestroy(vDispatcher_);
        vDispatcher_ = nullptr;
    }
    HCCL_DEBUG("~HcclCommunicator success.");
}

HcclResult HcclCommunicator::Init(HcclCommParams &params, const RankTable_t &rankTable)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(attrCollector_.Init(params, rankTable));
    CHK_RET(InitRankInfo(rankTable));
    CHK_RET(InitNetResource(rankTable));
    CHK_RET(InitDebug());
    CHK_RET(InitNotifyManager());
    CHK_RET(InitStreamManager());
    CHK_RET(InitTransportManager());
    CHK_RET(InitMemoryManager());
    CHK_RET(InitCombinOpara());
/*--------------加锁区--------------*/
    std::unique_lock<std::mutex> lock(g_hcomInitMutex);
    CHK_RET(RegistTaskExceptionHandler());

    attrCollector_.GenCollectiveId(params, rankTable);
    collectiveId_ = attrCollector_.GetCollectiveId();

    // 初始化参数(需要放置在ranktable解析之后)
    HcclResult ret = InitPara();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][Init]errNo[0x%016llx] collectiveid[%s] parameter initialization failed",
        HCCL_ERROR_CODE(ret), params.id.internal), ret);
    lock.unlock();
/*--------------加锁区--------------*/
    if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93) {
        CHK_RET(RegisterKernel(deviceType_));
    }
    CHK_RET(InitHDCommunicate());
    CHK_RET(InitOpRetry());
    CHK_RET(InitOpResPara());
    CHK_RET(InitOneSidedService(rankTable));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
    WorldGroupInfo &groupCommonData)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(attrCollector_.Init(params, rankList, groupCommonData));
    CHK_RET(InitRankInfoSubGroup(rankList, groupCommonData));
    CHK_RET(InitDebugSubGroup());
    CHK_RET(InitNotifyManager());
    CHK_RET(InitDispatcher());
    CHK_RET(InitStreamManager());
    CHK_RET(InitRaResource());
    CHK_RET(InitTransportManager());
    CHK_RET(InitMemoryManagerSubGroup());
    CHK_RET(InitHcclAlg());
    CHK_RET(InitHDCommunicate());
    CHK_RET(InitOpRetry());
    CHK_RET(InitOpResPara());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOneSidedService(const RankTable_t &rankTable)
{
    if (g_hcclOneSidedServiceCallback == nullptr) {
        HCCL_WARNING("[%s]g_hcclOneSidedServiceCallback is not registered, do not need to init "
        "oneSidedService.", __func__);
        return HCCL_SUCCESS;
    }
    g_hcclOneSidedServiceCallback(oneSideService_, socketManager_, notifyPool_);
    hcclRankLinkInfo_.userRank = userRank_;
    hcclRankLinkInfo_.devicePhyId = devicePhyId_;

    if (devIpAddr_.empty()) {
        HCCL_ERROR("[%s] device ip is invalid, please set device ip first.", __func__);
        return HCCL_E_NOT_FOUND;
    }
    hcclRankLinkInfo_.ip = devIpAddr_[0];
    if (nicRanksPort_.size() <= userRank_) {
        HCCL_ERROR("[%s] userRank_[%u] port is invalid, please set port first", __func__, userRank_);
        return HCCL_E_NOT_FOUND;
    }
    hcclRankLinkInfo_.port = nicRanksPort_[userRank_];
    hcclRankLinkInfo_.socketsPerLink = 1;
    HCCL_DEBUG("[%s]hcclRankLinkInfo_ userRank[%u], devicePhyId[%u], ip[%s], port[%u]", __func__,
        hcclRankLinkInfo_.userRank, hcclRankLinkInfo_.devicePhyId, hcclRankLinkInfo_.ip.GetReadableIP(),
        hcclRankLinkInfo_.port);
    CHK_RET(oneSideService_->Config(dispatcher_, hcclRankLinkInfo_, &rankTable));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
{
    if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        // 单边操作当前只支持Device网卡，不支持host
        HCCL_ERROR("[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.",
            __func__, nicDeployment_, userRankSize_);
        return HCCL_E_INTERNAL;
    }

    std::string localServerId = serverId_;
    std::string localSuperPodId = superPodId_;
    std::string remoteServerId = rankInfoList_.at(remoteRankId).serverId;
    std::string remoteSuperPodId = rankInfoList_.at(remoteRankId).superPodId;
    u32 intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    bool useRdma = false;
    if (intraRoceSwitch ||
        (!useSuperPodMode_ && localServerId != remoteServerId) ||
        (localSuperPodId != remoteSuperPodId)) {
        // 1. 初始化网口
        CHK_RET(InitNic());
        isOneSidedServiceNicInited = true;

        // 2. 单边操作SetNetDevCtx, RDMA
        if (netDevCtxMap_.find(devIpAddr_[0]) == netDevCtxMap_.end()) {
            HCCL_ERROR("[%s] nicDeployment_[%d], device nic init fail, please check", __func__, nicDeployment_);
            return HCCL_E_NOT_FOUND;
        }
        useRdma = true;
        oneSideService_->SetNetDevCtx(netDevCtxMap_[devIpAddr_[0]], useRdma);
        HCCL_INFO("[%s]init device Nic for oneSidedService success.", __func__);
    } else {
        // 单边操作SetNetDevCtx, IPC
        oneSideService_->SetNetDevCtx(netDevCtxMap_[localVnicIp_], useRdma);
        HCCL_INFO("[%s]init vNic for oneSidedService success.", __func__);
    }
    isOneSidedServiceNetDevCtxInited = true;
    HCCL_DEBUG("[%s]nicDeployment_[%d], intraRoceSwitch[%u]", __func__, nicDeployment_, intraRoceSwitch);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeInitOneSidedServiceNetDevCtx()
{
    if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        // 单边操作当前只支持Device网卡，不支持host
        HCCL_ERROR("[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.",
            __func__, nicDeployment_, userRankSize_);
        return HCCL_E_INTERNAL;
    }
    u32 intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    if (isOneSidedServiceNicInited) {
        // 1. close sockets
        if (raResourceInit_) {
            socketManager_->DestroySockets();
        }
        // 2. 去初始化网口
        CHK_RET(DeinitNic());
        isOneSidedServiceNicInited = false;
        HCCL_INFO("[%s]Deinit device Nic for oneSidedService success.", __func__);
    }
    isOneSidedServiceNetDevCtxInited = false;
    HCCL_DEBUG("[%s]nicDeployment_[%d], intraRoceSwitch[%u]", __func__, nicDeployment_, intraRoceSwitch);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetOneSidedService(IHcclOneSidedService** service)
{
    *service = oneSideService_.get();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOpResPara()
{
    CHK_SAFETY_FUNC_RET(
        memset_s(reinterpret_cast<void *>(&opResPara_), sizeof(HcclOpResParam), 0, sizeof(HcclOpResParam)));
    ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
    opResPara_.remoteResNum = 0;
    CHK_RET(GetOpCountInfo(opResPara_.opCounterInfo));
    CHK_RET(CreateWorkSpace(sizeof(HcclOpResParam), opResDevicePara_));

    opResDeviceParaPtr_ = static_cast<HcclOpResParam *>(opResDevicePara_.ptr());

    hostDeviceLock_.reset(new (std::nothrow) PetersonLock(PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC));
    CHK_SMART_PTR_NULL(hostDeviceLock_);
    CHK_RET(hostDeviceLock_->Init());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::PrepareZeroCopy(HcclCMDType opType, OpParam &opParam)
{
    // 默认关闭该特性
    opParam.isZeroCopy = false;

    // 目前zeroCopy只支持这些算子
    if (opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER &&
        opType != HcclCMDType::HCCL_CMD_ALLGATHER &&
        opType != HcclCMDType::HCCL_CMD_ALLREDUCE &&
        opType != HcclCMDType::HCCL_CMD_BROADCAST) {
        HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] opType[%s] not support zero copy", GetCMDTypeEnumStr(opType).c_str());
        return HCCL_SUCCESS;
    }

    // yxg-debug 不支持确定性计算，讨论是否有算法要使用scratchMem
    // 目前只支持A3单机内通信域，且在AICPU展开场景下，不支持重执行
    u64 minDataSize = 32 * 1024 * 1024; // 32MB以下不使能零拷贝
    u64 inputSize = (opType == HcclCMDType::HCCL_CMD_ALLGATHER) ? opParam.outputSize : opParam.inputSize;
    if (!opParam.aicpuUnfoldMode || retryEnable_ || serverNum_ != 1
        || GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        || deviceType_ != DevType::DEV_TYPE_910_93
        || userRankSize_ > MAX_MODULE_DEVICE_NUM
        || inputSize < minDataSize
        ) {
        HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] other scenes not support zero copy "
            "aicpuUnfold:%d retryEnable:%d serverNum:%u workflowMode:%d deviceType:%d rankSize:%u "
            "inputSize[%lu B]:minSize[%lu B]", opParam.aicpuUnfoldMode, retryEnable_, serverNum_, GetWorkflowMode(),
            deviceType_, userRankSize_, inputSize, minDataSize);
        return HCCL_SUCCESS;
    }

    // 判断输入输出地址是否都是支持零Copy特性的
    if (!ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(opParam.inputPtr, opParam.inputSize) ||
        !ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(opParam.outputPtr, opParam.outputSize)) {
        HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] input[%p] or output[%p] is not support zero copy", opParam.inputPtr, opParam.outputPtr);
        return HCCL_SUCCESS;
    }

    // 如果自己侧的共享内存没有申请，那么进行申请，并设置给transportManager，后续p2p建链时进行交换
    if (zeroCopyLocalBuffer_.ptr() == nullptr) {
        zeroCopyLocalBuffer_ = DeviceMem::alloc(ZERO_COPY_IPC_BUFFER_LENGTH);
        CHK_PRT_RET(!zeroCopyLocalBuffer_, HCCL_ERROR("[HcclCommunicator][PrepareZeroCopy]Create buffer size[%llu] fail,",
            ZERO_COPY_IPC_BUFFER_LENGTH), HCCL_E_PTR);
        CHK_RET(hrtMemSet(zeroCopyLocalBuffer_.ptr(), zeroCopyLocalBuffer_.size(), zeroCopyLocalBuffer_.size()));
        zeroCopyIpcPtrs_[userRank_] = zeroCopyLocalBuffer_.ptr();

        HCCL_RUN_INFO("[HCCL_TRACE][PrepareZeroCopy]Create ZeroCopy buffer success. buffer ptr[%p] size[%llu]",
            zeroCopyLocalBuffer_.ptr(), zeroCopyLocalBuffer_.size());
    }

    // 在图模式中算法中会使用零copy算法，因此我们在这里更新成图模式，这样能够选择零copy算法
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    opParam.isZeroCopy = true;
    HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] success to use zero copy feature");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &resp)
{
    if (!opParam.isZeroCopy) {
        return HCCL_SUCCESS;
    }

    // 遍历所有transport，找出里面的p2p链路对应的对端地址
    for (auto &levelNSubCommTransport : resp.opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
                LINK link = singleSubCommTransport.links[i];
                if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid) {
                    // 无效或者不支持的链路
                    continue;
                }

                // 在使能零拷贝场景，我们使用控制面内存做OpenIpc交换，因此这里取出input即可
                u32 remoteRank = link->GetRemoteRank();
                void *remotePtr = nullptr;
                CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remotePtr));
                CHK_PRT_RET(remoteRank >= MAX_MODULE_DEVICE_NUM || remotePtr == nullptr,
                    HCCL_ERROR("[BuildZeroCopyParam] invalid remoteRank[%u] or remotePtr[%p]", remoteRank, remotePtr), HCCL_E_PARA);
                CHK_PRT_RET(zeroCopyIpcPtrs_[remoteRank] != nullptr && zeroCopyIpcPtrs_[remoteRank] != remotePtr,
                    HCCL_ERROR("[BuildZeroCopyParam] zeroCopyIpcPtrs_[%u] is [%p] not equal to %p", remoteRank, zeroCopyIpcPtrs_[remoteRank],
                    remotePtr), HCCL_E_PARA);
                
                zeroCopyIpcPtrs_[remoteRank] = remotePtr;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildZeroCopyParam()
{
    // 不支持ZeroCopy
    if (zeroCopyLocalBuffer_.ptr() == nullptr) {
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; ++i) {
        opResPara_.zeroCopyIpcPtrs[i] = reinterpret_cast<u64>(zeroCopyIpcPtrs_[i]);
    }

    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        opResPara_.zeroCopyDevicePhyId[i] = rankInfoList_[i].devicePhyId;
    }

    CHK_RET(ZeroCopyMemoryAgent::GetRingBufferAddr(opResPara_.zeroCopyRingBuffer,
        opResPara_.zeroCopyHeadPtr, opResPara_.zeroCopyTailPtr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCommParams(HcclCommParams &params)
{
    commHandle_ = params.commHandle;
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    profilingOption_ = params.profilingOption;
    profilingInitiated_ = params.profilingInitiated;
    deviceType_ = params.deviceType;
    commWorkMode_ = params.commWorkMode;
    hcomGroupNicInit_ = params.hcomGroupNicInit;
    identifier_ = params.identifier;
    collectiveId_ = params.id.internal;
    ranktableCrc_ = params.ranktableCrc;
    commConnections_ = params.commConnections;
    commPortConfig_ = params.commPortConfig;

    HCCL_DEBUG(
        " userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
        userRank_,
        realUserRank_,
        userRankSize_,
        deviceLogicId_,
        deviceType_,
        commWorkMode_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRankInfo(const RankTable_t &rankTable)
{
    CHK_RET(InitTcpMode(rankTable));
    SetAttrs();
    localRank_ = attrCollector_.GetLocalRank();
    deviceLogicId_ = attrCollector_.GetDeviceLogicId();
    // 按通信域配置是否使用算子级重执行
    SetRetryEnable(deviceType_, superPodNum_, serverNum_, deviceNumPerAggregation_, isDiffDeviceType_, retryEnable_);
    // 校验A+X单机双module场景下通信能否建立
    CHK_RET(CheckSingleServerComm(rankTable.rankList));
    // 解析rank和port的映射信息
    CHK_RET(SetRanksPort(rankTable.rankList));
    return HCCL_SUCCESS;
}

bool HcclCommunicator::Is310PDuoCard()
{
    return (Is310P3Common(isHaveCpuRank_, deviceType_) &&
        (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == userRankSize_));
}
// 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
HcclResult HcclCommunicator::CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const
{
    if (serverNum_ == 1 && moduleNum_ == HCCL_MODULE_NUM_TWO && GetExternalInputIntraRoceSwitch() == 0) {
        std::vector<u32> devIdList0;
        std::vector<u32> devIdList1;
        for (RankInfo_t rankInfo : rankList) {
            if (rankInfo.deviceInfo.devicePhyId == HOST_DEVICE_ID) {
                HCCL_ERROR("[Check][SingleServerComm]not support cpu rank");
                return HCCL_E_NOT_SUPPORT;
            }
            if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                devIdList0.push_back(rankInfo.deviceInfo.devicePhyId);
            } else {
                devIdList1.push_back(rankInfo.deviceInfo.devicePhyId);
            }
        }
        std::sort(devIdList0.begin(), devIdList0.end());
        std::sort(devIdList1.begin(), devIdList1.end());

        if (devIdList0.size() != devIdList1.size()) {
            char errorLogBuffer[LOG_TMPBUF_SIZE];
            s32 ret = snprintf_s(errorLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                "errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "\
                "deviceNum in module0:[%d] not equal to deviceNum in module1:[%d], "\
                "you can export HCCL_INTRA_ROCE_ENABLE=1 to enable this scenario.",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),  serverNum_, moduleNum_, devIdList0.size(), devIdList1.size());
            CHK_PRT_CONT(ret == -1, HCCL_ERROR("Failed to build log info"));
            HCCL_ERROR("[Check][SingleServerComm]%s", errorLogBuffer);
            RPT_INPUT_ERR(true, "EI0010", std::vector<std::string>({"reason"}), \
                std::vector<std::string>({std::string(errorLogBuffer)}));
            return HCCL_E_NOT_SUPPORT;
        }
        for (size_t i = 0; i < devIdList0.size(); i++) {
            if (devIdList0[i] % DEVICE_PER_MODULE != devIdList1[i] % DEVICE_PER_MODULE) {
                char errorLogBuffer[LOG_TMPBUF_SIZE];
                s32 ret = snprintf_s(errorLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                    "errNo[0x%016llx]. In A+X serverNum_[%d], moduleNum_[%d] case: "\
                    "deviceId[%d] in module0 and deviceId[%d] in module1 are not on the same plane, "\
                    "you can export HCCL_INTRA_ROCE_ENABLE=1 to enable this scenario.",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), serverNum_, moduleNum_, devIdList0[i], devIdList1[i]);
                CHK_PRT_CONT(ret == -1, HCCL_ERROR("Failed to build log info"));
                HCCL_ERROR("[Check][SingleServerComm]%s", errorLogBuffer);
                RPT_INPUT_ERR(true, "EI0010", std::vector<std::string>({"reason"}), \
                    std::vector<std::string>({std::string(errorLogBuffer)}));
                return HCCL_E_NOT_SUPPORT;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetRanksPort(const std::vector<RankInfo_t> &rankList)
{
    bool devicePortSwitchOn = commPortConfig_.devPortSwitchOn;
    if (devicePortSwitchOn) {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        vnicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto &rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT
                ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
            vnicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.vnicPort == HCCL_INVALID_PORT
                ? HETEROG_CCL_PORT : rankInfo.deviceInfo.vnicPort;
        }
    } else {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto &rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT
                || rankInfo.deviceInfo.port < SINGLE_PROCESS_MIN_PORT
                || rankInfo.deviceInfo.port > SINGLE_PROCESS_MAX_PORT
                ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
        }
    }
    isUseRankPort_ = ((devicePortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) || isHaveCpuRank_)
        ? true : isUseRankPort_;
    HCCL_INFO("[HcclCommunicator][SetRanksPort] devicePortSwitchOn[%u], isHaveCpuRank[%u], isUseRankPort[%u], "
        "nicRanksPort size[%u], vnicRanksPort size[%u].",
        devicePortSwitchOn, isHaveCpuRank_, isUseRankPort_, nicRanksPort_.size(), vnicRanksPort_.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNetResource(const RankTable_t &rankTable)
{
    CHK_RET(InitPreResource(rankTable));
    CHK_RET(InitRaResource());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDebug()
{
    CHK_RET(InitProfiling());
    CHK_RET(InitATraceInfo());
    return HCCL_SUCCESS;
}

std::string HcclCommunicator::GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = { HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
        HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32 };
    if (needReduce) {
        if (!Is310P3Common(isHaveCpuRank_, deviceType_)) {
            supportList.insert(supportList.end(), { HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_INT64 });
        }
    } else {
        supportList.insert(supportList.end(), { HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
            HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64 });
        if (!Is310P3Common(isHaveCpuRank_, deviceType_)) {
            supportList.push_back(HCCL_DATA_TYPE_BFP16);
        }
    }

    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult HcclCommunicator::CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const vector<string> infoTitle({"ccl_op", "parameter", "value", "tips"});
    if (needReduce) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            if ((dataType == HCCL_DATA_TYPE_INT64) || (dataType == HCCL_DATA_TYPE_BFP16)) {
                RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType",
                    GetDataTypeEnumStr(dataType), "please check dataType"}));
                HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                    GetSupportDataType(needReduce).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        }

        if ((dataType == HCCL_DATA_TYPE_UINT64) ||
            (dataType == HCCL_DATA_TYPE_UINT8) || (dataType == HCCL_DATA_TYPE_UINT16) ||
            (dataType == HCCL_DATA_TYPE_UINT32) || (dataType == HCCL_DATA_TYPE_FP64) ||
            (dataType == HCCL_DATA_TYPE_RESERVED)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType",
                GetDataTypeEnumStr(dataType), "please check dataType"}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8) ||
            (Is310P3Common(isHaveCpuRank_, deviceType_) && dataType == HCCL_DATA_TYPE_BFP16)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, vector<string>({"CheckDataType", "dataType",
                GetDataTypeEnumStr(dataType), "please check dataType"}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitATraceInfo()
{
    /* 申请trace资源信息 */
    std::string logInfo = "HCCL_";
    logInfo.append(to_string(SalGetTid()));
    logInfo.append("_");
    logInfo.append(to_string(deviceLogicId_));
    opBaseAtraceInfo_.reset(new (std::nothrow) HcclTraceInfo());
    CHK_PTR_NULL(opBaseAtraceInfo_);
    CHK_RET(opBaseAtraceInfo_->Init(logInfo));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDebugSubGroup()
{
    CHK_RET(InitATraceInfo());
    CHK_RET(InitProfiler());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNotifyManager()
{
    queueNotifyManager_.reset(new (std::nothrow) QueueNotifyManager());
    CHK_SMART_PTR_NULL(queueNotifyManager_);
    CHK_RET(queueNotifyManager_->Init());
    queueNotifyManagerRefac_.reset(new (std::nothrow) QueueNotifyManager());
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    CHK_RET(queueNotifyManagerRefac_->Init());

    return HCCL_SUCCESS;
}

void TaskProfilerCallBack(void *userPtr, void *param, u32 length)
{
    static_cast<ProfilerManager *>(userPtr)->TaskProfilerHandle(param, length);
}

void TaskAivProfilerCallBack(void *userPtr, void *param, u32 length)
{
    static_cast<ProfilerManager *>(userPtr)->TaskAivProfilerHandle(param, length);
}

HcclResult HcclCommunicator::InitDispatcher()
{
    // 根据设备ID创建dispatcher
    if ((deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93) &&
        GetExternalInputHcclEnableFfts()) {
        CHK_PRT_CONT(GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !GetExternalInputHcclAicpuUnfold(),
            HCCL_RUN_INFO("Will use ffts mode."));
    } else {
        // 不满足ffts+特性开启条件。
        SetFftsSwitch(false);
    }
    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devicePhyId_, &dispatcher_));
    CHK_SMART_PTR_NULL(dispatcher_);

    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_VIRTURAL, devicePhyId_, &vDispatcher_));
    CHK_SMART_PTR_NULL(vDispatcher_);

    (void)RegisterLoadTaskCallBack(dispatcher_, static_cast<void *>(profilerManager_.get()), TaskProfilerCallBack);
    RegisterAlgCallBack(static_cast<void *>(profilerManager_.get()), TaskAivProfilerCallBack, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitStreamManager()
{
    opStreamManager_.reset(static_cast<OpBaseStreamManager *>(new (std::nothrow) OpBaseStreamManager));
    CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitSocketManager()
{
    socketManager_.reset(new (std::nothrow) HcclSocketManager(nicDeployment_, deviceLogicId_, devicePhyId_, userRank_));
    CHK_PTR_NULL(socketManager_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTransportManager()
{
    std::vector<u32> &nicRanksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32> &vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    transportManager_.reset(static_cast<TransportManager *>(new (std::nothrow) TransportManager(
        cclBufferManager_, socketManager_, dispatcher_, notifyPool_,
        rankInfoList_, userRank_, identifier_,
        deviceLogicId_, nicDeployment_, isHaveCpuRank_,
        static_cast<const void*>(&transportResInfo_), sizeof(transportResInfo_),
        isUseRankPort_, isUsedRdmaLevel0_, nicRanksPorts, vnicRanksPorts, useSuperPodMode_,
        devIpAddr_, hostIp_, localVnicIp_, netDevCtxMap_)));
    (void) transportManager_->SetPortConfig(commPortConfig_.devPortSwitchOn);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitZeroCopyMemoryAgent()
{
    CHK_PRT_RET(zeroCopyMemoryAgent_ != nullptr,
        HCCL_ERROR("[HcclCommunicator][InitZeroCopyMemoryAgent] ipc memory agent has init"), HCCL_E_INTERNAL);

    zeroCopyMemoryAgent_.reset(static_cast<ZeroCopyMemoryAgent *>(new (std::nothrow) ZeroCopyMemoryAgent(socketManager_, devicePhyId_,
        deviceLogicId_, localVnicIp_, rankInfoList_, userRank_, useSuperPodMode_, identifier_)));
    CHK_PTR_NULL(zeroCopyMemoryAgent_);
    CHK_RET(zeroCopyMemoryAgent_->Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitZeroCopyMemoryAgent(bool inDestructor)
{
    if (zeroCopyMemoryAgent_ != nullptr) {
        if (!inDestructor) {
            // 析构函数释放场景不做barrier close
            CHK_RET(zeroCopyMemoryAgent_->BarrierClose());
        }
        CHK_RET(zeroCopyMemoryAgent_->DeInit());
        zeroCopyMemoryAgent_ = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMemoryManager()
{
    CHK_RET(MrManagerInit());
    // server数量不为1且非TCP模式时初始化RDMA资源
    if (serverNum_ != SINGLE_SERVER_NUM && !GetExternalInputHcclIsTcpMode()) {
        CHK_RET(InitRecvMsgAndRequestBuffer());
        CHK_RET(InitMemBlocksAndRecvWrMem());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMemoryManagerSubGroup()
{
    CHK_RET(MrManagerInit());
    CHK_RET(InitRecvMsgAndRequestBuffer());
    CHK_RET(InitMemBlocksAndRecvWrMem());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitHcclAlg()
{
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));

    callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_,
        dispatcher_, nicDeployment_));
    CHK_SMART_PTR_NULL(callbackTask_);

    workSpaceRes_.reset(new (std::nothrow) WorkspaceResource(devicePhyId_, deviceLogicId_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    HcclTopoAttr topoAttr{};
    attrCollector_.GetTopoAttr(topoAttr);

    HcclAlgoAttr algoAttr{};
    attrCollector_.GetAlgoAttr(algoAttr);

    implAlg_ = new (std::nothrow) HcclAlg(cclBufferManager_, dispatcher_, vDispatcher_);
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->Init(static_cast<const void*>(&transportResInfo_), sizeof(transportResInfo_),
        workSpaceRes_, notifyPool_, netDevCtxMap_, queueNotifyManager_,
        algoAttr, topoAttr, false));
    return HCCL_SUCCESS;
}
void HcclCommunicator::SetAttrs()
{
    serverId_ = attrCollector_.GetServerId();
    superPodId_ = attrCollector_.GetSuperPodId();
    superDeviceId_ = attrCollector_.GetSuperDeviceId();
    // GetServerNum
    serverNum_ = attrCollector_.GetServerNum();
    // IsSuperPodMode
    useSuperPodMode_ = attrCollector_.GetSuperPodMode();
    // GetSuperPodNum
    superPodNum_ = attrCollector_.GetSuperPodNums();
    // GetInnerServerAverageDevice
    deviceNumPerAggregation_ = attrCollector_.GetDeviceNumPerAggregation();
    deviceNumPerServer_ = attrCollector_.GetDeviceNumPerServer();
    isHaveCpuRank_ = attrCollector_.GetHaveCpuRank();
    // TransformRankInfoByServerId
    servRankInfo_ = attrCollector_.GetServRankInfo();
    // GetModuleInfo
    isDiffDeviceModule_ = attrCollector_.GetDiffDeviceModule();
    isDiffDeviceType_ = attrCollector_.GetDiffDeviceType();
    gcdDeviceNumPerAggregation_ = attrCollector_.GetGcdDeviceNumPerAggregation();
    moduleNum_ = attrCollector_.GetModuleNum();
    multiModuleDiffDeviceNumMode_ = attrCollector_.GetMultiModuleDiffDeviceNumMode();
    multiSuperPodDiffServerNumMode_ = attrCollector_.GetMultiSuperPodDiffServerNumMode();
    // 生成nicList
    nicList_ = attrCollector_.GetNicList();
    // InitTopoInfo
    isSingleMeshAggregation_ = attrCollector_.GetSingleMeshAggregation();
    isAllRankSamePlane_ = attrCollector_.GetAllRankSamePlane();
    isStandardCard_ = attrCollector_.GetStandardCard();
    is310PDuoCard_ = attrCollector_.Get310PDuoCard();
    isCommon310P3DUO_ = attrCollector_.GetIsCommon310P3DUO();
    hccsPortNum_ = attrCollector_.GetHccsPortNum();
    attrCollector_.GetPairLinkCounter(pairLinkCounter_);
    attrCollector_.GetPairLinkInfo(pairLinkInfo_);
    // SetInterModeInSuperPod
    isUsedInterHccsMode_ = attrCollector_.GetUsedInterHccsMode();
    // GetRankInfoList
    rankInfoList_ = attrCollector_.GetRankInfoList();
    // Localinfo
    devIpAddr_ = attrCollector_.GetDevIpAddr();
    devBackupIpAddr_ = attrCollector_.GetDevBackupIpAddr();
    devBackupPort_ = attrCollector_.GetBackupDevPort();
    devBackupPort_ = devBackupPort_ == HCCL_INVALID_PORT ? AICPU_RETRY_BACKUP_PORT : devBackupPort_;
    devicePhyId_ = attrCollector_.GetDevicePhyId();
    hostIp_ = attrCollector_.GetHostIp();
    hostPort_ = attrCollector_.GetHostPort();

    interServer_ = attrCollector_.GetInterServe();
    nicDeployment_ = attrCollector_.GetNicDeployment(); //29
}
HcclResult HcclCommunicator::InitRankInfoSubGroup(const std::vector<RankInfo> &rankList,
    WorldGroupInfo &groupCommonData)
{
    SetAttrs();
    // inline reduce 开关
    inlineReduceSwitchOn_ = attrCollector_.GetInlineReduceSwitchOn();
    // CalAndSetMeshAggRankSize
    meshAggregationRankSize_ = attrCollector_.GetMeshAggregationRankSize();
    // IsUsedRdmaLevel0AndIpInvalid
    isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();

    CHK_RET(SetWorldGroupInfo(groupCommonData.phyIdNicInfoMap, groupCommonData.worldRankInfoList,
        groupCommonData.ranksPort, groupCommonData.vnicRanksPort));
    for (auto &rankInfo : worldRankInfoList_) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isUseRankPort_ = true;
            break;
        }
    }
    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));
    // 按通信域配置是否使用算子级重执行
    SetRetryEnable(deviceType_, superPodNum_, serverNum_, deviceNumPerAggregation_, isDiffDeviceType_, retryEnable_);
    groupNicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
    if (nicRanksPort_.size()) {
        for (auto &rankInfo : rankInfoList_) {
            groupNicRanksPort_[rankInfo.userRank] = nicRanksPort_[rankInfo.worldRank];
            HCCL_INFO("hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                "nic port[%u], devicePhyId[%d]",
                rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(),
                rankInfo.userRank, rankInfo.worldRank, groupNicRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
        }
    }
    commPortConfig_.devPortSwitchOn = groupCommonData.devPortSwitchOn;
    if (commPortConfig_.devPortSwitchOn) {
        groupVnicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
        if (vnicRanksPort_.size()) {
            for (auto &rankInfo : rankInfoList_) {
                groupVnicRanksPort_[rankInfo.userRank] = vnicRanksPort_[rankInfo.worldRank];
                HCCL_INFO("hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                    "vnic port[%u], devicePhyId[%d]",
                    rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(),
                    rankInfo.userRank, rankInfo.worldRank, groupVnicRanksPort_[rankInfo.userRank],
                    rankInfo.devicePhyId);
            }
        }
    }
    isUseRankPort_ = ((commPortConfig_.devPortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE)
        || isHaveCpuRank_) ? true : isUseRankPort_;
    for (auto &rank : rankInfoList_) {
        if (hostIp_ != rank.hostIp) {
            isServerInter_ = true;
            HCCL_DEBUG(" isServerInter_ is true");
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearResMap(const std::string &tag, bool &findTag)
{
    auto resIter = resMap_.find(tag);
    if (resIter != resMap_.end()) {
        findTag = true;
        DestroyAlgResource(resIter->second);
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(resIter->second.slaveStreams));
        resMap_.erase(resIter);
        HCCL_INFO("[%s] clear resMap[%s]", __func__, tag.c_str());
    }
return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearOpResource(const std::string &tag)
{
    bool findTag = false;
    CHK_RET(ClearResMap(tag, findTag));
    CHK_RET(ClearResMap(tag + "_host", findTag));
    CHK_RET(ClearResMap(tag + "_device", findTag));
    if (!findTag) {
        HCCL_WARNING("[%s] not find tag[%s] in resMap", __func__, tag.c_str());
    }

    tagCommInfo_.erase(tag);
    // stream解绑定
    auto iterStream = tagStreamInfo_.find(tag);
    if (iterStream != tagStreamInfo_.end()) {
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(iterStream->second.ringStreams));
    }
    tagStreamInfo_.erase(tag);
    if (opRetryStreamPtr_ != nullptr) {
        opRetryStreamPtr_->erase(tag);
    }

    if (implAlg_ != nullptr) {
        CHK_RET(implAlg_->ClearOpResource(tag));
    }
    DestroyWorkspaceResource(tag);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearAivSyncBuf(bool aivClearEnable)
{
    aivClearEnable_ = aivClearEnable;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
    const HcomCollOpInfo &opInfo)
{
    return workSpaceRes_->CreateOpBasedResources(opType, tag, opInfo);
}

HcclResult HcclCommunicator::CreateRemoteOpBasedResources(u64 memSize, const std::string &tag)
{
    return workSpaceRes_->CreateRemoteOpBasedResources(memSize, tag);
}

HcclResult HcclCommunicator::DestroyRemoteOpBasedMem(const std::string &tag)
{
    return workSpaceRes_->DestroyRemoteOpBasedMem(tag);
}

bool HcclCommunicator::IsAtomicInit()
{
    if (!initializedFlag_.test_and_set()) {
        initializedFlag_.clear();
        return false;
    }
    return true;
}

bool HcclCommunicator::IsNeedNicInit()
{
    return ((nicInitialized_ == 0) && (!hcomGroupNicInit_) && (userRankSize_ > 1) && !isSingleMeshAggregation_ &&
        (superPodNum_ > 1 || !isUsedInterHccsMode_));
}

HcclResult HcclCommunicator::GetBandWidthPerNPU(u32 level, float &bandWidth)
{
    return hccl::GetBandWidthPerNPU(level, userRankSize_, deviceNumPerAggregation_, bandWidth);
}

HcclResult HcclCommunicator::GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation)
{
    deviceNumPerAggregation = deviceNumPerAggregation_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op)
{
    if ((deviceType_ == DevType::DEV_TYPE_910B) || (deviceType_ == DevType::DEV_TYPE_910_93)) {
        if ((op == HCCL_REDUCE_PROD) &&
        ((dataType == HCCL_DATA_TYPE_INT16) || (dataType == HCCL_DATA_TYPE_BFP16))) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "CheckReduceDataType",
                "dataType",
                GetDataTypeEnumStr(dataType),
                "please check dataType when optype is prod"
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] and data "\
                "type[%s] for Op[%s]", HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(HCCL_DATA_TYPE_BFP16).c_str(),
                GetDataTypeEnumStr(HCCL_DATA_TYPE_INT16).c_str(),
                GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        if (dataType == HCCL_DATA_TYPE_INT16) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "CheckReduceDataType",
                "dataType",
                GetDataTypeEnumStr(dataType),
                "please check the data type when the device type is 910."
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s]",\
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(dataType).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        if (dataType == HcclDataType::HCCL_DATA_TYPE_INT16 && op != HcclReduceOp::HCCL_REDUCE_SUM) {
            RPT_INPUT_ERR(true, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
                std::vector<std::string>({
                "CheckReduceDataType",
                "op",
                GetReduceOpEnumStr(op),
                "please check operation type when the data type is int16."
                }));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] device type[%d] does not support the data type[%s] for Op[%s]",\
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_INT16).c_str(),
                GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgType(AlgType &algType, HcclCMDType opType)
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAlgType(algType, opType);
}

u32 HcclCommunicator::GetRankTableCrc()
{
    return ranktableCrc_;
}

u32 HcclCommunicator::GetServerNum()
{
    return serverNum_;
}

u32 HcclCommunicator::GetModuleNum()
{
    return moduleNum_;
}

HcclResult HcclCommunicator::GetCommParams(HcclCommParams &params)
{
    params.commHandle = commHandle_;
    params.rank = userRank_;
    params.userRank = realUserRank_;
    params.totalRanks = userRankSize_;
    params.logicDevId = deviceLogicId_;
    params.deviceType = deviceType_;
    params.hcomGroupNicInit = hcomGroupNicInit_;
    params.identifier = identifier_;
    params.ranktableCrc = ranktableCrc_;
    params.commConnections = commConnections_;
    params.commPortConfig.devPortSwitchOn = commPortConfig_.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetCommRankTable(RankTable_t &rankTable)
{
    for (auto &server : servRankInfo_) {
        for (auto &rank : server.second) {
            rankTable.rankList.emplace_back(rank);
        }
    }
    rankTable.serverNum = serverNum_;
    rankTable.superPodNum = superPodNum_;
    rankTable.nicDeploy = nicDeployment_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitPara()
{
    // 检查当前user_rank 对应的devid和rt查到的一致
    CHK_RET(attrCollector_.CheckLocalRankInfo());
    CHK_RET(attrCollector_.CalAndSetMeshAggRankSize());
    meshAggregationRankSize_ = attrCollector_.GetMeshAggregationRankSize();

    CHK_RET(InitProfiler());

    CHK_RET(InitDispatcher());

    // 初始化计数任务
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));

    callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_,
        dispatcher_, nicDeployment_));
    CHK_SMART_PTR_NULL(callbackTask_);

    workSpaceRes_.reset(new (std::nothrow)
                            WorkspaceResource(devicePhyId_, deviceLogicId_, &cclBufferManager_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    HcclTopoAttr topoAttr{};
    attrCollector_.GetTopoAttr(topoAttr);

    HcclAlgoAttr algoAttr{};
    attrCollector_.GetAlgoAttr(algoAttr);

    implAlg_ = new (std::nothrow) HcclAlg(cclBufferManager_, dispatcher_, vDispatcher_);
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->Init(static_cast<const void*>(&transportResInfo_), sizeof(transportResInfo_),
        workSpaceRes_, notifyPool_, netDevCtxMap_, queueNotifyManager_,
        algoAttr, topoAttr, false));

    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsStandardCard()
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_INFO("The current device just support this StandardCard case.");
        return true;
    }

    return ((pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0) &&
           (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size() == 0) &&
           (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size() == 0));
}

HcclResult HcclCommunicator::InitHDCommunicate()
{
    if ((GetExternalInputHcclAicpuUnfold() == true) ||
        ((deviceType_ == DevType::DEV_TYPE_910_93) || (deviceType_ == DevType::DEV_TYPE_910B) ||
          Is310P3Common(isHaveCpuRank_, deviceType_))) {
        EXECEPTION_CATCH((kfcControlTransferH2D_ =
            std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl))),
            return HCCL_E_PTR);
        CHK_RET(kfcControlTransferH2D_->InitHost());
        EXECEPTION_CATCH((kfcStatusTransferD2H_ =
            std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus))),
            return HCCL_E_PTR);
        CHK_RET(kfcStatusTransferD2H_->InitHost());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOpRetry()
{
    EXECEPTION_CATCH((opRetryStreamPtr_ = std::make_shared<HcclOpStreamRes>()), return HCCL_E_PTR);
    if (retryEnable_) {
        opRetryManager_.reset(new (std::nothrow) OpRetryManager());
        HcclIpAddress hostIp = !rankInfoList_.empty() ? rankInfoList_[0].hostIp : HcclIpAddress();
        u32 hostPort = !rankInfoList_.empty() ? rankInfoList_[0].hostPort : HCCL_INVALID_PORT;
        s32 hostDevId = !rankInfoList_.empty() ? rankInfoList_[0].devicePhyId : 0;
        HcclIpAddress localIp = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].hostIp : HcclIpAddress();
        auto notifyResetCallback = [this](bool isSendRecv, s64 destRank){
            return isSendRecv? this->ResetNotifyForDestRank(destRank) : this->ResetNotify(); };

        auto setTransportStatusCallback = [this](const HcclOpIdentifier &opId, bool statusStop,
            const std::map<u32, bool> &remoteRankPortMap, const std::map<u32, bool> &isChangeLinkMap, bool isChangeLinkFlag){
                return this->SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag); };

        HcclNetDevCtx netDevCtx = netDevCtxMap_[devIpAddr_[0]];
        HcclNetDevCtx backUpNetDevCtx;
        if (IsEnableBackupLink() && netDevCtxMap_.find(devBackupIpAddr_[0]) != netDevCtxMap_.end()) {
            backUpNetDevCtx = netDevCtxMap_[devBackupIpAddr_[0]];
        }
        OpRetryServerInfo serverInfo = {hostIp, hostPort, hostDevId};
        OpRetryAgentInfo agentInfo = {userRank_, deviceLogicId_, localIp, devIpAddr_[0], netDevCtx, backUpNetDevCtx};

        CHK_RET(opRetryManager_->RegisterOpRetryMachine(identifier_, userRankSize_, commConnections_.isRoot,
            commConnections_.agentConnection, commConnections_.serverConnections,
            kfcControlTransferH2D_, kfcStatusTransferD2H_, opRetryStreamPtr_, notifyResetCallback,
            setTransportStatusCallback, IsEnableBackupLink(), serverInfo, agentInfo));
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicator::CompareWithServerId(const ServerInfo_t &left, const ServerInfo_t &right)
{
    return (strcmp(left.serverId.c_str(), right.serverId.c_str()) < 0);
}

bool HcclCommunicator::CompareWithNicName(const NetworkInfo_t &left, const NetworkInfo_t &right)
{
    return (strcmp(left.ethName.c_str(), right.ethName.c_str()) < 0);
}

bool HcclCommunicator::CompareWithUserRank(const RankInfo &left, const RankInfo &right)
{
    return left.userRank < right.userRank;
}

HcclResult HcclCommunicator::InitPreResource(const RankTable_t &rankTable)
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[Init][PreResource]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    }
    (void)rankTable;
    // 查询本rank所在服务器
    auto iterServ = servRankInfo_.find(serverId_);

    bool check = (iterServ == servRankInfo_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Init][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
        HCCL_E_NOT_FOUND);

    for (u32 i = 0; i < iterServ->second.size(); i++) {
        if (iterServ->second[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            enableP2PDevices_.push_back(iterServ->second[i].deviceInfo.devicePhyId);
        }
    }
    if (deviceType_ != DevType::DEV_TYPE_310P3) {
        HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Init][PreResource]Enable P2P Failed, deviceLogicId[%d], ret[%u]",
            deviceLogicId_, ret), ret);
    }

    drvInit_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTcpMode(const RankTable_t &rankTable) const
{
    bool isTcpMode = false;
    HCCL_INFO("[TcpMode][%u] [1:TCP, 2:RDMA, 3:RESERVED]", GetExternalInputProtocolType());
    if (GetExternalInputProtocolType() == ProtocolType::TCP) {
        isTcpMode = true;
    } else if (GetExternalInputProtocolType() == ProtocolType::RDMA) {
    // 通信协议选择RDMA
    } else {
        isTcpMode = (rankTable.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST &&
            !GetExternalInputHcclHostRdmaEnable());
        HCCL_INFO("[Init][TcpMode]isTcpMode[%d] nicDeploy[%d] hcclDeviceNicDisable[%d] hcclHostRdmaEnable[%d]",
            isTcpMode, rankTable.nicDeploy, GetExternalInputHcclDeviceNicDisable(),
            GetExternalInputHcclHostRdmaEnable());
    }
    SetTcpMode(isTcpMode);

    // 异构场景解析外部输入,放在SetTcpMode前防止Tcp用例走错分支，放在RecordProtocolType确保hdc模式下建链通信协议校验正确
    CHK_RET(InitExternalInputHeterog());
    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsEnableRoce()
{
    return attrCollector_.IsEnableRoce();
}

bool HcclCommunicator::IsEnableBackupLink()
{
    return deviceType_ == DevType::DEV_TYPE_910_93 && IsEnableRoce() && GetExternalInputHcclAicpuUnfold() &&
        GetExternalInputInterSuperPodRetryEnable() && !devBackupIpAddr_[0].IsInvalid() && rtsSupportChangeLink_ &&
        !isDiffDeviceType_;
    return false;
}

HcclResult HcclCommunicator::InitRaResource()
{
    /* 本通信域内只有1个device时，不需要初始化ra资源 */
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device.");
        return HCCL_SUCCESS;
    }

    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

    if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID ||
        nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_, false));
        if (IsEnableBackupLink()) {
            // 超节点 && level2支持重执行 && Aicpu -> 初始化主备hccp资源(Pid粒度)
            CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
            if (hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackUpLogicId_) != HCCL_SUCCESS) {
                rtsSupportChangeLink_ = false;
                HCCL_RUN_WARNING("[%s]Runtime does not support changelink, deviceLogicId_[%d], devicePhyId_[%u], "
                "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]"
                "rtsSupportChangeLink_[%d]", __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_,
                deviceBackUpLogicId_, nicDeployment_, IsEnableBackupLink(), rtsSupportChangeLink_);
            } else {
                CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_, deviceBackUpLogicId_,
                    false, true));
                HCCL_DEBUG("[%s]Default & backup NetworkManager Init, deviceLogicId[%d], devicePhyId[%u], "
                    "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]",
                    __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_,
                    nicDeployment_, IsEnableBackupLink());
            }
        }
    }

    if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_) ||
        (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ||
        (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
        u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_, false));
    }

    CHK_RET(InitSocketManager());

    if (Is310PDevice()) {
        CHK_RET(InitNic());
    } else if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
        std::shared_ptr<HcclSocket> &devVnicSocket = commPortConfig_.devVnicListen.first;
        if (devVnicSocket) {
            localVnicIp_ = devVnicSocket->GetLocalIp();
            localVnicListenPort_ = devVnicSocket->GetLocalPort();
            HcclNetDevCtx &devVnicCtx = commPortConfig_.devVnicListen.second;
            CHK_PTR_NULL(devVnicCtx);
            netDevCtxMap_.insert(std::make_pair(localVnicIp_, devVnicCtx));
            CHK_RET(socketManager_->ServerInit(devVnicCtx, localVnicListenPort_));
            commPortConfig_.devVnicListen.second = nullptr;
            HCCL_INFO("[HcclCommunicator][InitRaResource] init vnic with listened socket sucess, "
                "listened ip[%s] port[%u]", localVnicIp_.GetReadableAddress(), localVnicListenPort_);
        } else {
            localVnicListenPort_ = GetLocalNicPort(NicType::VNIC_TYPE);
            localVnicIp_ = HcclIpAddress(devicePhyId_);
            if (useSuperPodMode_) {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, superDeviceId_, localVnicIp_));
            } else {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devicePhyId_, localVnicIp_));
            }
            HcclNetDevCtx vnicPortCtx;
            CHK_RET(HcclNetOpenDev(&vnicPortCtx, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
            CHK_PTR_NULL(vnicPortCtx);
            netDevCtxMap_.insert(std::make_pair(localVnicIp_, vnicPortCtx));
            CHK_RET(socketManager_->ServerInit(vnicPortCtx, localVnicListenPort_));
            HCCL_INFO("[HcclCommunicator][InitRaResource] init vnic with ip[%s] port[%u] success",
                localVnicIp_.GetReadableAddress(), localVnicListenPort_);
        }

        if (isHaveCpuRank_) {
            HcclNetDevCtx hostPortCtx;
            CHK_RET(HcclNetOpenDev(&hostPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, loopBackIp_));
            CHK_PTR_NULL(hostPortCtx);
            netDevCtxMap_.insert(std::make_pair(loopBackIp_, hostPortCtx));
            CHK_RET(socketManager_->ServerInit(hostPortCtx, hostPort_));
        }

        if (IsEnableRoce()) {
            CHK_RET(InitNic()); // isUsedRdmaLevel0_默认为false，若初始化网卡时，网卡IP有效才根据环境变量配置
        }
    }

    HCCL_INFO("isUsedRdmaLevel0_[%u] nicNum[%u] hostIP[%s], nicDeployment[%d].",
        isUsedRdmaLevel0_, devIpAddr_.size(), hostIp_.GetReadableAddress(), nicDeployment_);

    raResourceInit_ = true; // 全局通信域会初始化，子通信域不会初始化，但是析构均会进入此逻辑，需要标记
    attrCollector_.GenSupportRdmaLite();
    isSupportRdmaLite_ = attrCollector_.GetSupportRdmaLite(); // 是否支持Rdma Lite

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DisablePreResource()
{
    // 查询本rank所在服务器
    auto iterServ = servRankInfo_.find(serverId_);
    bool check = (iterServ == servRankInfo_.end());
    CHK_PRT_RET(check, HCCL_ERROR("[Disable][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
        HCCL_E_NOT_FOUND);
    HcclResult ret = P2PMgmtPub::DisableP2P(enableP2PDevices_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Disable][PreResource]Disable all P2P Failed, deviceLogicId[%d], ret[%u]",
        deviceLogicId_, ret), ret);
    enableP2PDevices_.clear();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType opType)
{
    AlgType algType;

    CHK_RET(GetAlgType(algType, opType));

    std::map<HcclCMDType, u64> gapMap = {
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HCCL_SMALL_COUNT_2_MB + HCCL_SMALL_COUNT_512_KB},
        {HcclCMDType::HCCL_CMD_ALLGATHER, HCCL_SMALL_COUNT_2_MB + HCCL_SMALL_COUNT_512_KB},
        {HcclCMDType::HCCL_CMD_ALLREDUCE, (HCCL_SMALL_COUNT_1_MB + HCCL_SMALL_COUNT_512_KB) * userRankSize_}
    };

    if (serverNum_ == 1 && deviceType_ == DevType::DEV_TYPE_910_93 && gapMap.find(opType) != gapMap.end() &&
        !GetExternalInputHcclAicpuUnfold() && dataSize <= gapMap[opType] &&
        deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) {
        streamNum = deviceNumPerAggregation_ - HCCL_SUB_STREAM_NP_MESH;
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93 Single Server, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        streamNum = HCCL_SUB_STREAM_NUM_DOUBLE_RING + RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
        if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING && GetExternalInputHcclAicpuUnfold()) {
            streamNum += 1U; // semi_ring算法server内增加一条从流，需要2条从流
        }
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    // 根据所用算法，选择所需的从stream数目
    switch (algType.algoLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
            streamNum = userRankSize_ / moduleNum_ - HCCL_SUB_STREAM_NP_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
            streamNum = HCCL_SUB_STREAM_NUM_8P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            streamNum = (GetExternalInputEnableRdmaSdmaConcurrent() == false) ? HCCL_SUB_STREAM_NUM_DOUBLE_RING :
            HCCL_SUB_STREAM_NUM_DOUBLE_RING + RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
            streamNum = HCCL_SUB_STREAM_NUM_4P_MESH;
            break;
        default:
            streamNum = (GetExternalInputEnableRdmaSdmaConcurrent() == false) ? HCCL_SUB_STREAM_NUM_ZERO :
            HCCL_SUB_STREAM_NUM_DOUBLE_RING;
            break;
    }

    if (SatisfyIntraSuperPod(deviceType_, userRankSize_, useSuperPodMode_, superPodNum_)) {
        streamNum = std::max(static_cast<u64>(userRankSize_ - 1u), streamNum);
    } else if (FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(deviceType_,
        meshAggregationRankSize_, useSuperPodMode_)) {
        streamNum = std::max(static_cast<u64>(meshAggregationRankSize_ - 1u), streamNum);
    }

    auto iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
    CHK_PRT_RET(iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(),
        HCCL_ERROR("[GetWorkspaceSubStreamNum]level0: algType[%u] is invalid.", algType.algoLevel0),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[GetWorkspaceSubStreamNum]hccl algorithm: In level0, using %s algo, the streamNum is %llu",
        iter->second.c_str(), streamNum);

    u64 sliceNum = CalculatePiplineSliceNum(opType, dataSize, algType, deviceType_, deviceNumPerServer_, serverNum_);
    // 图模式下数据量固定, 按照当前数据量判断是否支持pipline切分并申请从流
    if (implAlg_ != nullptr && sliceNum >= MIN_PIPLINE_SLICE_NUM) {
        streamNum++;
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::DestroyOpTransportResponse(OpCommTransport &opTransportResponse)
{
    std::lock_guard<std::mutex> commLock(linkResMapMutex_);
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.virtualLinks.size();i++) {
                if (singleSubCommTransport.virtualLinks[i] != nullptr) {
                    linkResMap_.erase(singleSubCommTransport.virtualLinks[i].get());
                    singleSubCommTransport.virtualLinks[i]->DeInit();
                }
            }
            for (u32 i = 0; i < singleSubCommTransport.links.size();i++) {
                if (singleSubCommTransport.transportRequests[i].isValid
                    && singleSubCommTransport.links[i] != nullptr) {
                    linkResMap_.erase(singleSubCommTransport.links[i].get());
                    singleSubCommTransport.links[i]->DeInit();
                }
            }
        }
    }
}

void HcclCommunicator::DestroyAlgResource(AlgResourceResponse &res)
{
    DestroyOpTransportResponse(res.opTransportResponse);
    if (IsEnableBackupLink()) {
        DestroyOpTransportResponse(res.opTransportResponseBackUp);
        HCCL_INFO("[%s]finish DestroyOpTransportResponse", __func__);
    }
}

HcclResult HcclCommunicator::ReleasePreemptSocket()
{
    if (commPortConfig_.devNicListen.first) {
        CHK_RET(PreemptPortManager::GetInstance(deviceLogicId_).Release(commPortConfig_.devNicListen.first));
        commPortConfig_.devNicListen.first.reset();
        if (commPortConfig_.devNicListen.second) {
            HcclNetCloseDev(commPortConfig_.devNicListen.second);
            commPortConfig_.devNicListen.second = nullptr;
        }
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
        HCCL_INFO("[HcclCommunicator][ReleasePreemptSocket] release preempt socket of device nic success, "
            "comm id[%s].", identifier_.c_str());
    }

    if (commPortConfig_.devVnicListen.first) {
        CHK_RET(PreemptPortManager::GetInstance(deviceLogicId_).Release(commPortConfig_.devVnicListen.first));
        commPortConfig_.devVnicListen.first.reset();
        if (commPortConfig_.devVnicListen.second) {
            HcclNetCloseDev(commPortConfig_.devVnicListen.second);
            commPortConfig_.devVnicListen.second = nullptr;
        }
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
        HCCL_INFO("[HcclCommunicator][ReleasePreemptSocket] release preempt socket of device vnic success, "
            "comm id[%s].", identifier_.c_str());
    }

    if (commPortConfig_.backupDevNicListen.first) {
        CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
        if (hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackUpLogicId_) == HCCL_SUCCESS) {
            CHK_RET(PreemptPortManager::GetInstance(deviceBackUpLogicId_)
                .Release(commPortConfig_.backupDevNicListen.first));
            commPortConfig_.backupDevNicListen.first.reset();
            if(commPortConfig_.backupDevNicListen.second) {
                HcclNetCloseDev(commPortConfig_.backupDevNicListen.second);
                commPortConfig_.backupDevNicListen.second = nullptr;
            }
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_,
                deviceBackUpLogicId_, true));
            HCCL_INFO("[HcclCommunicator][ReleasePreemptSocket] release preempt socket of backup nic success, "
                "comm id[%s].", identifier_.c_str());
        }
    }

    HCCL_INFO("[HcclCommunicator][ReleasePreemptSocket] release all preempt socket success, comm id[%s].",
        identifier_.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DestroyNetworkResources()
{
    transportManager_ = nullptr;
    if (raResourceInit_) {
        socketManager_->DestroySockets();
    }

    /* 本通信域内只有1个device时，不需要卸载ra资源 */
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device");
        return HCCL_SUCCESS;
    }

    // nic的初始化独立调用，在此单独判断是否需要解初始化
    if (nicInitialized_ > 0) {
        CHK_RET(DeinitNic());
    }

    if (raResourceInit_ && (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) && !Is310PDevice()) {
        if (isHaveCpuRank_) {
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[loopBackIp_], hostPort_));
            HcclNetCloseDev(netDevCtxMap_[loopBackIp_]);
            netDevCtxMap_.erase(loopBackIp_);
        }
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[localVnicIp_], localVnicListenPort_));
        HcclNetCloseDev(netDevCtxMap_[localVnicIp_]);
        netDevCtxMap_.erase(localVnicIp_);
    }

    CHK_RET(ReleasePreemptSocket());

    if (raResourceInit_) {
        if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID ||
            nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            if (IsEnableBackupLink()) {
                // 超节点 && level2支持重执行 && Aicpu -> 释放主备hccp资源
                CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
                CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_,
                    deviceBackUpLogicId_, true));
                HCCL_DEBUG("[%s]Default & backup HcclNetDeInit, deviceLogicId[%d], devicePhyId[%u], "
                    "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]",
                    __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_,
                    nicDeployment_, IsEnableBackupLink());
            } else {
                CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
            }
        }

        if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_) ||
            (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) ||
            (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
            u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_));
        }

        socketManager_ = nullptr;
    }

    raResourceInit_ = false;
    return HCCL_SUCCESS;
}


HcclResult HcclCommunicator::SetWorkspaceResource(const std::string &tag, void *memPtr, u64 &maxSize,
    std::vector<rtStream_t> &stream)
{
    return workSpaceRes_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
}

void HcclCommunicator::DestroyWorkspaceResource(const std::string &tag)
{
    workSpaceRes_->DestroyWorkspaceResource(tag);
}

HcclResult HcclCommunicator::AtomicInitSet()
{
    CHK_PRT_RET(initializedFlag_.test_and_set(),
        HCCL_ERROR("[HcclCommunicator][AtomicInitSet]errNo[0x%016llx] instance "
                   "already been initialized",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void HcclCommunicator::AtomicInitClear()
{
    initializedFlag_.clear();
}

u32 HcclCommunicator::GetUserRank()
{
    return realUserRank_;
}

u32 HcclCommunicator::GetGroupRank()
{
    return userRank_;
}

u32 HcclCommunicator::GetRankSize()
{
    return userRankSize_;
}

bool HcclCommunicator::GetNicInitialized()
{
    return nicInitialized_ > 0;
}

HcclResult HcclCommunicator::CheckDeviceType(const DevType deviceType) const
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR("[Check][DeviceType]errNo[0x%016llx] deivce Type[%d] out of range[%d, %d]",
            HCCL_ERROR_CODE(HCCL_E_PARA), deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckReductionOp(const HcclReduceOp op) const
{
    if ((op >= HCCL_REDUCE_RESERVED) || (op < HCCL_REDUCE_SUM)) {
        HCCL_ERROR("[Check][ReductionOp]errNo[0x%016llx] op:[%d] not supported", HCCL_ERROR_CODE(HCCL_E_PARA), op);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckUserRank(const u32 userRank) const
{
    if (userRankSize_ <= userRank) {
        HCCL_ERROR("[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]",
            HCCL_ERROR_CODE(HCCL_E_PARA), userRank, userRankSize_);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckCount(const u64 count) const
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupRanksInfo(const std::vector<u32> &groupRanks, std::vector<RankInfo> &ranksInfo)
{
    ranksInfo.clear();
    std::vector<RankInfo> tmpRankInfoList;
    tmpRankInfoList.assign(rankInfoList_.begin(), rankInfoList_.end());

    for (u32 index = 0; index < groupRanks.size(); index++) {
        if (tmpRankInfoList.size() <= groupRanks[index]) {
            HCCL_ERROR("[Get][GroupRanksInfo]errNo[0x%016llx] groupRanks[%u]=[%u], >= rankinfolist size[%zu]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, groupRanks[index], tmpRankInfoList.size());
            return HCCL_E_PARA;
        }
        tmpRankInfoList[groupRanks[index]].userRank = index;
        ranksInfo.push_back(tmpRankInfoList[groupRanks[index]]);
        HCCL_DEBUG("index: %d userRank: %dhost ip: %s host port: %u dev phy id: %d serverIdx:%d",
            index,
            tmpRankInfoList[groupRanks[index]].userRank,
            tmpRankInfoList[groupRanks[index]].hostIp.GetReadableAddress(),
            tmpRankInfoList[groupRanks[index]].hostPort,
            tmpRankInfoList[groupRanks[index]].devicePhyId,
            tmpRankInfoList[groupRanks[index]].serverIdx);
    }

    // 按rank id从小到大的顺序返回
    std::sort(ranksInfo.begin(), ranksInfo.end(), CompareWithUserRank);

    for (u32 index = 0; index < ranksInfo.size(); ++index) {
        if (index != ranksInfo[index].userRank) {
            HCCL_ERROR("[Get][GroupRanksInfo]errNo[0x%016llx] index[%u] !=  user rank[%u]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, ranksInfo[index].userRank);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupCommonData(WorldGroupInfo &groupCommonData) const
{
    groupCommonData.inlineReduceSwitchOn = inlineReduceSwitchOn_;
    groupCommonData.deviceType = deviceType_;
    groupCommonData.deviceLogicId = deviceLogicId_;
    groupCommonData.profilingInitiated = profilingInitiated_;
    groupCommonData.serverId = serverId_;
    groupCommonData.phyIdNicInfoMap = rankDevicePhyIdNicInfoMap_;
    groupCommonData.worldRankInfoList = rankInfoList_;
    groupCommonData.ranksPort = nicRanksPort_;
    groupCommonData.vnicRanksPort = vnicRanksPort_;
    groupCommonData.useSuperPodMode = useSuperPodMode_;
    groupCommonData.devPortSwitchOn = commPortConfig_.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
    u32 &rankSize, u64 &memSize, DevType &deviceType) const
{
    return workSpaceRes_->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize, deviceType);
}

DeviceMem HcclCommunicator::GetWorkspaceScracthMem(const std::string &tag, u64 allocMemSize)
{
    return workSpaceRes_->AllocDeviceMem(tag, allocMemSize);
}

std::vector<Stream> HcclCommunicator::GetWorkspaceSubStreams(const std::string &tag, u32 num)
{
    return workSpaceRes_->AllocSlaveStreams(tag, num);
}

HcclResult HcclCommunicator::InitProfiling()
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[Init][Profiling]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(profilingInitiated_, HCCL_DEBUG("Profiling plugin has already been Initiated."), HCCL_SUCCESS);

    if (profilingMode_ != HcomProfilingMode::PROFILING_OPEN && GetExternalInputProfilingMode()) {
        profilingMode_ = HcomProfilingMode::PROFILING_OPEN;
        profilingOption_ = GetExternalInputProfilingOption();
    }
    HCCL_INFO("profiling config information:options[%s], mode[%d]", profilingOption_.c_str(), profilingMode_);

    // profilingInitiated_会广播给所有子通信域，用于避免taskInfoSaver的重复初始化
    profilingInitiated_ = true;
    // isExecuteProfilingInit_用于记录本impl是否执行了taskInfoSaver的初始化，用于进行对应的释放
    isExecuteProfilingInit_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitProfiling()
{
    CHK_PRT_RET(!profilingInitiated_, HCCL_DEBUG("Profiling plugin has not been Initiated"), HCCL_SUCCESS);
    profilingInitiated_ = false;
    HCCL_INFO("Profiling is deinitiated.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegistTaskExceptionHandler() const
{
    CHK_RET(TaskExceptionHandler::Init());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnRegistTaskExceptionHandler() const
{
    CHK_RET(TaskExceptionHandler::DeInit());

    return HCCL_SUCCESS;
}

ErrorMessageReport HcclCommunicator::GetAicpuTaskException()
{
    HcclResult ret = HCCL_SUCCESS;
    ErrorMessageReport errorMessage;
    if (kfcStatusTransferD2H_ != nullptr) {
        ret = kfcStatusTransferD2H_->Get(sizeof(HcclOpIdentifier) + sizeof(ExecStatusDef),
            sizeof(errorMessage), reinterpret_cast<uint8_t *>(&errorMessage));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("GetAicpuTaskException get aicpu task exception failed.ret[%u]", ret);
        }
    }
    return errorMessage;
}

HcclResult HcclCommunicator::UnRegisterBackGroundThread()
{
    HCCL_INFO("start to stop the backGround Thread");
    if (deviceType_ == DevType::DEV_TYPE_910 || deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93) {
        if (GetMC2EnvFlag()) {
            if (kfcControlTransferH2D_ != nullptr) {
                BackgroundCommand request = BackgroundCommand::kStop;
                CHK_RET(kfcControlTransferH2D_->Put(sizeof(KfcCommand),
                        sizeof(BackgroundCommand),
                        reinterpret_cast<uint8_t *>(&request))); //下的停止命令仅仅只修改BackGroundCommand
                auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
                auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
                auto startTime = std::chrono::steady_clock::now();
                while (true) {
                    if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                        HCCL_WARNING("[NsRecovery]~HcclCommunicator is timeout [%u ms]", waitStopExecCmdTimeoutMs);
                        return HCCL_E_INTERNAL;
                    }
                    KfcExecStatus status;
                    CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&status)));
                    if (status.execStatus.backgroundStatus == BackgroundStatus::kStop) {
                        break;
                    }
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DestroyAicpuComm()
{
    HCCL_INFO("[HcclCommunicator][%s]start to destroy the aicpu comm, group[%s].", __func__, identifier_.c_str());
    if (deviceType_ != DevType::DEV_TYPE_910_93) {
        HCCL_WARNING("[HcclCommunicator][%s]Device type is not A3, no needs to destroy the aicpu comm.", __func__);
        return HCCL_SUCCESS;
    }
    if (kfcControlTransferH2D_ == nullptr) {
        HCCL_WARNING("[HcclCommunicator][%s]kfcControlTransferH2D_ is nullptr, can not destroy the aicpu comm.",
            __func__);
        return HCCL_SUCCESS;
    }
    if (!GetMC2EnvFlag()) {
        HCCL_WARNING("[HcclCommunicator][%s]Not mc2 or aicpu environment, no needs to destroy the aicpu comm.",
            __func__);
        return HCCL_SUCCESS;
    }
    KfcCommand destroyCmd = KfcCommand::kDestroyComm;
    CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t *>(&destroyCmd)));
    KfcExecStatus status;
    auto waitCmdTimeoutMs = HcclGetCmdTimeout();
    auto waitCmdTimeout = std::chrono::milliseconds(waitCmdTimeoutMs);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&status)));
        if (status.execStatus.kfcStatus == KfcStatus::kDestroyComm) {
            HCCL_RUN_INFO("[HcclCommunicator][%s]ExecStatus[%d]", __func__, status.execStatus.kfcStatus);
            return HCCL_SUCCESS;
        } else {
            if((std::chrono::steady_clock::now() - startTime) >= waitCmdTimeout){
                HCCL_ERROR("[HcclCommunicator][%s]Wait DestroyExec response status timeout[%u ms] and get the "
                "ExecState is [%d].", __func__, waitCmdTimeoutMs, status.execStatus.kfcStatus);
                return HCCL_E_INTERNAL;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInCCLbuffer(void* &buffer, u64 &size)
{
    return cclBufferManager_.GetInCCLbuffer(buffer, size);
}

HcclResult HcclCommunicator::GetOutCCLbuffer(void* &buffer, u64 &size)
{
    return cclBufferManager_.GetOutCCLbuffer(buffer, size);
}

void HcclCommunicator::ReleaseCommCCLbuffer()
{
    cclBufferManager_.ReleaseCommCCLbuffer();
}

HcclResult HcclCommunicator::ReleaseCommInfos()
{
    if (implAlg_ != nullptr) {
        return implAlg_->ReleaseCommInfos();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitProfiler()
{
    profilerManager_.reset(new (std::nothrow) ProfilerManager(devicePhyId_, deviceLogicId_, realUserRank_));
    CHK_SMART_PTR_NULL(profilerManager_);
    HcclResult ret = profilerManager_->InitProfiler();
    CHK_PRT_RET((ret != HCCL_SUCCESS), HCCL_ERROR("[BASE][InitProfiler]profilerManager_ InitProfiler failed."),
        HCCL_E_PARA);

    HCCL_INFO("[BASE][InitProfiler]Register CtrlCallBack success.");

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommCCLbuffer()
{
    return cclBufferManager_.CreateCommCCLbuffer();
}

HcclResult HcclCommunicator::CreateCommExpBuffer()
{
    return cclBufferManager_.CreateCommExpBuffer();
}

HcclResult HcclCommunicator::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
{
    return cclBufferManager_.InitCCLbuffer(inCCLbufferSize, outCCLbufferSize);
}

u32 HcclCommunicator::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

u32 HcclCommunicator::GetLocalNicPort(NicType nicType)
{
    u32 port = HCCL_INVALID_PORT;
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        return GetHostPort(devicePhyId_);
    }
    // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
    // groupRanksPort_为空说明此时处于全局通信域，要从ranksPort_取监听端口；否则取groupRanksPort_
    bool devicePortSwitchOn = commPortConfig_.devPortSwitchOn;
    if (nicType == NicType::HOST_NIC_TYPE) {
        port = GetHostPort(devicePhyId_);
    } else if (devicePortSwitchOn && nicType == NicType::VNIC_TYPE) {
        // vnic ports仅在开启device侧端口配置时单独配置
        std::vector<u32> &ranksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
        port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    } else {
        // 1. 开启device侧端口配置时的nic port时使用ranksPorts
        // 2. 异构场景使用ranksPorts
        // 3. 其余场景场景isUseRankPort_应当为false，使用默认port
        std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    }
    HCCL_INFO("[HcclCommunicator][GetLocalNicPort] nicType[%u], devicePortSwitchOn[%u], isUseRankPort[%u], "
        "get port[%u], devId[%u]", nicType, devicePortSwitchOn, isUseRankPort_, port, devicePhyId_);
    return port;
}

HcclResult HcclCommunicator::InitNic(bool isMC2ReInit)
{
    if (!GetExternalInputIntraRoceSwitch() && servRankInfo_.size() == 1 && isDiffDeviceModule_ && !isMC2ReInit) {
        return HCCL_SUCCESS;
    }

    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        std::shared_ptr<HcclSocket> &devNicSocket = commPortConfig_.devNicListen.first;
        if (devNicSocket) {
            HcclNetDevCtx &devNicCtx = commPortConfig_.devNicListen.second;
            CHK_PTR_NULL(devNicCtx);
            netDevCtxMap_.insert(std::make_pair(devNicSocket->GetLocalIp(), devNicCtx));
            CHK_RET(socketManager_->ServerInit(devNicCtx, devNicSocket->GetLocalPort()));
            commPortConfig_.devNicListen.second = nullptr;
            HCCL_INFO("[HcclCommunicator][InitNic] init nic with listened socket sucess, "
                "listened ip[%s] port[%u]",
                devNicSocket->GetLocalIp().GetReadableAddress(), devNicSocket->GetLocalPort());
        } else {
            u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
            u32 nicNum = devIpAddr_.size();
            for (u32 i = 0; i < nicNum; i++) {
                if (devIpAddr_[i].IsInvalid()) {
                    HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
                    continue;
                }
                attrCollector_.GenUsedRdmaLevel0();
                isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();
                HcclNetDevCtx nicPortCtx;
                CHK_RET(HcclNetOpenDev(&nicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, devIpAddr_[i]));
                CHK_PTR_NULL(nicPortCtx);
                netDevCtxMap_.insert(std::make_pair(devIpAddr_[i], nicPortCtx));
                CHK_RET(socketManager_->ServerInit(nicPortCtx, port));
                HCCL_INFO("[HcclCommunicator][InitNic] init nic with ip[%s] port[%u] success",
                    devIpAddr_[i].GetReadableAddress(), port);
            }
        }

        if (IsEnableBackupLink()) {
            std::shared_ptr<HcclSocket> &backupNicSocket = commPortConfig_.backupDevNicListen.first;
            if (backupNicSocket) {
                HcclNetDevCtx &backupNicCtx = commPortConfig_.backupDevNicListen.second;
                CHK_PTR_NULL(backupNicCtx);
                netDevCtxMap_.insert(std::make_pair(backupNicSocket->GetLocalIp(), backupNicCtx));
                CHK_RET(socketManager_->ServerInit(backupNicCtx, backupNicSocket->GetLocalPort()));
                commPortConfig_.backupDevNicListen.second = nullptr;
                HCCL_INFO("[HcclCommunicator][InitNic] init backup nic with listened socket sucess, "
                    "listened ip[%s] port[%u]",
                    backupNicSocket->GetLocalIp().GetReadableAddress(), backupNicSocket->GetLocalPort());
            } else {
                // 超节点 && level2支持重执行 && Aicpu -> 备用网卡 initRdma
                HcclNetDevCtx nicPortBackUpCtx;
                CHK_RET(HcclNetOpenDev(&nicPortBackUpCtx, NicType::DEVICE_NIC_TYPE, deviceBackUpPhyId_,
                    deviceBackUpLogicId_, devBackupIpAddr_[0], devIpAddr_[0]));
                CHK_PTR_NULL(nicPortBackUpCtx);
                netDevCtxMap_.insert(std::make_pair(devBackupIpAddr_[0], nicPortBackUpCtx));
                CHK_RET(socketManager_->ServerInit(nicPortBackUpCtx, devBackupPort_));
                HCCL_DEBUG("[%s]finish backup ServerInit, deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], "
                    "devBackupIpAddr_[%s], devBackupPort_[%u], nicDeployment_[%d], IsEnableBackupLink[%d], "
                    "netDevCtxMap_.size[%d]",
                    __func__, deviceBackUpPhyId_, deviceBackUpLogicId_, devBackupIpAddr_[0].GetReadableAddress(),
                    devBackupPort_, nicDeployment_, IsEnableBackupLink(), netDevCtxMap_.size());
                HCCL_INFO("[HcclCommunicator][InitNic] init backup nic with ip[%s] port[%u] success",
                    devBackupIpAddr_[0].GetReadableAddress(), devBackupPort_);
            }
        }
    }  else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        u32 port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
        CHK_PRT_RET((hostIp_.IsInvalid()), HCCL_ERROR("[Init][Nic] host ip is invalid when NIC "
        "deployment is host. "), HCCL_E_PARA);
        attrCollector_.GenUsedRdmaLevel0();
        isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();
        u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
        HCCL_INFO("[Init][Nic], hostPort[%u], devicePhyID[%u]", port, devicePhyID);
        HcclNetDevCtx hostnicPortCtx;
        CHK_RET(HcclNetOpenDev(&hostnicPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, hostIp_));
        CHK_PTR_NULL(hostnicPortCtx);
        netDevCtxMap_.insert(std::make_pair(hostIp_, hostnicPortCtx));
        CHK_RET(socketManager_->ServerInit(hostnicPortCtx, port));
    } else {
        HCCL_ERROR("[Init][Nic]nic deployment[%d] is not supported", nicDeployment_);
        return HCCL_E_PARA;
    }
    isNeedInitNic_ = true;
    attrCollector_.SetNeedInitNicFlag(isNeedInitNic_);
    nicInitialized_++;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitNic()
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
        u32 nicNum = devIpAddr_.size();
        for (u32 i = 0; i < nicNum; i++) {
            if (devIpAddr_[i].IsInvalid()) {
                HCCL_INFO("continue invalid devIp %s", devIpAddr_[i].GetReadableAddress());
                continue;
            }
            if (netDevCtxMap_.find(devIpAddr_[i]) == netDevCtxMap_.end()) {
                HCCL_INFO("devIp[%s] not found in netDevCtxMap_", devIpAddr_[i].GetReadableAddress());
                continue;
            }
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devIpAddr_[i]], port));
            // 最后一次调用才删除netCtx
            if (nicInitialized_ - 1 <= 0) {
                HcclNetCloseDev(netDevCtxMap_[devIpAddr_[i]]);
                netDevCtxMap_.erase(devIpAddr_[i]);
            }
        }
        if (IsEnableBackupLink() && netDevCtxMap_.find(devBackupIpAddr_[0]) != netDevCtxMap_.end()) {
            // 超节点 && level2支持重执行 && Aicpu -> 备用网卡 deinit
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devBackupIpAddr_[0]], devBackupPort_));
            if (nicInitialized_ - 1 <= 0) {
                HcclNetCloseDev(netDevCtxMap_[devBackupIpAddr_[0]]);
                netDevCtxMap_.erase(devBackupIpAddr_[0]);
                HCCL_DEBUG("[%s]finish backup ServerDeInit devBackupIpAddr_[%s], port[%u], IsEnableBackupLink[%d]",
                    __func__, devBackupIpAddr_[0].GetReadableAddress(), devBackupPort_, IsEnableBackupLink());
            }
        }
    } else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        u32 port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
        CHK_PRT_RET((hostIp_.IsInvalid()), HCCL_ERROR("[DeInit][Nic] host ip is invalid when NIC "
        "deployment is host. "), HCCL_E_PARA);
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[hostIp_], port));
        HcclNetCloseDev(netDevCtxMap_[hostIp_]);
        netDevCtxMap_.erase(hostIp_);
    } else {
        HCCL_ERROR("[Deinit][Nic]nic deployment[%d] is not supported", nicDeployment_);
        return HCCL_E_PARA;
    }
    nicInitialized_--;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegisterToHeartBeat()
{
    u32 localPort = commPortConfig_.devPortSwitchOn ? HCCL_INVALID_PORT : GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
    return Heartbeat::GetInstance(deviceLogicId_).RegisterToHeartBeat(userRank_, deviceType_, rankInfoList_,
        localPort, isNeedInitNic_, identifier_, useSuperPodMode_, isUsedRdmaLevel0_, retryEnable_,
        IsEnableBackupLink());
}

HcclResult HcclCommunicator::RegisterToHeartBeat(u32 peerRankId, string &tag)
{
    u32 localPort = commPortConfig_.devPortSwitchOn ? HCCL_INVALID_PORT : GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
    return Heartbeat::GetInstance(deviceLogicId_).RegisterToHeartBeat(userRank_, deviceType_, rankInfoList_,
        localPort, isNeedInitNic_, peerRankId, identifier_, tag, useSuperPodMode_, isUsedRdmaLevel0_,
        retryEnable_, IsEnableBackupLink());
}

void HcclCommunicator::UnRegisterToHeartBeat()
{
    for (auto tag : hbSendRecvTags_) {
        Heartbeat::GetInstance(deviceLogicId_).UnRegisterToHeartBeat(deviceType_, identifier_, tag);
    }
    Heartbeat::GetInstance(deviceLogicId_).UnRegisterToHeartBeat(deviceType_, identifier_);
}

HcclResult HcclCommunicator::SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr)
{
    CHK_RET(HcclSetGlobalWorkSpace(dispatcher_, globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAttachedStream(const std::vector<rtStream_t> &streams)
{
    // 在图模式下，通信使用的附属从流可能不同，所以这里直接刷新所有
    attachedStreams_.clear();

    for (auto &s : streams) {
        if (s != nullptr) {
            HCCL_DEBUG("[HcclCommunicator][SetAttachedStream] stream ptr [%p]", s);
            attachedStreams_.push_back(Stream(s, false));
        }
    }

    HCCL_INFO("[HcclCommunicator][SetAttachedStream] input streams[%llu] actual streams[%llu]",
        streams.size(), attachedStreams_.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo)
{
    if (profilerManager_ != nullptr) {
        CHK_RET(profilerManager_->GetandClearOverFlowTasks(hcclDumpInfo));
    } else {
        HCCL_WARNING("[impl][GetDumpTask] profilerManager_ not set");
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetDeviceId(s32 &deviceId) const
{
    deviceId = deviceLogicId_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetQosCfg(const u32 qosCfg)
{
    CHK_PTR_NULL(dispatcher_);
    return HcclSetQosCfg(dispatcher_, qosCfg);
}

HcclResult HcclCommunicator::ResetQosCfg()
{
    CHK_PTR_NULL(dispatcher_);
    return HcclResetQosCfg(dispatcher_);
}

HcclResult HcclCommunicator::GetQosCfg(u32& qosCfg)
{
    CHK_PTR_NULL(dispatcher_);
    return HcclGetQosCfg(dispatcher_, &qosCfg);
}

HcclResult HcclCommunicator::GetCqeError(HcclResult &result)
{
    CHK_RET(Heartbeat::GetInstance(deviceLogicId_).CheckErrorCqe(identifier_, result));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::MrManagerInit()
{
    // 拉远、下沉、推理场景(ps、worker)支持使用mrManager
    if (!GetExternalInputHcclIsTcpMode() && (Is310PDevice())) {
        mrManager_.reset(new (std::nothrow) MrManager(netDevCtxMap_[devIpAddr_[0]]));
        CHK_SMART_PTR_NULL(mrManager_);

        CHK_RET(mrManager_->Init());
        mrManagerInit_ = true;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::MrManagerDeInit()
{
    if (mrManagerInit_) {
        CHK_SMART_PTR_NULL(mrManager_);
        CHK_RET(mrManager_->DeInit());
        mrManager_ = nullptr;
        mrManagerInit_ = false;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SupportDeterministicOptim(bool &isDeterministicOptim)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SupportDeterministicOptim(isDeterministicOptim));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetHccsLinkNum(u32 &numHccsLink)
{
    auto iter = pairLinkInfo_.find(static_cast<u32>(LinkTypeInServer::HCCS_TYPE));
    if (iter == pairLinkInfo_.end()) {
        HCCL_ERROR("[HcclCommunicator][GetHccsLinkNum]HCCS_TYPE is not found");
        return HCCL_E_PARA;
    }
    numHccsLink = iter->second.size();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
    HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    bool aicpuUnfoldMode = false;
if (GetExternalInputHcclAicpuUnfold() == true &&
    (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AllGather]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = inputCount * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize * userRankSize_;
    opParam.DataDes.count = inputCount;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER,
        tag, inputCount, dataType, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherV(const std::string &tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
    const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclRtStream stream)
{
    bool aicpuUnfoldMode = false;

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AllGatherV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = sendCount * perDataSize;

    u64 outputSize = 0;
    const u64* counts = static_cast<const u64 *>(recvCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        outputSize += counts[i] * perDataSize;
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.inputSize = totalSize;
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.outputSize = outputSize;
    opParam.VDataDes.dataType = dataType;
    opParam.VDataDes.counts = const_cast<void *>(recvCounts);
    opParam.VDataDes.displs = const_cast<void *>(rdispls);
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER_V,
        tag, recvCounts, rdispls, userRankSize_, dataType, cclBufferManager_.GetInCCLbufferSize(),
        cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType cmdType)
{
    Stream streamObj(stream);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    CHK_RET(ProfilerAdd(opParam, algType));
    HcclResult ret = HCCL_SUCCESS;
    if (!IsExistCommRes(identifier_)) {
        HCCL_INFO("[AicpuUnfold] tag[%s] count[%llu] dataType[%s] op[%s].", identifier_.c_str(),
            count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(stream, &streamMode));

        rtStream_t aicpuStream;
        ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
        void *commContext = nullptr;
        ret = CreateCommResource(identifier_, stream, true, &commContext);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]",
                identifier_.c_str(), ret);
            return ret;
        }
    }

    std::string kernelName = "RunAicpuRpcSrvLaunch";
    AicpuOpTiling opTilingInfo;
    ret = AicpuKfcTilingDataLaunch(opParam, cmdType, commContext_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"\
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]", ret, inputPtr, outputPtr, count,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        return ret;
    }
    CHK_RET(ProfilerDel(opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 inputCount, HcclDataType dataType, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGatherOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = inputCount * perDataSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = inputCount;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER,
        tag, inputCount, dataType, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, 
    u64 inputCount, const void *outputCounts, const void *outputDispls, HcclDataType dataType, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ == 1) {
        // rankSize为1时，退化为AllGather
        return AllGather(tag, inputPtr, outputPtr, inputCount, dataType, stream);
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGatherVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = 0;
    const u64* counts = static_cast<const u64 *>(outputCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        outputSize += counts[i] * perDataSize;
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputCount * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = outputSize;
    opParam.VDataDes.counts = const_cast<void *>(outputCounts);
    opParam.VDataDes.displs = const_cast<void *>(outputDispls);
    opParam.VDataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = false;

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLGATHER_V,
        tag, outputCounts, outputDispls, userRankSize_, dataType, cclBufferManager_.GetInCCLbufferSize(),
        cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

void HcclCommunicator::GetAndSetSyncMode(SyncMode& preSyncMode, SyncMode newSyncMode)
{
    if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            HCCL_WARNING("310P don't support unlimited notify wait mode");
        } else {
            HcclGetNotifyWaitMode(dispatcher_, &preSyncMode);
            HcclSetNotifyWaitMode(dispatcher_, newSyncMode);
        }
    }
}

void HcclCommunicator::RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode)
{
    if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE && !Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HcclSetNotifyWaitMode(dispatcher_, preSyncMode);
    }
}

HcclResult HcclCommunicator::AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
    SyncMode syncMode, const HcomCollOpInfo *opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) &&
        deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AllReduce]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    // 设置notify wait模式
    SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    GetAndSetSyncMode(preSyncMode, syncMode);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    /* 将输入数据量按照字节对齐扩展，占用图模式512Byte尾内存，在不支持InlineReduce场景下,
       reduce scatter 可以并发从对端接收 */
    if (GetExternalInputHcclHighPerfEnable() != 0 &&
        userRankSize_ <= HCCL_DEVICE_NUM_FOUR && deviceType_ == DevType::DEV_TYPE_910) {
        u64 alignSize = HCCL_MIN_SLICE_ALIGN * userRankSize_;
        u64 remainder = totalSize % alignSize;
        if (remainder != 0) {
            count = count - remainder / perDataSize + alignSize / perDataSize;
            totalSize = count * perDataSize;
        }
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = syncMode;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    // 用于inplace支持重执行场景的图模式归一至单算子模式
    retryOrigWorkflowMode_ = GetWorkflowMode();
    bool isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, userRank_, userRankSize_,
        isInplaceStatus_);
    if (aicpuUnfoldMode && retryEnable_ && isHcclOpInplace) {
        HCCL_DEBUG("The retry with inplace case is expected to be supported, "
            "aicpuUnfoldMode[%d], retryEnable_[%d], isHcclOpInplace[%d], "
            "therefore HcclWorkflowMode is converted from [%d] to HCCL_WORKFLOW_MODE_OP_BASE",
            aicpuUnfoldMode, retryEnable_, isHcclOpInplace, static_cast<u8>(retryOrigWorkflowMode_));
        CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    }

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE,
        tag, count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
    RestorePreSyncMode(preSyncMode, syncMode);
    CHK_RET(SetWorkflowMode(retryOrigWorkflowMode_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceAicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    Stream streamObj(stream);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    CHK_RET(ProfilerAdd(opParam, algType));
    HcclResult ret;
    if (!IsExistCommRes(tag)) {
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(stream, &streamMode));

        rtStream_t aicpuStream;
        ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
        void *commContext = nullptr;
        ret = CreateCommResource(tag, stream, true, &commContext);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]",
                tag.c_str(), ret);
            return ret;
        }
    }

    AicpuOpTiling opTilingInfo;
    std::string kernelName = "RunAicpuRpcSrvLaunch";
    ret = AicpuKfcTilingDataLaunch(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, commContext_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"\
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]", ret, inputPtr, outputPtr, count,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        return ret;
    }
    CHK_RET(ProfilerDel(opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
    SyncMode syncMode)
{
    CHK_RET(CheckSuspendingStatus());
    const u32 RANK_SIZE_TWO = 2;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && userRankSize_ >= RANK_SIZE_TWO &&
        Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HcclResult ret = AllReduceAicpuUnfold(tag, inputPtr, outputPtr, count, dataType, op, stream);
        CHK_PRT_RET((ret != HCCL_SUCCESS),
            HCCL_ERROR("[HcclCommunicator][AllReduce]errNo[0x%016llx]  tag[%s],all reduce aicpu unfold failed",
            HCCL_ERROR_CODE(ret), tag.c_str()), ret);

        return HCCL_SUCCESS;
    }

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) &&
        (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    // 设置notify wait模式
    SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    GetAndSetSyncMode(preSyncMode, syncMode);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = syncMode;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_ALLREDUCE,
        tag, count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
    RestorePreSyncMode(preSyncMode, syncMode);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AlltoAllV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = const_cast<void *>(sendCounts);
    opParam.All2AllDataDes.recvCounts = const_cast<void *>(recvCounts);
    opParam.All2AllDataDes.sdispls = const_cast<void *>(sdispls);
    opParam.All2AllDataDes.rdispls = const_cast<void *>(rdispls);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
    HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
    rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[AlltoAllVOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = const_cast<void *>(sendCounts);
    opParam.All2AllDataDes.recvCounts = const_cast<void *>(recvCounts);
    opParam.All2AllDataDes.sdispls = const_cast<void *>(sdispls);
    opParam.All2AllDataDes.rdispls = const_cast<void *>(rdispls);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][AlltoAllVC]AlltoAllVC is not supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AlltoAllVC]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = const_cast<void *>(sendCountMatrix);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
    const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[AlltoAllVCOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVCOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = const_cast<void *>(sendCountMatrix);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    opParam.aicpuUnfoldMode = deviceType_ == DevType::DEV_TYPE_910_93 && GetExternalInputHcclAicpuUnfold();

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
    const void *recvBuf, u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag)
{
    CHK_RET(CheckSuspendingStatus());
    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][AlltoAll]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    std::vector<u64> sendCountMatrix(userRankSize_ * userRankSize_, sendCount);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void *>(sendBuf);
    opParam.outputPtr = const_cast<void *>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCount = sendCount;
    opParam.All2AllDataDes.sendCountMatrix = static_cast<void *>(sendCountMatrix.data());
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    opParam.aicpuUnfoldMode = false;
    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        opParam.aicpuUnfoldMode = GetExternalInputHcclAicpuUnfold();
    }

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALL, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
    HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Broadcast]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    if (isHaveCpuRank_ && !isSetHDCModeInfo_ && isServerInter_) {
        isSetHDCModeInfo_ = true;
    }
    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Broadcast(tag, ptr, count, dataType, root, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = ptr;
        opParam.outputPtr = ptr;
        opParam.inputSize = totalSize;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.root = root;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.opType = HcclCMDType::HCCL_CMD_BROADCAST;

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType,
    u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[BroadcastOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][BroadcastOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Broadcast(tag, ptr, count, dataType, root, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = ptr;
        opParam.outputPtr = ptr;
        opParam.inputSize = totalSize;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.root = root;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opType = HcclCMDType::HCCL_CMD_BROADCAST;

        // 记录指令信息用于一致性校验
        CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BROADCAST, tag, count,
            dataType, root, cclBufferManager_.GetInCCLbufferSize(), 0, identifier_.c_str(), ranktableCrc_));

        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));

        // 移除tag对应的指令信息
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][Scatter]Scatter Not Supported Yet");
        return HCCL_E_NOT_SUPPORT;
    }
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Scatter]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 totalSize = outputSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = recvCount;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.root = root;
    opParam.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
    HcclDataType dataType, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][ScatterOutPlace]ScatterOutPlace Not Supported Yet");
        return HCCL_E_NOT_SUPPORT;
    }

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][ScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 totalSize = outputSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = recvCount;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.root = root;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.opType = HcclCMDType::HCCL_CMD_SCATTER;
    /* 记录指令信息用于一致性校验 */
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SCATTER, tag,
        recvCount, dataType, root, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][Reduce]Reduce Not Supported Yet");
        return HCCL_E_NOT_SUPPORT;
    }
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Reduce]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE;

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE,
        tag, count, dataType, op, root, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[ReduceOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.inputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE;
    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE,
        tag, count, dataType, op, root, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatter]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = userRankSize_ * count * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = count * perDataSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    // 用于inplace支持重执行场景的图模式归一至单算子模式
    retryOrigWorkflowMode_ = GetWorkflowMode();
    bool isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, userRank_, userRankSize_,
        isInplaceStatus_);
    if (aicpuUnfoldMode && retryEnable_ && isHcclOpInplace) {
        HCCL_DEBUG("The retry with inplace case is expected to be supported, "
            "aicpuUnfoldMode[%d], retryEnable_[%d], isHcclOpInplace[%d], "
            "therefore HcclWorkflowMode is converted from [%d] to HCCL_WORKFLOW_MODE_OP_BASE",
            aicpuUnfoldMode, retryEnable_, isHcclOpInplace, static_cast<u8>(retryOrigWorkflowMode_));
        CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    }
    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, tag,
        count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));
    CHK_RET(SetWorkflowMode(retryOrigWorkflowMode_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
    u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ > 1) {
        CHK_RET(CreateCommCCLbuffer());
        CHK_RET(CreateCommExpBuffer());
    }

    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]ReduceScatterOutPlace is not supported");
        return HCCL_E_NOT_SUPPORT;
    }

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true &&
        IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(),
        dataType, op) && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = userRankSize_ * count * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = count * perDataSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, tag,
        count, dataType, op, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterV(const std::string &tag, void *inputPtr,
    const void *inputCounts, const void *inputDispls, void *outputPtr, u64 outputCount,
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ == 1) {
        // rankSize为1时，退化为ReduceScatter
        return ReduceScatter(tag, inputPtr, outputPtr, outputCount, dataType, op, stream);
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatterV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 inputSize = 0;
    const u64* counts = static_cast<const u64*>(inputCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        inputSize += counts[i] * perDataSize;
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = outputCount * perDataSize;
    opParam.VDataDes.counts = const_cast<void *>(inputCounts);
    opParam.VDataDes.displs = const_cast<void *>(inputDispls);
    opParam.VDataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    opParam.aicpuUnfoldMode = false;

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, tag,
        inputCounts, inputDispls, userRankSize_, dataType, op, cclBufferManager_.GetInCCLbufferSize(),
        cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
    const void *inputCounts, const void *inputDispls, u64 outputCount, 
    HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ == 1) {
        // rankSize为1时，退化为ReduceScatter
        return ReduceScatterOutPlace(tag, inputPtr, outputPtr, outputCount, dataType, op, stream);
    }

    CHK_RET(CreateCommCCLbuffer());
    CHK_RET(CreateCommExpBuffer());
    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][ReduceScatterVOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 inputSize = 0;
    const u64* counts = static_cast<const u64*>(inputCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        inputSize += counts[i] * perDataSize;
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = outputCount * perDataSize;
    opParam.VDataDes.counts = const_cast<void *>(inputCounts);
    opParam.VDataDes.displs = const_cast<void *>(inputDispls);
    opParam.VDataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = false;

    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, tag,
        inputCounts, inputDispls, userRankSize_, dataType, op, cclBufferManager_.GetInCCLbufferSize(),
        cclBufferManager_.GetInCCLbufferSize(), identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BatchSendRecv(const std::string &tag, HcclSendRecvItem* sendRecvItemsPtr, u32 itemNum,
    rtStream_t stream)
{
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }
    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
    OpParam opParam;
    opParam.tag = tag;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.BatchSendRecvDataDes.sendRecvItemsPtr = sendRecvItemsPtr;
    opParam.BatchSendRecvDataDes.itemNum = itemNum;
    opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    // 记录指令信息用于一致性校验
    CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV,
        tag, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetInCCLbufferSize(),
        identifier_.c_str(), ranktableCrc_));

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, opParam));

    // 移除tag对应的指令信息
    CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(tag));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Send]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Send(tag, inputPtr, count, dataType, destRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = inputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.dstRank = destRank;
        opParam.opType = HcclCMDType::HCCL_CMD_SEND;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
    u32 destRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (static_cast<u8>(GetExternalInputHcclAicpuUnfold()) == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][SendOutPlace]SendOutPlace is not supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][SendOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    // 记录指令信息用于一致性校验
    HcclResult ret = RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_SEND, tag, count,
        dataType, cclBufferManager_.GetInCCLbufferSize(), 0, identifier_.c_str(), ranktableCrc_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->SendOutPlace(tag, inputPtr, count, dataType, destRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = inputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.dstRank = destRank;
        opParam.opType = HcclCMDType::HCCL_CMD_SEND;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));
    }

    // 移除tag对应的指令信息
    ret = RankConsistentcyChecker::GetInstance().DelOpPara(tag);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret),
        tag.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
    u32 srcRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Receive]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->Receive(tag, outputPtr, count, dataType, srcRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = outputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.srcRank = srcRank;
        opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count,
    HcclDataType dataType, u32 srcRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetExternalInputHcclAicpuUnfold() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][ReceiveOutPlace]ReceiveOutPlace is not supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReceiveOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);

    // 记录指令信息用于一致性校验
    HcclResult ret = RankConsistentcyChecker::GetInstance().RecordOpPara(HcclCMDType::HCCL_CMD_RECEIVE, tag, count,
        dataType, cclBufferManager_.GetInCCLbufferSize(), 0, identifier_.c_str(), ranktableCrc_);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("errNo[0x%016llx] record CMD with parameter error", HCCL_ERROR_CODE(ret)), ret);

    if (isHaveCpuRank_) {
        CHK_RET(implAlg_->ReceiveOutPlace(tag, outputPtr, count, dataType, srcRank, streamObj));
    } else {
        u32 perDataSize = SIZE_TABLE[dataType];
        u64 totalSize = count * perDataSize;

        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = outputPtr;
        opParam.inputSize = totalSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = totalSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.stream = streamObj;
        opParam.aicpuUnfoldMode = aicpuUnfoldMode;
        opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
        opParam.srcRank = srcRank;
        opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));
    }

    // 移除tag对应的指令信息
    ret = RankConsistentcyChecker::GetInstance().DelOpPara(tag);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("errNo[0x%016llx] delete CMD with parameters error. tag[%s]", HCCL_ERROR_CODE(ret),
        tag.c_str()), ret);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank,
    u64 inputCount, HcclDataType dataType, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (!IsAtomicInit()) {
        HCCL_ERROR("[HcclCommunicator][Gather]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    if (isHaveCpuRank_ && !isSetHDCModeInfo_ && isServerInter_) {
        isSetHDCModeInfo_ = true;
    }
    std::vector<u32> &ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, isSetHDCModeInfo_, isUseRankPort_);
    CHK_RET(implAlg_->Gather(tag, inputPtr, outputPtr, rootRank, inputCount, dataType, streamObj));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetInfoToDevice(const OpParam &opParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
    const HcclWorkflowMode &mode, Stream &stream)
{
    auto inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
    auto outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    if ((inAlltoAllvParaBuffer.ptr() == nullptr) || (outAlltoAllvParaBuffer.ptr() == nullptr)) {
        CHK_RET(
            cclBufferManager_.InitAlltoAllvParaBuffer(preMetaInfo->inputSize, preMetaInfo->outputSize));
        inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
        outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    }

    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    if ((inCCLbuffer.ptr() == nullptr) || (outCCLbuffer.ptr() == nullptr)) {
        CHK_RET(CreateCommCCLbuffer());
        inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    }
    auto expBuffer = cclBufferManager_.GetCommExpBuffer();
    if (expBuffer.ptr() == nullptr) {
        CHK_RET(CreateCommExpBuffer());
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(hcclStreamSynchronize(stream.ptr()));
    CHK_RET(hrtMemSyncCopy(inAlltoAllvParaBuffer.ptr(), preMetaInfo->inputSize, preMetaInfo->inputData.data(),
        preMetaInfo->inputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInfoFromDevice(const OpParam &opParam,
    const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
    const HcclWorkflowMode &mode, Stream &stream, HostMem& hostCollectBuffer)
{
    CHK_RET(hrtMemSyncCopy(hostCollectBuffer.ptr(), preMetaInfo->outputSize,
        cclBufferManager_.GetOutAlltoAllvParaBuffer().ptr(), preMetaInfo->outputSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    // 非单算子场景，中转内存使用完之后直接释放
    if (mode != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        cclBufferManager_.ReleaseAlltoAllvParaBuffer();
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator* &alltoAllOperator, const OpParam &opParam,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo)
{
    HCCL_INFO("Run with Graph, alloc new stream");
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    return RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, stream);
}

HcclResult HcclCommunicator::RegressCalPreOp(AlltoAllOperator* &alltoAllOperator, const OpParam &opParam,
    std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream)
{
    OpParam preProcessOpParam;
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]",
        mode), HCCL_E_INTERNAL);

    // h to d
    CHK_RET(SetInfoToDevice(opParam, preMetaInfo, mode, preProcessStream));
    // opParam准备
    CHK_RET(alltoAllOperator->PreparePreOpParam(preProcessOpParam, preMetaInfo, preProcessStream));

    // 回归调用其它算子
    HCCL_INFO("[HcclCommunicator][RegressCalPreOp] Regression calls other operators and opType[%u]",
        preMetaInfo->opType);
    CHK_RET(ExecOp(preMetaInfo->opType, preProcessOpParam));
    CHK_RET(hcclStreamSynchronize(preProcessStream.ptr()));
    HCCL_DEBUG("[HcclCommunicator][RegressCalPreOp] preProcess tag[%s].", preProcessOpParam.tag.c_str());
    SetWorkflowMode(mode);

    // d to h
    HostMem hostCollectBuffer = HostMem::alloc(preMetaInfo->outputSize);
    CHK_PTR_NULL(hostCollectBuffer.ptr());
    CHK_RET(GetInfoFromDevice(opParam, preMetaInfo, mode, preProcessStream, hostCollectBuffer));

    alltoAllOperator->SetPreProcessResult(std::move(hostCollectBuffer));
    HCCL_INFO("[HcclCommunicator][RegressCalPreOp] run success!");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOp(HcclCMDType opType, OpParam &opParam)
{
    CHK_RET(PrepareZeroCopy(opType, opParam));
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);

    // 算法选择
    std::string algName;
    std::string newTag;
    if (opParam.aicpuUnfoldMode) {
        // 用于inplace支持重执行判断
        HCCL_ERROR("aicpuUnfold Mode is online")
        CHK_RET(algOperator->SetRetryEnable(retryEnable_));
    }
    if (GetExternalInputHcclAivMode()) {
        // 用于判断图模式是否清零
        CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
    }
    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, algName, newTag)); 

    newTag += !opParam.isZeroCopy ? "" : "_ZeroCopy"; // 使能零拷贝特性需要使用新的Tag，避免影响已有算法
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && userRankSize_ > 1) {
        CHK_RET(CreateCommCCLbuffer());
    }
    CHK_RET(CreateCommExpBuffer());

    // 资源创建
    InsertNewTagToTagMap(newTag, opParam.tag);
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
        if (!isHaveCpuRank_) {
            if (isUseRankPort_) {
                std::vector<u32> &nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                std::vector<u32> &vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                Heartbeat::GetInstance(deviceLogicId_).SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts,
                    commPortConfig_.devPortSwitchOn);
            }
            if (opType == HcclCMDType::HCCL_CMD_SEND) {
                CHK_RET(RegisterToHeartBeat(opParam.dstRank, newTag));
                hbSendRecvTags_.emplace(newTag);
            } else if (opType == HcclCMDType::HCCL_CMD_RECEIVE) {
                CHK_RET(RegisterToHeartBeat(opParam.srcRank, newTag));
                hbSendRecvTags_.emplace(newTag);
            } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && retryEnable_) {
                CHK_RET(RegisterToHeartBeat(userRank_, newTag));
                hbSendRecvTags_.emplace(newTag);
            } else if (opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
                CHK_RET(RegisterToHeartBeat());
            }
        }
        CHK_RET(UpdateZeroCopy(opParam, resMap_[newTag]));
    } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        // batchsendrecv需要根据任务来确定和哪些卡建链，因此复用tag，并在此基础上实现增量建链
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcIncreLinkRequest(algName, opParam, resRequest));
        CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
    }

    // 算法执行
    bool selectA3AivAlg = algName.find("Aiv") != std::string::npos;
    if (selectA3AivAlg) {
        opParam.aicpuUnfoldMode = false;
        CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
        if (aivClearEnable_) {
            // 用于判断图模式是否清零
            CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
        }
    }
    // 头计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_));
    if (!opParam.aicpuUnfoldMode) {
    //if (opParam.aicpuUnfoldMode) {
        isInplaceStatus_ = 0;
        inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
        // algOperator->SupportRetryWithInplaceCheck 依赖 algOperator->SetRetryEnable 才能正确返回是否支持inplace
        
        inplaceSupportRetry_ = algOperator->SupportRetryWithInplaceCheck(
            opType, opParam, algName, isInplaceStatus_, inPlaceSupportRetryStatus_);
        auto algType = algOperator->GetAlgType();
        HCCL_INFO("[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%s], inplaceSupportRetry_[%d], opType[%d], "
            "isInplaceStatus_[%d], inPlaceSupportRetryStatus_[%d]",
            AlgTypeToStr(algType).c_str(), inplaceSupportRetry_, opType, isInplaceStatus_, inPlaceSupportRetryStatus_);
        CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType));
    } else {
        // HOST展开aclgraph场景，capture从流
        if (!selectA3AivAlg) {
            CHK_RET(CaptureSlaveStreams(opParam.stream.ptr(), resMap_[newTag].slaveStreams));
        }
        CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
        if (hostResMap_.find(newTag) == hostResMap_.end()) {
            hostResMap_.insert(newTag);
        }
        CHK_RET(algOperator->GetBlockDim(blockDim_));
    }
    // 尾计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_));
    if (selectA3AivAlg) {
        CHK_RET(algOperator->SetAivClearEnable(false));
        aivClearEnable_ = false;
    }

    if (opParam.isZeroCopy) {
        // 恢复opbase
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::FreeScratchMemOnOpBaseMode(DeviceMem &scratchMem, const OpParam &opParam,
    const HcclCMDType &opType)
{
    // 当前单算子模式下scratch内存为手动申请，需要手动进行释放
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || IsForceAicpuOpBaseMode(opParam, opType)) {
        scratchMem.free();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyAivCommInfoToDevice(AlgResourceResponse &algResResp)
{
    void* buffersInOut[MAX_RANK_SIZE_A3 * 2] = {};
    SingleSubCommTransport &transportInfo =
        const_cast<SingleSubCommTransport&>(algResResp.opTransportResponse[COMM_COMBINE_ORDER][COMM_INDEX_0]);
    u32 localRank = transportInfo.userRank2subCommRank[userRank_];
    u32 localRankSize = transportInfo.transportRequests.size();
    bool isOpbaseMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;

    for (u32 i = 0; i < localRankSize; i++) {
        u32 idx = (i << 1);
        if (i != localRank) {
            CHK_RET(transportInfo.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(buffersInOut[idx])));
            CHK_RET(transportInfo.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(buffersInOut[idx + 1])));
        } else {
            buffersInOut[idx] = isOpbaseMode ? algResResp.cclInputMem.ptr() : algResResp.paramInputMem.ptr();
            buffersInOut[idx + 1] = algResResp.aivOutputMem.ptr();
        }
    }
    const u32 bufferNum = 2;
    CHK_RET(hrtMemSyncCopy(static_cast<u8 *>(algResResp.aivOutputMem.ptr()) + (AIV_FLAG_SIZE - COMM_INFO_OFFSET),
        sizeof(u64) * localRankSize * bufferNum, buffersInOut, sizeof(u64) * localRankSize * bufferNum,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOpAlltoAll(HcclCMDType opType, OpParam &opParam)
{
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
    CHK_PTR_NULL(alltoAllOperator);

    // 算法选择
    std::string algName;
    std::string newTag;
    if (opParam.aicpuUnfoldMode) {
        // 用于inplace支持重执行判断
        CHK_RET(algOperator->SetRetryEnable(retryEnable_));
    }
    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    CHK_SMART_PTR_NULL(preMetaInfo);

    bool preProcessFlag = alltoAllOperator->JudgeIfNeedPreProcessAndGetParam(opParam, preMetaInfo);
    if (preProcessFlag) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, const_cast<Stream&>(opParam.stream)));
        } else {
            CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));
        }
    }

    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, algName, newTag));
    bool supportAicpuAlg = algName == "RunAlltoAllVFullMesh" || algName == "RunAlltoAllDirectFullmesh";
    bool isOpbaseMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    if ((opParam.aicpuUnfoldMode && supportAicpuAlg) || (isOpbaseMode && userRankSize_ > 1)) {
        CHK_RET(CreateCommCCLbuffer());
    }
    CHK_RET(CreateCommExpBuffer());

    // 资源创建
    InsertNewTagToTagMap(newTag, opParam.tag);
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));

        if (alltoAllOperator->IsSatisfyAlltoAllAivCondition(opParam) && deviceType_ == DevType::DEV_TYPE_910_93
            && serverNum_ > 1) {
            CHK_RET(CopyAivCommInfoToDevice(resMap_[newTag]));
        }

        if (!isHaveCpuRank_) {
            if (isUseRankPort_) {
                std::vector<u32> &nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                std::vector<u32> &vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                Heartbeat::GetInstance(deviceLogicId_).SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts,
                    commPortConfig_.devPortSwitchOn);
            }
             CHK_RET(RegisterToHeartBeat());
        }
    } else {
        bool needRecreateAlltoallComm = false;
        CHK_RET(alltoAllOperator->CheckNeedRecreateComm(algName, opParam, resMap_[newTag].scratchMem.size(),
            needRecreateAlltoallComm));
        HCCL_INFO("resMap_ find this newTag[%s], and need to judge whether recreate comm [%d]", newTag.c_str(),
            needRecreateAlltoallComm);
        if (needRecreateAlltoallComm) {
            CHK_RET(hcclStreamSynchronize(opParam.stream.ptr()));
            AlgResourceRequest resRequest;
            CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
            // alltoall算子重分配内存前需清除scratchMMem，防止内存泄漏
            CHK_RET(FreeScratchMemOnOpBaseMode(resMap_[newTag].scratchMem, opParam, opType));
            CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag]));
            if (!isHaveCpuRank_) {
                if (isUseRankPort_) {
                    std::vector<u32> &nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                    std::vector<u32> &vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                    Heartbeat::GetInstance(deviceLogicId_).SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts,
                        commPortConfig_.devPortSwitchOn);
                }
                CHK_RET(RegisterToHeartBeat());
            }
        } else {
            DeviceMem tinySendRecvMem;
            CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
            CHK_RET(CalcTinySendRecvMem(opParam, resMap_[newTag], tinySendRecvMem));
        }
    }
    // 算法执行
    bool selectA3AivAlg = algName.find("Aiv") != std::string::npos;
    if (selectA3AivAlg) {
        opParam.aicpuUnfoldMode = false;
        CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
        if (aivClearEnable_) {
            // 用于判断图模式是否清零
            CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
        }
    }
    // 头计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_));
    // 算法执行
    if (opParam.aicpuUnfoldMode && supportAicpuAlg) {
        isInplaceStatus_ = 0;
        inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
        // algOperator->SupportRetryWithInplaceCheck 依赖 algOperator->SetRetryEnable 才能正确返回是否支持inplace
        
        inplaceSupportRetry_ = algOperator->SupportRetryWithInplaceCheck(
            opType, opParam, algName, isInplaceStatus_, inPlaceSupportRetryStatus_);
        auto algType = algOperator->GetAlgType();
        HCCL_INFO("[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%s], inplaceSupportRetry_[%d], opType[%d], "
            "isInplaceStatus_[%d], inPlaceSupportRetryStatus_[%d]",
            AlgTypeToStr(algType).c_str(), inplaceSupportRetry_, opType, isInplaceStatus_, inPlaceSupportRetryStatus_);
        CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType));
    } else {
        // HOST展开aclgraph场景，capture从流
        if (!selectA3AivAlg) {
            CHK_RET(CaptureSlaveStreams(opParam.stream.ptr(), resMap_[newTag].slaveStreams));
        }
        CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
        // for profiling, blockDim upload 
        CHK_RET(algOperator->GetBlockDim(blockDim_));
    }
    // 尾计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_));
    if (selectA3AivAlg) {
        CHK_RET(algOperator->SetAivClearEnable(false));
        aivClearEnable_ = false;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HandleAclGraphFirstOpAivBuff(rtStream_t mainStream)
{
    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    u32 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
    if (isCapture) {
        CHK_PTR_NULL(rtModel);
        // 获取不到modelId会报错
        CHK_RET(GetModelId(rtModel, modelId));
        if (captureModelIds_.find(modelId) == captureModelIds_.end()) {
            // aclgraph场景，首算子清理AIV buff
            aivClearEnable_ = true;
            captureModelIds_.insert(modelId);
            HCCL_INFO("[HcclCommunicator][%s] modelId[%u] is inserted to captureModelIds_", __func__, modelId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CaptureSlaveStreams(rtStream_t mainStream, vector<Stream> &slaveStreams)
{
    if (deviceType_ != DevType::DEV_TYPE_910_93) {
        HCCL_INFO("[HcclCommunicator][%s]Only A3 device in host expan mode need to capture slave streams.", __func__);
        return HCCL_SUCCESS;
    }
    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    u32 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
    if (isCapture) {
        if (GetExternalInputHcclEnableFfts()) {
            // A3 host展开 aclgraph不支持FFTS+
            HCCL_ERROR("[HcclCommunicator][%s]A3 device unsupport acl graph in FFTS+.", __func__);
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_PTR_NULL(rtModel);
        CHK_RET(GetModelId(rtModel, modelId));
        for (auto slaveStream : slaveStreams) {
            CHK_RET(AddStreamToModel(slaveStream.ptr(), rtModel));
            HCCL_DEBUG("[HcclCommunicator][%s]Add stream[%d] to model[%u] success.", __func__, slaveStream.id(),
                modelId);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::updateList(u64 size, void *buffer) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpLocalScratchMemResParam(
    const AlgResourceResponse &algResource, const std::string &newTag, LocalResInfoV2 *localResHostPtr)
{
    if (algResource.scratchMem.size() > 0) {
        hostMemVec_.resize(hostMemVec_.size() + 1);
        CHK_RET(AllocAndClearHostMem(sizeof(HccltagLocalResV2), hostMemVec_.back()));
        HccltagLocalResV2 *tagLocalResHostPtr = static_cast<HccltagLocalResV2 *>(hostMemVec_.back().get()->ptr());

        deviceMemVec_.resize(deviceMemVec_.size() + 1);
        CHK_RET(AllocAndClearDeviceMem(sizeof(HccltagLocalResV2), deviceMemVec_.back()));
        HccltagLocalResV2 *tagLocalResDevicePtr = static_cast<HccltagLocalResV2 *>(deviceMemVec_.back().get()->ptr());

        // 初始化HcclRankRelationResV2中的tagRes链表
        ListCommonInit(&tagLocalResDevicePtr->nextTagRes, &tagLocalResHostPtr->nextTagRes);
        // 刷新host空间内容
        CHK_SAFETY_FUNC_RET(
            memcpy_s(tagLocalResHostPtr->tag, sizeof(tagLocalResHostPtr->tag), newTag.c_str(), newTag.length() + 1));
        tagLocalResHostPtr->ScratchmemSize = algResource.scratchMem.size();
        tagLocalResHostPtr->Scratchmem = reinterpret_cast<u64>(algResource.scratchMem.ptr());

        // 3、将节点插入链表头
        ListCommonAddHead(&tagLocalResDevicePtr->nextTagRes,
            &tagLocalResHostPtr->nextTagRes,
            &localResHostPtr->nextTagRes,
            &opResDeviceParaPtr_->localRes.nextTagRes);
        HCCL_DEBUG("[HcclCommunicator][BuildOpLocalScratchMemResParam] LocalResHostPtr head addr[%p], nextHost[%p], "
                   "preHost[%p]",
            &localResHostPtr->nextTagRes,
            localResHostPtr->nextTagRes.nextHost,
            localResHostPtr->nextTagRes.preHost);

        HCCL_DEBUG("[HcclCommunicator][BuildOpLocalScratchMemResParam] tag LocalResHostPtr head addr[%p], nextHost[%p],"
                   "preHost[%p]",
            &tagLocalResHostPtr->nextTagRes,
            tagLocalResHostPtr->nextTagRes.nextHost,
            tagLocalResHostPtr->nextTagRes.preHost);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetMC2EnvFlag()
{
    isNsRecovery_ = true;
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetMC2EnvFlag()
{
    return isNsRecovery_;
}

HcclResult HcclCommunicator::SetStopFlag(bool value)
{
    if (socketManager_ != nullptr) {
        CHK_RET(socketManager_->SetStopFlag(value));
    }

    if (transportManager_ != nullptr) {
        CHK_RET(transportManager_->SetStopFlag(value));
    }

    for (auto& entry : resMap_) {   // map
        for (auto& levelNSubCommTransport : entry.second.opTransportResponse) { // vector
            for (auto& singleSubCommTransport : levelNSubCommTransport) {   // vector
                for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {   // vector
                    if (singleSubCommTransport.transportRequests[i].isValid
                        && i < singleSubCommTransport.links.size()) {
                        auto transport = singleSubCommTransport.links[i];
                        if (transport != nullptr) {
                            CHK_RET(transport->SetStopFlag(value));
                        }
                    }
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetState(HcclCommState state)
{
    state_.store(state);
    return HCCL_SUCCESS;
}

HcclCommState HcclCommunicator::GetState()
{
    return state_.load();
}

u32 HcclCommunicator :: HcclGetCmdTimeout(){
    return HCCL_AICPU_HOST_BASE_TIME_MS;
}

HcclResult HcclCommunicator::Suspend(){
    isSuspending = true;
    if (GetMC2EnvFlag()) {
        HCCL_DEBUG("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecControl execCommand;
        execCommand.kfcCmd = KfcCommand::NsStopLaunch;
        execCommand.bgCmd = BackgroundCommand::kNone;
        execCommand.suspendingStatus = HcclComSuspendingFlag::isSuspending;
        HCCL_RUN_INFO("[NsRecovery][SetOpExecCmd]set the suspending flag [%d] and set KfcCommand [%d].",
                      execCommand.suspendingStatus, execCommand.kfcCmd);
        CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t *>(&execCommand)));
        KfcExecStatus opInfo;
        auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
        auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&opInfo)));
            if (opInfo.execStatus.kfcStatus == KfcStatus::kStoplaunch) {
                HCCL_RUN_INFO("[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                return HCCL_E_SUSPENDING;
            } else if (opInfo.execStatus.kfcStatus == KfcStatus::kError){
                return HCCL_E_INTERNAL;
            } else {
                if((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout){
                    HCCL_ERROR("[NsRecovery]Wait suspend reponse status timeout[%u ms] and get the opExecState is [%u] and opId[%u].", waitStopExecCmdTimeoutMs,
                               opInfo.execStatus.kfcStatus, opInfo.opId.index);

                    return HCCL_E_INTERNAL;
                }
                continue;
            }
        }
    } else {
        HCCL_DEBUG("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::StopExec(){
    isSuspending = true;
    if (GetMC2EnvFlag()) {
        HCCL_DEBUG("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&opInfo)));
        HCCL_DEBUG("[NsRecovery][GetOpExecInfo] opExeState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
        if (opInfo.execStatus.kfcStatus == KfcStatus::kStoplaunch) {
            KfcCommand opCmd = KfcCommand::NsStopExec;
            HCCL_RUN_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d]", opCmd);
            CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t *>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while (true) {
                CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&opInfo)));
                if (opInfo.execStatus.kfcStatus == KfcStatus::kStopExec) {
                    HCCL_RUN_INFO("[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kEnd){
                    HCCL_RUN_INFO("[NsRecovery]opExecState[%d], opId[%u]",  opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_SUCCESS;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kError){
                    return HCCL_E_INTERNAL;
                } else {
                    if((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout){
                        HCCL_ERROR("[NsRecovery]Wait stopExec reponse status timeout[%u ms] and get the opExecState is [%u] and opId[%u].", waitStopExecCmdTimeoutMs,
                                   opInfo.execStatus.kfcStatus, opInfo.opId.index);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        } else {
            return HCCL_SUCCESS;
        }
    } else {
        HCCL_DEBUG("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::Clean(){
    isSuspending = true;
    if (GetMC2EnvFlag()) {
        HCCL_DEBUG("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&opInfo)));
        HCCL_DEBUG("[NsRecovery][GetOpExecInfo] opExeState[%d], opId[%u]", opInfo.execStatus.kfcStatus,opInfo.opId.index);
        if (opInfo.execStatus.kfcStatus == KfcStatus::kStopExec || opInfo.execStatus.kfcStatus == KfcStatus::kEnd) {
            KfcCommand opCmd = KfcCommand::NsClear;
            HCCL_RUN_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d]", opCmd);
            CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t *>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while (true) {
                CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t *>(&opInfo)));
                if (opInfo.execStatus.kfcStatus == KfcStatus::kClear) {
                    HCCL_RUN_INFO("[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kEnd){
                    HCCL_RUN_INFO("[NsRecovery]opExecState[%d], opId[%u]",  opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_SUCCESS;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kError){
                    return HCCL_E_INTERNAL;
                } else {
                    if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                        HCCL_ERROR("[NsRecovery]Wait clean reponse status timeout[%u ms] and get the opExecState is [%u] and opId[%u].", waitStopExecCmdTimeoutMs,
                                   opInfo.execStatus.kfcStatus, opInfo.opId.index);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        } else {
            return HCCL_SUCCESS;
        }
    } else {
        HCCL_DEBUG("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::BuildOpLocalResParam(const AlgResourceResponse &algResource, const std::string &newTag)
{
    LocalResInfoV2 *localResHostPtr = &opResPara_.localRes;
    ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
    if (algResource.slaveDevStreams.size() > LOCAL_STREAM_MAX_NUM) {
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign stream for tag[%s]", newTag.c_str());
        return HCCL_E_PARA;
    }
    auto signalM2SNum = algResource.notifiesDevMain.size();
    auto signalS2MNum = algResource.notifiesDevAux.size();
    auto signalNum = signalM2SNum + signalS2MNum;
    if (signalNum > LOCAL_NOTIFY_MAX_NUM) {
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign local notify for tag[%s]", newTag.c_str());
        return HCCL_E_PARA;
    }

    localResHostPtr->streamNum = algResource.slaveDevStreams.size();
    for (u32 i = 0; i < algResource.slaveDevStreams.size(); i++) {
        localResHostPtr->streamInfo[i].streamIds = algResource.slaveDevStreams[i].id();
        localResHostPtr->streamInfo[i].sqIds = algResource.slaveDevStreams[i].sqId();
        localResHostPtr->streamInfo[i].cqIds = algResource.slaveDevStreams[i].cqId();
        localResHostPtr->streamInfo[i].logicCqids = algResource.slaveDevStreams[i].logicCqId();
    }

    localResHostPtr->signalNum = signalNum;

    for (u32 i = 0; i < signalM2SNum; i++) {
        algResource.notifiesDevMain[i]->GetNotifyData(localResHostPtr->localSignals[i << 1]);
        algResource.notifiesDevAux[i]->GetNotifyData(localResHostPtr->localSignals[(i << 1) + 1]);
    }
    HcclResult ret = HCCL_SUCCESS;
    ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[0], localResHostPtr->aicpuOpNotify[0]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 0 error,"
                   "errNo[0x%016llx]",
            HCCL_ERROR_CODE(ret)),
        ret);
    ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[1], localResHostPtr->aicpuOpNotify[1]);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 1 error,errNo[0x%016llx]", HCCL_ERROR_CODE(ret)),
        ret);
    if (opMainStream_.ptr() == nullptr) {
        opMainStream_ = Stream(StreamType::STREAM_TYPE_DEVICE);
    }
    localResHostPtr->mainStreamInfo.streamIds = opMainStream_.id();
    localResHostPtr->mainStreamInfo.sqIds = opMainStream_.sqId();
    localResHostPtr->mainStreamInfo.cqIds = opMainStream_.cqId();
    localResHostPtr->mainStreamInfo.logicCqids = opMainStream_.logicCqId();

    CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::CopyVectorToDeviceMem(const u64 len, DeviceMem &dstDeviceMem, const std::vector<T> &srcVec)
{
    CHK_PRT_RET(!len,
        HCCL_INFO("[HcclCommunicator][CopyVectorToDeviceMem] space size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((len > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][CopyVectorToDeviceMem] space size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    CHK_RET(CreateWorkSpace(len, dstDeviceMem));
    std::shared_ptr<HostMem> srcHostMem;
    CHK_RET(AllocAndClearHostMem(len, srcHostMem));
    std::copy(srcVec.begin(), srcVec.end(), static_cast<T *>(srcHostMem.get()->ptr()));
    CHK_RET(hrtMemSyncCopy(
        dstDeviceMem.ptr(), len, srcHostMem.get()->ptr(), len, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResTlvParam(const std::string &algName,
    const std::vector<std::vector<std::vector<u32>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
{
    vector<u32> tlv;
    CommonTlv commonTlv;
    HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] input vector size[%lu], group[%s]",
        inputVectorInfo.size(), identifier_.c_str());
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            commonTlv.type = ((level0Idx << TOP_COMM_LEVEL0_SHIFT) | level1Idx);
            commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE)) +
                                    inputVectorInfo[level0Idx][level1Idx].size() * sizeof(RANK_TYPE);
            tlv.push_back(commonTlv.type);
            tlv.push_back(commonTlv.length);
            tlv.insert(tlv.end(), inputVectorInfo[level0Idx][level1Idx].begin(),
                       inputVectorInfo[level0Idx][level1Idx].end());
        }
    }
    for (u64 idx = 0; idx < tlv.size(); idx++) {
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResVectorTlvParam(const std::string &algName,
    const std::vector<std::vector<std::vector<std::vector<u32>>>> &inputVectorInfo, DeviceMem &dstTlvDeviceMem, u64 &tlvLen)
{
    vector<u32> tlv;
    CommonTlv commonTlv;
    HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResVectorTlvParam] input vector size[%lu], group[%s]",
        inputVectorInfo.size(), identifier_.c_str());
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            for (u16 level2Idx = 0; level2Idx < inputVectorInfo[level0Idx][level1Idx].size(); level2Idx++) {
                commonTlv.type = (((level0Idx << TOP_HIERARCHICAL_COMM_LEVEL0_SHIFT) | level1Idx) << TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT) | level2Idx;
                commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE)) +
                                        inputVectorInfo[level0Idx][level1Idx][level2Idx].size() * sizeof(RANK_TYPE);
                tlv.push_back(commonTlv.type);
                tlv.push_back(commonTlv.length);
                tlv.insert(tlv.end(), inputVectorInfo[level0Idx][level1Idx][level2Idx].begin(),
                        inputVectorInfo[level0Idx][level1Idx][level2Idx].end());
            }
        }
    }
    for (u64 idx = 0; idx < tlv.size(); idx++) {
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResVectorTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildPairLinkCounter(const std::string &algName)
{
    constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
    if (pairLinkCounterDevice_.ptr() == nullptr) {
        u64 pairLinkCounterSize = pairLinkCounter_.size();
        HCCL_DEBUG("[HcclCommunicator][BuildPairLinkCounter] pairLinkCounter size[%lu], group[%s]",
            pairLinkCounterSize, identifier_.c_str());
        std::vector<u32> pairLinkCounterVec(pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS);
        u64 index = 0;
        for (auto& kt : pairLinkCounter_){
            pairLinkCounterVec[index] = kt.first;
            pairLinkCounterVec[index + 1] = kt.second;
            index += KEY_VALUE_TO_VECTOR_MODULUS;  // 每次根据
        }
        u64 len = pairLinkCounterSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS;  // key-value，都为u32
        CHK_RET(CopyVectorToDeviceMem(len, pairLinkCounterDevice_, pairLinkCounterVec));
        opResPara_.topoInfo.pairLinkCounter = reinterpret_cast<u64>(pairLinkCounterDevice_.ptr());
        opResPara_.topoInfo.pairLinkCounterNum = pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildIsUsedRdmaRank(const std::string &algName)
{
    constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
    if (isUsedRdmaRankPairDevice_.ptr() == nullptr) {
        std::unordered_map<u32, bool> isUsedRdmaMap;
        CHK_RET(implAlg_->GetIsUsedRdmaMap(isUsedRdmaMap));
        u64 isUsedRdmaMapSize = isUsedRdmaMap.size();
        HCCL_DEBUG("[HcclCommunicator][BuildIsUsedRdmaRank] is used Rdma rank size[%lu], group[%s]",
            isUsedRdmaMapSize, identifier_.c_str());
        std::vector<u32> isUsedRdmaPairVec(isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS);
        u64 index = 0;
        for (auto &kt : isUsedRdmaMap) {
            isUsedRdmaPairVec[index] = kt.first;
            isUsedRdmaPairVec[index + 1] = static_cast<u32>(kt.second);
            index += KEY_VALUE_TO_VECTOR_MODULUS;
        }
        u64 len = isUsedRdmaMapSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS;  // key-value，都为u32
        CHK_RET(CopyVectorToDeviceMem(len, isUsedRdmaRankPairDevice_, isUsedRdmaPairVec));
        opResPara_.topoInfo.isUsedRdmaRankPair = reinterpret_cast<u64>(isUsedRdmaRankPairDevice_.ptr());
        opResPara_.topoInfo.isUsedRdmaRankPairNum = isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildNicList(const std::string &algName)
{
    if (nicListDevice_.ptr() == nullptr) {
        u64 len = nicList_.size() * sizeof(u32);
        HCCL_DEBUG("[HcclCommunicator][BuildNicList] niclist size[%lu], group[%s]",
            nicList_.size(), identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, nicListDevice_, nicList_));
        opResPara_.topoInfo.nicList = reinterpret_cast<u64>(nicListDevice_.ptr());
        opResPara_.topoInfo.nicNum = nicList_.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildBridgeRank(const std::string &algName)
{
    if (bridgeRankDevice_.ptr() == nullptr) {
        std::vector<bool> isBridgeVector;
        CHK_RET(implAlg_->GetIsBridgeVector(isBridgeVector));
        u64 len = isBridgeVector.size() * sizeof(bool);
        HCCL_DEBUG("[HcclCommunicator][BuildBridgeRank] Bridge size[%lu], group[%s]",
            isBridgeVector.size(), identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, bridgeRankDevice_, isBridgeVector));
        opResPara_.topoInfo.bridgeRank = reinterpret_cast<u64>(bridgeRankDevice_.ptr());
        opResPara_.topoInfo.bridgeRankNum = isBridgeVector.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCommPlanRank(const std::string &algName)
{
    opResPara_.topoInfo.complanRank = 0;
    opResPara_.topoInfo.complanRankLength = 0;
    if (complanRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
        CHK_RET(implAlg_->GetCommPlaneRanks(commPlaneRanks));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResTlvParam(algName, commPlaneRanks, complanRankDevice_, tlvLen));
        opResPara_.topoInfo.complanRank = reinterpret_cast<u64>(complanRankDevice_.ptr());
        opResPara_.topoInfo.complanRankLength = tlvLen;
        HCCL_DEBUG("[HcclCommunicator][BuildCommPlanRank] comm plane ranks tlv length[%lu], ptr[%p], group[%s], "
                   "local user rankId[%u] ", tlvLen, complanRankDevice_.ptr(), identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildServerAndsuperPodRank(const std::string &algName)
{
    opResPara_.topoInfo.serverAndsuperPodRank = 0;
    opResPara_.topoInfo.serverAndsuperPodRankLength = 0;
    if (serverAndsuperPodToRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
        CHK_RET(implAlg_->GetRankVecInfo(serverAndsuperPodToRank));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResTlvParam(algName, serverAndsuperPodToRank, serverAndsuperPodToRankDevice_, tlvLen));
        opResPara_.topoInfo.serverAndsuperPodRank = reinterpret_cast<u64>(serverAndsuperPodToRankDevice_.ptr());
        opResPara_.topoInfo.serverAndsuperPodRankLength = tlvLen;
        HCCL_DEBUG("[HcclCommunicator][BuildServerAndsuperPodRank] server and super pod ranks tlv length[%lu], ptr[%p], "
                   "group[%s],  local user rankId[%u] ", tlvLen, serverAndsuperPodToRankDevice_.ptr(),
                   identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRetryParam(const AlgResourceResponse &algResource, const std::string &newTag)
{
    opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
    opResPara_.config.retryHoldTime = GetExternalInputRetryHoldTime();
    opResPara_.config.retryIntervalTime = GetExternalInputRetryIntervalTime();
    opResPara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    opResPara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();

    CHK_SMART_PTR_NULL(opRetryStreamPtr_);
    if (opRetryStreamPtr_->find(newTag) == opRetryStreamPtr_->end()) {
        std::vector<Stream> retryStreams(algResource.slaveDevStreams.begin(), algResource.slaveDevStreams.end());
        retryStreams.push_back(opMainStream_);
        opRetryStreamPtr_->insert(std::make_pair(newTag, retryStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCommPlaneSubGroupRank(const std::string &algName)
{
    opResPara_.hierarchicalAlgInfo.commplaneSubGroupRank = 0;
    opResPara_.hierarchicalAlgInfo.commplaneSubGroupRankLength = 0;
    if (commplaneSubGroupRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<std::vector<u32>>>> commplaneSubGroupVector;
        CHK_RET(implAlg_->GetCommPlaneSubGroupVector(commplaneSubGroupVector));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResVectorTlvParam(algName, commplaneSubGroupVector, commplaneSubGroupRankDevice_, tlvLen));
        opResPara_.hierarchicalAlgInfo.commplaneSubGroupRank = reinterpret_cast<u64>(commplaneSubGroupRankDevice_.ptr());
        opResPara_.hierarchicalAlgInfo.commplaneSubGroupRankLength = tlvLen;
        HCCL_DEBUG("[HcclCommunicator][BuildCommPlaneSubGroupRank] comm plane subGroups ranks tlv length[%lu], ptr[%p], "
                   "group[%s],  local user rankId[%u] ", tlvLen, commplaneSubGroupRankDevice_.ptr(),
                   identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildHierarchicalAlgOption(const std::string &algName)
{
    std::map<std::string, std::string> hierarchicalAlgOption;
    CHK_RET(implAlg_->GetAHCAlgOption(hierarchicalAlgOption));
    if (hierarchicalAlgOptionDevice_.ptr() == nullptr) {
        std::vector<u32> hierarchicalAlgOptionVec;
        std::vector<std::string> hierarchicalAlgOptionList = {"LEVEL0INTRA", "LEVEL0INTER", "LEVEL1INTRA", "LEVEL1INTER"};
        for (u32 i = 0; i < hierarchicalAlgOptionList.size(); i++) {
            hierarchicalAlgOptionVec.push_back(i);
            if (hierarchicalAlgOption.find(hierarchicalAlgOptionList[i]) == hierarchicalAlgOption.end()) {
                hierarchicalAlgOption.insert(std::make_pair(hierarchicalAlgOptionList[i], "Ring"));
            }
            hierarchicalAlgOptionVec.push_back(hierarchicalAlgOption[hierarchicalAlgOptionList[i]].size());
            for (u32 j = 0; j < hierarchicalAlgOption[hierarchicalAlgOptionList[i]].size(); j++) {
                hierarchicalAlgOptionVec.push_back(static_cast<u32>(hierarchicalAlgOptionList[i][j]));
            }
        }
        u64 len = hierarchicalAlgOptionVec.size() * sizeof(u32);
        HCCL_DEBUG("[HcclCommunicator][BuildHierarchicalAlgOption] hierarchicalAlgOptionVec size[%lu], group[%s]",
            hierarchicalAlgOptionVec.size(), identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, hierarchicalAlgOptionDevice_, hierarchicalAlgOptionVec));
        opResPara_.hierarchicalAlgInfo.hierarchicalAlgOptionVec = reinterpret_cast<u64>(hierarchicalAlgOptionDevice_.ptr());
        opResPara_.hierarchicalAlgInfo.hierarchicalAlgOptionNum = hierarchicalAlgOptionVec.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResParam(const std::string &algName, const AlgResourceResponse &algResource)
{
    opResPara_.topoInfo.userRank = userRank_;
    opResPara_.topoInfo.userRankSize = userRankSize_;
    opResPara_.topoInfo.deviceLogicId = deviceLogicId_;
    opResPara_.topoInfo.isSingleMeshAggregation = isSingleMeshAggregation_;
    opResPara_.topoInfo.deviceNumPerAggregation = deviceNumPerAggregation_;
    opResPara_.topoInfo.superPodNum = superPodNum_;
    opResPara_.topoInfo.devicePhyId = devicePhyId_;
    opResPara_.topoInfo.deviceType = static_cast<u32>(deviceType_);
    TopoType topoType;
    CHK_RET(implAlg_->GetTopoType(topoType));
    opResPara_.topoInfo.topoType = static_cast<u32>(topoType);
	opResPara_.topoInfo.serverNum = serverNum_;
    opResPara_.topoInfo.meshAggregationRankSize = meshAggregationRankSize_;
    opResPara_.topoInfo.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;
    opResPara_.topoInfo.multiSuperPodDiffServerNumMode = multiSuperPodDiffServerNumMode_;
    opResPara_.topoInfo.realUserRank = realUserRank_;
    opResPara_.topoInfo.isDiffDeviceModule = isDiffDeviceModule_;
    opResPara_.topoInfo.isDiffDeviceType = isDiffDeviceType_;
    opResPara_.topoInfo.gcdDeviceNumPerAggregation = gcdDeviceNumPerAggregation_;
    opResPara_.topoInfo.moduleNum = moduleNum_;
    CHK_RET(BuildPairLinkCounter(algName));
    CHK_RET(BuildIsUsedRdmaRank(algName));
    CHK_RET(BuildNicList(algName));
    CHK_RET(BuildBridgeRank(algName));
    CHK_RET(BuildCommPlanRank(algName));
    CHK_RET(BuildServerAndsuperPodRank(algName));
    CHK_RET(BuildCommPlaneSubGroupRank(algName));
    CHK_RET(BuildHierarchicalAlgOption(algName));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckNotifyOrQPMaxNum(u64 &existNum, const u64 &MaxNum, const bool &isNotifyRes)
{
    std::string resType = isNotifyRes ? "Notify" : "QP";
    if (existNum + 1 > MaxNum) {
        HCCL_ERROR("[%s]%s resources are insufficient, existNum[%llu], MaxNum is [%llu]",
            __func__, resType.c_str(), existNum, MaxNum);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[%s]%s resources are sufficient, existNum[%llu], MaxNum is [%llu]",
            __func__, resType.c_str(), existNum, MaxNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteLinkP2pResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes)
{
    HcclLinkP2pV2 *linkp2p = &(tagRemoteRes.tagRemoteResPtr->linkP2p);
    if (linkp2p->localIpcSignal[0].resId != INVALID_U64) {
        HCCL_INFO("[%s]the linkP2p is existed, no need to refresh transport resource, resId[%llu]",
            __func__, linkp2p->localIpcSignal[0].resId);
        return HCCL_SUCCESS;
    }
    // localMem & remoteMem
    void *inbufferPtr = nullptr;
    void *outbufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
    (linkp2p->remoteMem)[INPUT].addr = reinterpret_cast<u64>(inbufferPtr);
    (linkp2p->remoteMem)[OUTPUT].addr = reinterpret_cast<u64>(outbufferPtr);
    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, (linkp2p->remoteMem)[INPUT].size));
    CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, (linkp2p->remoteMem)[OUTPUT].size));
    MemDetails localMem; // 暂时预留，赋值为空
    (linkp2p->localMem)[0] = localMem;
    (linkp2p->localMem)[1] = localMem;
    HCCL_DEBUG("[%s] finish set localMem & remoteMem info", __func__);
    // localnotify & remotenotify
    u64 notifyNum = 0;
    std::vector<HcclSignalInfo> locIpcSignals;
    std::vector<HcclSignalInfo> rmtIpcSignals;
    CHK_RET(link->GetLocalNotify(locIpcSignals));
    CHK_RET(link->GetRemoteNotify(rmtIpcSignals));

    for (size_t i = 0; i < locIpcSignals.size(); i++) {
        CHK_RET(CheckNotifyOrQPMaxNum(notifyNum, LINK_P2P_MAX_NUM, true));
        linkp2p->localIpcSignal[notifyNum] = locIpcSignals[i];
        linkp2p->remoteIpcSignal[notifyNum] = rmtIpcSignals[i];
        notifyNum++;
    }
    tagRemoteRes.p2pNotifyNum = notifyNum;
    HCCL_DEBUG("[%s] finish set localnotify & remotenotify info, notifyNum[%llu]", __func__, notifyNum);
    // transportAttr
    CHK_RET(link->GetTransportAttr(linkp2p->transportAttr));
    HCCL_DEBUG("[%s] finish set RemoteLinkP2pResParam info", __func__);
    return HCCL_SUCCESS;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteLinkRoceResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes,
    bool isBackup, bool isRetry, bool IsSecondBuild)
{
    u32 iter = IsSecondBuild ? 2 : 0;
    HcclLinkRoceV2 *linkRoce = isBackup ? &(tagRemoteRes.tagRemoteResPtr->linkRoce[AICPU_RETRY_LINKROCE_BACKUP + iter])
        : &(tagRemoteRes.tagRemoteResPtr->linkRoce[AICPU_RETRY_LINKROCE_DEFAULT + iter]);
    if (!isRetry && linkRoce->localNotifyList != 0) {
        HCCL_INFO("[%s]the linkRoce is existed, no need to refresh transport resource, localNotifyListPtr[%p], iter[%u]",
            __func__, reinterpret_cast<void*>(linkRoce->localNotifyList), iter);
        return HCCL_SUCCESS;
    }
    // localMem & remoteMem
    CHK_RET(link->GetLocalMemDetails(UserMemType::INPUT_MEM, (linkRoce->localMem)[INPUT]));
    CHK_RET(link->GetLocalMemDetails(UserMemType::OUTPUT_MEM, (linkRoce->localMem)[OUTPUT]));
    void *inbufferPtr = nullptr;
    void *outbufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
    HCCL_DEBUG("[%s]inbufferPtr[%p], outbufferPtr[%p]", __func__, inbufferPtr, outbufferPtr);
    if (inbufferPtr == nullptr || outbufferPtr == nullptr) {
        HCCL_ERROR("[%s]inbufferPtr[%p], outbufferPtr[%p]", __func__, inbufferPtr, outbufferPtr);
        return HCCL_E_INTERNAL;
    }
    (linkRoce->remoteMem)[INPUT].addr = reinterpret_cast<u64>(inbufferPtr);
    (linkRoce->remoteMem)[OUTPUT].addr = reinterpret_cast<u64>(outbufferPtr);
    CHK_RET(link->GetRemoteMemKey(UserMemType::INPUT_MEM, &((linkRoce->remoteMem)[INPUT].key)));
    CHK_RET(link->GetRemoteMemKey(UserMemType::OUTPUT_MEM, &((linkRoce->remoteMem)[OUTPUT].key)));
    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, (linkRoce->remoteMem)[INPUT].size));
    CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, (linkRoce->remoteMem)[OUTPUT].size));
    HCCL_DEBUG("[%s] finish set localMem & remoteMem info", __func__);
    // notifyValue & Key
    std::vector<AddrKey> notifyValueAddrKey;
    CHK_RET(link->GetLocalNotifyValueAddrKey(notifyValueAddrKey));
    linkRoce->notifyValue = notifyValueAddrKey[0].addr;
    linkRoce->notifyValueKey = notifyValueAddrKey[0].key;
    // chipId
    CHK_RET(link->GetChipId(linkRoce->chipId));
    // QPInfo
    std::vector<HcclQpInfoV2> aiQpInfos;
    CHK_RET(link->GetAiQpInfo(aiQpInfos));
    u32 qpNum = aiQpInfos.size();
    if(qpNum > RDMA_QP_MAX_NUM || qpNum < 1)
    {
        return HCCL_E_INTERNAL;
    }
    std::copy_n(aiQpInfos.begin(), qpNum, linkRoce->QpInfo);
    linkRoce->qpsPerConnection = qpNum - static_cast<u32>(qpNum > 1);//多QP数量或单QP模式

    // localnotify & remotenotify
    std::vector<AddrKey> notifyAddrKey;
    std::vector<HcclSignalInfo> signalInfos;
    CHK_RET(link->GetLocalRdmaNotify(signalInfos));
    CHK_RET(link->GetRemoteRdmaNotifyAddrKey(notifyAddrKey));
    constexpr u32 RDMA_NOTIFY_MIN_NUM = 3;
    constexpr u32 RDMA_NOTIFY_MAX_NUM = 8192;
    if((signalInfos.size() != notifyAddrKey.size()) || (signalInfos.size() < RDMA_NOTIFY_MIN_NUM) || 
        (signalInfos.size() > RDMA_NOTIFY_MAX_NUM) || (notifyAddrKey.size() < RDMA_NOTIFY_MIN_NUM) || 
        (notifyAddrKey.size() > RDMA_NOTIFY_MAX_NUM ) ||
        ((signalInfos.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce->qpsPerConnection) ||
        ((notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce->qpsPerConnection)) {
        return HCCL_E_INTERNAL;
    }
    u64 notifyNum = (notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) / linkRoce->qpsPerConnection 
        - static_cast<u32>(linkRoce->qpsPerConnection > 1);
    linkRoce->singleQPNotifyNum = notifyNum;

    u64 len = signalInfos.size() * sizeof(HcclSignalInfo);
    DeviceMem localNotifyListMem;
    CHK_RET(CopyVectorToDeviceMem(len, localNotifyListMem, signalInfos));
    linkRoce->localNotifyList = reinterpret_cast<u64>(localNotifyListMem.ptr());
    ibverbsLocalNotify_.emplace_back(std::move(localNotifyListMem));

    std::vector<u64> notifyAddr;
    for(const auto& iter : notifyAddrKey)
    {
        notifyAddr.push_back(iter.addr);
    }
    len = notifyAddr.size() * sizeof(u64);
    DeviceMem remoteNotifyListMem;
    CHK_RET(CopyVectorToDeviceMem(len, remoteNotifyListMem, notifyAddr));
    linkRoce->remoteNotifyList = reinterpret_cast<u64>(remoteNotifyListMem.ptr());
    ibverbsRemoteNotify_.emplace_back(std::move(remoteNotifyListMem));
    linkRoce->remoteNotifyKey = notifyAddrKey[0].key; //只需要刷新一次，remote notify共用

    HCCL_DEBUG("[%s] finish set localnotify & remotenotify info, notifyNum[%llu], linkNotifyNum[%llu]",
        __func__, notifyNum, signalInfos.size());
   
    if (isBackup) {
        tagRemoteRes.roceNotifyNumBackup = linkRoce->singleQPNotifyNum;
        tagRemoteRes.qpNumBackup = linkRoce->qpsPerConnection;
    } else {
        tagRemoteRes.roceNotifyNum = linkRoce->singleQPNotifyNum;
        tagRemoteRes.qpNum = linkRoce->qpsPerConnection;
    }
    HCCL_DEBUG("[%s] finish set Qp info & chipId info[%lld], qpNum[%u], linkRoce->localNotifyList[0].resId[%llu], "
        "notifyNum[%u], isBackup[%d], isSecond[%d], qpPtr[%llu]", __func__, linkRoce->chipId, linkRoce->qpsPerConnection, 
        signalInfos[0].resId, linkRoce->singleQPNotifyNum, isBackup, IsSecondBuild, 
        linkRoce->QpInfo[0].qpPtr);
    return HCCL_SUCCESS;
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::CreateListNode(T **resHostPtr, T **resDevicePtr)
{
    hostMemVec_.resize(hostMemVec_.size() + 1);
    CHK_RET(AllocAndClearHostMem(sizeof(T), hostMemVec_.back()));
    *resHostPtr = static_cast<T *>(hostMemVec_.back().get()->ptr());

    deviceMemVec_.resize(deviceMemVec_.size() + 1);
    CHK_RET(AllocAndClearDeviceMem(sizeof(T), deviceMemVec_.back()));

    *resDevicePtr = static_cast<T *>(deviceMemVec_.back().get()->ptr());
    // 初始化HcclRankRelationResV2中的tagRes链表
    ListCommonInit(&((*resDevicePtr)->nextTagRes), &((*resHostPtr)->nextTagRes));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildRemoteResByTag(const std::string &newTag, const u32 &usrRankId,
    HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr, bool isBackup,
    bool isRetry)
{
    HCCL_DEBUG("[%s]start to add RemoteRes with newtag[%s] and remoteRankId[%u] to list",
        __func__, newTag.c_str(), usrRankId);
    if (rankTagRemoteRes_.find(usrRankId) == rankTagRemoteRes_.end() ||
        rankTagRemoteRes_[usrRankId].find(newTag) == rankTagRemoteRes_[usrRankId].end()) {
        HccltagRemoteResV2 *tagRemoteResHostPtr = nullptr;
        HccltagRemoteResV2 *tagRemoteResDevicePtr = nullptr;
        CHK_RET(CreateListNode(&tagRemoteResHostPtr, &tagRemoteResDevicePtr));
        CHK_SAFETY_FUNC_RET(memcpy_s(tagRemoteResHostPtr->tag, sizeof(tagRemoteResHostPtr->tag),
            newTag.c_str(), newTag.length() + 1));
        tagRemoteResHostPtr->linkP2p.localIpcSignal[0].resId = INVALID_U64;
        tagRemoteResHostPtr->linkRoce[0].localNotifyList = 0;
        tagRemoteResHostPtr->linkRoce[1].localNotifyList = 0;
        tagRemoteResHostPtr->linkRoce[2].localNotifyList = 0;
        tagRemoteResHostPtr->linkRoce[3].localNotifyList = 0;
        ListCommonAddHead(&tagRemoteResDevicePtr->nextTagRes, &tagRemoteResHostPtr->nextTagRes,
            &rankRelationResHostPtr->nextTagRes, &rankRelationResDevicePtr->nextTagRes);
        HccltagRemoteResV3 tempTagRemoteRes;
        tempTagRemoteRes.tagRemoteResPtr = tagRemoteResHostPtr;
        rankTagRemoteRes_[usrRankId][newTag] = tempTagRemoteRes;
        HCCL_DEBUG("[%s] successfully add RemoteRes to list with newtag[%s], remoteRankId[%u]"
            "rankRelationResHostPtr head addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
            "tagRemoteResDevicePtr head addr[%p]", __func__, newTag.c_str(), usrRankId,
            &rankRelationResHostPtr->nextTagRes, rankRelationResHostPtr->nextTagRes.nextHost,
            rankRelationResHostPtr->nextTagRes.preHost, rankRelationResHostPtr->nextTagRes.nextDevice,
            rankRelationResHostPtr->nextTagRes.preDevice, &tagRemoteResDevicePtr->nextTagRes);
    } else {
        HCCL_DEBUG("[%s] the RemoteRes with usr rankid[%u] tag[%s] has been added list",
            __func__, usrRankId, newTag.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildRelationResByRemoteRankId(const TransportRequest &transportRequest, const LINK &link,
    HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr)
{
    const u32 usrRankId = transportRequest.remoteUserRank;
    HCCL_INFO("[%s]start to add RelationRes with remote usr rankid[%u] to list", __func__, usrRankId);
    if (opResPara_.remoteRes[usrRankId].nextHostPtr != 0 && opResPara_.remoteRes[usrRankId].nextDevicePtr != 0) {
        rankRelationResHostPtr =
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[usrRankId].nextHostPtr);
        rankRelationResDevicePtr =
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[usrRankId].nextDevicePtr);
        HCCL_DEBUG("[%s] RelationRes with remote usr rankid[%u] has been added to list, "
            "rankRelationResHostPtr[%p], rankRelationResDevicePtr[%p]",
            __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr);
    } else {
        CHK_RET(CreateListNode(&rankRelationResHostPtr, &rankRelationResDevicePtr));
        opResPara_.remoteRes[usrRankId].nextHostPtr = reinterpret_cast<u64>(rankRelationResHostPtr);
        opResPara_.remoteRes[usrRankId].nextDevicePtr = reinterpret_cast<u64>(rankRelationResDevicePtr);
        rankRelationResHostPtr->remoteUsrRankId = usrRankId;
        rankRelationResHostPtr->remoteWorldRank = rankInfoList_[usrRankId].worldRank;
        HCCL_DEBUG("[%s]successfully add RelationRes with remote usr rankid[%u] to list, rankRelationResHostPtr[%p],"
            "rankRelationResDevicePtr[%p]", __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr);
    }
    // 刷新远端对应的cclbuffer
    std::vector<void*> extraMemVector;
    if (transportRequest.inputMemType == TransportMemType::CCL_INPUT && rankRelationResHostPtr->windowsIn == 0) {
        void *inbufferPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
        rankRelationResHostPtr->windowsIn = reinterpret_cast<u64>(inbufferPtr);
    }
    if (transportRequest.outputMemType == TransportMemType::CCL_OUTPUT && rankRelationResHostPtr->windowsOut == 0) {
        void *outbufferPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
        rankRelationResHostPtr->windowsOut = reinterpret_cast<u64>(outbufferPtr);
    }
    if (rankRelationResHostPtr->windowsExp == 0) {
        std::vector<void *> memPtrVec = {};
        CHK_RET(link->GetRemoteMem(&memPtrVec));
        if (memPtrVec.size() != 0) {
            rankRelationResHostPtr->windowsExp = reinterpret_cast<u64>(memPtrVec[0]);
        }
    }
    HCCL_INFO("group[%s] successfully set windowsIn & windowsOut & windowsExp info: userRank[%u], groupRank[%u], "\
              "remoteRank[%u], windowsIn[0x%llx], InSize[0x%llx], windowOut[0x%llx], OutSize[0x%llx], "\
              "windowExp[0x%llx], ExpSize[0x%llx]",
              identifier_.c_str(), GetUserRank(), GetGroupRank(), transportRequest.remoteUserRank,
              rankRelationResHostPtr->windowsIn, cclBufferManager_.GetInCCLbufferSize(),
              rankRelationResHostPtr->windowsOut, cclBufferManager_.GetOutCCLbufferSize(),
              rankRelationResHostPtr->windowsExp, cclBufferManager_.GetExpBufferSize());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseRemoteDataToMem(const OpCommTransport &opTransportResponse, const std::string &newTag,
    const HcclCMDType opType, bool isBackup, bool isRetry)
{
    HCCL_INFO("[%s] entry process newtag[%s], isBackup[%d]", __func__, newTag.c_str(), isBackup);
    std::set<u32> bsrTansportRank;
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            u32 linkIdx = 0;
            for (auto &transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    auto tempLink = singleSubCommTransport.links[linkIdx];
                    HCCL_INFO("[%s]transportRequest.isUsedRdma[%d], isBackup[%d]", __func__,
                        transportRequest.isUsedRdma, isBackup);
                    if ((!transportRequest.isUsedRdma || tempLink->GetLinkType() == LinkType::LINK_SIO) &&
                        (isBackup || isRetry)) {
                        HCCL_INFO("[%s]no need to add p2p backup Link resource, transportRequest.isUsedRdma[%d], "
                            "isBackup[%d]", __func__,transportRequest.isUsedRdma, isBackup);
                        linkIdx++;
                        continue;
                    }
                    HcclRankRelationResV2 *rankRelationResHostPtr = nullptr;
                    HcclRankRelationResV2 *rankRelationResDevicePtr = nullptr;
                    CHK_RET(BuildRelationResByRemoteRankId(transportRequest, tempLink, rankRelationResHostPtr,
                        rankRelationResDevicePtr));
                    const u32 usrRankId = transportRequest.remoteUserRank;
                    HCCL_INFO("[%s]successfully BuildRelationResByRemoteRankId with remote usr rankid[%u], "
                        "rankRelationResHostPtr[%p], rankRelationResDevicePtr[%p], newTage[%s]",
                        __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr, newTag.c_str());
                    CHK_RET(BuildRemoteResByTag(newTag, usrRankId, rankRelationResHostPtr,
                        rankRelationResDevicePtr, isBackup, isRetry));
                    // transport信息保存（notify、qp）
                    if (!transportRequest.isUsedRdma || tempLink->GetLinkType() == LinkType::LINK_SIO) {
                        // sdma -> P2P
                        CHK_RET(BuildOpRemoteLinkP2pResParam(tempLink, rankTagRemoteRes_[usrRankId][newTag]));
                    } else {
                        // rdma -> roce
                        bool isSecondBuild = false;
                        if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && 
                            bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()){
                            isSecondBuild = true;
                        }
                        bsrTansportRank.insert(transportRequest.remoteUserRank);
                        CHK_RET(BuildOpRemoteLinkRoceResParam(tempLink, rankTagRemoteRes_[usrRankId][newTag],
                            isBackup, isRetry, isSecondBuild));
                    }
                    HCCL_INFO("[%s] successfully add RemoteRes to list with newtag[%s] rankRelationResHostPtr "
                        "head addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
                        "rankRelationResDevicePtr head addr[%p]", __func__, newTag.c_str(),
                        &rankRelationResHostPtr->nextTagRes, rankRelationResHostPtr->nextTagRes.nextHost,
                        rankRelationResHostPtr->nextTagRes.preHost, rankRelationResHostPtr->nextTagRes.nextDevice,
                        rankRelationResHostPtr->nextTagRes.preDevice, &rankRelationResDevicePtr->nextTagRes);
                    HCCL_INFO("[%s] create link success with newtag[%s], linkIdx[%u], isBackup[%d], usrRankId[%u]",
                        __func__, newTag.c_str(), linkIdx, isBackup, usrRankId);
                }
                linkIdx++;
            }
        }
    }
    HCCL_DEBUG("[%s] process success newtag[%s]", __func__, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteResParam(const AlgResourceResponse &algResource, const std::string &newTag,
    const HcclCMDType opType, bool isRetry)
{
    HCCL_DEBUG("[%s]start ParseRemoteDataToMem, IsEnableBackupLink[%d]", __func__, IsEnableBackupLink());
    CHK_RET(ParseRemoteDataToMem(algResource.opTransportResponse, newTag, opType, false, isRetry));
    if (IsEnableBackupLink()) {
        HCCL_DEBUG("[%s]start Parse backupRemoteDataToMem, IsEnableBackupLink[%d]", __func__, IsEnableBackupLink());
        CHK_RET(ParseRemoteDataToMem(algResource.opTransportResponseBackUp, newTag, opType, true, isRetry));
    }
    if (deviceType_ == DevType::DEV_TYPE_910_93 || deviceType_ == DevType::DEV_TYPE_910B) {
        opResPara_.notifysize = 4; // 910B & 910_93 每个notify占4个字节
    } else {
        opResPara_.notifysize = 8; // 其他芯片类型每个notify占8个字节
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostListResToDeviceParam(const std::string &newTag, const ListCommon *headHostList, const u64 size)
{
    ListCommon *nextHostList = reinterpret_cast<ListCommon *>(headHostList->nextHost);
    ListCommon *nextDeviceList = reinterpret_cast<ListCommon *>(headHostList->nextDevice);

    while (nextHostList != headHostList) {
        HCCL_INFO(
            "[HcclCommunicator][CopyHostListResToDeviceParam] remote resource, tag[%s], head Host List[%p], next "
            "Host List[%p],next Device List[%p]", newTag.c_str(), headHostList, nextHostList, nextDeviceList);
        CHK_RET(hrtMemSyncCopy(reinterpret_cast<void *>(nextDeviceList), size, reinterpret_cast<void *>(nextHostList),
            size, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        nextDeviceList = reinterpret_cast<ListCommon *>(nextHostList->nextDevice);
        nextHostList = reinterpret_cast<ListCommon *>(nextHostList->nextHost);
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::CopyHostOpRemoteResToDeviceParam(const std::string &newTag)
{
    HCCL_DEBUG("[%s] remote resource, tag[%s]", __func__, newTag.c_str());
    for (u32 userRankIdx = 0; userRankIdx < AICPU_MAX_RANK_NUM; userRankIdx++) {
        if (opResPara_.remoteRes[userRankIdx].nextHostPtr == 0 &&
            opResPara_.remoteRes[userRankIdx].nextDevicePtr == 0) {
            continue;
        }
        // 1、将rank公共资源，H2D到device
        HcclRankRelationResV2 *remoteResHostPtr =
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[userRankIdx].nextHostPtr);
        HcclRankRelationResV2 *remoteResDevicePtr =
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[userRankIdx].nextDevicePtr);
        CHK_RET(hrtMemSyncCopy(static_cast<void *>(remoteResDevicePtr), sizeof(HcclRankRelationResV2),
            static_cast<void *>(remoteResHostPtr), sizeof(HcclRankRelationResV2),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        HCCL_DEBUG("[%s] remote resource, tag[%s], userRankIx[%u], "
            "cclinbuffer[%p], ccloutbuffer[%p], opResPara_.remoteRes[userRankIdx].nextDevicePtr[%p], "
            "opResPara_.remoteRes[userRankIdx].nextHostPtr[%p]", __func__,
            newTag.c_str(), userRankIdx, remoteResHostPtr->windowsIn, remoteResHostPtr->windowsOut,
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[userRankIdx].nextDevicePtr),
            reinterpret_cast<HcclRankRelationResV2 *>(opResPara_.remoteRes[userRankIdx].nextHostPtr));
        CHK_RET(CopyHostListResToDeviceParam(
            newTag, reinterpret_cast<ListCommon *>(&remoteResHostPtr->nextTagRes), sizeof(HccltagRemoteResV2)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostOpResToDeviceParam(const std::string &newTag)
{
    // 1、将opResPara_，H2D到device
    CHK_RET(hrtMemSyncCopy(opResDevicePara_.ptr(), sizeof(HcclOpResParam), reinterpret_cast<void *>(&opResPara_),
        sizeof(HcclOpResParam), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] tag[%s] local rankId[%u] workspace[%p] "
               "workspacesize[%lu] ranksize[%u], cclbuffersize[%lu], cclinbuffer[%p], ccloutbuffer[%p], "
               "remote winStart[%u], remote rWinOffset[%u], hostStateInfo[%p], aicpuStateInfo[%p], notifysize[%u]",
        newTag.c_str(), userRank_, opResPara_.mc2WorkSpace.workSpace, opResPara_.mc2WorkSpace.workSpaceSize,
        opResPara_.rankSize, opResPara_.winSize, opResPara_.localWindowsIn, opResPara_.localWindowsOut,
        opResPara_.rWinStart, opResPara_.rWinOffset, opResPara_.hostStateInfo, opResPara_.aicpuStateInfo,
        opResPara_.notifysize);
    // 2、将opResPara_中localres的tagRes，H2D到device
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] local resource, tag[%s] streamNum[%u] signalNum[%u]",
        newTag.c_str(), opResPara_.localRes.streamNum, opResPara_.localRes.signalNum);
    CHK_RET(CopyHostListResToDeviceParam(
        newTag, reinterpret_cast<ListCommon *>(&opResPara_.localRes.nextTagRes), sizeof(HccltagLocalResV2)));
    // 3、遍历rank中tag资源，H2D到device
    CHK_RET(CopyHostOpRemoteResToDeviceParam(newTag));
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] copy host resource success!, tag[%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpResParam(
    const std::string &algName, const AlgResourceResponse &algResource, const std::string &newTag,
    const HcclCMDType opType)
{
    CHK_RET(InitWorkSpace());
    HcclResult ret = GetWorkSpace(&(opResPara_.mc2WorkSpace.workSpaceSize), &(opResPara_.mc2WorkSpace.workSpace));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][BuildOpResParam]errNo[0x%016llx] size[%llu] space[%llu]", HCCL_ERROR_CODE(ret),
            opResPara_.mc2WorkSpace.workSpaceSize, opResPara_.mc2WorkSpace.workSpace), ret);

    opResPara_.localUsrRankId = userRank_;
    opResPara_.rankSize = userRankSize_;

    opResPara_.winSize = algResource.cclInputMem.size();
    opResPara_.localWindowsIn = reinterpret_cast<u64>(algResource.cclInputMem.ptr());
    opResPara_.localWindowsOut = reinterpret_cast<u64>(algResource.cclOutputMem.ptr());
    //填充Exp相关信息 当前该块内存大小恒为1M
    opResPara_.winExpSize = EXP_BUFFER_SIZE;
    opResPara_.localWindowsExp = reinterpret_cast<u64>(cclBufferManager_.GetCommExpBuffer().ptr());

    CHK_SAFETY_FUNC_RET(
        memcpy_s(opResPara_.hcomId, sizeof(opResPara_.hcomId), identifier_.c_str(), identifier_.length() + 1));

    opResPara_.config.deterministic = GetDeterministicConfig();
    opResPara_.config.highPerfEnable = GetExternalInputHcclHighPerfEnable();
    rtFloatOverflowMode_t floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
    CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
    opResPara_.config.floatOverflowMode = floatOverflowMode;
    opResPara_.config.notifyWaitTime =
        (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET)
            ? GetExternalInputHcclExecTimeOut()
            : NOTIFY_DEFAULT_WAIT_TIME;
    opResPara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
    opResPara_.config.interHccsDisable = GetExternalInputInterHccsDisable();
    opResPara_.config.multiQpThreshold = GetExternalInputMultiQpThreshold();
    opResPara_.rWinStart = offsetof(HcclOpResParam, remoteRes);
    opResPara_.rWinOffset = sizeof(RemoteResPtr);
    opResPara_.notifysize = 0;
    opResPara_.lockAddr = hostDeviceLock_->GetDevMemAddr();
    opResPara_.utraceStatusFlag = GetExternalInputHcclEnableEntryLog();
    DeviceMem tinySendRecvMem;
    CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
    opResPara_.tinyMem = reinterpret_cast<u64>(tinySendRecvMem.ptr());
    opResPara_.tinyMemSize = reinterpret_cast<u64>(tinySendRecvMem.size());

    CHK_RET(BuildOpLocalResParam(algResource, newTag));
    CHK_RET(BuildOpRemoteResParam(algResource, newTag, opType));
    CHK_RET(BuildOpTopoResParam(algName, algResource));
    CHK_RET(BuildOpRetryParam(algResource, newTag));
    CHK_RET(BuildZeroCopyParam());
    CHK_RET(CopyHostOpResToDeviceParam(newTag));
    HCCL_RUN_INFO("[%s]build aicpu unfold resource success!, tag[%s] rWinStart[%u] rWinOffset[%u]",
        __func__, newTag.c_str(), opResPara_.rWinStart, opResPara_.rWinOffset);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceInit(const std::string &algName,
    const AlgResourceResponse &algResource, const std::string &newTag, const rtStream_t &aicpuStream,
    const HcclCMDType opType)
{
    HCCL_RUN_INFO("[%s] start to init group[%s] aicpu resources newTag[%s] local rankId[%u]",
        __func__, identifier_.c_str(), newTag.c_str(), userRank_);
    isContextLaunched_ = true;
    CHK_RET(BuildOpResParam(algName, algResource, newTag, opType)); //构建context结构体
    std::string kernelName = "RunAicpuKfcResInitV2";
    //在这里构建suspending状态码的HDC通道初始化，并且在host侧进行init
    //（这个主要是针对hcomId；对算子通信域的复用；也就是多个算子复用（tag+Identifier）这个通信域的情况）
    CHK_RET(Mc2AiCpuKernelLaunch(aicpuStream, reinterpret_cast<u64>(opResDevicePara_.ptr()), kernelName));
    SetMC2EnvFlag();//并且只有资源初始化调用成功后
    newTagResAlloced_.insert(newTag);
    // 图模多档位场景，需要保证执行序上优先下资源初始化的kernel
    CHK_RET(hcclStreamSynchronize(aicpuStream));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceRefresh(const AlgResourceResponse &algResource, const std::string &newTag,
    const HcclCMDType opType)
{
    HCCL_INFO("[HcclCommunicator][AicpuResourceRefresh] start refresh aicpu resources newTag[%s] local rankId[%u]",
        newTag.c_str(), userRank_);
    LocalResInfoV2 *localResHostPtr = &opResPara_.localRes;
    opResPara_.winSize = algResource.cclInputMem.size();
    opResPara_.localWindowsIn = reinterpret_cast<u64>(algResource.cclInputMem.ptr());
    opResPara_.localWindowsOut = reinterpret_cast<u64>(algResource.cclOutputMem.ptr());
    CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
    CHK_RET(BuildOpRemoteResParam(algResource, newTag, opType));
    CHK_RET(BuildZeroCopyParam());
    CHK_RET(CopyHostOpResToDeviceParam(newTag));
    newTagResAlloced_.insert(newTag);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ProfilerAdd(const OpParam &param, AlgType algType)
{
    HCCL_PROFILER_ADD_TAG(param.tag, identifier_, GetWorkflowMode());
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType);
    u64 count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        count = param.All2AllDataDes.sendCount;
        dataType = param.All2AllDataDes.sendType;
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        dataType = param.All2AllDataDes.sendType;
    } else {
        count = param.DataDes.count;
        dataType = param.DataDes.dataType;
    }
    HCCL_PROFILER_ADD_OPDATA_OP(param.tag, count, param.inputPtr, param.outputPtr, dataType, param.root, identifier_,
        param.reduceType);
    HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ProfilerDel(const OpParam &param)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        !Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
        HCCL_PROFILER_DEL_TAG(param.tag);
        HCCL_PROFILER_DEL_OPDATA(param.tag);
        HCCL_PROFILER_DEL_GROUPRANK(identifier_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetReportHcclMC2Info(const Stream &kfcStream, const std::vector<Stream> &aicpuStreams)
{
    hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
    hcclMc2Info_.rankSize = userRankSize_;
    hcclMc2Info_.rankId = userRank_;
    hcclMc2Info_.usrRankId = realUserRank_;
    hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(kfcStream.id());
    hcclMc2Info_.reserve = 0;
    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    for (uint32_t streamIndex = 0, reportId = 0; streamIndex < aicpuStreams.size(); streamIndex++) {
        HCCL_INFO("streamIndex:%u, reportId:%u, streamId:%d", streamIndex, reportId, aicpuStreams[streamIndex].id());
        hcclMc2Info_.commStreamIds[reportId++] = aicpuStreams[streamIndex].id();
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info_.commStreamSize = reportId;
            CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
                sizeof(hcclMc2Info_)));
            reportId = 0;
        }
        if (streamIndex == (aicpuStreams.size() - 1)) {
            HCCL_INFO("streamIndex:%u, reportId:%u, streamId:%d", streamIndex, reportId, opMainStream_.id());
            hcclMc2Info_.commStreamIds[reportId++] = opMainStream_.id();
            hcclMc2Info_.commStreamSize = reportId;
            CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
                sizeof(hcclMc2Info_)));
            reportId = 0;
        }
    }
    if (aicpuStreams.empty()) {
        HCCL_INFO("only exist main stream, streamId:%d", opMainStream_.id());
        hcclMc2Info_.commStreamIds[0] = opMainStream_.id();
        hcclMc2Info_.commStreamSize = 1; // 只有主流1条
        CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
            sizeof(hcclMc2Info_)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OrchestrateAicpu(const HcclCMDType &opType, const std::string &algName,
    const OpParam &param, const AlgResourceResponse &algResource, const std::string &newTag, AlgType algType)
{
     uint64_t streamMode = 0;
    CHK_RET(hrtStreamGetMode(param.stream.ptr(), &streamMode));
    rtStream_t aicpuStream;
    Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream); // aicpuStream需要在首次下发时申请
    CHK_RET(ProfilerAdd(param, algType));
    if (!isContextLaunched_) {
        // 1、通信域内首次下发，从algResource中获取资源，H2D刷新资源，launch init
        rtStream_t aicpuInitStream;
        Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuInitStream); // 使用aicpuInitStream_下初始化kernel
        Stream tmpStream(aicpuInitStream);
        HCCL_DEBUG("%s ContextLaunched, aicpuInitStream:%p, aicpuStream:%p", __func__, aicpuInitStream, aicpuStream);
        CHK_RET(AicpuResourceInit(algName, algResource, newTag, aicpuInitStream, opType));
        CHK_RET(GetReportHcclMC2Info(tmpStream, algResource.slaveDevStreams));
    } else if (newTagResAlloced_.find(newTag) == newTagResAlloced_.end() ||
        opType  == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        // 2、通信域内非首次，但是有新的newTag，查看是否需要补充资源。
        PetersonLockGuard guard(hostDeviceLock_.get());
        CHK_PRT_RET(guard.IsLockFailed(),
            HCCL_ERROR("[HcclCommunicator][OrchestrateAicp] hostDeviceLock lock failed"), HCCL_E_INTERNAL);
        CHK_RET(AicpuResourceRefresh(algResource, newTag, opType));
    }
    bool isUsedMainStream = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || param.isZeroCopy) ? false : true;
    // inplace支持重执行的stream资源处理逻辑
    bool isHcclOpInplace = IsHcclOpInplace(opType, param, userRank_, userRankSize_, isInplaceStatus_);
    if ((retryOrigWorkflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) &&
        retryEnable_ && isHcclOpInplace &&
        (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER)) {
        isUsedMainStream = true;
    }
    AicpuOpTiling opTilingInfo;
    opTilingInfo.algName = algName;
    opTilingInfo.newTag = newTag;
    opTilingInfo.algType = algType;
    opTilingInfo.isUsedMainStream = isUsedMainStream;
    opTilingInfo.dumpDebug = GetExternalInputHcclDumpDebug();
    rtFloatOverflowMode_t floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
    CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
    opTilingInfo.floatOverflowMode = floatOverflowMode;
    HcclResult ret = HCCL_SUCCESS;
    std::string kernelName = "RunAicpuRpcSrvLaunchV2";
    ret = AicpuKfcTilingDataLaunchExt(param, opType, opResDevicePara_, kernelName, opTilingInfo);
    CHK_RET(ProfilerDel(param));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommunicator][OrchestrateAicpu]aicpu unfold launch kernel failed. return[%d] inputPtr[%p]"
                   "outputPtr[%p] count[%llu] dataType[%s] op[%s]", ret, param.inputPtr, param.outputPtr,
                    param.DataDes.count, GetDataTypeEnumStr(param.DataDes.dataType).c_str(),
                    GetReduceOpEnumStr(param.reduceType).c_str());
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CalcTinySendRecvMem(const OpParam &opParam, AlgResourceResponse &algResResponse,
    DeviceMem &tinySendRecvMem)
{
    u64 sendCount = 0;
    u64 recvCount = 0;
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        for (u32 i = 0; i < userRankSize_; i++) {
            u64 curSendCount = *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i) +
                *(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i);
            sendCount = std::max(sendCount, curSendCount);
            u64 curRecvCount = *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i) +
                *(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i);
            recvCount = std::max(recvCount, curRecvCount);
        }
    } else {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendCount += *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) +
                            userRank_ * userRankSize_ + i);
            recvCount += *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) +
                            userRank_ + userRankSize_ * i);
        }
    }

    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.recvType, recvTypeSize));

    // 在sendCount/recvCount全0时, 使用tinySendRecvMem, 避免使用空deviceMem
    algResResponse.paramInputMem = sendCount == 0 ?
        DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) :
        DeviceMem::create(opParam.inputPtr, sendCount * sendTypeSize);
    algResResponse.paramOutputMem = recvCount == 0 ?
        DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) :
        DeviceMem::create(opParam.outputPtr, recvCount * recvTypeSize);

    HCCL_INFO("[HcclCommunicator][CalcTinySendRecvMem] senMem addr[%p], sendSize[%llu], "
        "RecvMem addr[%p], RecvSize[%llu],", algResResponse.paramInputMem.ptr(),
        algResResponse.paramInputMem.size(), algResResponse.paramOutputMem.ptr(),
        algResResponse.paramOutputMem.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgNotifys(const std::string &tag, const NotifyLoadType notifyLoadType, const u32 notifyNum,
    std::vector<std::shared_ptr<LocalNotify> > &notifiesMain, std::vector<std::shared_ptr<LocalNotify> > &notifiesAux)
{
    std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
    queueNotifyManagerRefac_->Alloc(tag, notifyNum, notifys, notifyLoadType);

    u32 signalNum = notifyNum >> 1;
    notifiesMain.resize(signalNum);
    notifiesAux.resize(signalNum);
    for (u32 i = 0; i < signalNum; i++) {
        notifiesMain[i] = notifys[i << 1];
        notifiesAux[i] = notifys[(i << 1) + 1];
    }
    return HCCL_SUCCESS;
}

// 判断AICPU展开是否需要都走OpBase模式
bool HcclCommunicator::IsForceAicpuOpBaseMode(const OpParam &opParam, const HcclCMDType &opType)
{
    // 目前alltoall系列算子在aicpu展开场景下仍走原有的OpBase模式
    // ZeroCopy特性也强制走OpBase流程
    if (opParam.aicpuUnfoldMode &&
        (opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
         opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
         opType == HcclCMDType::HCCL_CMD_ALLTOALLVC ||
         opParam.isZeroCopy
         )) {
        return true;
    }

    return false;
}

HcclResult HcclCommunicator::AllocOpBaseModeScratchMem(HcclCMDType opType, const OpParam &opParam,
    AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
{
    if (resRequest.scratchMemSize == 0) {
        return HCCL_SUCCESS;
    }

    if (opParam.isZeroCopy) {
        if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            // 零拷贝场景不需要进行scratchMem申请
            DeviceMem tmpBuffer = DeviceMem::create(opParam.inputPtr, resRequest.scratchMemSize + CCE_REDUCE_ALIGN_SIZE);
            // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
            u32 addOffset = (reinterpret_cast<uintptr_t>(tmpBuffer.ptr())) % CCE_REDUCE_ALIGN_SIZE;
            u64 totalSize = userRankSize_ * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
            algResResponse.scratchMem = addOffset == 0 ? tmpBuffer.range(addOffset, totalSize) :
                tmpBuffer.range(CCE_REDUCE_ALIGN_SIZE - addOffset, totalSize);
            deviceResOrigMem_.emplace_back(std::move(tmpBuffer));
        } else {
            algResResponse.scratchMem = DeviceMem::create(opParam.inputPtr, resRequest.scratchMemSize);
        }
    } else {
        if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            DeviceMem tmpBuffer = DeviceMem::alloc(resRequest.scratchMemSize + CCE_REDUCE_ALIGN_SIZE);
            CHK_PTR_NULL(tmpBuffer.ptr());
            // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
            u32 addOffset = (reinterpret_cast<uintptr_t>(tmpBuffer.ptr())) % CCE_REDUCE_ALIGN_SIZE;
            algResResponse.scratchMem = addOffset == 0 ? tmpBuffer.range(addOffset, cclBufferManager_.GetInCCLbufferSize()) :
                tmpBuffer.range(CCE_REDUCE_ALIGN_SIZE - addOffset, cclBufferManager_.GetInCCLbufferSize());
            deviceResOrigMem_.emplace_back(std::move(tmpBuffer));
        } else {
            algResResponse.scratchMem = DeviceMem::alloc(resRequest.scratchMemSize);
            CHK_PTR_NULL(algResResponse.scratchMem.ptr());
        } 
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgResource(const std::string &newTag, HcclCMDType opType, const OpParam &opParam,
    AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
{
    HcclResult ret = HCCL_SUCCESS;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        !IsForceAicpuOpBaseMode(opParam, opType)) {
        if (resRequest.scratchMemSize > 0) {
            algResResponse.scratchMem = GetWorkspaceScracthMem(opParam.tag, resRequest.scratchMemSize);
            if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
                // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
                u32 addOffset = (reinterpret_cast<uintptr_t>(algResResponse.scratchMem.ptr())) % CCE_REDUCE_ALIGN_SIZE;
                u64 totalSize = userRankSize_ * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
                algResResponse.scratchMem = algResResponse.scratchMem.range(addOffset, totalSize);
            }
        }
        if (resRequest.streamNum > 0) {
            algResResponse.slaveStreams = GetWorkspaceSubStreams(opParam.tag, resRequest.streamNum);
        }
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE ||
        IsForceAicpuOpBaseMode(opParam, opType)) {
        CHK_RET(AllocOpBaseModeScratchMem(opType, opParam, resRequest, algResResponse));
        if (resRequest.streamNum > 0) {
            CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
             algResResponse.slaveStreams =
                opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_ONLINE, resRequest.streamNum);
        }
    } else {
        HCCL_ERROR("[AllocAlgResource]WorkflowMode is not set.");
        return HCCL_E_PARA;
    }

    if (opParam.aicpuUnfoldMode && ((userRankSize_ != 1) || IsForceAicpuOpBaseMode(opParam, opType))) {
        CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
        algResResponse.slaveDevStreams =
            opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_DEVICE, LOCAL_STREAM_MAX_NUM);
        CHK_RET(AllocAlgNotifys(opParam.tag, NotifyLoadType::DEVICE_NOTIFY, LOCAL_NOTIFY_MAX_NUM,
            algResResponse.notifiesDevMain, algResResponse.notifiesDevAux));
    }
    CHK_RET(AllocAlgNotifys(opParam.tag, NotifyLoadType::HOST_NOTIFY, resRequest.notifyNum, algResResponse.notifiesMain,
        algResResponse.notifiesAux));

    algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
    algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
    DeviceMem expMem = cclBufferManager_.GetCommExpBuffer(); //获取拓展内存
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        DeviceMem tinySendRecvMem;
        CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
        CHK_RET(CalcTinySendRecvMem(opParam, algResResponse, tinySendRecvMem));
    } else {
        algResResponse.paramInputMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
        algResResponse.paramOutputMem = DeviceMem::create(opParam.outputPtr, opParam.outputSize);
    }

    if (resRequest.needAivBuffer) {
        ret = cclBufferManager_.CreateCommAIVbuffer();
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Alloc][AlgResource]Create CommAIVbuffer failed"), ret);
        algResResponse.aivInputMem = cclBufferManager_.GetInAIVbuffer();
        algResResponse.aivOutputMem = cclBufferManager_.GetOutAIVbuffer();
    }

    TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
        algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
        algResResponse.aivInputMem, algResResponse.aivOutputMem, expMem};
    HCCL_DEBUG("algResResponse.cclInputMem[%p], size[%llu]; algResResponse.cclOutputMem[%p], "
        "size[%llu]; algResResponse.paramInputMem[%p], size[%llu]; algResResponse.paramOutputMem[%p], size[%llu].",
        algResResponse.cclInputMem.ptr(), algResResponse.cclInputMem.size(),
        algResResponse.cclOutputMem.ptr(), algResResponse.cclOutputMem.size(),
        algResResponse.paramInputMem.ptr(), algResResponse.paramInputMem.size(),
        algResResponse.paramOutputMem.ptr(), algResResponse.paramOutputMem.size());
    algResResponse.opTransportResponse = resRequest.opTransport;

    // 零拷贝场景这里只借助P2p的openIpc能力交换控制面zeroCopyLocalBuffer_，不交换实际用户的输出输出
    if (opParam.isZeroCopy) {
        HCCL_INFO("[AllocAlgResource] zero copy change paramInput[%p] paramOutput[%p] scratchMem[%p] to localBuffer[%p]",
            transMem.paramInputMem.ptr(), transMem.paramOutputMem.ptr(), transMem.scratchMem.ptr(), zeroCopyLocalBuffer_.ptr());
        transMem.scratchMem = zeroCopyLocalBuffer_;
        transMem.paramInputMem = zeroCopyLocalBuffer_;
        transMem.paramOutputMem = zeroCopyLocalBuffer_;
    }

    ClearOpTransportResponseLinks(algResResponse.opTransportResponse);
    if (IsEnableBackupLink()) {
        algResResponse.opTransportResponseBackUp = resRequest.opTransport;
        ClearOpTransportResponseLinks(algResResponse.opTransportResponseBackUp);
        HCCL_DEBUG("[%s]IsEnableBackupLink[%d] init backup & default opTransportResponse", __func__,
            IsEnableBackupLink());
    }

    if (!GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u32 slaveNum = algResResponse.slaveStreams.size();
        algResResponse.threadManage.resize(slaveNum);
        for (u32 ringIndex = 0; ringIndex < slaveNum; ringIndex ++) {
            algResResponse.threadManage[ringIndex].reset(new (std::nothrow) ThreadManage(deviceLogicId_,
                                                                        userRank_,
                                                                        dispatcher_));
            CHK_SMART_PTR_NULL(algResResponse.threadManage[ringIndex]);
            HcclResult ret = algResResponse.threadManage[ringIndex]->Init();
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][MultiRingResource]ringIndex[%u] ThreadManage failed,return[%d]",
                    ringIndex, ret), ret);
            HCCL_INFO("ringThreadsManage Init success[%u]", ringIndex);
        }
    }
    {
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(opParam.tag, transMem, algResResponse.opTransportResponse,
            opParam.aicpuUnfoldMode);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]Alloc transports failed, tag[%s]", __func__, newTag.c_str()), ret);
    if (retryEnable_) {
        // 获取当前rdma相连的所有对端rankList
        std::vector<u32> rankList;
        CHK_RET(transportManager_->GetRemoteRankList(algResResponse.opTransportResponse, rankList,
            TransportType::TRANS_TYPE_IBV_EXP));
        std::string rankListStr = "";
        for (auto remoteRank: rankList) {
            rankListStr += (std::to_string(remoteRank) + ";");
        }
        HCCL_DEBUG("identifier[%s] newTag[%s] rankList[%s]", identifier_.c_str(), newTag.c_str(), rankListStr.c_str());
        CHK_RET(OpRetryManager::AddLinkInfoByIdentifier(deviceLogicId_, identifier_, newTag, rankList));
    }

    if (IsEnableBackupLink()) {
        // 超节点 && level2支持重执行 && Aicpu：创建备用Transport资源
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(opParam.tag, transMem, algResResponse.opTransportResponseBackUp,
            opParam.aicpuUnfoldMode, true);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]Alloc backup transports failed, tag[%s]", __func__, newTag.c_str()), ret);
    }

    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::IncreAllocLink(const std::string &newTag, const OpParam &opParam,
    AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse)
{
    algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
    algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
    DeviceMem expMem = cclBufferManager_.GetCommExpBuffer();

    TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
        algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
        algResResponse.aivInputMem, algResResponse.aivOutputMem, expMem};
    {
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        CHK_RET(transportManager_->IncreAlloc(opParam.tag, transMem, resRequest.opTransport,
            algResResponse.opTransportResponse, opParam.aicpuUnfoldMode));
    }

    if (retryEnable_) {
        // 获取当前rdma相连的所有对端rankList
        std::vector<u32> rankList;
        CHK_RET(transportManager_->GetIncreRemoteRankList(resRequest.opTransport,
            algResResponse.opTransportResponse, rankList, TransportType::TRANS_TYPE_IBV_EXP));
        std::string rankListStr = "";
        for (auto remoteRank: rankList) {
            rankListStr += (std::to_string(remoteRank) + ";");
        }
        HCCL_DEBUG("identifier[%s] newTag[%s] rankList[%s]", identifier_.c_str(), newTag.c_str(), rankListStr.c_str());
        CHK_RET(OpRetryManager::AddLinkInfoByIdentifier(deviceLogicId_, identifier_, newTag, rankList, true));
    }

    if (IsEnableBackupLink()) {
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        CHK_RET(transportManager_->IncreAlloc(opParam.tag, transMem, resRequest.opTransport,
            algResResponse.opTransportResponseBackUp, opParam.aicpuUnfoldMode, true));
    }

    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRecvMsgAndRequestBuffer()
{
    CHK_RET(CheckSuspendingStatus());
    // 拉远、下沉、推理场景(ps、worker)支持使用msg/request内存池
    if (pMsgInfosMem_ == nullptr) {
        pMsgInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclMessageInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pMsgInfosMem_);
        CHK_RET(pMsgInfosMem_->Init());
        HCCL_INFO("InitRecvMsgBuffer Success!");
    }

    if (pReqInfosMem_ == nullptr) {
        pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pReqInfosMem_);
        CHK_RET(pReqInfosMem_->Init());
        HCCL_INFO("InitRequestBuffer Success!");
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMemBlocksAndRecvWrMem()
{
    u32 memBlockNum = MEM_BLOCK_NUM;
    CHK_PRT(GetMemBlockNum(devicePhyId_, memBlockNum));

    if (!GetExternalInputHcclIsTcpMode() && (Is310PDevice() || isHostUseDevNic_)) {
        // 注册mr,hdc模式下在通信类内进行
        if (!isHostUseDevNic_) {
            // 初始化信封内存
            memBlocksManager_.reset(new (std::nothrow) HeterogMemBlocksManager());
            CHK_SMART_PTR_NULL(memBlocksManager_);
            CHK_RET(memBlocksManager_->Init(memBlockNum));

            // 信封内存注册
            CHK_RET(mrManager_->GetKey(memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize(),
                transportResInfo_.lkey));
        }

        // 初始化wr内存
        pRecvWrInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<RecvWrInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pRecvWrInfosMem_);
        CHK_RET(pRecvWrInfosMem_->Init());
        HCCL_INFO("InitMemBlocksAndRecvWrMem Success!");
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDevicePid(s32 devicePid)
{
    devicePid_ = devicePid;
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseWorkSpacebuffer()
{
    workSpace_.free();
}

HcclResult HcclCommunicator::AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem> &bufferPtr) const
{
    CHK_PRT_RET(!size,
        HCCL_INFO("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    DeviceMem tmpBuffer = DeviceMem::alloc(size);
    EXECEPTION_CATCH((bufferPtr = std::make_shared<DeviceMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

    CHK_PRT_RET(size && !bufferPtr.get()->ptr(),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]Create DeviceMem size[%llu] fail,"
                   "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_RET(hrtMemSet(bufferPtr.get()->ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem> &bufferPtr) const
{
    CHK_PRT_RET(!size,
        HCCL_INFO("[HcclCommunicator][AllocAndClearHostMem] host memory size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem] host memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    HostMem tmpBuffer = HostMem::alloc(size);
    EXECEPTION_CATCH((bufferPtr = std::make_shared<HostMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

    CHK_PRT_RET(size && !bufferPtr.get()->ptr(),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem]host memory space size[%llu] fail,"
                   "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memset_s(bufferPtr.get()->ptr(), size, 0, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateWorkSpace(u64 size, DeviceMem &buffer) const
{
    CHK_PRT_RET(!size, HCCL_INFO("[Create][WorkSpace]work space size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX), \
        HCCL_ERROR("[Create][WorkSpace]work space size is greater than %llu",
            ULONG_MAX), HCCL_E_PARA);

    u64 memSize = size;
    buffer = DeviceMem::alloc(memSize);
    CHK_PRT_RET(size && !buffer, HCCL_ERROR("[Create][WorkSpace]Create work space size[%llu] fail,"\
        "please check workspace size.", size), HCCL_E_PTR);
    CHK_RET(hrtMemSet(buffer.ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkSpace(u64 *workSpaceSize, u64 *workSpace) const
{
    *workSpaceSize = workSpaceSize_;
    *workSpace = reinterpret_cast<u64>(workSpace_.ptr());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitWorkSpace()
{
    if (workSpace_.ptr() == nullptr) {
        workSpaceSize_ = COMM_MAX_WORK_SPACE_SIZE;
        CHK_RET(CreateWorkSpace(workSpaceSize_, workSpace_));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgInfo(const std::string &algConfig, const std::string &tag,
    HcclCMDType commType, std::string &algName, std::string &newTag)
{
    // 查表
    CHK_PRT_RET((ALGCFG_TO_NAME.find(algConfig) == ALGCFG_TO_NAME.end()),
        HCCL_ERROR("[%s] invalid algConfig=[%s]", __func__, algConfig.c_str()),
        HCCL_E_PARA);

    auto iter = CMDTYPE_TO_KEYWORD.find(commType);
    CHK_PRT_RET((iter == CMDTYPE_TO_KEYWORD.end()),
        HCCL_ERROR("[%s] invalid commType=[%d]", __func__, static_cast<int>(commType)),
        HCCL_E_PARA);
    CHK_PRT_RET((algConfig.find(iter->second) == algConfig.npos),
        HCCL_ERROR("[%s] commType=[%d] not support algConfig=[%s]",
        __func__, static_cast<int>(commType), algConfig.c_str()),
        HCCL_E_PARA);

    algName = ALGCFG_TO_NAME[algConfig];
    newTag = tag + algName + "_device";
    HCCL_INFO("[%s] tag=[%s], algName=[%s], newTag=[%s]",
              __func__, tag.c_str(), algName.c_str(), newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::FillOpParam(const HcclCMDType commType, OpParam& opParam,
        const uint64_t count, void *pCount, void *pDispls)
{
    if (commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        commType == HcclCMDType::HCCL_CMD_ALLGATHER ||
        commType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16; //按照fp16配置
    } else if (commType == HcclCMDType::HCCL_CMD_ALLTOALLV||
               commType == HcclCMDType::HCCL_CMD_ALLTOALL ||
               commType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        opParam.All2AllDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_FP16;
        opParam.All2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_FP16;
        opParam.All2AllDataDes.sendCounts = pCount;
        opParam.All2AllDataDes.recvCounts = pCount;
        opParam.All2AllDataDes.sdispls = pDispls;
        opParam.All2AllDataDes.rdispls = pDispls;
        opParam.All2AllDataDes.sendCountMatrix = pCount;
    } else {
        HCCL_ERROR("[%s] invalid commType=[%u]",
                   __func__, static_cast<uint32_t>(commType));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocComResource(const string &newTag, const string &algName,
        const HcclCMDType commType, const OpParam& opParam, rtStream_t stream)
{
    if (resMap_.find(newTag) == resMap_.end()) { // 计算&申请通信资源
        unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(commType);
        CHK_PRT_RET(algOperator == nullptr,
            HCCL_ERROR("[%s] algOperator is nullptr", __func__), HCCL_E_INTERNAL);
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        CHK_RET(AllocAlgResource(newTag, commType, opParam, resRequest, resMap_[newTag]));
        CHK_RET(RegisterToHeartBeat());
    }

    if (!isContextLaunched_) { // 通信域内首次下发
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(opParam.stream.ptr(), &streamMode));
        rtStream_t aicpuStream;
        Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream); // aicpuStream需要在首次下发时申请

        rtStream_t aicpuInitStream;
        Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuInitStream);
        Stream tmpStream(aicpuInitStream);
        HCCL_DEBUG("%s ContextLaunched, aicpuInitStream:%p, aicpuStream:%p", __func__, aicpuInitStream, aicpuStream);
        CHK_RET(AicpuResourceInit(algName, resMap_[newTag], newTag, stream, commType));
        CHK_RET(GetReportHcclMC2Info(tmpStream, resMap_[newTag].slaveDevStreams));
    } else if (newTagResAlloced_.find(newTag) == newTagResAlloced_.end()) {
        // 通信域内非首次，但是有新的newTag
        PetersonLockGuard guard(hostDeviceLock_.get());
        CHK_PRT_RET(guard.IsLockFailed(),
            HCCL_ERROR("[%s] hostDeviceLock lock failed", __func__), HCCL_E_INTERNAL);
        CHK_RET(AicpuResourceRefresh(resMap_[newTag], newTag, commType));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocComResourceByTiling(const string &algConfig,
    const string &tag, uint32_t opType, uint32_t reduceType, rtStream_t stream)
{
    string algName, newTag;
    HcclCMDType commType = static_cast<HcclCMDType>(opType);
    CHK_RET(GetAlgInfo(algConfig, tag, commType, algName, newTag));
    CHK_RET(CreateAndGetAiCpuNotifyWithNotifyRes(combinOpara_.signalInfo.aicpuNotify));
    HCCL_INFO("Create aicpu notify %p.", localAiCpuNotifyRes_[0]->ptr());

    // 只有第一次创建，此处通过CCL Buffer地址有效来防止通信域内非首次重新申请内存
    CHK_RET(CreateCommCCLbuffer());
    CHK_RET(CreateCommExpBuffer());

    Stream streamObj(stream);
    // 根据Mc2HcommCfg和通信域中的信息，生成 opParams，其中数据量按照 cclbuffer 配置
    OpParam opParam;
    opParam.tag = tag;
    opParam.opType = commType;
    opParam.reduceType = static_cast<HcclReduceOp>(reduceType);
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuUnfoldMode = true;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(opParam.inputPtr, opParam.inputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(opParam.outputPtr, opParam.outputSize));

    // 按照 ccl buffer size 折算，不同算子折算方式不同, allreduce和cclbuffer size相同
    // allgather、reducescatter、alltoall需除以rank size
    uint64_t count = opParam.outputSize / SIZE_TABLE[HcclDataType::HCCL_DATA_TYPE_FP16];
    if (commType != HcclCMDType::HCCL_CMD_ALLREDUCE) {
        count = (count + userRankSize_ - 1) / userRankSize_;
    }
    HCCL_INFO("[%s] userRankSize=[%u], count=[%u]", __func__, userRankSize_, count);
    vector<uint64_t> countList(userRankSize_ * userRankSize_, count);
    vector<uint64_t> displsList(userRankSize_, 0);
    void *pCount = reinterpret_cast<void *>(&countList[0]);
    void *pDispls = reinterpret_cast<void *>(&displsList[0]);
    CHK_RET(FillOpParam(commType, opParam, count, pCount, pDispls));
    CHK_RET(AllocComResource(newTag, algName, commType, opParam, stream));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
    void **commContext)
{
    const std::string &suffix = HCCL_MC2_MULTISERVER_SUFFIX;
    if (tag.size() > suffix.size() && tag.compare(tag.size() - suffix.size(), suffix.size(), suffix) == 0) {
        HCCL_INFO("[HcclCommunicator][CreateCommResource] Set isA2MC2MultiServer_ to [true]");
        isA2MC2MultiServer_ = true;
    }
    if (isA2MC2MultiServer_ && !isNeedInitNic_ ) {
        InitNic(true);
    }

    if ((deviceType_ != DevType::DEV_TYPE_910_93 && moduleNum_ > 1 && !isA2MC2MultiServer_) ||
        (deviceType_ == DevType::DEV_TYPE_910_93 && superPodNum_ > 1)) {
        HCCL_ERROR("[HcclCommunicator][CommResource]MC2 does not support in the current scenario, "
            "device type[%d] moduleNum[%d] serverNum[%d] superPodNum[%d], isMC2MultiServer[%d].",
            deviceType_, moduleNum_, serverNum_, superPodNum_, isA2MC2MultiServer_);
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO("[HcclCommunicator][CommResource]tag[%s] aicpu stream[%p] isOpbaseMode[%u]", tag.c_str(), aiCpuStream,
        isOpbaseMode);

    Stream stream(aiCpuStream);
    CHK_RET(CreateCommAndStreamRes(tag, stream));

    CHK_RET(Mc2CreateAndLaunchContext(aiCpuStream, isOpbaseMode, commContext));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2CreateAndLaunchContext(rtStream_t aiCpuStream, bool isOpbaseMode, void **commContext)
{
    u32 qosCfg = INVALID_QOSCFG;
    CHK_RET(GetQosCfg(qosCfg));
    CHK_RET(InitWorkSpace());
    HcclResult ret = GetWorkSpace(&(combinOpara_.mc2WorkSpace.workSpaceSize), &(combinOpara_.mc2WorkSpace.workSpace));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][CommResource]errNo[0x%016llx] size[%llu] space[%llu]",
        HCCL_ERROR_CODE(ret), combinOpara_.mc2WorkSpace.workSpaceSize, combinOpara_.mc2WorkSpace.workSpace), ret);

    CHK_SAFETY_FUNC_RET(memcpy_s(combinOpara_.hcomId, sizeof(combinOpara_.hcomId),
        identifier_.c_str(), identifier_.length() + 1));

    Stream tmpStream(aiCpuStream);
    CHK_RET(CreateAndGetAiCpuNotifyWithNotifyRes(combinOpara_.signalInfo.aicpuNotify));
    CHK_RET(CreateAndGetAiCpuNotify(localAiCpuOpNotify_[0], combinOpara_.signalInfo.aicpuOpNotify[0]));
    CHK_RET(CreateAndGetAiCpuNotify(localAiCpuOpNotify_[1], combinOpara_.signalInfo.aicpuOpNotify[1]));
    // 申请集合通信域存储context的device空间
    CHK_RET(CreateDeviceCommContext(sizeof(HcclCombinOpParam), commContext_));
    combinOpara_.config.deterministic = GetDeterministicConfig();
    // retryEnable 写入aicpu_ctx
    combinOpara_.config.retryEnable = static_cast<u8>(retryEnable_);
    combinOpara_.config.retryHoldTime = GetExternalInputRetryHoldTime();
    combinOpara_.config.retryIntervalTime = GetExternalInputRetryIntervalTime();
    combinOpara_.config.notifyWaitTime =
        (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) ?
            GetExternalInputHcclExecTimeOut() : NOTIFY_DEFAULT_WAIT_TIME;
    combinOpara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    combinOpara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    combinOpara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();

    void *overflowAddr = nullptr;
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        CHK_RET(hrtCtxGetOverflowAddr(&overflowAddr));
        combinOpara_.overFlowAddr = reinterpret_cast<u64>(overflowAddr);
        HCCL_INFO("[HcclImplBase][Mc2CreateAndLaunchContext]get combinOpara_.overFlowAddr %llx",
            combinOpara_.overFlowAddr);
        // 非整卡 (2DUO卡各取1芯的场景) 因为受到PCIE限制，不可以使用读操作进行数据拷贝
        if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() != userRankSize_) {
            combinOpara_.onlyRead = 1;
        }
    }
    HCCL_INFO("read only is set to %u", combinOpara_.onlyRead);

    if (isA2MC2MultiServer_) {
        // 拷贝normal transport信息到device侧
        const u64 ibverbsDataSize = transDevIbverbsData_.size() * sizeof(TransportDeviceNormalData);
        ibverbsDataBuffer_ = DeviceMem::alloc(ibverbsDataSize);
        CHK_PTR_NULL(ibverbsDataBuffer_.ptr());
        CHK_RET(hrtMemAsyncCopyByQos(ibverbsDataBuffer_.ptr(),
            ibverbsDataBuffer_.size(),
            transDevIbverbsData_.data(),
            ibverbsDataSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,
            aiCpuStream,
            qosCfg));
        combinOpara_.ibverbsData = reinterpret_cast<u64>(ibverbsDataBuffer_.ptr());
        combinOpara_.ibverbsDataSize = ibverbsDataSize;
        combinOpara_.multiServerFlag = static_cast<u8>(true);
        HCCL_INFO("[HcclImplBase][Mc2CreateAndLaunchContext] set combinOpara_.ibverbsData to [%llu], "
                  "combinOpara_.multiServerFlag to [%u]",
            combinOpara_.ibverbsData,
            combinOpara_.multiServerFlag);
    }

    HostMem src = HostMem::create(&combinOpara_, sizeof(HcclCombinOpParam));
    // 将通信数据拷贝到device侧，供AICPU算法编排使用
    CHK_RET(hrtMemAsyncCopyByQos(commContext_.ptr(), commContext_.size(), src.ptr(), src.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream, qosCfg));

    std::string kernelName = "RunAicpuKfcResInit";
    CHK_RET(Mc2AiCpuKernelLaunch(tmpStream.ptr(), reinterpret_cast<u64>(commContext_.ptr()), kernelName));
    SetMC2EnvFlag();
    if (isOpbaseMode == true) {
        CHK_RET(hcclStreamSynchronize(tmpStream.ptr()));
    }

    *commContext = commContext_.ptr();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAiCpuNotifyData(const std::shared_ptr<LocalNotify> &localNotify,
    HcclSignalInfo &notifyInfo)
{
    if (localNotify == nullptr) {
        HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]notifyHandle is null");
        notifyInfo.resId = INVALID_U64;
        return HCCL_SUCCESS;
    }

    CHK_RET(localNotify->GetNotifyData(notifyInfo));
    HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]esId[%lld], addr[%lld], devId[%u], tsId[%u].",
        notifyInfo.resId, notifyInfo.addr, notifyInfo.devId, notifyInfo.tsId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify> &localNotify,
    HcclSignalInfo &notifyInfo)
{
    if (localNotify != nullptr) {
        CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));
        HCCL_INFO("[HcclCommunicator][CreateAndGetAiCpuNotify]aicpu notify allready create ptr[%p]",
            localNotify->ptr());
        return HCCL_SUCCESS;
    }

    EXECEPTION_CATCH((localNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_RET(localNotify->Init(NotifyLoadType::DEVICE_NOTIFY));
    CHK_RET(localNotify->SetIpc());

    CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateAndGetAiCpuNotifyWithNotifyRes(HcclSignalInfo &notifyInfo)
{
    if (localAiCpuNotifyRes_.size() > 0) {
        CHK_RET(CreateAndGetAiCpuNotify(localAiCpuNotifyRes_[0], notifyInfo));
    } else {
        std::shared_ptr<LocalNotify> localNotify = {nullptr};
        CHK_RET(CreateAndGetAiCpuNotify(localNotify, notifyInfo));
        localAiCpuNotifyRes_.push_back(localNotify);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
{
    if (opStream_.ptr() != nullptr) {
        HCCL_INFO("%s allready alloc, group:%s, stream id:%u", __func__, identifier_.c_str(), opStream_.id());
        aiCpuStream = opStream_.ptr();
        return HCCL_SUCCESS;
    }

    constexpr u32 aicpuStreamMode = 1; // 单独申请的kernel流，使能遇错即停，避免出错后流卡住不退
    opStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(hrtStreamSetMode(opStream_.ptr(), aicpuStreamMode));
    aiCpuStream = opStream_.ptr();
    HCCL_RUN_INFO("%s alloc success, group:%s, stream id:%u, mainStreamMode:%u, aicpuStreamMode:%u",
        __func__, identifier_.c_str(), opStream_.id(), streamMode, aicpuStreamMode);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuInitStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream)
{
    if (aicpuInitStream_.ptr() != nullptr) {
        HCCL_INFO("%s allready alloc, group:%s, stream id:%u", __func__, identifier_.c_str(), aicpuInitStream_.id());
        aiCpuStream = aicpuInitStream_.ptr();
        return HCCL_SUCCESS;
    }

    constexpr u32 aicpuStreamMode = 1; // 单独申请的kernel流，使能遇错即停，避免出错后流卡住不退
    aicpuInitStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(hrtStreamSetMode(aicpuInitStream_.ptr(), aicpuStreamMode));
    aiCpuStream = aicpuInitStream_.ptr();
    HCCL_RUN_INFO("%s alloc success, group:%s, stream id:%u, mainStreamMode:%u, aicpuStreamMode:%u",
        __func__, identifier_.c_str(), aicpuInitStream_.id(), streamMode, aicpuStreamMode);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string &kernelName)
{
    uint64_t beginTime = hrtMsprofSysCycleTime();
    const std::string profName = "hcomAicpuInit";
    rtAicpuArgsEx_t argsInfo;
    struct ApiParamDef {
        u64 commContext;
        char kernelName[64] = "";
        char soName[64] = "libccl_kernel.so";
        char opName[64] = "HcclAicpuOp";
    };
    struct ApiParamDef apiParam;
    CHK_SAFETY_FUNC_RET(
         memcpy_s(apiParam.kernelName, sizeof(apiParam.kernelName), kernelName.c_str(), kernelName.length() + 1));
    apiParam.commContext = static_cast<uint64_t>(addr);

    if (mc2DeviceMem_.ptr() == nullptr) {
        mc2DeviceMem_ = DeviceMem::alloc(sizeof(apiParam));
    }
    CHK_SMART_PTR_NULL(mc2DeviceMem_);
    CHK_RET(hrtMemSyncCopy(mc2DeviceMem_.ptr(), sizeof(apiParam), &apiParam, sizeof(apiParam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    argsInfo.args = mc2DeviceMem_.ptr();
    argsInfo.hostInputInfoPtr = nullptr;
    argsInfo.kernelOffsetInfoPtr = nullptr;
    argsInfo.argsSize = sizeof(apiParam);
    argsInfo.hostInputInfoNum = 0;
    argsInfo.kernelOffsetInfoNum = 0;
    argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.soName) -
        reinterpret_cast<const char *>(&apiParam));

    argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.kernelName) -
        reinterpret_cast<const char *>(&apiParam));
    argsInfo.isNoNeedH2DCopy = true;
    CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU, apiParam.opName, 1, &argsInfo, nullptr, stm, 0));
    uint64_t endTime = hrtMsprofSysCycleTime();
    s32 threadId = SalGetTid();
    CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunch(const OpParam &opParam, const HcclCMDType &opType,
    const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo)
{
    HCCL_DEBUG("AicpuKfcTilingDataLaunch count %llu dataType %s op %s opType %u", opParam.DataDes.count,
        GetDataTypeEnumStr(opParam.DataDes.dataType).c_str(), GetReduceOpEnumStr(opParam.reduceType).c_str(), opType);
    struct HcclKFCTilingData tilingDate = {0};
    tilingDate.sendCnt = opParam.DataDes.count;
    tilingDate.dataType = opParam.DataDes.dataType;
    tilingDate.commType = static_cast<uint8_t>(opType);
    tilingDate.reduceOp = opParam.reduceType;
    tilingDate.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE;
    tilingDate.totalCnt = 1;
    tilingDate.turnNum = 1;
    tilingDate.hasCommOut = 1;
    u32 tempDebugMode = GetExternalInputMc2DebugMode();
    const u32 mC2DebugWaitComm = 8;
    tilingDate.debugMode = (tempDebugMode == mC2DebugWaitComm) ? static_cast<uint8_t>(tempDebugMode) : 0;
    CHK_RET(SetNormalMode(dispatcher_));
    HcclWorkflowMode mode = GetWorkflowMode();
    Stream mainStream(opParam.stream.ptr());
    CHK_RET(LocalNotify::Post(mainStream, dispatcher_, localAiCpuOpNotify_[0], INVALID_VALUE_STAGE));
    rtStream_t kfcOpStream = opStream_.ptr();
    if (opTilingInfo.isUsedMainStream) {
        kfcOpStream = opParam.stream.ptr();
    }
    CHK_RET(AicpuUnfoldKernelLaunch(opParam.inputPtr, opParam.outputPtr, kfcOpStream,
                                    reinterpret_cast<u64>(deviceContext.ptr()), &tilingDate, sizeof(HcclKFCTilingData),
                                    kernelName, mode, opParam.tag));
    CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, localAiCpuOpNotify_[1], INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}


HcclResult HcclCommunicator::SetDynamicTilingDataAlltoall(const OpParam &opParam, HostMem &dynamicDataMem)
{
    struct OpTilingAllToAllDataDes* a2ADataPtr =
        reinterpret_cast<struct OpTilingAllToAllDataDes*>(dynamicDataMem.ptr());
    a2ADataPtr->sendType = static_cast<u8>(opParam.All2AllDataDes.sendType);
    a2ADataPtr->recvType = static_cast<u8>(opParam.All2AllDataDes.recvType);
    a2ADataPtr->sendCount = opParam.All2AllDataDes.sendCount;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDynamicTilingDataAlltoallv(const OpParam &opParam, HostMem &dynamicDataMem)
{
    struct OpTilingAlltoallvDataDes* alltoallvDataPtr =
        reinterpret_cast<struct OpTilingAlltoallvDataDes*>(dynamicDataMem.ptr());
    alltoallvDataPtr->sendType = static_cast<u8>(opParam.All2AllDataDes.sendType);
    alltoallvDataPtr->recvType = static_cast<u8>(opParam.All2AllDataDes.recvType);
    u32 rankSize = GetRankSize();
    u64* sendCountsPtr = static_cast<u64 *>(alltoallvDataPtr->sendRecvInfos);
    u64* recvCountsPtr = sendCountsPtr + rankSize;
    u64* sdisplsPtr = recvCountsPtr + rankSize;
    u64* rdisplsPtr = sdisplsPtr + rankSize;
    for (u32 i = 0 ; i < rankSize; i++) {
        CHK_PTR_NULL(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i);
        sendCountsPtr[i] = *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i);
        CHK_PTR_NULL(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i);
        recvCountsPtr[i] = *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i);
        CHK_PTR_NULL(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i);
        sdisplsPtr[i] = *(static_cast<const u64 *>(opParam.All2AllDataDes.sdispls) + i);
        CHK_PTR_NULL(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i);
        rdisplsPtr[i] = *(static_cast<const u64 *>(opParam.All2AllDataDes.rdispls) + i);
        HCCL_DEBUG("[SetDynamicTilingDataAlltoallv] sendCounts[%llu], recvCounts[%llu], sdispls[%llu], rdispls[%llu]",
            sendCountsPtr[i], recvCountsPtr[i], sdisplsPtr[i], rdisplsPtr[i]);
    }
    HCCL_DEBUG("[SetDynamicTilingDataAlltoallv] set dynamic tiling data for alltoallv successs, alltoallvDataPtr[%p]", alltoallvDataPtr);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDynamicTilingDataAlltoallvc(const OpParam &opParam, HostMem &dynamicDataMem)
{
    struct OpTilingAlltoallvcDataDes* a2ADataPtr =
        reinterpret_cast<struct OpTilingAlltoallvcDataDes*>(dynamicDataMem.ptr());
    a2ADataPtr->sendType = static_cast<u8>(opParam.All2AllDataDes.sendType);
    a2ADataPtr->recvType = static_cast<u8>(opParam.All2AllDataDes.recvType);
    u32 rankSize = GetRankSize();
    for (u64 i = 0 ; i < rankSize * rankSize; i++) {
        a2ADataPtr->sendCountMatrix[i] = *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCountMatrix) + i);
    }
    return HCCL_SUCCESS;
}

u64 HcclCommunicator::CalcOpTilingDynamicDataSize(
    const OpParam &opParam, const HcclCMDType &opType, const u32 &rankSize)
{
    u64 dynamicDataSize = 0ULL;
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        dynamicDataSize = sizeof(struct OpTilingBatchSendRecvDataDes) +
            opParam.BatchSendRecvDataDes.itemNum * sizeof(HcclSendRecvItem);
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        dynamicDataSize = sizeof(struct OpTilingAllToAllDataDes);
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        dynamicDataSize = sizeof(struct OpTilingAlltoallvDataDes) + rankSize * ALLTOALL_INFO_MATRIX_SIZE * sizeof(u64);
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dynamicDataSize = sizeof(struct OpTilingAlltoallvcDataDes) + rankSize * rankSize * sizeof(u64);
    } else {
        dynamicDataSize = sizeof(struct OpTilingDataDes);
    }
    return dynamicDataSize;
}

HcclResult HcclCommunicator::AicpuInitOpTilingDataFromOpParam(const OpParam &opParam, const HcclCMDType &opType,
    struct OpTilingData* opTilingData)
{
    opTilingData->workflowMode = (IsForceAicpuOpBaseMode(opParam, opType) && !opParam.isZeroCopy) ?
        static_cast<u8>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) : static_cast<u8>(GetWorkflowMode());
    opTilingData->inputPtr = reinterpret_cast<u64>(opParam.inputPtr);
    opTilingData->outputPtr = reinterpret_cast<u64>(opParam.outputPtr);
    opTilingData->reduceType = static_cast<u8>(opParam.reduceType);
    opTilingData->syncMode = static_cast<u8>(opParam.syncMode);
    opTilingData->root = opParam.root;
    opTilingData->dstRank = opParam.dstRank;
    opTilingData->srcRank = opParam.srcRank;
    opTilingData->opType = static_cast<u8>(opType);
    opTilingData->inplaceSupportRetry = static_cast<u8>(inplaceSupportRetry_);
    opTilingData->retryEnable = static_cast<u8>(retryEnable_);
    opTilingData->inPlaceSupportRetryStatus = static_cast<u8>(inPlaceSupportRetryStatus_);
    opTilingData->isInplacePreSync = static_cast<u8>(isInplacePreSync_);
    opTilingData->isPostSync = static_cast<u8>(isPostSync_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuInitOpTilingDataBuf(const OpParam &opParam, const HcclCMDType &opType,
    const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 dynamicDataSize)
{
    u32 opTilingDataSize = sizeof(struct OpTilingData) + dynamicDataSize;

    if (opTilingDataBuf_.ptr() == nullptr) {
        opTilingDataBuf_ = HostMem::alloc(TILINGDATA_BUF_SIZE);
        CHK_PRT_RET(opTilingDataBuf_.ptr() == nullptr,
            HCCL_ERROR("[HcclCommunicator][AicpuInitOpTilingDataBuf] Alloc opTilingDataBuf failed!"),
            HCCL_E_INTERNAL);
    }

    if(opTilingDataBuf_.ptr() != nullptr && opTilingDataSize > opTilingDataBuf_.size()) {
        opTilingDataBuf_.free();
        opTilingDataBuf_ = HostMem::alloc(opTilingDataSize);
        CHK_PRT_RET(opTilingDataBuf_.ptr() == nullptr,
            HCCL_ERROR("[HcclCommunicator][AicpuInitOpTilingDataBuf] increate opTilingDataBuf len[%llu] failed!",
            opTilingDataSize), HCCL_E_INTERNAL);
    }

    //填充固定内容
    HostMem opTilingDataMem = opTilingDataBuf_.range(0, opTilingDataSize);
    struct OpTilingData*  opTilingData = static_cast<struct OpTilingData*>(opTilingDataMem.ptr());
    u32 algTypeTranfer =  (static_cast<u32>(opTilingInfo.algType.algoLevel2) << (HCCL_LEVEL_ALGO_WIDTH + HCCL_LEVEL_ALGO_WIDTH)) +
        (static_cast<u32>(opTilingInfo.algType.algoLevel1) << HCCL_LEVEL_ALGO_WIDTH) +
        static_cast<u32>(opTilingInfo.algType.algoLevel0);
    opTilingData->algType = static_cast<u64>(algTypeTranfer);
    opTilingData->floatOverflowMode = opTilingInfo.floatOverflowMode;
    opTilingData->dumpDebug = opTilingInfo.dumpDebug;
    CHK_RET(AicpuInitOpTilingDataFromOpParam(opParam, opType, opTilingData));
    opTilingData->length = dynamicDataSize;
    ProfilerBase::GetSubmittedOpCnt(opTilingData->index);
    u32 tempDebugMode = GetExternalInputMc2DebugMode();
    const u32 mC2DebugWaitComm = 8;
    opTilingData->debugMode = (tempDebugMode == mC2DebugWaitComm) ? static_cast<uint8_t>(tempDebugMode) : 0;
    opTilingData->isZeroCopy = opParam.isZeroCopy;

    //填充动态内容
    HostMem dynamicDataMem = opTilingDataBuf_.range(sizeof(struct OpTilingData), dynamicDataSize);
    CHK_PTR_NULL(dynamicDataMem.ptr());
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        struct OpTilingBatchSendRecvDataDes* batchSendRecvDataPtr =
            reinterpret_cast<struct OpTilingBatchSendRecvDataDes*>(dynamicDataMem.ptr());
        batchSendRecvDataPtr->itemNum = opParam.BatchSendRecvDataDes.itemNum;
        for (u32 i = 0; i < opParam.BatchSendRecvDataDes.itemNum; i++) {
            CHK_PTR_NULL(opParam.BatchSendRecvDataDes.sendRecvItemsPtr + i);
            batchSendRecvDataPtr->batchSendRecvItem[i] = *(opParam.BatchSendRecvDataDes.sendRecvItemsPtr + i);
        }
    } else if(opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(SetDynamicTilingDataAlltoall(opParam, dynamicDataMem));
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(SetDynamicTilingDataAlltoallv(opParam, dynamicDataMem));
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(SetDynamicTilingDataAlltoallvc(opParam, dynamicDataMem));
    } else {
        struct OpTilingDataDes* opDataDesPtr = reinterpret_cast<struct OpTilingDataDes*>(dynamicDataMem.ptr());
        opDataDesPtr->count = opParam.DataDes.count;
        opDataDesPtr->dataType = static_cast<u8>(opParam.DataDes.dataType);
    }
    HCCL_INFO("[HcclCommunicator][AicpuInitOpTilingDataBuf]algType[%lu]", opTilingData->algType);
    CHK_SAFETY_FUNC_RET(memcpy_s(opTilingData->algName, sizeof(opTilingData->algName), opTilingInfo.algName.c_str(),
        opTilingInfo.algName.length() + 1));
    CHK_SAFETY_FUNC_RET(memcpy_s(opTilingData->newTag, sizeof(opTilingData->newTag),
        opTilingInfo.newTag.c_str(), opTilingInfo.newTag.length() + 1));
    CHK_SAFETY_FUNC_RET(memcpy_s(opTilingData->tag, sizeof(opTilingData->tag), opParam.tag.c_str(),
        opParam.tag.length() + 1));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchIn(const OpParam &opParam,const DeviceMem &deviceContext,
    const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 opTilingDataSize)
{
    HostMem opTilingDataMem = opTilingDataBuf_.range(0, opTilingDataSize);
    CHK_RET(SetNormalMode(dispatcher_));
    Stream &mainStream = const_cast<Stream&>(opParam.stream);
    CHK_RET(LocalNotify::Post(mainStream, dispatcher_, localAiCpuOpNotify_[0], INVALID_VALUE_STAGE));

    rtStream_t kfcOpStream = opTilingInfo.isUsedMainStream ? opParam.stream.ptr() : opStream_.ptr();
    HcclWorkflowMode mode = GetWorkflowMode();
    // 如果是图模式，则尝试从附属从流中获取一下stream，如果能拿到则使用，否则用原有的
    if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        !attachedStreams_.empty() && attachedStreams_[0].ptr() != nullptr) {
        kfcOpStream = attachedStreams_[0].ptr();
        HCCL_INFO("[HcclCommunicator][AicpuKfcTilingDataLaunchExt] Use attached stream [%p]", kfcOpStream);
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    std::string profName = GetCMDTypeEnumStr(opParam.opType);
    if (profName == "Invalid HcclCMDType" || profName == "invalid") {
        profName = "HcclOpAicpuKernel";
    } else {
        profName += "AicpuKernel";
    }
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(kfcOpStream, streamId));
    auto getAicpuTaskExceptionCallBack = [this]() {return this->GetAicpuTaskException();};
    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId_, getAicpuTaskExceptionCallBack);
    if (streamId != opParam.stream.id()) {
        RegisterGetAicpuTaskExceptionCallBack(opParam.stream.id(), deviceLogicId_, getAicpuTaskExceptionCallBack);
    }
    HCCL_INFO("profName:%s streamId[%d] opParam streamId[%d]", profName.c_str(), streamId, opParam.stream.id());

    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    CHK_RET(GetStreamCaptureInfo(opParam.stream.ptr(), rtModel, isCapture));
    if (isCapture) {
        CHK_PTR_NULL(rtModel);
        CHK_RET(AddStreamToModel(kfcOpStream, rtModel));
        HCCL_INFO("[HcclCommunicator][%s]Add stream[%d] to model success.", __func__, streamId);
    }

    CHK_RET(AicpuUnfoldKernelLaunchV2(opParam.inputPtr, opParam.outputPtr, kfcOpStream,
                                    reinterpret_cast<u64>(deviceContext.ptr()), opTilingDataMem.ptr(), opTilingDataSize,
                                    kernelName, mode, opParam.tag));
    uint64_t endTime = hrtMsprofSysCycleTime();
    s32 threadId = SalGetTid();
    CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));
    u32 timeOut = (opResPara_.config.notifyWaitTime == 0) ? opResPara_.config.notifyWaitTime :
        (opResPara_.config.notifyWaitTime + AICPU_H2D_TIMEOUT_INC);
    CHK_RET(LocalNotify::Wait(
        mainStream, dispatcher_, localAiCpuOpNotify_[1], INVALID_VALUE_STAGE, timeOut));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchExt(const OpParam &opParam, const HcclCMDType &opType,
    const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo)
{
    HCCL_DEBUG("AicpuKfcTilingDataLaunchExt count %llu dataType %s op %s opType %u retryEnable_ %d, "
        "inPlaceSupportRetryStatus_ %d",
        opParam.DataDes.count,
        GetDataTypeEnumStr(opParam.DataDes.dataType).c_str(), GetReduceOpEnumStr(opParam.reduceType).c_str(), opType,
        retryEnable_, inPlaceSupportRetryStatus_);

    u32 severNum4PostSync = 4;
    bool needPostSync = superPodNum_ > 1 || serverNum_ >= severNum4PostSync; // reduce/reduce scatter算子是否需要PostSync
    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE &&
        retryEnable_ && (inPlaceSupportRetryStatus_ == InplaceSupportRetryStatus::USER_LARGER_THAN_CCL)) {
        u32 itemNum = 2;
        for (u32 i = 0; i < itemNum; i++) {
            if (i == 0) {
                isInplacePreSync_ = true;
            } else {
                isInplacePreSync_ = false;
            }
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isInplacePreSync_[%d].",
                isInplacePreSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize());
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                sizeof(struct OpTilingData) + dynamicDataSize));
            isInplacePreSync_ = false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE && retryEnable_ && needPostSync) {
            isPostSync_ = true;
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isPostSync_[%d].",
                isPostSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize());
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                sizeof(struct OpTilingData) + dynamicDataSize));
            isPostSync_ = false;
    } else if (retryEnable_ && opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        if (inPlaceSupportRetryStatus_ == InplaceSupportRetryStatus::USER_LARGER_THAN_CCL) {
            isInplacePreSync_ = true;
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isInplacePreSync_[%d].",
                isInplacePreSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize());
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
                sizeof(struct OpTilingData) + dynamicDataSize));
            isInplacePreSync_ = false;
        }
        isInplacePreSync_ = false;
        if (needPostSync) {
            isPostSync_ = true;
        }
        HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with "
            "isInplacePreSync_[%d], isPostSync_[%d].",
            isInplacePreSync_, isPostSync_);
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize());
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
            sizeof(struct OpTilingData) + dynamicDataSize));
        isPostSync_ = false;
    } else if (retryEnable_ &&
        (opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
        opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
        isPostSync_ = true;
        HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with "
            "isInplacePreSync_[%d], isPostSync_[%d].",
            isInplacePreSync_, isPostSync_);
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize());
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
            sizeof(struct OpTilingData) + dynamicDataSize));
        isPostSync_ = false;
    } else {
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize());
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(opParam, deviceContext, kernelName, opTilingInfo,
            sizeof(struct OpTilingData) + dynamicDataSize));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunch(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
    void* tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
{
    struct ApiParamDef {
        uint64_t x1; // 算子sendbuffer地址
        uint64_t y = 0;
        uint64_t gatherOut; // 算子recvbuffer地址
        uint64_t context; // 通信资源准备的地址
        uint64_t workspace; // 消息区地址
        uint64_t tilingDataPtr; // tilingData地址
        uint8_t tilingData[2048];
        char soName[32] = "libccl_kernel.so";
        char kernelName[32] = "";
        char opName[32] = "HcclAicpuOp";
        char hostInputInfo[16];
    };

    struct ApiParamDef apiParam;
    CHK_SAFETY_FUNC_RET(
        memcpy_s(apiParam.kernelName, sizeof(apiParam.kernelName), kernelName.c_str(), kernelName.length() + 1));
    apiParam.x1 = reinterpret_cast<uint64_t>(inputPtr);
    apiParam.gatherOut = reinterpret_cast<uint64_t>(outputPtr);
    apiParam.context = addr;
    apiParam.workspace = (u64)workSpace_.ptr();
    CHK_SAFETY_FUNC_RET(memcpy_s(apiParam.tilingData, sizeof(apiParam.tilingData), tilingDataPtr,
        tilingDataSize));

    rtAicpuArgsEx_t argsInfo;

    argsInfo.args = (void*)&apiParam;
    apiParam.tilingDataPtr = reinterpret_cast<uint64_t>(apiParam.tilingData);

    rtHostInputInfo_t* hostInfo = (rtHostInputInfo_t*)apiParam.hostInputInfo;
    hostInfo->addrOffset = 5 * sizeof(void*); // aclnn与aicore协定，addr地址偏移时5*(void*）
    hostInfo->dataOffset = 6 * sizeof(void*); // aclnn与aicore协定，data偏移6*(void*)
    argsInfo.hostInputInfoPtr = hostInfo;
    argsInfo.kernelOffsetInfoPtr = nullptr;
    argsInfo.argsSize = sizeof(apiParam);
    argsInfo.hostInputInfoNum = 1;
    argsInfo.kernelOffsetInfoNum = 0;
    argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.soName) -
        reinterpret_cast<const char *>(&apiParam));

    argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&apiParam.kernelName) -
        reinterpret_cast<const char *>(&apiParam));
    argsInfo.isNoNeedH2DCopy = false;

    CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, apiParam.opName, 1, &argsInfo, nullptr, stm, 0));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchV2(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
    void* tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode, const std::string &tag)
{
    struct ApiParamDef {
        uint64_t context;
        uint64_t tilingDataPtr; //tilingData 地址
        char soName[32];
        char kernelName[32];
        char opName[32];
        char hostInputInfo[16];
        uint8_t tilingData[];
    };

    u32 apiTilingDataSize = sizeof(struct ApiParamDef) + tilingDataSize;
    if (apiTilingDataMem_.ptr() == nullptr ) {
        apiTilingDataMem_ = HostMem::alloc(TILINGDATA_BUF_SIZE);
        CHK_PRT_RET(apiTilingDataMem_.ptr() == nullptr,
            HCCL_ERROR("[HcclCommunicator][AicpuUnfoldKernelLaunch] Alloc apiTilingDataMem_ failed!"),
            HCCL_E_INTERNAL);
        std::string opName = "HcclAicpuOp";
        std::string soName = "libccl_kernel.so";
        struct ApiParamDef* apiParamTmp = static_cast<struct ApiParamDef*>(apiTilingDataMem_.ptr());
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParamTmp->soName, sizeof(apiParamTmp->soName), soName.c_str(), soName.length() + 1));
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParamTmp->opName, sizeof(apiParamTmp->opName), opName.c_str(), opName.length() + 1));
    }
    void* pytr;
    HostMem apiTilingDataMemTmp_;
    bool isNeedReAlloc = apiTilingDataSize > TILINGDATA_BUF_SIZE;
    if (isNeedReAlloc){
        apiTilingDataMemTmp_= HostMem::alloc(apiTilingDataSize);
        CHK_PRT_RET(apiTilingDataMemTmp_.ptr() == nullptr,
            HCCL_ERROR("[HcclCommunicator][AicpuUnfoldKernelLaunch] Alloc apiTilingDataMemTmp_ failed!"),
            HCCL_E_INTERNAL);
        HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunch] apiTilingDataSize is larger than "\
                "the size of TILINGDATA_BUF_SIZE.");
        std::string opName = "HcclAicpuOp";
        std::string soName = "libccl_kernel.so";
        struct ApiParamDef* apiParamTmp = static_cast<struct ApiParamDef*>(apiTilingDataMemTmp_.ptr());
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParamTmp->soName, sizeof(apiParamTmp->soName), soName.c_str(), soName.length() + 1));
        CHK_SAFETY_FUNC_RET(
            memcpy_s(apiParamTmp->opName, sizeof(apiParamTmp->opName), opName.c_str(), opName.length() + 1));
        pytr = apiTilingDataMemTmp_.range(0, apiTilingDataSize).ptr();
    } else {
        pytr = apiTilingDataMem_.range(0, apiTilingDataSize).ptr();
    }
    struct ApiParamDef* apiParam = static_cast<struct ApiParamDef*>(pytr);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(apiParam->kernelName, sizeof(apiParam->kernelName), kernelName.c_str(), kernelName.length() + 1));
    CHK_SAFETY_FUNC_RET(
        memcpy_s(apiParam->tilingData, tilingDataSize, tilingDataPtr, tilingDataSize));
    apiParam->context = addr;

    rtAicpuArgsEx_t argsInfo;
    argsInfo.args = (void*)apiParam;
    apiParam->tilingDataPtr = reinterpret_cast<uint64_t>(apiParam->tilingData);
    rtHostInputInfo_t* hostInfo = (rtHostInputInfo_t*)(apiParam->hostInputInfo);
    hostInfo->addrOffset = 1 * sizeof(void*); //tiliingDataPtr, addr地址偏移时1*(void*）
    hostInfo->dataOffset = 16 * sizeof(void*); //tilingData，data偏移16*(void*)
    argsInfo.hostInputInfoPtr = hostInfo;
    argsInfo.kernelOffsetInfoPtr = nullptr;
    argsInfo.argsSize = apiTilingDataSize;
    argsInfo.hostInputInfoNum = 1;
    argsInfo.kernelOffsetInfoNum = 0;
    argsInfo.soNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&(apiParam->soName)) -
        reinterpret_cast<const char *>(apiParam));
    argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(reinterpret_cast<const char *>(&(apiParam->kernelName)) -
        reinterpret_cast<const char *>(apiParam));
    argsInfo.timeout = static_cast<u16>((opResPara_.config.notifyWaitTime == 0) ?
        opResPara_.config.notifyWaitTime : (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC));
    argsInfo.isNoNeedH2DCopy = false;
    CHK_RET(hrtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, apiParam->opName, 1, &argsInfo, nullptr, stm,
        RT_KERNEL_USE_SPECIAL_TIMEOUT));
    HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunchV2] exec succ.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCombinOpara()
{
    CHK_SAFETY_FUNC_RET(memset_s(&combinOpara_, sizeof(combinOpara_), 0, sizeof(combinOpara_)));

    combinOpara_.rankId = INVALID_UINT;
    combinOpara_.signalInfo.aicpuNotify.rankId = INVALID_UINT;

    for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.noIpcNotifys) / sizeof(combinOpara_.signalInfo.noIpcNotifys[0]);
        i++) {
        combinOpara_.signalInfo.noIpcNotifys[i].rankId = INVALID_UINT;
    }

    for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.ipcNotifys) / sizeof(combinOpara_.signalInfo.ipcNotifys[0]);
        i++) {
        combinOpara_.signalInfo.ipcNotifys[i].rankId = INVALID_UINT;
    }

    for (u32 i = 0; i < sizeof(combinOpara_.signalInfo.noIpcEvents) / sizeof(combinOpara_.signalInfo.noIpcEvents[0]);
        i++) {
        combinOpara_.signalInfo.noIpcEvents[i].rankId = INVALID_UINT;
    }

    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetCommResource(const std::string &tag, void **commContext)
{
    if (LIKELY(IsExistCommRes(tag))) {
        *commContext = commContext_.ptr();
        return true;
    }
    return false;
}

bool HcclCommunicator::GetCommResource(void *&commContext)
{
    commContext = opResDevicePara_.ptr();
    return true;
}

HcclResult HcclCommunicator::GetAicpuOpStreamNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    CHK_RET(GetAicpuOpStreamAndNotify(opStream, aicpuNotifyNum, aicpuNotify));
    HCCL_INFO("[HcclCommunicator][GetAicpuOpStreamNotify]opStream %p aicpuNotify %p.", *opStream, *aicpuNotify);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAicpuOpStreamAndNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    *opStream = opStream_.ptr();
    if (localAiCpuNotifyRes_.size() <  aicpuNotifyNum) {
        for (u16 i = localAiCpuNotifyRes_.size(); i < aicpuNotifyNum; i++) {
            std::shared_ptr<LocalNotify> localNotify = {nullptr};
            HcclSignalInfo aicpuNotify;
            CHK_RET(CreateAndGetAiCpuNotify(localNotify, aicpuNotify));
            localAiCpuNotifyRes_.push_back(localNotify);
        }
    }

    for(u16 i = 0; i < aicpuNotifyNum; i++) {
        *(aicpuNotify + i) = localAiCpuNotifyRes_[i]->ptr();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuNotifyInvaild()
{
    combinOpara_.signalInfo.aicpuNotify.resId = INVALID_U64;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo)
{
    std::unique_lock<std::mutex> replLock(commLock_);
    tagCommInfo_.erase(tag);
    tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(*commInfo)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateMutiStreamResFor310P(const std::string &tag, level1StreamInfo_t &streamInfo)
{
    u32 rankSize = GetRankSize();
    u32 pid;
    if (SalGetBareTgid(&pid) != HCCL_SUCCESS) {
        HCCL_DEBUG("get pid fail");
    }
    HCCL_INFO("[HcclCommunicator][CreateMutiStreamRes]tag[%s] ranksize[%u] comminfo ranksize[%u] "\
        "auxRingCommStreamsDev_ size[%u] ringDeviceSignalAux size[%u] ringDeviceSignal size[%u] "\
        "ringDeviceStreams size[%u]", tag.c_str(), rankSize, tagCommInfo_[tag].commIntraServer->RankSize(),
        auxRingCommStreamsDev_.size(), streamInfo.ringDeviceSignalAux.size(),
        streamInfo.ringDeviceSignal.size(), streamInfo.ringDeviceStreams.size());
    if (auxRingCommStreamsDev_.empty() || auxRingCommStreamsDev_.size() < rankSize) {
        auxRingCommStreamsDev_.resize(rankSize);
        u32 resNum = rankSize - 1;
        streamInfo.ringDeviceSignalAux.resize(resNum);
        streamInfo.ringDeviceSignal.resize(resNum);
        for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
            auxRingCommStreamsDev_[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
            // 给device侧申请的流不需要setmode，否则rts会捕获流成员Flags为1024的异常
        }
        for (auto &signal : streamInfo.ringDeviceSignal) {
            signal = nullptr;
        }
        for (auto &signal : streamInfo.ringDeviceSignalAux) {
            signal = nullptr;
        }

        u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
        std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
        CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY));
        for (u32 i = 0; i < resNum; i++) {
            streamInfo.ringDeviceSignal[i] = notifys[2 * i];
            streamInfo.ringDeviceSignalAux[i] = notifys[2 * i + 1];
        }
    }

    if (streamInfo.ringDeviceStreams.empty() || streamInfo.ringDeviceStreams.size() < rankSize) {
        streamInfo.ringDeviceStreams.resize(rankSize);
        for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
            streamInfo.ringDeviceStreams[ringIndex] = auxRingCommStreamsDev_[ringIndex];
            CHK_SMART_PTR_NULL(streamInfo.ringDeviceStreams[ringIndex]);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommAndStreamRes(const std::string &tag, Stream &stream)
{
    CHK_SMART_PTR_NULL(implAlg_);
    void *commInputPtr = nullptr;
    void *commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    HcclResult ret = CreateCommCCLbuffer();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclImplBase][CreateCommAndStreamRes]errNo[0x%016llx],create cclbuff failed",
            HCCL_ERROR_CODE(ret)), ret);

    ret = CreateCommExpBuffer();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclImplBase][CreateCommAndStreamRes]errNo[0x%016llx],create expbuff failed",
            HCCL_ERROR_CODE(ret)), ret);

    if (isA2MC2MultiServer_) {
        // 该场景下ccl buffer有一块区域在上层会被用作flag区，因此需要先清理一下
        CHK_RET(cclBufferManager_.CleanCCLbuffer());
    }

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    DeviceMem expMem = cclBufferManager_.GetCommExpBuffer();
    DeviceMem inputMem = DeviceMem::create(commInputPtr, commInputSize);
    DeviceMem outputMem = DeviceMem::create(commOutputPtr, commOutputSize);
    AlgType algType;
    AlgType algTypeTmp;

    CHK_RET(GetAlgType(algType, HcclCMDType::HCCL_CMD_ALL));
    algTypeTmp = algType;

    CHK_RET(notifyPool_->RegisterOp(tag));

    // 根据tag创建comm和流资源
    if (!(IsExistCommRes(tag))) {
        std::unique_ptr<CommInfo> commInfo = nullptr;
        HcclResult ret = implAlg_->CreateComm(tag, inputMem, outputMem, algType, commInfo,
                                              INVALID_VALUE_RANKID, false, true);

        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create comm failed",
                HCCL_ERROR_CODE(ret),
                tag.c_str()),
            ret);

        CHK_RET(ReplaceCommInfoByTag(tag, commInfo));
    }

    if (!(IsExistMutiStreamRes(tag))) {
        level1StreamInfo_t streamInfo;
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        // 2p场景下，mc2当前algType为518，streamInfo.ringNum走默认流程值为1导致资源申请不足，910_93 mc2固定在节点内默认用mesh
        constexpr u32 RANK_SIZE_TWO = 2;
        if ((GetRankSize() == RANK_SIZE_TWO && !isA2MC2MultiServer_) || (deviceType_ == DevType::DEV_TYPE_910_93)) {
            algTypeTmp.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
            algTypeTmp.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        }
        HcclResult ret = HCCL_SUCCESS;
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            ret = CreateMutiStreamResFor310P(tag, streamInfo);
        } else {
            ret = implAlg_->CreateMutiStreamRes(tag, stream, streamInfo, algTypeTmp, true);
        }
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create stream "
                       "resource",
                HCCL_ERROR_CODE(ret),
                tag.c_str()),
            ret);
        tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(streamInfo)));
        opRetryStreamPtr_->insert(std::make_pair(tag, tagStreamInfo_[tag].ringDeviceStreams));
        mutiStreamLock.unlock();
    }

    HCCL_INFO("resource creation (allreduce) success, tag[%s]", tag.c_str());
    CHK_RET(notifyPool_->UnregisterOp(tag));
    CHK_RET(RegisterToHeartBeat());

    CommBase *comm = nullptr;
    CHK_RET(GetComm(tag, &comm));
    if (comm == nullptr) {
        HCCL_ERROR("comm get err, comm %p", comm);
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommResource(commInputSize, commInputPtr, commOutputPtr, expMem.ptr(),
                            comm, tagStreamInfo_[tag], stream));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetComm(const std::string &tag, CommBase **comm)
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        *comm = tagCommInfo_[tag].commIntraServer.get();
    } else if (isA2MC2MultiServer_) {
        // 使用打平RDMA Mesh子通信域
        *comm = tagCommInfo_[tag].commLevel1Rdma[0].get();
    } else {
        *comm = tagCommInfo_[tag].commLevel0[0].get();
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetCommResource(u64 commBufferSize, void *commInPtr, void *commOutPtr, void *commExpPtr,
    CommBase *comm, level1StreamInfo_t &streamInfo, Stream &stream)
{
    u32 rankSize = comm->RankSize();
    u32 curRankId = comm->Rank();
    u32 usrRankId = comm->UserRank();
    combinOpara_.rankId = curRankId;
    combinOpara_.signalInfo.aicpuNotify.rankId = curRankId;
    combinOpara_.rankNum = rankSize;
    combinOpara_.winSize = commBufferSize;
    combinOpara_.winExpSize = EXP_BUFFER_SIZE;
    combinOpara_.config.deterministic = GetDeterministicConfig();
    combinOpara_.config.notifyWaitTime =
        (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET) ?
            GetExternalInputHcclExecTimeOut() : NOTIFY_DEFAULT_WAIT_TIME;
            hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
    combinOpara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    hcclMc2Info_.rankSize = rankSize;
    hcclMc2Info_.rankId = curRankId;
    hcclMc2Info_.usrRankId = usrRankId;
    hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(stream.id());
    hcclMc2Info_.commStreamSize = rankSize;
    hcclMc2Info_.reserve = 0;
    rtEvent_t event = nullptr;
    u32 eventId = 0;
    u32 idx = 0;
    u32 txSigleBase = 2;
    u32 rxSigleBase = 3;

    if (isA2MC2MultiServer_) {
        // MoE融合算子优化，MC2多机场景
        // 判断是否支持NormalQP创建，若不支持，需要额外下发敲Doorbell任务
        bool isSupportNormalQP = false;
        CHK_RET(IsSupportAicpuNormalQP(devicePhyId_, isSupportNormalQP));
        CHK_RET(SetDevIbverbsData(comm, isSupportNormalQP, commBufferSize, commInPtr, commOutPtr));

        // 非NormalQP场景需要传一条流，用于敲Doorbell
        combinOpara_.streamInfo[0].streamIds = streamInfo.ringDeviceStreams[0].id();
        combinOpara_.streamInfo[0].sqIds = streamInfo.ringDeviceStreams[0].sqId();
        combinOpara_.streamInfo[0].cqIds = streamInfo.ringDeviceStreams[0].cqId();
        combinOpara_.streamInfo[0].logicCqids = streamInfo.ringDeviceStreams[0].logicCqId();
        HCCL_DEBUG("[SetCommResource] Set streamInfo[0].streamIds[%u].sqIds[%u].cqIds[%u].logicCqids[%u]",
            combinOpara_.streamInfo[0].streamIds,
            combinOpara_.streamInfo[0].sqIds,
            combinOpara_.streamInfo[0].cqIds,
            combinOpara_.streamInfo[0].logicCqids);
    } else {
        for (u32 i = 0; i < rankSize; i++) {
            if (i != curRankId) {
                void* bufferIn;
                void* bufferOut;
                std::vector<void *> remotePtrVec;
            CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(UserMemType::INPUT_MEM, &bufferIn));
                combinOpara_.windowsIn[i] = reinterpret_cast<u64>(bufferIn);

                CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(UserMemType::OUTPUT_MEM, &bufferOut));
                combinOpara_.windowsOut[i] = reinterpret_cast<u64>(bufferOut);

            CHK_RET(comm->GetTransportByRank(i)->GetRemoteMem(&remotePtrVec));
            if (remotePtrVec.size() != 0) {
                combinOpara_.windowsExp[i] = reinterpret_cast<u64>(remotePtrVec[0]);
            }

                CHK_RET(comm->GetTransportByRank(i)-> \
                    GetTxAckDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i]));
                CHK_RET(comm->GetTransportByRank(i)-> \
                    GetRxAckDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize]));
                CHK_RET(comm->GetTransportByRank(i)-> \
                    GetTxDataSigleDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase]));
                CHK_RET(comm->GetTransportByRank(i)-> \
                    GetRxDataSigleDevNotifyInfo(combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase]));
                CHK_RET(GetAiCpuNotifyData(streamInfo.ringDeviceSignalAux[idx],
                    combinOpara_.signalInfo.noIpcNotifys[i]));

                CHK_RET(GetAiCpuNotifyData(streamInfo.ringDeviceSignal[idx],
                    combinOpara_.signalInfo.noIpcNotifys[i + rankSize]));
                idx++;
            } else {
                combinOpara_.windowsIn[i] = reinterpret_cast<u64>(commInPtr);
                combinOpara_.windowsOut[i] = reinterpret_cast<u64>(commOutPtr);
                combinOpara_.windowsExp[i] = reinterpret_cast<u64>(commExpPtr);

                // 在与aicpu商议后，本卡不再防止无效值。后续代码要删掉
                combinOpara_.signalInfo.ipcNotifys[i].resId = INVALID_U64;
                combinOpara_.signalInfo.ipcNotifys[i + rankSize].resId = INVALID_U64;
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId = INVALID_U64;
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId = INVALID_U64;
            }
        HCCL_INFO("group[%s] successfully set windowsIn & windowsOut & windowsExp info: userRank[%u], groupRank[%u], "\
                  "windowsIn[0x%llx], InSize[0x%llx], windowOut[0x%llx], OutSize[0x%llx], windowExp[0x%llx], ExpSize[0x%llu]",
                  identifier_.c_str(), GetUserRank(), GetGroupRank(),
                  combinOpara_.windowsIn[i], cclBufferManager_.GetInCCLbufferSize(),
                  combinOpara_.windowsOut[i], cclBufferManager_.GetOutCCLbufferSize(),
                  combinOpara_.windowsExp[i], cclBufferManager_.GetExpBufferSize());

            combinOpara_.signalInfo.ipcNotifys[i].rankId = i;
            combinOpara_.signalInfo.ipcNotifys[i + rankSize].rankId = i;
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId = i;
            combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId = i;
            combinOpara_.signalInfo.noIpcNotifys[i].rankId = i;

            hcclMc2Info_.commStreamIds[i] = streamInfo.ringDeviceStreams[i].id();
            combinOpara_.streamInfo[i].streamIds = streamInfo.ringDeviceStreams[i].id();
            combinOpara_.streamInfo[i].sqIds = streamInfo.ringDeviceStreams[i].sqId();
            combinOpara_.streamInfo[i].cqIds = streamInfo.ringDeviceStreams[i].cqId();
            combinOpara_.streamInfo[i].logicCqids = streamInfo.ringDeviceStreams[i].logicCqId();
            HCCL_DEBUG("[hccl_Mc2_Info] commStreamIds[%u]:[%u]", i, streamInfo.ringDeviceStreams[i].id());

            CHK_RET(hrtEventCreateWithFlag(&event));

            CHK_RET(hrtGetEventID(event, &eventId));
            aiCpuNoIpcEvnet_.push_back(event);
            combinOpara_.signalInfo.noIpcEvents[i].resId = eventId;
            HCCL_DEBUG("SetCommResource ipc notify info pre record local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.",
                curRankId, combinOpara_.signalInfo.ipcNotifys[i].rankId, combinOpara_.signalInfo.ipcNotifys[i].resId,
                combinOpara_.signalInfo.ipcNotifys[i].devId, combinOpara_.signalInfo.ipcNotifys[i].tsId,
                combinOpara_.signalInfo.ipcNotifys[i].addr);
            HCCL_DEBUG("SetCommResource ipc notify info pre wait local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.", curRankId, combinOpara_.signalInfo.ipcNotifys[i + rankSize].rankId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize].resId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize].devId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize].tsId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize].addr);
            HCCL_DEBUG("SetCommResource ipc notify info post record local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.", curRankId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].devId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].tsId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * txSigleBase].addr);
            HCCL_DEBUG("SetCommResource ipc notify info post wait local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.", curRankId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].devId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].tsId,
                combinOpara_.signalInfo.ipcNotifys[i + rankSize * rxSigleBase].addr);
        }
    }
    HCCL_DEBUG("[hccl_Mc2_Info] groupname:[%s][%llu], rankSize[%u], rankId[%u], usrRankId[%u], aicpuKfcStreamId[%u], "
        "commStreamSize[%u]", identifier_.c_str(), hcclMc2Info_.groupName, rankSize, curRankId, usrRankId,
        static_cast<uint32_t>(stream.id()), rankSize);
    CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_,
        sizeof(hcclMc2Info_)));
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseCommContextbuffer()
{
    commContext_.free();
}

HcclResult HcclCommunicator::CreateDeviceCommContext(u64 size, DeviceMem &buffer) const
{
    CHK_PRT_RET(!size, HCCL_INFO("[Create][DeviceCommContext]device commContext size is zero. "\
        "not need to malloc memory"), HCCL_SUCCESS);

    CHK_PRT_RET((size > ULONG_MAX), \
        HCCL_ERROR("[Create][DeviceCommContext]device commContext size %llu is large than ULONG_MAX",
            size), HCCL_E_PARA);

    if (!buffer.ptr()) {
        u64 memSize = size;
        buffer = DeviceMem::alloc(memSize);
        CHK_PRT_RET(size && !buffer, HCCL_ERROR("[Create][DeviceCommContext]Create device commContext size[%llu] fail,"\
            "please check deviceCommContext size.", size), HCCL_E_PTR);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SaveTraceInfo(std::string &logInfo)
{
    opBaseAtraceInfo_->SaveTraceInfo(logInfo, AtraceOption::Opbasekey);
    return HCCL_SUCCESS;
}

void HcclCommunicator::Break()
{
    if (implAlg_ != nullptr) {
        implAlg_->Break();
    }
    return;
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
    u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize)
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        RPT_ENV_ERR(true, "EI0001", vector<string>({"env", "tips"}),
            vector<string>({ "310P", std::string(__func__) + " is not supported"}));
        HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(implAlg_);
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(HcclCMDType::HCCL_CMD_ALLTOALLV);
    AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator *>(algOperator.get());
    CHK_PTR_NULL(alltoAllOperator);

    OpParam opParam;
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = static_cast<void *>(sendCounts);
    opParam.All2AllDataDes.recvCounts = static_cast<void *>(recvCounts);
    opParam.All2AllDataDes.sdispls = static_cast<void *>(sdispls);
    opParam.All2AllDataDes.rdispls = static_cast<void *>(rdispls);
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = false;

    if (alltoAllOperator->IsSatisfyAlltoAllAivCondition(opParam) ||
        alltoAllOperator->IsSatisfy91093OffloadCondition()) {
        memSize = 0;
        HCCL_INFO("Calculate workSpace MemSize for aiv alltoall done, memSize[%llu]", memSize);
        return HCCL_SUCCESS;
    }

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    CHK_SMART_PTR_NULL(preMetaInfo);

    CHK_RET(alltoAllOperator->PrepareAlltoAllAddrInfo(opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.sdispls,
            opParam.All2AllDataDes.sendType, opParam.All2AllDataDes.recvCounts, opParam.All2AllDataDes.rdispls,
            opParam.All2AllDataDes.recvType, preMetaInfo));

    preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));

    return alltoAllOperator->GetAlltoAllStagedWorkSpaceMemSize(opParam, memSize);
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo, u64 &memSize)
{
    CHK_PRT_RET(Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!"), HCCL_E_NOT_SUPPORT);

    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
}

HcclResult HcclCommunicator::GetAllReduceScratchSize(
    const u32 count, const HcclDataType dataType, u64 &scratchSize) const
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAllReduceScratchSize(count, dataType, scratchSize);
}
std::unordered_map<std::string, std::map<u32, HcclIpAddress>> HcclCommunicator::GetPhyIdNicInfo()
{
    return rankDevicePhyIdNicInfoMap_;
}

vector<u32> HcclCommunicator::GetRanksPort()
{
    return nicRanksPort_;
}

vector<RankInfo> HcclCommunicator::GetRanksList()
{
    return rankInfoList_;
}

HcclResult HcclCommunicator::SetWorldGroupInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
    vector<RankInfo> worldRankInfoList, vector<u32> &nicRanksPort, vector<u32> &vnicRanksPort)
{
    for (auto &ipInfo : phyIdNicInfoMap) {
        for (auto &devInfo : ipInfo.second) {
            rankDevicePhyIdNicInfoMap_[ipInfo.first][devInfo.first] = devInfo.second;
            HCCL_DEBUG("phyIdNicInfoMap print hostIp[%s] devId[%u] devIp[%s]",
                ipInfo.first.c_str(), devInfo.first, devInfo.second.GetReadableAddress());
        }
    }

    for (auto &rankInfo : worldRankInfoList) {
        worldRankInfoList_.push_back(rankInfo);
    }

    for (auto &port : nicRanksPort) {
        nicRanksPort_.push_back(port);
        HCCL_DEBUG("nicRanksPort port[%u]", port);
    }
    for (auto &port : vnicRanksPort) {
        vnicRanksPort_.push_back(port);
        HCCL_DEBUG("vnicRanksPort port[%u]", port);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize)
{
    if (topoSize < static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_MAX)) {
        HCCL_ERROR("topoDescs size is not enough, please check topoSize[%u]", topoSize);
        return HCCL_E_PARA;
    }

    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_SWITCH | HCCL_ALG_RING;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = HCCL_ALG_RING;
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_MESH;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_RING;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
    }

    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].rankSize = userRankSize_;
    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].rankSize = 0;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable)
{
    u32 curRankId = params.rank;
    u32 curRankPort = HCCL_INVALID_PORT;
    // 获取当前rank的device port
    for (auto &rankInfo : rankTable.rankList) {
        if (rankInfo.rankId == curRankId) {
            curRankPort = rankInfo.deviceInfo.port;
        }
    }
    // 判断当前device是否已经监听
    if (netDevCtxMap_.find(HcclIpAddress(devicePhyId_)) != netDevCtxMap_.end()) {
        // 先将停止监听错误的port
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[HcclIpAddress(devicePhyId_)], localVnicListenPort_));
        // 将真正的端口号监听
        CHK_RET(socketManager_->ServerInit(netDevCtxMap_[HcclIpAddress(devicePhyId_)], curRankPort));
    }
    return HCCL_SUCCESS;
}

std::string HcclCommunicator::GetUniqueId(void)
{
    static std::atomic<u32> idCounter(0);

    std::string uniqueId("");
    uniqueId += std::to_string(SalGetPid());
    uniqueId += '-';
    uniqueId += std::to_string(idCounter.fetch_add(1));
    uniqueId += '-';
    uniqueId += std::to_string(SalGetSysTime());

    return uniqueId;
}

u8 HcclCommunicator::GetDeterministicConfig() const
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetDeterministicConfig();
}

HcclResult HcclCommunicator::SetDeterministicConfig(const u8 deterministic)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAivModeConfig(const bool aivMode)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetAivModeConfig(aivMode));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuUnfoldConfig(const bool aicpuUnfold)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetAicpuUnfoldConfig(aicpuUnfold));
    return HCCL_SUCCESS;
}

void HcclCommunicator::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
{
    transportManager_->SetQpQosAttr(trafficClass, serviceLevel);
}

HcclResult HcclCommunicator::MigrateLinkToStopOrResume(LINK &link, bool isStop)
{
    if (isStop) {
        return link->Stop();
    }
    return link->Resume();
}

HcclResult HcclCommunicator::MigrateLinkVectorToStopOrResume(const std::vector<LINK> &links, bool isStop)
{
    for (auto it : links) {
        if (it) {
            CHK_RET(MigrateLinkToStopOrResume(it, isStop));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseLinkVector(std::vector<std::unique_ptr<CommBase> > &commBaseVector, bool isStop)
{
    for (unsigned int i = 0; i < commBaseVector.size(); i++) {
        auto commBase = commBaseVector[i].get();
        if(commBase == nullptr) {
            continue;
        }
        const std::vector<LINK> &ret = commBase->TransportInfo();
        CHK_RET(MigrateLinkVectorToStopOrResume(ret, isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseSingleSubCommTransport(SingleSubCommTransport &commTransport, bool isStop)
{
    for (unsigned int i = 0; i < commTransport.transportRequests.size(); i++) {
        if (!commTransport.transportRequests[i].isValid) {
            continue;
        }
        if (commTransport.links[i] == nullptr) {
            continue;
        }

        if (isStop) {
            CHK_RET(commTransport.links[i]->Stop());
        } else {
            CHK_RET(commTransport.links[i]->Resume());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseLevelNSubCommTransport(LevelNSubCommTransport &levelNSubCommTransport, bool isStop)
{
    for (unsigned int jj = 0; jj < levelNSubCommTransport.size(); jj++) {
        CHK_RET(TraverseSingleSubCommTransport(levelNSubCommTransport[jj], isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseOpCommTransport(OpCommTransport &opCommTransport, bool isStop)
{
    for (unsigned int ii = 0; ii < opCommTransport.size(); ii++) {
        CHK_RET(TraverseLevelNSubCommTransport(opCommTransport[ii], isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseAlgResourceResponse(bool isStop)
{
    for (auto &it : resMap_) {
        CHK_RET(TraverseOpCommTransport(it.second.opTransportResponse, isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ResetNotify()
{
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    notifyPool_->ResetNotify();
    queueNotifyManagerRefac_->ResetNotify();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ResetNotifyForDestRank(s64 destRank)
{
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    notifyPool_->ResetNotifyForDestRank(destRank);
    return HCCL_SUCCESS;
}

void HcclCommunicator::InsertNewTagToTagMap(std::string &newTag, std::string &tag)
{
    const auto &mapIt = newTagToTagMap_.find(newTag);
    if (mapIt == newTagToTagMap_.end()) {
        newTagToTagMap_.insert({newTag, tag});
    } else {
        mapIt->second = tag;
    }
    return ;
}

HcclResult HcclCommunicator::GetTagFromNewTag(const std::string &newTag, std::string &tag)
{
    const auto &mapIt = newTagToTagMap_.find(newTag);
    if (mapIt == newTagToTagMap_.end()) {
        HCCL_ERROR("[OpRetry]newTag[%s] is not in newTagToTagMap_", newTag.c_str());
        return HCCL_E_INTERNAL;
    } else {
        tag = mapIt->second;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetSignalTransport(SingleSubCommTransport &singleSubCommTransport,
    u32 linkIdx, bool statusStop)
{
    RankId loc = singleSubCommTransport.transportRequests[linkIdx].localUserRank;
    RankId rmt = singleSubCommTransport.transportRequests[linkIdx].remoteUserRank;
    if (statusStop) {
        if (singleSubCommTransport.links[linkIdx]->GetLinkType() == LinkType::LINK_ROCE) {
            CHK_RET(singleSubCommTransport.links[linkIdx]->Stop());
            singleSubCommTransport.status[linkIdx] = TransportStatus::STOP;
            HCCL_INFO("[SetTransportStatus]set transport status to stop, loc[%u], rmt[%u]", loc, rmt);
        }
    } else {
        if (singleSubCommTransport.status[linkIdx] == TransportStatus::STOP) {
            HCCL_INFO("[SetTransportStatus]set transport status to resume, loc[%u], rmt[%u]", loc, rmt);
                CHK_RET(singleSubCommTransport.links[linkIdx]->DeInit());
                singleSubCommTransport.links[linkIdx] = nullptr;  // 赋值为nullptr, 供后面重新建链
                singleSubCommTransport.status[linkIdx] = TransportStatus::INIT;
            }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetBsrTransportStatusImpl(OpCommTransport &opCommTransport, bool statusStop,
    const HcclOpIdentifier &opId, u32 remoteRank)
{
    u32 commIndex = 0;
    if ((userRank_ == opId.detRank && remoteRank > userRank_) ||
        (userRank_ == opId.srcRank && remoteRank < userRank_)){
        commIndex = COMM_INDEX_0;
    } else {
        commIndex = COMM_INDEX_1;
    }
    CHK_PRT_RET(commIndex >= opCommTransport[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR("[SetBsrTransportStatusImpl] batchsendrecv op commIndex[%u] is larger than "\
        "opTransportResponse size[%zu]",
        remoteRank, opCommTransport[COMM_COMBINE_ORDER].size()), HCCL_E_PARA);
    SingleSubCommTransport &commCombined =
    const_cast<SingleSubCommTransport&>(opCommTransport[COMM_COMBINE_ORDER][commIndex]);
    u32 Rank = commCombined.userRank2subCommRank[remoteRank];
    CHK_PRT_RET(Rank >= commCombined.links.size(),
        HCCL_ERROR("[SetBsrTransportStatusImpl] batchsendrecv op remoteRank[%u], get Rank[%u]," \
        "the size of combinedComm links is [%zu]", remoteRank, Rank, commCombined.links.size()),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(commCombined.links[Rank]);

    RankId loc = commCombined.transportRequests[Rank].localUserRank;
    RankId rmt = commCombined.transportRequests[Rank].remoteUserRank;
    if (!commCombined.transportRequests[Rank].isValid){
        return HCCL_SUCCESS;
    }
    if (statusStop) {
        if (commCombined.links[Rank]->GetLinkType() == LinkType::LINK_ROCE) {
            CHK_RET(commCombined.links[Rank]->Stop());
            commCombined.status[Rank] = TransportStatus::STOP;
            HCCL_INFO("[SetBsrTransportStatusImpl]set bsr transport status to stop, comindex[%u] loc[%u], rmt[%u]",
                commIndex, loc, rmt);
        }
    } else {
        if (commCombined.status[Rank] == TransportStatus::STOP) {
            HCCL_INFO("[SetBsrTransportStatusImpl]set bsr transport status to resume, comindex[%u] loc[%u], rmt[%u]",
                commIndex, loc, rmt);
                CHK_RET(commCombined.links[Rank]->DeInit());
                commCombined.links[Rank] = nullptr;  // 赋值为nullptr, 供后面重新建链
                commCombined.status[Rank] = TransportStatus::INIT;
            }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetBsrTransportStatusImplforchange(OpCommTransport &opCommTransport, 
    const HcclOpIdentifier &opId, u32 remoteRank, const std::map<u32, bool> &remoteRankPortMap, bool isUseDefault, 
    const std::map<u32, bool> &isChangeLinkMap, bool isCurTag)
{
    bool isPortSatisfy = (remoteRankPortMap.find(remoteRank) != remoteRankPortMap.end() &&
                    remoteRankPortMap.find(remoteRank)->second == isUseDefault);
    bool isChangeLink = (isChangeLinkMap.find(remoteRank) != isChangeLinkMap.end() &&
                            isChangeLinkMap.find(remoteRank)->second);
    HCCL_INFO("[SetBsrTransportStatusImplforchange]remoteRank[%u], isUseDefault[%d], "
        "isPortSatisfy[%d], isChangeLink[%d], isCurTag[%d]",
        remoteRank, isUseDefault, isPortSatisfy, isChangeLink, isCurTag);
    if (!isPortSatisfy || !(isChangeLink || isCurTag)){
        return HCCL_SUCCESS;
    }

    u32 commIndex = 0;
    if ((userRank_ == opId.detRank && remoteRank > userRank_) ||
        (userRank_ == opId.srcRank && remoteRank < userRank_)){
        commIndex = COMM_INDEX_0;
    } else {
        commIndex = COMM_INDEX_1;
    }
    CHK_PRT_RET(commIndex >= opCommTransport[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR("[SetBsrTransportStatusImplforchange] batchsendrecv op commIndex[%u] is larger than "\
        "opTransportResponse size[%zu]",
        commIndex, opCommTransport[COMM_COMBINE_ORDER].size()), HCCL_E_PARA);
    SingleSubCommTransport &commCombined =
        static_cast<SingleSubCommTransport&>(opCommTransport[COMM_COMBINE_ORDER][commIndex]);
    u32 rank = commCombined.userRank2subCommRank[remoteRank];
    CHK_PRT_RET(rank >= commCombined.links.size(),
        HCCL_ERROR("[SetBsrTransportStatusImplforchange] batchsendrecv op remoteRank[%u], get Rank[%u]," \
        "the size of combinedComm links is [%zu]", remoteRank, rank, commCombined.links.size()),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(commCombined.links[rank]);

    RankId loc = commCombined.transportRequests[rank].localUserRank;
    RankId rmt = commCombined.transportRequests[rank].remoteUserRank;
    if (!commCombined.transportRequests[rank].isValid){
        return HCCL_SUCCESS;
    }

    if (commCombined.status[rank] == TransportStatus::STOP) {
        HCCL_INFO("[SetBsrTransportStatusImplforchange]set bsr transport status to resume, comindex[%u] loc[%u], rmt[%u]",
            commIndex, loc, rmt);
            CHK_RET(commCombined.links[rank]->DeInit());
            commCombined.links[rank] = nullptr;  // 赋值为nullptr, 供后面重新建链
            commCombined.status[rank] = TransportStatus::INIT;
        }
    return HCCL_SUCCESS;
}


HcclResult HcclCommunicator::SetTransportStatusImpl(OpCommTransport &opCommTransport, bool statusStop,
    const HcclOpIdentifier &opId, u32 remoteRank, const std::map<u32, bool> &remoteRankPortMap, bool isUseDefault)
{
    bool isSendRecv = opId.isSendRecv;
    
    // stop阶段及原地重执行的resume阶段
    //bsr判断当前故障的send、recv是否remoterank是否相同的情况，如果是相同只操作故障op，如果不同都操作
    u32 sendRemoteRank = userRank_ == opId.bsrInfo[HCCL_SEND].detRank ? opId.bsrInfo[HCCL_SEND].srcRank : opId.bsrInfo[HCCL_SEND].detRank;
    u32 recvRemoteRank = userRank_ == opId.bsrInfo[HCCL_RECV].detRank ? opId.bsrInfo[HCCL_RECV].srcRank : opId.bsrInfo[HCCL_RECV].detRank;
    bool isBsrPortSatisfy = (remoteRankPortMap.find(remoteRank) != remoteRankPortMap.end() &&
    	remoteRankPortMap.find(remoteRank)->second == isUseDefault);
    if (opId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && sendRemoteRank == recvRemoteRank && isBsrPortSatisfy){
        CHK_RET(SetBsrTransportStatusImpl(opCommTransport, statusStop, opId, remoteRank));
        return HCCL_SUCCESS;
    }

    for (auto &levelNSubCommTransport: opCommTransport) {
        for (auto &singleSubCommTransport: levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                u32 transportRemoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
                bool isValid = singleSubCommTransport.transportRequests[i].isValid;
                bool isRankSatisfy = ((!isSendRecv ) || (isSendRecv && remoteRank == transportRemoteRank));
                // isPortSatisfy表示当前对端使用的主备网口是否和changeLinkInfo一致
                bool isPortSatisfy = (remoteRankPortMap.find(transportRemoteRank) != remoteRankPortMap.end() &&
                        remoteRankPortMap.find(transportRemoteRank)->second == isUseDefault);
                HCCL_INFO("[SetTransportStatus]remoteRank[%u], isUseDefault[%d], isValid[%d], isRankSatisfy[%d], isPortSatisfy[%d]",
                    transportRemoteRank, isUseDefault, isValid, isRankSatisfy, isPortSatisfy);
                if (isValid && isRankSatisfy && isPortSatisfy) {
                    CHK_RET(SetSignalTransport(singleSubCommTransport, i, statusStop));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetTransportStatusImplForChange(OpCommTransport &opCommTransport, const HcclOpIdentifier &opId,
    u32 remoteRank, const std::map<u32, bool> &remoteRankPortMap, bool isUseDefault, const std::map<u32, bool> &isChangeLinkMap,
    bool isCurTag)
{
    bool isSendRecv = opId.isSendRecv;
    
    //bsr判断当前故障的send、recv是否remoterank是否相同的情况，如果是相同只操作故障op，如果不同都操作
    u32 sendRemoteRank = userRank_ == opId.bsrInfo[HCCL_SEND].detRank ? opId.bsrInfo[HCCL_SEND].srcRank : opId.bsrInfo[HCCL_SEND].detRank;
    u32 recvRemoteRank = userRank_ == opId.bsrInfo[HCCL_RECV].detRank ? opId.bsrInfo[HCCL_RECV].srcRank : opId.bsrInfo[HCCL_RECV].detRank;
    if (opId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && sendRemoteRank == recvRemoteRank){
        CHK_RET(SetBsrTransportStatusImplforchange(opCommTransport, opId, remoteRank, remoteRankPortMap, isUseDefault,
            isChangeLinkMap, isCurTag));
        return HCCL_SUCCESS;
    }

    // 借轨的resume阶段
    for (auto &levelNSubCommTransport: opCommTransport) {
        for (auto &singleSubCommTransport: levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                u32 transportRemoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
                bool isValid = singleSubCommTransport.transportRequests[i].isValid;
                bool isRankSatisfy = (!isSendRecv || (isSendRecv && remoteRank == transportRemoteRank));
                // isPortSatisfy表示当前对端使用的主备网口是否和changeLinkInfo一致
                bool isPortSatisfy = (remoteRankPortMap.find(transportRemoteRank) != remoteRankPortMap.end() &&
                    remoteRankPortMap.find(transportRemoteRank)->second == isUseDefault);
                bool isChangeLink = (isChangeLinkMap.find(transportRemoteRank) != isChangeLinkMap.end() &&
                                     isChangeLinkMap.find(transportRemoteRank)->second);
                HCCL_INFO("[SetTransportStatus]remoteRank[%u], isUseDefault[%d], isValid[%d], isRankSatisfy[%d], "
                    "isPortSatisfy[%d], isChangeLink[%d], isCurTag[%d]",
                    transportRemoteRank, isUseDefault, isValid, isRankSatisfy, isPortSatisfy, isChangeLink, isCurTag);
                if (isValid && isRankSatisfy && isPortSatisfy && (isChangeLink || isCurTag)) {
                    CHK_RET(SetSignalTransport(singleSubCommTransport, i, false));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetTransportStatus(const HcclOpIdentifier &opId, bool statusStop,
    const std::map<u32, bool> &remoteRankPortMap, const std::map<u32, bool> &isChangeLinkMap, bool isChangeLinkFlag)
{
    std::string newTag(reinterpret_cast<const char *>(opId.newTag));
    u32 remoteRank = userRank_ == opId.detRank ? opId.srcRank : opId.detRank;

    if (resMap_.find(newTag) == resMap_.end()) {
        HCCL_ERROR("HcclCommunicator SetTransportStatus failed: newTag[%s] is not in resMap", newTag.c_str());
        return HCCL_E_INTERNAL;
    }

    if (statusStop) {
        CHK_RET(SetTransportStatusImpl(resMap_[newTag].opTransportResponse, statusStop, opId, remoteRank,
            remoteRankPortMap, true));
        CHK_RET(SetTransportStatusImpl(resMap_[newTag].opTransportResponseBackUp, statusStop, opId,
             remoteRank, remoteRankPortMap, false));
    } else {
        if (isChangeLinkFlag) {
            // 借轨场景
            for (auto &resMapIt: resMap_) {
                bool isCurTag = false;
                if (resMapIt.first == newTag) {
                    isCurTag = true;
                }
                if (hostResMap_.find(resMapIt.first) != hostResMap_.end()) {
                    // 若当前tag未进行aicpu展开，则不重新build资源
                    continue;
                }

                CHK_RET(SetTransportStatusImplForChange(resMapIt.second.opTransportResponse, opId, remoteRank,
                    remoteRankPortMap, true, isChangeLinkMap, isCurTag));
                CHK_RET(SetTransportStatusImplForChange(resMapIt.second.opTransportResponseBackUp, opId,
                    remoteRank, remoteRankPortMap, false, isChangeLinkMap, isCurTag));

                std::string tag;
                CHK_RET(GetTagFromNewTag(resMapIt.first, tag));
                CHK_RET(ReAllocTransports(tag, resMapIt.first));
                CHK_RET(BuildOpRemoteResParam(resMapIt.second, resMapIt.first, opId.opType, true));
                HCCL_RUN_INFO("[%s]success to set status of [%s] resume", __func__, resMapIt.first.c_str());
            }
            CHK_RET(CopyHostOpResToDeviceParam(newTag));
        } else {
            // 原地重执行
            CHK_RET(SetTransportStatusImpl(resMap_[newTag].opTransportResponse, statusStop, opId,
                remoteRank, remoteRankPortMap, true));
            CHK_RET(SetTransportStatusImpl(resMap_[newTag].opTransportResponseBackUp, statusStop, opId,
                 remoteRank, remoteRankPortMap, false));
            std::string tag(reinterpret_cast<const char *>(opId.tag));
            CHK_RET(ReAllocTransports(tag, newTag));
            CHK_RET(BuildOpRemoteResParam(resMap_[newTag], newTag, opId.opType, true));
            CHK_RET(CopyHostOpResToDeviceParam(newTag));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReAllocTransports(const std::string &tag, const std::string &newTag)
{
    HcclResult ret = HCCL_SUCCESS;

    AlgResourceResponse &algResResponse = resMap_[newTag];
    DeviceMem expMem = cclBufferManager_.GetCommExpBuffer();

    TransportIOMem transMem{algResResponse.cclInputMem, algResResponse.cclOutputMem,
        algResResponse.paramInputMem, algResResponse.paramOutputMem, algResResponse.scratchMem,
        algResResponse.aivInputMem, algResResponse.aivOutputMem, expMem};

    {
        // Transport资源 重建链, 一定是AICPU展开，所以 isAicpuModeEn=true
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(tag, transMem, algResResponse.opTransportResponse, true);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Realloc transports failed, tag[%s]", __func__, newTag.c_str()),
                    ret);
    }

    if (IsEnableBackupLink()) {
        // 超节点 && level2支持重执行 && Aicpu：备用Transport资源 重建链
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(tag, transMem, algResResponse.opTransportResponseBackUp, true, true);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[%s]Alloc backup transports failed, tag[%s]", __func__, newTag.c_str()), ret);
    }
    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Stop()
{
    HcclUs startut = TIME_NOW();
    isSuspending = true;
    HCCL_DEBUG("HcclCommunicator Stop begin.");
    for (auto &it : tagCommInfo_) {
        CHK_RET(TraverseLinkVector(it.second.commLevel1, true));
        CHK_RET(TraverseLinkVector(it.second.commLevel0, true));
        CHK_RET(TraverseLinkVector(it.second.commLevel2, true));
        CHK_RET(TraverseLinkVector(it.second.commP2P, true));
        if(it.second.commIntraServer) {
            const std::vector<LINK> &ret = it.second.commIntraServer->TransportInfo();
            CHK_RET(MigrateLinkVectorToStopOrResume(ret, true));
        }
    }
    CHK_RET(TraverseAlgResourceResponse(true));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommunicator::Stop, Stop take time:[%lld]us",
        DURATION_US(endut - startut).count());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HostMC2EnvResume(){
    if (GetMC2EnvFlag()) {
        HCCL_DEBUG("[NsRecovery]reset the suspending flag");
        KfcExecControl controlCmd;
        controlCmd.kfcCmd = KfcCommand::kNone;
        controlCmd.bgCmd = BackgroundCommand::kNone;
        controlCmd.suspendingStatus = HcclComSuspendingFlag::isResume;
        return kfcControlTransferH2D_->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t *>(&controlCmd));
    } else {
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::Resume()
{
    HcclUs startut = TIME_NOW();
    HCCL_DEBUG("HcclCommunicator Resume begin.");
    for (auto &it : tagCommInfo_) {
        CHK_RET(TraverseLinkVector(it.second.commLevel1, false));
        CHK_RET(TraverseLinkVector(it.second.commLevel0, false));
        CHK_RET(TraverseLinkVector(it.second.commLevel2, false));
        CHK_RET(TraverseLinkVector(it.second.commP2P, false));
        if(it.second.commIntraServer) {
            const std::vector<LINK> &ret = it.second.commIntraServer->TransportInfo();
            CHK_RET(MigrateLinkVectorToStopOrResume(ret, false));
        }
    }
    CHK_RET(TraverseAlgResourceResponse(false));
    HcclUs cleanNotifyStart = TIME_NOW();
    CHK_RET(hrtResourceClean(deviceLogicId_, RT_NOTIFY_ID));
    HcclUs cleanNotifyEnd = TIME_NOW();
    HCCL_RUN_INFO("HcclCommunicator::Resume, hrtResourceClean notify take time:[%lld]us",
        DURATION_US(cleanNotifyEnd - cleanNotifyStart).count());
        CHK_RET(HostMC2EnvResume());
    isSuspending = false;
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommunicator::Resume, Resume take time:[%lld]us",
        DURATION_US(endut - startut).count());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckSuspendingStatus()
{
    if (isSuspending) {
        return HCCL_E_SUSPENDING;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetMemoryRange(void *baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    CHK_PRT_RET(serverNum_ != 1, HCCL_ERROR("[HcclCommunicator][SetMemoryRange] serverNum[%u] not support zero copy",
        serverNum_), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(deviceType_ != DevType::DEV_TYPE_910_93,
        HCCL_ERROR("[HcclCommunicator][SetMemoryRange] deviceType[%d] not support zero copy", deviceType_), HCCL_E_NOT_SUPPORT);
    if (zeroCopyMemoryAgent_ == nullptr) {
        CHK_RET(InitZeroCopyMemoryAgent());
    }
    CHK_RET(zeroCopyMemoryAgent_->SetMemoryRange(baseVirPtr, size, alignment, flags));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnsetMemoryRange(void *baseVirPtr)
{
    CHK_PRT_RET(zeroCopyMemoryAgent_ == nullptr,
        HCCL_ERROR("[HcclCommunicator][UnsetMemoryRange] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
    CHK_RET(zeroCopyMemoryAgent_->UnsetMemoryRange(baseVirPtr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ActivateCommMemory(void *virPtr, size_t size, size_t offset, void* handle, uint64_t flags)
{
    CHK_PRT_RET(zeroCopyMemoryAgent_ == nullptr,
        HCCL_ERROR("[HcclCommunicator][ActivateCommMemory] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
    CHK_RET(zeroCopyMemoryAgent_->ActivateCommMemory(virPtr, size, offset, handle, flags));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeactivateCommMemory(void *virPtr)
{
    CHK_PRT_RET(zeroCopyMemoryAgent_ == nullptr,
        HCCL_ERROR("[HcclCommunicator][DeactivateCommMemory] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
    CHK_RET(zeroCopyMemoryAgent_->DeactivateCommMemory(virPtr));
    return HCCL_SUCCESS;
}

void HcclCommunicator::SaveLinkRes(const OpCommTransport &opTransportResponse)
{
    std::lock_guard<std::mutex> commLock(linkResMapMutex_);
    for (auto &opCommTransport : opTransportResponse) {
        for (auto &transports : opCommTransport) {
            for (u32 i = 0; i < transports.transportRequests.size(); i++) {
                if (transports.links[i] != nullptr &&
                    transports.links[i]->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                    linkResMap_.emplace(transports.links[i].get(),
                        std::make_pair(identifier_, transports.transportRequests[i].remoteUserRank));
                }
            }
        }
    }
    return;
}

HcclResult HcclCommunicator::GetTransportCqeErrors(const HcclNetDevCtx netDevCtx,
    std::vector<ErrCqeInfo> &infos, u32 &num)
{
    if (netDevCtx == nullptr || linkResMap_.empty()) {
        return HCCL_SUCCESS;
    }
    HcclIpAddress localIp;
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));

    u32 qpn = 0;
    std::vector<std::pair<Transport*, CqeInfo>> infolist;
    Transport::GetTransportErrorCqe(netDevCtx, infolist, num);
    std::lock_guard<std::mutex> commLock(linkResMapMutex_);
    for (auto &info : infolist) {
        auto iter = linkResMap_.find(info.first);
        if (iter != linkResMap_.end()) {
            CHK_RET((info.first)->GetTransportId(qpn));
            infos.push_back(ErrCqeInfo(info.second, iter->second.first, iter->second.second, qpn));
        } else {
            HCCL_RUN_WARNING("[GetTransportCqeErrors]get err failed, transport is not find, localIp[%s], remoteIp[%s]",
                localIp.GetReadableAddress(), info.second.remoteIp.GetReadableAddress());
        }
    }
    num = infos.size();
    return HCCL_SUCCESS;
}

void HcclOneSidedServiceCallbackInstall(HcclResult (*func)(std::unique_ptr<IHcclOneSidedService> &,
    std::unique_ptr<HcclSocketManager> &, std::unique_ptr<NotifyPool> &))
{
    g_hcclOneSidedServiceCallback = func;
}

void HcclOneSidedServiceCallbackUninstall()
{
    g_hcclOneSidedServiceCallback = nullptr;
}

void HcclCommunicator::ClearOpTransportResponseLinks(OpCommTransport &opTransportResponse)
{
    for (auto &levelNSubCommTransport : opTransportResponse) {
        for (auto &singleSubCommTransport : levelNSubCommTransport) {
            u32 size = singleSubCommTransport.transportRequests.size();
            singleSubCommTransport.links.resize(size, nullptr);
            singleSubCommTransport.status.resize(size, TransportStatus::INIT);
            HCCL_INFO("[%s] size[%u], linksSize[%d]", __func__, size, singleSubCommTransport.links.size());
        }
    }
}

HcclResult HcclCommunicator::SetDevIbverbsData(CommBase *comm, bool isSupportNormalQP, u64 commBufferSize,
    void *commInPtr, void *commOutPtr)
{
    const u32 curRankId = comm->Rank();
    const u32 rankSize = comm->RankSize();
    transDevIbverbsData_.resize(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        auto &data = transDevIbverbsData_[i];
        if (i != curRankId) {
            // 对端link的信息
            const auto transport = comm->GetTransportByRank(i);
            void* bufferIn = nullptr;
            void* bufferOut = nullptr;
            u32 remoteInMemKey = 0;
            u32 remoteOutMemKey = 0;
            CHK_RET(transport->GetRemoteMem(UserMemType::INPUT_MEM, &bufferIn));
            CHK_RET(transport->GetRemoteMem(UserMemType::OUTPUT_MEM, &bufferOut));
            data.remoteInputMem.addr = reinterpret_cast<uint64_t>(bufferIn);
            data.remoteOutputMem.addr = reinterpret_cast<uint64_t>(bufferOut);
            CHK_RET(transport-> \
                GetRemoteMemSize(UserMemType::INPUT_MEM, data.remoteInputMem.size));
            CHK_RET(transport-> \
                GetRemoteMemSize(UserMemType::OUTPUT_MEM, data.remoteOutputMem.size));
            // IBV链路需要的资源
            if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP){
                CHK_RET(transport->GetRemoteMemKey(UserMemType::INPUT_MEM, &remoteInMemKey));
                CHK_RET(transport->GetRemoteMemKey(UserMemType::OUTPUT_MEM, &remoteOutMemKey));
                data.remoteInputMem.key = remoteInMemKey;
                data.remoteOutputMem.key = remoteOutMemKey;
                CHK_RET(transport-> \
                    GetLocalMemDetails(UserMemType::INPUT_MEM, data.localInputMem));
                CHK_RET(transport-> \
                    GetLocalMemDetails(UserMemType::OUTPUT_MEM, data.localOutputMem));
                std::vector<HcclQpInfoV2> qpInfos;
                CHK_RET(transport->GetAiQpInfo(qpInfos));
                data.qpInfo = qpInfos[0];
            }
        } else {
            // 本rank的信息
            data.localInputMem.addr = reinterpret_cast<uint64_t>(commInPtr);
            data.localInputMem.size = commBufferSize;
            data.localOutputMem.addr = reinterpret_cast<uint64_t>(commOutPtr);
            data.localOutputMem.size = commBufferSize;
        }
        
        if (isSupportNormalQP) {
            data.qpMode = QPMode::NORMAL;
        }
	// Debugging info
        data.Print();
    }
    return HCCL_SUCCESS;
}
}
