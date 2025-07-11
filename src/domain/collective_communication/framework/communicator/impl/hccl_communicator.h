/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMUNICATOR_H
#define HCCL_COMMUNICATOR_H

#include <atomic>
#include <memory>
#include <hccl/hccl_types.h>
#include "hccl_communicator_attrs.h"
#include "hccl/base.h"
#include "hccl_impl_pub.h"
#include "opexecounter_pub.h"
#include "op_base_stream_manager_pub.h"
#include "offload_stream_manager_pub.h"
#include "profiler_manager.h"

#include "topoinfo_parse.h"
#include "hccl_alg.h"
#include "ccl_buffer_manager.h"
#include "hccl_trace_info.h"
#include "hccl_callback_task.h"
#include "aicpu_operator_pub.h"
#include "mr_manager.h"
#include "transport_heterog_def.h"
#include "resource_manager/queue_notify_manager.h"
#include "hccl_network_pub.h"
#include "comm.h"
#include "device_capacity.h"
#include "transport_manager.h"
#include "zero_copy/zero_copy_memory_agent.h"
#include "coll_alg_operator.h"
#include "alltoall_operator.h"
#include "peterson_lock.h"
#include "coll_alg_utils.h"
#include "heartbeat.h"
#include "i_hccl_one_sided_service.h"
#include "opretry_manager.h"

namespace hccl {
using ServRankInfo_t = std::map<std::string, std::vector<RankInfo_t> >;
constexpr u32 COMM_MAX_WORK_SPACE_SIZE = 16 * 1024 * 1024; // 默认16MB
constexpr u32 INPUT = 0;
constexpr u32 OUTPUT = 1;
const std::string COMM_LOOPBACK_IP = "127.0.0.1";
constexpr u8 INPLACE_PRESYNC_STATUS_SEVEN = 7;
struct RemoteRes {
    u64 inbufferSize;
    u64 outbufferSize;
    u64 inbuffer;
    u32 inbufferKey;
    u64 outbuffer;
    u32 outbufferKey;
};
#define HCCL_AICPU_HOST_BASE_TIME_MS 10*1000 // 10秒
struct AicpuOpTiling {
    std::string newTag;
    std::string algName;
    AlgType  algType;
    bool isUsedMainStream = false;
    u8 floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
    u8 dumpDebug = false;
};
using rankTagSignalInfo_t = std::unordered_map<u32, std::unordered_map<std::string, std::vector<HcclSignalInfo>>>;
using rankTagKey_t = std::unordered_map<u32, std::unordered_map<std::string, std::vector<u32>>>;
using rankTagAddr_t = std::unordered_map<u32, std::unordered_map<std::string, std::vector<u64>>>;
using rankTagChipId_t = std::unordered_map<u32, std::unordered_map<std::string, s64>>;
class HcclCommunicator {
public:
    explicit HcclCommunicator();

    virtual ~HcclCommunicator();

    virtual HcclResult Stop();
    virtual HcclResult Resume();
    HcclResult Suspend();
    HcclResult TraverseAlgResourceResponse(bool isStop);
    HcclResult TraverseOpCommTransport(OpCommTransport &opCommTransport, bool isStop);
    HcclResult TraverseLevelNSubCommTransport(LevelNSubCommTransport &levelNSubCommTransport, bool isStop);
    HcclResult TraverseSingleSubCommTransport(SingleSubCommTransport &commTransport, bool isStop);

    // 对外接口
    virtual HcclResult StopExec();
    virtual HcclResult Clean();
    virtual HcclResult Init(HcclCommParams &params, const RankTable_t &rankTable);
    virtual HcclResult Init(HcclCommParams &params, const std::vector<RankInfo> &rankList,
        WorldGroupInfo &globalData);

    virtual HcclResult GetAlgType(AlgType &algType, HcclCMDType opType);

    virtual HcclResult GetDeviceNumPerAggregation(u32 &deviceNumPerAggregation);

    virtual HcclResult GetBandWidthPerNPU(u32 level, float &bandWidth);

    u32 GetRankTableCrc();

    u32 GetServerNum();

    u32 GetModuleNum();

    HcclResult GetCommParams(HcclCommParams &params); // 逆向解析获取HcclCommParams参数

    HcclResult GetCommRankTable(RankTable_t &rankTable); // 逆向解析获取RankTable_t参数

    virtual HcclResult AllGather(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, HcclRtStream stream, HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult AllGatherV(const std::string &tag, const void *sendBuf, u64 sendCount, const void *recvBuf,
        const void *recvCounts, const void *rdispls, HcclDataType dataType, HcclRtStream stream);

    virtual HcclResult AllGatherOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 inputCount,
        HcclDataType dataType, HcclRtStream stream);

    virtual HcclResult AllGatherVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, 
        u64 inputCount, const void *outputCounts, const void *outputDispls, HcclDataType dataType, HcclRtStream stream);

    virtual HcclResult AllReduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
        SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE, const HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult AllReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
        SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);

    virtual HcclResult AlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls,
        HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
        rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAllVOutPlace(const void *sendBuf, const void *sendCounts, const void *sdispls,
        HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
        rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
        const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAllVCOutPlace(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType,
        const void *recvBuf, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    virtual HcclResult AlltoAll(const void *sendBuf, u64 sendCount, HcclDataType sendType,
        const void *recvBuf, u64 recvCount, HcclDataType recvType, rtStream_t stream, const std::string &tag);

    virtual HcclResult Broadcast(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        HcclRtStream stream);

    virtual HcclResult BroadcastOutPlace(const std::string &tag, void *ptr, u64 count, HcclDataType dataType, u32 root,
        HcclRtStream stream);

    virtual HcclResult Scatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, HcclRtStream stream);

    virtual HcclResult ScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 recvCount,
        HcclDataType dataType, u32 root, HcclRtStream stream);

    virtual HcclResult Reduce(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream);

    virtual HcclResult ReduceOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, u32 root, HcclRtStream stream);

    virtual HcclResult ReduceScatter(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);

    virtual HcclResult ReduceScatterV(const std::string &tag, void *inputPtr,
        const void *inputCounts, const void *inputDispls, void *outputPtr, u64 outputCount,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo *opInfo = nullptr);

    virtual HcclResult ReduceScatterVOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
        const void *inputCounts, const void *inputDispls, u64 outputCount, 
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);

    virtual HcclResult BatchSendRecv(const std::string &tag, HcclSendRecvItem* sendRecvItemsPtr,
        u32 itemNum, rtStream_t stream);

    virtual HcclResult Send(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, rtStream_t stream);

    virtual HcclResult SendOutPlace(const std::string &tag, void *inputPtr, u64 count, HcclDataType dataType,
        u32 destRank, rtStream_t stream);

    virtual HcclResult Receive(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, rtStream_t stream);

    virtual HcclResult ReceiveOutPlace(const std::string &tag, void *outputPtr, u64 count, HcclDataType dataType,
        u32 srcRank, rtStream_t stream);

    virtual HcclResult Gather(const std::string &tag, void *inputPtr, void *outputPtr, u32 rootRank, u64 inputCount,
        HcclDataType dataType, rtStream_t stream);

    virtual HcclResult SupportDeterministicOptim(bool &isDeterministicOptim);

    virtual HcclResult GetCqeError(HcclResult &result);

    //  对内接口
    virtual HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

    virtual HcclResult CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op);

    virtual HcclResult ReleaseCommInfos();

    virtual HcclResult GetAlltoAllStagedWorkSpaceMemSize(u64 *sendCounts, u64 *sdispls, HcclDataType sendType,
        u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize);

    virtual HcclResult GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        u64 &memSize);

    virtual HcclResult GetAllReduceScratchSize(const u32 count, const HcclDataType dataType, u64 &scratchSize) const;

    virtual bool IsStandardCard();

    virtual bool Is310PDuoCard();

    HcclResult CheckDeviceType(const DevType deviceType) const;

    HcclResult CheckReductionOp(const HcclReduceOp op) const;

    HcclResult CheckUserRank(const u32 userRank) const;

    HcclResult CheckCount(const u64 count) const;

    HcclResult GetGroupCommonData(WorldGroupInfo &groupCommonData) const;

    HcclResult GetHccsLinkNum(u32 &numHccsLink);

    HcclResult GetGroupRanksInfo(const std::vector<u32> &groupRanks, std::vector<RankInfo> &ranksInfo);

    static bool CompareWithUserRank(const RankInfo &left, const RankInfo &right);

    static bool CompareWithServerId(const ServerInfo_t &left, const ServerInfo_t &right);

    static bool CompareWithNicName(const NetworkInfo_t &left, const NetworkInfo_t &right);

    HcclResult GetOneSidedService(IHcclOneSidedService** service);
    HcclResult InitOneSidedServiceNetDevCtx(u32 remoteRankId);
    HcclResult DeInitOneSidedServiceNetDevCtx();

    u32 GetUserRank();
    u32 GetGroupRank();
    u32 GetRankSize();
    /* * 以下两函数用于防止重复初始化 */
    HcclResult AtomicInitSet();
    HcclResult HostMC2EnvResume();
    void AtomicInitClear();
    bool GetNicInitialized();
    void DestroyAlgResource(AlgResourceResponse &res);
    void DestroyOpTransportResponse(OpCommTransport &opTransportResponse);
    HcclResult ReleasePreemptSocket();
    HcclResult DestroyNetworkResources();
    HcclResult DisablePreResource();
    HcclResult GetWorkspaceSubStreamNum(u64 &streamNum, u64 dataSize, HcclCMDType opType);
    HcclResult GetWorkspaceMemSize(const std::string &opType, u64 count, HcclDataType dataType,
                                   u32 &rankSize, u64 &size, DevType &deviceType) const;
    HcclResult SetWorkspaceResource(const std::string &tag, void *memPtr, u64 &maxSize,
        std::vector<rtStream_t> &stream);
    HcclResult CreateOpBasedResources(const HcclCMDType &opType, const std::string &tag,
        const HcomCollOpInfo &opInfo);
    HcclResult CreateRemoteOpBasedResources(u64 memSize, const std::string &tag);
    HcclResult DestroyRemoteOpBasedMem(const std::string &tag);
    void DestroyWorkspaceResource(const std::string &tag);
    HcclResult GetInCCLbuffer(void* &buffer, u64 &size);
    HcclResult GetOutCCLbuffer(void* &buffer, u64 &size);
    void ReleaseCommCCLbuffer();
    HcclResult CreateCommCCLbuffer();
    HcclResult CreateCommExpBuffer();
    HcclResult InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize);
    HcclResult SetQosCfg(const u32 qosCfg);
    HcclResult ResetQosCfg();
    HcclResult GetQosCfg(u32& qosCfg);

    // 目前支持按tag对资源释放、解绑定
    HcclResult  ClearResMap(const std::string &tag, bool &findTag);
    virtual HcclResult ClearOpResource(const std::string &tag);
    HcclResult ClearAivSyncBuf(bool aivClearEnable);

    HcclResult SetGlobalWorkSpace(std::vector<void *> &globalWorkSpaceAddr);
    HcclResult SetAttachedStream(const std::vector<rtStream_t> &streams);
    // 获得rdma with reduce算子溢出的task信息后清除
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo> &hcclDumpInfo);

    HcclResult GetDeviceId(s32 &deviceId) const;
    virtual void Break();
    HcclResult SetDevicePid(s32 devicePid);
    HcclResult DestroyCDomainResource(s32 tag);

    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> GetPhyIdNicInfo();
    std::vector<u32> GetRanksPort();
    std::vector<RankInfo> GetRanksList();
    HcclResult SetWorldGroupInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
        std::vector<RankInfo> worldRankInfoList, std::vector<u32> &ranksPort, std::vector<u32> &vnicRanksPort);
    virtual HcclResult SaveTraceInfo(std::string &logInfo);
    virtual bool GetCommResource(const std::string &tag, void **commContext);
    virtual bool GetCommResource(void *&commContext);

    virtual HcclResult GetAicpuOpStreamNotify(HcclRtStream *opStream, u8 aicpuNotifyNUm, void** aicpuNotify);

    HcclResult GetAlgInfo(const std::string &algConfig, const std::string &tag, HcclCMDType commType,
        std::string &algName, std::string &newTag);
    HcclResult FillOpParam(const HcclCMDType commType, OpParam& opParam,
        const uint64_t count, void *pCount, void *pDispls);
    HcclResult AllocComResource(const std::string &newTag, const std::string &algName,
        const HcclCMDType commType, const OpParam& opParam, rtStream_t stream);
    HcclResult AllocComResourceByTiling(const std::string &algConfig, const std::string &tag, 
        uint32_t opType, uint32_t reduceType, rtStream_t stream);

    virtual HcclResult CreateCommResource(const std::string &tag, rtStream_t aiCpuStream, bool isOpbaseMode,
        void **commContext);
    virtual HcclResult Mc2AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string &kernelName);
    virtual HcclResult AicpuUnfoldKernelLaunch(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
        void* tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode,
        const std::string &tag);
    virtual HcclResult AicpuUnfoldKernelLaunchV2(void *inputPtr, void *outputPtr, const rtStream_t stm, u64 addr,
        void* tilingDataPtr, u32 tilingDataSize, const std::string &kernelName, HcclWorkflowMode mode,
        const std::string &tag);
    virtual HcclResult Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream);
    HcclResult Mc2AiCpuInitStreamAllocAndGet(u32 streamMode, rtStream_t &aiCpuStream);
    HcclResult GetTopoDesc(HcclTopoDescs *topoDescs, uint32_t topoSize);
    HcclResult ReStartVnic(const HcclCommParams &params, const RankTable_t &rankTable);
    static std::string GetUniqueId(void);

    u8 GetDeterministicConfig() const;  // 获取确定性计算配置
    HcclResult SetDeterministicConfig(const u8 deterministic);  // 设置确定性计算配置
    HcclResult SetAivModeConfig(const bool aivMode);  // 设置aiv模式配置
    HcclResult SetAicpuUnfoldConfig(const bool aicpuUnfold);  // 设置aicpu配置
    void SetQpQosAttr(u32 trafficClass, u32 serviceLevel); // 设置TC/SL配置
    bool GetMC2EnvFlag();
    HcclResult SetStopFlag(bool value);
    HcclResult SetState(HcclCommState state);
    HcclCommState GetState();
    HcclResult ResetNotify();
    HcclResult ResetNotifyForDestRank(s64 destRank);
    HcclResult InitZeroCopyMemoryAgent();
    HcclResult DeinitZeroCopyMemoryAgent(bool inDestructor = false);
    HcclResult SetMemoryRange(void *baseVirPtr, size_t size, size_t alignment, uint64_t flags);
    HcclResult UnsetMemoryRange(void *baseVirPtr);
    HcclResult ActivateCommMemory(void *virPtr, size_t size, size_t offset, void* handle, uint64_t flags);
    HcclResult DeactivateCommMemory(void *virPtr);
    HcclResult GetBlockDim(u32& blockDim){
        blockDim = blockDim_;
        return HCCL_SUCCESS;
    }
    HcclResult SetTransportStatus(const HcclOpIdentifier &opId, bool statusStop,
        const std::map<u32, bool> &remoteRankPortMap, const std::map<u32, bool> &isChangeLinkMap, bool isChangeLinkFlag);
    static HcclResult GetTransportCqeErrors(const HcclNetDevCtx netDevCtx, std::vector<ErrCqeInfo> &infos, u32 &num);
    ErrorMessageReport GetAicpuTaskException();

private:

    bool IsEnableRoce();
    bool IsEnableBackupLink();
    void SetAttrs();
    u32 HcclGetCmdTimeout();
    HcclResult SetMC2EnvFlag();
    HcclResult InitCommParams(HcclCommParams &params);
    HcclResult InitRankInfo(const RankTable_t &rankTable);
    HcclResult InitRankInfoSubGroup(const std::vector<RankInfo> &rankList, WorldGroupInfo &groupCommonData);
    HcclResult CheckSingleServerComm(const std::vector<RankInfo_t> &rankList) const;
    HcclResult SetRanksPort(const std::vector<RankInfo_t> &rankList);
    HcclResult InitNetResource(const RankTable_t &rankTable);
    HcclResult InitDebug();
    HcclResult InitDebugSubGroup();
    HcclResult InitATraceInfo();
    HcclResult InitNotifyManager();
    HcclResult InitDispatcher();
    HcclResult InitStreamManager();
    HcclResult InitSocketManager();
    HcclResult InitTransportManager();
    HcclResult InitMemoryManager();
    HcclResult InitMemoryManagerSubGroup();
    HcclResult InitHcclAlg();
    HcclResult InitProfiling();
    HcclResult DeinitProfiling();
    HcclResult InitProfiler();
    HcclResult InitOneSidedService(const RankTable_t &rankTable);
    HcclResult RegistTaskExceptionHandler() const;
    HcclResult UnRegistTaskExceptionHandler() const;
    HcclResult UnRegisterBackGroundThread();
    HcclResult DestroyAicpuComm();
    HcclResult InitPreResource(const RankTable_t &rankTable);
    HcclResult InitTcpMode(const RankTable_t &rankTable) const;
    HcclResult InitRaResource();
    bool IsNeedNicInit();
    HcclResult InitNic(bool isMC2ReInit = false);
    HcclResult DeinitNic();
    HcclResult RegisterToHeartBeat();
    HcclResult RegisterToHeartBeat(u32 peerRankId, std::string &tag);
    void UnRegisterToHeartBeat();
    HcclResult MrManagerInit();
    HcclResult MrManagerDeInit();
    HcclResult InitRecvMsgAndRequestBuffer();
    HcclResult InitMemBlocksAndRecvWrMem();
    HcclResult PrintOpbaseKeyTraceInfo(void);
    HcclResult InitPara();
    HcclResult GetComm(const std::string &tag, CommBase **comm);
    HcclResult Mc2CreateAndLaunchContext(rtStream_t aiCpuStream, bool isOpbaseMode, void **commContext);
    HcclResult SetCommResource(u64 commBufferSize, void *commInPtr, void *commOutPtr, void *commExpPtr,
        CommBase *comm, level1StreamInfo_t &streamInfo, Stream &stream);
    HcclResult GetAicpuOpStreamAndNotify(HcclRtStream *opStream, u8 aicpuNotifyNum, void** aicpuNotify);
    HcclResult SetAicpuNotifyInvaild();
    HcclResult AicpuKfcTilingDataLaunch(const OpParam &opParam, const HcclCMDType &opType, const DeviceMem &deviceContext,
    const std::string &kernelName, const AicpuOpTiling opTilingInfo);
    HcclResult AicpuKfcTilingDataLaunchExt(const OpParam &opParam, const HcclCMDType &opType,
        const DeviceMem &deviceContext, const std::string &kernelName, const AicpuOpTiling opTilingInfo);
    u64 CalcOpTilingDynamicDataSize(
        const OpParam &opParam, const HcclCMDType &opType, const u32 &rankSize);
    HcclResult AicpuInitOpTilingDataFromOpParam(const OpParam &opParam, const HcclCMDType &opType,
        struct OpTilingData* opTilingData);
    HcclResult AicpuInitOpTilingDataBuf(const OpParam &opParam, const HcclCMDType &opType, 
        const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 dynamicDataSize);
    HcclResult AicpuKfcTilingDataLaunchIn(const OpParam &opParam, const DeviceMem &deviceContext, 
        const std::string &kernelName, const AicpuOpTiling opTilingInfo, u64 opTilingDataSize);
    HcclResult AllReduceAicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);
    HcclResult CreateMutiStreamResFor310P(const std::string &tag, level1StreamInfo_t &streamInfo);
    HcclResult SetDynamicTilingDataAlltoall(const OpParam &opParam, HostMem &dynamicDataMem);
    HcclResult ProfilerDel(const OpParam &param);
    HcclResult ProfilerAdd(const OpParam &param, AlgType algType);
    HcclResult SetDynamicTilingDataAlltoallv(const OpParam &opParam, HostMem &dynamicDataMem);
    HcclResult SetDynamicTilingDataAlltoallvc(const OpParam &opParam, HostMem &dynamicDataMem);
    HcclResult GetReportHcclMC2Info(const Stream &kfcStream, const std::vector<Stream> &aicpuStreams);

    HcclResult ReAllocTransports(const std::string &tag, const std::string &newTag);
    HcclResult SetTransportStatusImpl(OpCommTransport &opCommTransport, bool statusStop,
        const HcclOpIdentifier &opId, u32 remoteRank, const std::map<u32, bool> &remoteRankPortMap,
        bool isUseDefault);
    HcclResult SetBsrTransportStatusImpl(OpCommTransport &opCommTransport, bool statusStop,
        const HcclOpIdentifier &opId, u32 remoteRank);
    HcclResult SetTransportStatusImplForChange(OpCommTransport &opCommTransport, const HcclOpIdentifier &opId, 
        u32 remoteRank, const std::map<u32, bool> &remoteRankPortMap, bool isUseDefault, 
        const std::map<u32, bool> &isChangeLinkMap, bool isCurTag);
    HcclResult SetBsrTransportStatusImplforchange(OpCommTransport &opCommTransport, 
        const HcclOpIdentifier &opId, u32 remoteRank, const std::map<u32, bool> &remoteRankPortMap, bool isUseDefault, 
        const std::map<u32, bool> &isChangeLinkMap, bool isCurTag);
    void ClearOpTransportResponseLinks(OpCommTransport &opTransportResponse);
    HcclResult SetSignalTransport(SingleSubCommTransport &singleSubCommTransport,
        u32 linkIdx, bool statusStop);
    void InsertNewTagToTagMap(std::string &newTag, std::string &tag);
    HcclResult GetTagFromNewTag(const std::string &newTag, std::string &tag);

    u32 deviceNumPerServer_;
    HcclDispatcher dispatcher_; // dispatcher放到最后析构
    HcclDispatcher vDispatcher_; // virtualDispatcher放到最后析构
    std::unique_ptr<NotifyPool> notifyPool_;
    std::unique_ptr<HcclCallbackTask> callbackTask_;
    std::atomic_flag initializedFlag_;
    u32 userRank_;  // 本group中的userrank
    u32 realUserRank_;  // world group中的userrank
    u32 userRankSize_;
    std::vector<RankInfo> rankInfoList_;  // world group内rank的信息, 按照rank id递增依次排列
    bool drvInit_;                          // ra是否初始化
    ServRankInfo_t servRankInfo_;
    std::string serverId_;
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo_; // server内所有device间的链路类型
    bool inlineReduceSwitchOn_;
    NICDeployment nicDeployment_;
    u32 devicePhyId_;
    u32 deviceBackUpPhyId_;
    s32 deviceLogicId_;
    u32 deviceBackUpLogicId_;
    std::vector<HcclIpAddress> devIpAddr_;
    std::vector<HcclIpAddress> devBackupIpAddr_;
    u32 devBackupPort_{HCCL_INVALID_PORT};
    HcclIpAddress hostIp_;
    u32 hostPort_{HCCL_INVALID_PORT};
    u32 localRank_;
    SocketHandle hostSocketHandle_;
    SocketHandle loopbackHeterogSocketHandle_;
    bool isUsedRdmaLevel0_; // 节点内是否使用rdma, 包括a+x和标卡
    std::atomic<s32> nicInitialized_;
    bool hcomGroupNicInit_;
    // profiling 相关资源
    HcomProfilingMode profilingMode_;
    std::string profilingOption_;
    ProfilingDeviceCommResInfo hcclMc2Info_;
    bool raResourceInit_;
    bool interServer_;
    std::unique_ptr<WorkspaceResource> workSpaceRes_;
    std::vector<u32> enableP2PDevices_;
    bool isSingleMeshAggregation_;
    CCLBufferManager cclBufferManager_;
    bool isExecuteProfilingInit_;
    DevType deviceType_;
    std::string collectiveId_;
    HcclComm commHandle_;
    std::vector<u32> nicRanksPort_;
    std::vector<u32> groupNicRanksPort_;
    std::vector<u32> vnicRanksPort_;
    std::vector<u32> groupVnicRanksPort_;
    std::unique_ptr<MrManager> mrManager_;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_;
    std::unordered_map<u32, HcclRtContext> rtCtxMap_; // {devPhyId, rtCtx}
    WorkMode commWorkMode_;
    u32 meshAggregationRankSize_;
    std::map<HcomOperationType, std::string> opTypeTagMap_;
    bool isHaveCpuRank_;
    bool isUseRankPort_{ true };
    bool isSetHDCModeInfo_{ false };
    std::map<std::string, HostMem> tagWorkSpaceMem_;
    std::string identifier_;
    u32 ranktableCrc_;
    s32 devicePid_;
    std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>> pMsgInfosMem_;
    std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>> pReqInfosMem_;
    std::unique_ptr<HeterogMemBlocksManager> memBlocksManager_;
    std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>> pRecvWrInfosMem_;
    TransportResInfo transportResInfo_;
    bool multiModuleDiffDeviceNumMode_;
    bool multiSuperPodDiffServerNumMode_;
    DeviceMem commContext_;
    std::shared_ptr<ProfilerManager> profilerManager_;
    bool isStandardCard_ = false;
    bool is310PDuoCard_ = false;
    bool isCommon310P3DUO_ = false;
    s32 hccsPortNum_ = -1;
    std::string superPodId_;
    u32 superDeviceId_ = INVALID_UINT;
    bool useSuperPodMode_ = false;
    bool isUsedInterHccsMode_ = false;
    bool isNeedInitNic_ = false;
    std::vector<RankInfo> worldRankInfoList_;
    std::unique_ptr<HcclTraceInfo> opBaseAtraceInfo_;
    bool aivClearEnable_ = false;
    u32 blockDim_ = 0;
    
private:
    bool IsAtomicInit();
    HcclResult MigrateLinkToStopOrResume(LINK &link, bool isStop);
    HcclResult MigrateLinkVectorToStopOrResume(const std::vector<LINK> &links, bool isStop);
    HcclResult TraverseLinkVector(std::vector<std::unique_ptr<CommBase> > &commBaseVector, bool isStop);
    HcclResult CheckSuspendingStatus();
    HcclResult InitCombinOpara();
    HcclResult InitWorkSpace();
    void ReleaseWorkSpacebuffer();
    HcclResult CreateWorkSpace(u64 size, DeviceMem &buffer) const;
    HcclResult GetWorkSpace(u64 *workSpaceSize, u64 *workSpace) const;
    void ReleaseCommContextbuffer();
    HcclResult CreateDeviceCommContext(u64 size, DeviceMem &buffer) const;
    HcclResult CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify> &localNotify, HcclSignalInfo &notifyInfo);
    HcclResult GetAiCpuNotifyData(const std::shared_ptr<LocalNotify> &localNotify, HcclSignalInfo &notifyInfo);
    HcclResult ReplaceCommInfoByTag(const std::string &tag, std::unique_ptr<CommInfo> &commInfo);
    HcclResult CreateCommAndStreamRes(const std::string &tag, Stream &stream);
    HcclResult SetInfoToDevice(const OpParam &opParam, const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
        const HcclWorkflowMode &mode, Stream &stream);
    HcclResult GetInfoFromDevice(const OpParam &opParam, const std::unique_ptr<PreProcessMetaInfo> &preMetaInfo,
        const HcclWorkflowMode &mode, Stream &stream, HostMem& hostCollectBuffer);
    HcclResult RegressCalPreOp(AlltoAllOperator* &alltoAllOperator, const OpParam &opParam,
        std::unique_ptr<PreProcessMetaInfo> &preMetaInfo);
    HcclResult RegressCalPreOp(AlltoAllOperator* &alltoAllOperator, const OpParam &opParam,
        std::unique_ptr<PreProcessMetaInfo> &preMetaInfo, Stream &preProcessStream);
    HcclResult ExecOp(HcclCMDType opType, OpParam &opParam);
    // alltoall专用
    HcclResult ExecOpAlltoAll(HcclCMDType opType, OpParam &opParam);
    HcclResult CopyAivCommInfoToDevice(AlgResourceResponse &algResResp);
    HcclResult FreeScratchMemOnOpBaseMode(DeviceMem &scratchMem, const OpParam &opParam,
        const HcclCMDType &opType);
    HcclResult CalcTinySendRecvMem(const OpParam &opParam, AlgResourceResponse &algResResponse,
        DeviceMem &tinySendRecvMem);
    bool IsForceAicpuOpBaseMode(const OpParam &opParam, const HcclCMDType &opType);
    HcclResult AllocOpBaseModeScratchMem(HcclCMDType opType, const OpParam &opParam,
        AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse);
    HcclResult AllocAlgResource(const std::string &tag, HcclCMDType opType, const OpParam &opParam,
        AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse);
    HcclResult IncreAllocLink(const std::string &newTag, const OpParam &opParam,
        AlgResourceRequest &resRequest, AlgResourceResponse &algResResponse);
    DeviceMem GetWorkspaceScracthMem(const std::string &tag, u64 allocMemSize);
    std::vector<Stream> GetWorkspaceSubStreams(const std::string &tag, u32 num);
    // HcclImplBase中Comm资源是否存在
    inline bool IsExistCommRes(const std::string &tag)
    {
        std::unique_lock<std::mutex> commLock(commLock_);
        return (tagCommInfo_.find(tag) != tagCommInfo_.end());
    }
    // HcclImplBase中MutiStream资源是否存在
    inline bool IsExistMutiStreamRes(const std::string &tag)
    {
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        return (tagStreamInfo_.find(tag) != tagStreamInfo_.end());
    }
    void GetAndSetSyncMode(SyncMode& preSyncMode, SyncMode newSyncMode);
    void RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode);
    HcclResult AicpuUnfold(const std::string &tag, void *inputPtr, void *outputPtr, u64 count,
        HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcclCMDType cmdType);
    u32 GetHostPort(s32 devicePhyId);
    u32 GetLocalNicPort(NicType nicType);
    std::string GetSupportDataType(bool needReduce);
    HcclResult InitHDCommunicate();
    HcclResult InitOpRetry();
    HcclResult InitOpResPara();
    HcclResult PrepareZeroCopy(HcclCMDType opType, OpParam &opParam);
    HcclResult UpdateZeroCopy(const OpParam &opParam, const AlgResourceResponse &algResource);
    HcclResult BuildZeroCopyParam();
    HcclResult AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem> &buffer) const;
    HcclResult AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem> &bufferPtr) const;
    HcclResult updateList(u64 size, void *buffer) const;
    HcclResult BuildOpLocalResParam(const AlgResourceResponse &algResource, const std::string &newTag);
    HcclResult BuildOpLocalScratchMemResParam(
        const AlgResourceResponse &algResource, const std::string &newTag, LocalResInfoV2 *localResHostPtr);
    HcclResult BuildOpTopoResTlvParam(const std::string &algName,
                                      const std::vector<std::vector<std::vector<u32>>> &inputVectorInfo,
                                      DeviceMem &dstTlvDeviceMem, u64 &tlvLen);
    HcclResult BuildOpTopoResVectorTlvParam(const std::string &algName,
                                      const std::vector<std::vector<std::vector<std::vector<u32>>>> &inputVectorInfo,
                                      DeviceMem &dstTlvDeviceMem, u64 &tlvLen);
    HcclResult BuildPairLinkCounter(const std::string &algName);
    HcclResult BuildIsUsedRdmaRank(const std::string &algName);
    HcclResult BuildNicList(const std::string &algName);
    HcclResult BuildBridgeRank(const std::string &algName);
    HcclResult BuildCommPlanRank(const std::string &algName);
    HcclResult BuildServerAndsuperPodRank(const std::string &algName);
    HcclResult BuildCommPlaneSubGroupRank(const std::string &algName);
    HcclResult BuildHierarchicalAlgOption(const std::string &algName);
    HcclResult BuildOpTopoResParam(
        const std::string &algName, const AlgResourceResponse &algResource);
    HcclResult BuildOpRemoteResParam(const AlgResourceResponse &algResource, const std::string &newTag,
        const HcclCMDType opType, bool isRetry = false);
    HcclResult BuildOpResParam(const std::string &algName, const AlgResourceResponse &algResource,
        const std::string &newTag, const HcclCMDType opType);
    HcclResult BuildOpRetryParam(const AlgResourceResponse &algResource, const std::string &newTag);
    HcclResult CopyHostListResToDeviceParam(const std::string &newTag, const ListCommon *headList, const u64 size);
    HcclResult CopyHostOpRemoteResToDeviceParam(const std::string &newTag);
    HcclResult CopyHostOpResToDeviceParam(const std::string &newTag);
    HcclResult AicpuResourceInit(const std::string &algName,
        const AlgResourceResponse &algResource, const std::string &newTag, const rtStream_t &aicpuStream,
        const HcclCMDType opType);
    HcclResult AicpuResourceRefresh(const AlgResourceResponse &algResource, const std::string &newTag, 
        const HcclCMDType opType);
    HcclResult OrchestrateAicpu(const HcclCMDType &opType, const std::string &algName, const OpParam &param,
        const AlgResourceResponse &algResource, const std::string &newTag, AlgType algType);
    template <typename T>
    HcclResult CopyVectorToDeviceMem(const u64 len, DeviceMem &dstDeviceMem, const std::vector<T> &srcVec);
    template <typename T>
    HcclResult CreateListNode(T **resHostPtr, T **resDevicePtr);
    HcclResult ParseRemoteDataToMem(const OpCommTransport &opTransportResponse, const std::string &newTag,
        const HcclCMDType opType, bool isBakup = false, bool isRetry = false);
    HcclResult BuildRelationResByRemoteRankId(const TransportRequest &transportRequest, const LINK &link,
        HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr);
    HcclResult BuildRemoteResByTag(const std::string &newTag, const u32 &usrRankId,
        HcclRankRelationResV2 *&rankRelationResHostPtr, HcclRankRelationResV2 *&rankRelationResDevicePtr,
        bool isBackup, bool isRetry);
    HcclResult BuildOpRemoteLinkP2pResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes);
    HcclResult BuildOpRemoteLinkRoceResParam(const LINK &link, HccltagRemoteResV3 &tagRemoteRes, bool isBakup,
        bool isRetry, bool isSecondBuild);
    HcclResult CheckNotifyOrQPMaxNum(u64 &existNum, const u64 &MaxNum, const bool &isNotifyRes);
    HcclResult AllocAlgNotifys(const std::string &tag, const NotifyLoadType notifyLoadType, const u32 notifyNum,
      std::vector<std::shared_ptr<LocalNotify> > &notifiesMain, std::vector<std::shared_ptr<LocalNotify> > &notifiesAux);
    HcclResult CreateAndGetAiCpuNotifyWithNotifyRes(HcclSignalInfo &notifyInfo);
    void SaveLinkRes(const OpCommTransport &opTransportResponse);
    HcclResult SetDevIbverbsData(CommBase *comm, bool isSupportNormalQP, u64 commBufferSize, void *commInPtr,
        void *commOutPtr);
    HcclResult CaptureSlaveStreams(rtStream_t mainStream, std::vector<Stream> &slaveStreams);
    HcclResult HandleAclGraphFirstOpAivBuff(rtStream_t mainStream);
    HcclIpAddress loopBackIp_;
    bool profilingInitiated_;
    u64 callbackThreadId_;
    u32 role_;
    bool mrManagerInit_;
    std::map<u64, std::vector<rtStream_t>> callbackStreamMap_;
    bool isHostUseDevNic_;
    std::mutex socketListenMutex_;

    HcclAlg *implAlg_ = nullptr;
    HcclCommunicatorAttrs attrCollector_;

    u32 deviceNumPerAggregation_;
    std::vector<u32> nicList_;
    std::unordered_map<u32, u32> pairLinkCounter_; // server内所有device间的链路类型计数
    bool isAllRankSamePlane_;
    std::unique_ptr<TopoInfoParse> topoInfoParse_; // 对rank table device选取的校验模块
    u32 serverNum_;
    u32 moduleNum_;
    u32 superPodNum_ = 0;
    bool isAlgoLevel1Default_ = false;
    HcclCombinOpParam combinOpara_;
    Stream opStream_;
    Stream aicpuInitStream_;
    std::vector<Stream> attachedStreams_;
    HcclRtNotify aicpuOpNotify_[2] = { nullptr };
    std::vector<std::shared_ptr<LocalNotify>> localAiCpuNotifyRes_;
    std::shared_ptr<LocalNotify> localAiCpuOpNotify_[2] = { nullptr };
    u32 workSpaceSize_;
    DeviceMem workSpace_;
    DeviceMem mc2DeviceMem_;
    std::vector<DeviceMem> extraMem_;
    std::vector<HcclRtEvent> aiCpuNoIpcEvnet_;
    bool isDiffDeviceModule_;
    bool isDiffDeviceType_;
    u32 gcdDeviceNumPerAggregation_;
    tagCommInfo_t tagCommInfo_;    // 以tag为粒度分配comm实例和资源
    std::mutex commLock_;
    tagStreamInfo_t tagStreamInfo_;
    std::mutex tagStreamInfoLock_;

    std::vector<Stream> auxRingCommStreamsDev_;
    bool isServerInter_{ false };
    bool isSupportRdmaLite_{ false };          // 是否支持RDMA Lite

    HcclIpAddress localVnicIp_;
    u32 localVnicListenPort_;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;

    std::unique_ptr<OpBaseStreamManager> opStreamManager_ = { nullptr };
    std::unique_ptr<QueueNotifyManager> queueNotifyManager_ = { nullptr };
    std::unique_ptr<QueueNotifyManager> queueNotifyManagerRefac_ = { nullptr };
    std::unique_ptr<HcclSocketManager> socketManager_;
    std::unique_ptr<TransportManager> transportManager_ = { nullptr };
    std::unique_ptr<ZeroCopyMemoryAgent> zeroCopyMemoryAgent_ = { nullptr };
    std::unordered_map<std::string, AlgResourceResponse> resMap_; // tag : AlgResourceResponse
    std::unordered_set<std::string> hostResMap_;
    std::unordered_set<std::string> hbSendRecvTags_;
    std::vector<DeviceMem> deviceResOrigMem_;
    bool isSuspending = false;
    bool retryEnable_ = false;
    bool rtsSupportChangeLink_ = true;  // RTS是否支持借轨（部分ASCEND_RT_VISIBLE_DEVICES自定义场景不支持访问同chip内的另一个die）
    bool inplaceSupportRetry_ = false; //inplace是否支持重执行
    u8 isInplaceStatus_ = 0; // 算子是不是inplace的状态
    // 算子在inplace时，是否支持重执行的状态
    InplaceSupportRetryStatus inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    bool isInplacePreSync_ = false;
    bool isPostSync_ = false;
    HcclWorkflowMode retryOrigWorkflowMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
    HcclOpResParam opResPara_{};
    DeviceMem opResDevicePara_;
    HcclOpResParam *opResDeviceParaPtr_;
    Stream opMainStream_;
    bool isContextLaunched_{false};
    std::vector<std::shared_ptr<DeviceMem>> deviceMemVec_;
    std::vector<std::shared_ptr<HostMem>> hostMemVec_;
    DeviceMem nicListDevice_;
    DeviceMem complanRankDevice_;
    DeviceMem pairLinkCounterDevice_;
    DeviceMem isUsedRdmaRankPairDevice_;
    std::unordered_set<std::string> newTagResAlloced_;
    DeviceMem bridgeRankDevice_;
    DeviceMem serverAndsuperPodToRankDevice_;
    DeviceMem commplaneSubGroupRankDevice_;
    DeviceMem hierarchicalAlgOptionDevice_;
    std::unique_ptr<IHcclOneSidedService> oneSideService_ = {nullptr};
    HcclRankLinkInfo hcclRankLinkInfo_{};
    std::atomic<bool> isOneSidedServiceNetDevCtxInited{false};
    std::atomic<bool> isOneSidedServiceNicInited{false};
    std::unique_ptr<OpRetryManager> opRetryManager_ = { nullptr };
    std::shared_ptr<HcclOpStreamRes> opRetryStreamPtr_;
    std::unordered_set<u32> captureModelIds_;
    std::unordered_map<u32, std::unordered_map<std::string, HccltagRemoteResV3>> rankTagRemoteRes_;  // 以rankid&tag粒度保存HccltagRemoteResV3
    std::shared_ptr<HDCommunicate> kfcControlTransferH2D_;
    std::shared_ptr<HDCommunicate> kfcStatusTransferD2H_;
    HcclCommConnections commConnections_;
    HcclSocketPortConfig commPortConfig_;
    std::shared_ptr<PetersonLock> hostDeviceLock_;
    bool isNsRecovery_{false};
    HostMem opTilingDataBuf_;
    HostMem apiTilingDataMem_;
    // 单机场景下多卡间能互相访问的共享buffer，除了自己rank是申请的，其余均是Ipc打开的
    DeviceMem zeroCopyLocalBuffer_;
    void *zeroCopyIpcPtrs_[MAX_MODULE_DEVICE_NUM] {};
    std::atomic<HcclCommState> state_{HcclCommState::IDLE};
    std::unordered_map<std::string, std::string> newTagToTagMap_;
    static std::mutex linkResMapMutex_;
    static std::unordered_map<Transport*, std::pair<std::string, u32>> linkResMap_;
    std::vector<TransportDeviceNormalData> transDevIbverbsData_;
    bool isA2MC2MultiServer_{false};
    DeviceMem ibverbsDataBuffer_;
    std::list<DeviceMem> ibverbsLocalNotify_;
    std::list<DeviceMem> ibverbsRemoteNotify_;
};

void HcclOneSidedServiceCallbackInstall(HcclResult (*func)(std::unique_ptr<IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &));
void HcclOneSidedServiceCallbackUninstall();
}  // end namespace hccl
#endif  // HCCL_IMPL_BASE_H
