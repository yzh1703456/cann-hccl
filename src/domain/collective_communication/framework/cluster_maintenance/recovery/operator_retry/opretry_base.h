/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RETRY_BASE_H
#define HCCL_RETRY_BASE_H

#include <memory>
#include "hccl_socket.h"
#include "notify_pool.h"
#include "hccl_op_retry_pub.h"
#include "hdc_pub.h"
#include "exception_handler.h"

namespace hccl {
constexpr u32 OP_RETRY_MAX_CNT = 3;
constexpr u32 OP_RETRY_WAIT_AICPU_TIMEOUT = 5; // 等待Aicpu的时长, 单位s
constexpr u32 OP_RETRY_POLL_AICPU_ERROR_INTERVAL = 1; // 正常状态轮询Aicpu错误码的间隔, 单位s
constexpr u32 OP_RETRY_POLL_RDMA_ERROR_INTERVAL = 1; // 正常状态轮询RDMA错误码的间隔, 单位s
constexpr u32 OP_RETRY_POLL_AICPU_STATE_INTERVAL = 10000; // 重执行状态轮询Aicpu状态的间隔, 单位us
constexpr u32 OP_RETRY_SEND_RECV_TIMEOUT = 205; // 发送和接收的超时时间, 单位s, 比aicpu状态机超时时间长5s
constexpr u32 OP_RETRY_SEND_RECV_INTERVAL = 10000; // 发送和接收的间隔时间, 单位us
constexpr u32 OP_RETRY_KEEP_INTERVAL = 1; // 保活时间间隔, 单位s
constexpr u32 TIME_MS_TO_US = 1000;
constexpr u32 OP_RETRY_WAIT_CAN_RETRY_RANK = 60;

struct LinkPortStatus {
    bool defaultPort = false;
    bool backupPort = false;
    u32 rankSize = 0;
    u32 rankList[AICPU_MAX_RANK_NUM] = {};
};

using HcclAgentRetryInfo = struct HcclAgentRetryInfoDef {
    std::shared_ptr<HcclSocket> socket{nullptr};
    RetryInfo retryInfo;
    ChangeLinkInfo changeLinkInfo;
    LinkPortStatus linkPortStatus;
};

inline const char *GetReadableState(RetryState retryState) {
    auto it = RETRY_STATE_STR_MAP.find(retryState);
    return (it != RETRY_STATE_STR_MAP.end()) ? it->second.c_str() : "unkown state";
}

inline const char *GetReadableCmd(RetryCommand retryCommand) {
    auto it = RETRY_COMMAND_STR_MAP.find(retryCommand);
    return (it != RETRY_COMMAND_STR_MAP.end()) ? it->second.c_str() : "unkown cmd";
}

class RetryContext;

// 状态基类
class OpRetryBase {
public:
    virtual HcclResult Handle(RetryContext* retryCtx);
    virtual HcclResult ProcessEvent(RetryContext* retryCtx) = 0;
    virtual HcclResult ProcessError(RetryContext* retryCtx) = 0;

    OpRetryBase() {};
    virtual ~OpRetryBase() {};

protected:
    /* server-agent 交互 */
    HcclResult IssueResponse(std::shared_ptr<HcclSocket> socket, RetryInfo &retryInfo); // agent向server发送数据
    HcclResult WaitResponse(std::shared_ptr<HcclSocket> socket, RetryInfo &retryInfo); // server等待agent回复

    HcclResult IssueCommand(std::shared_ptr<HcclSocket> socket, RetryCommand command); // server向agent发送命令
    HcclResult WaitCommand(std::shared_ptr<HcclSocket> socket, RetryCommand &command); // agent轮询命令

    // server向agent发送命令,携带opid
    HcclResult IssueCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo &commandInfo);
    // agent轮询命令,携带opid
    HcclResult WaitCommandWithOpId(std::shared_ptr<HcclSocket> socket, RetryCommandInfo &commandInfo);

    // server向agent发送借轨信息
    HcclResult IssueChangeLink(std::shared_ptr<HcclSocket> socket, ChangeLinkInfo &changeLinkInfo);
    // agent轮询借轨信息
    HcclResult WaitChangeLink(std::shared_ptr<HcclSocket> socket, ChangeLinkInfo &changeLinkInfo);
    // agent向server发送当前网口情况
    HcclResult IssueLinkPortCheckResult(std::shared_ptr<HcclSocket> socket, LinkPortStatus &linkPortStatus);
    // server接收当前网口情况
    HcclResult WaitLinkPortCheckResult(std::shared_ptr<HcclSocket> socket, LinkPortStatus &linkPortStatus);
    // agent向device发送借轨信息
    HcclResult SetOpChangeLinkInfo(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd,
        ChangeLinkInfo &changeLinkInfo);

    /* 校验 */
    HcclResult CheckRetryInfo(RetryContext &retryCtx); // 校验收到的N个RetryInfo
    HcclResult GetRetryInfo(RetryContext* retryCtx, RetryInfo &retryInfo);

    /* agent-device 交互 */
    HcclResult GetOpExecInfo(std::shared_ptr<HDCommunicate> hdcPtr, KfcExecStatus &opInfo);
    HcclResult SetOpExecCmd(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd);
    HcclResult ClearStream(std::shared_ptr<HcclOpStreamRes> opStreamPtr_, rtClearStep_t clearStep);
    HcclResult SetOpExecCmdWithOpId(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd, HcclOpIdentifier &opId);
    HcclResult ClearStreamWithOpId(std::shared_ptr<HcclOpStreamRes> opStreamPtr_, rtClearStep_t clearStep, 
        HcclOpIdentifier &opId, HcclOpIdentifier &curOpId);
    HcclResult ResetNotify(RetryContext* retryCtx);
    HcclResult SetTransportStatusForStop(RetryContext* retryCtx);
    HcclResult SetTransportStatusForResume(RetryContext* retryCtx);
    HcclResult GetLinkPortStatus(RetryContext* retryCtx, LinkPortStatus &linkPortStatus);
    HcclResult InitChangeLinkInfo(RetryContext* retryCtx, bool incre = false);
    /*获取batchsendrecv rdma重执行时的故障信息*/
    HcclResult SetBsrOpId(RetryContext* retryCtx, HcclSendRecvType type);
    HcclResult GetBsrOpId(RetryContext* retryCtx, HcclSendRecvType type);
private:
    // 阻塞式发送 && 非阻塞式接收, 接口内部不报错, 返回值在上层判断并打印日志, 避免未进入重执行时出现ERROR日志
    HcclResult Send(std::shared_ptr<HcclSocket> socket, void *data, u64 size);
    HcclResult Recv(std::shared_ptr<HcclSocket> socket, void *data, u64 size);

    HcclResult CheckOpName(const RetryInfo &opInfo1, const RetryInfo &opInfo2); // 校验算子一致
    HcclResult CheckMaxRetryCnt(const RetryInfo &retryInfo); // 校验重执行次数
    HcclResult CheckLinkStates(const RetryInfo &retryInfo); // 校验link状态
};

class RetryContext {
public:
    // agent状态机初始化
    RetryContext(const std::string& group, std::shared_ptr<HcclSocket> socket, std::shared_ptr<HDCommunicate> h2dPtr,
        std::shared_ptr<HDCommunicate> d2hPtr, std::shared_ptr<HcclOpStreamRes> opStreamPtr,
        OpRetryResetNotifyCallback notifyResetCallback, std::shared_ptr<OpRetryBase> retryBase,
        OpRetrySetTransprotStatusCallback setTransprotStatusCallback, bool isEnableBackupLink,
        const OpRetryAgentInfo& agentInfo):
        group_(group), agentSocket_(socket), h2dPtr_(h2dPtr), d2hPtr_(d2hPtr), opStreamPtr_(opStreamPtr),
        notifyResetCallback_(notifyResetCallback), setTransprotStatusCallback_(setTransprotStatusCallback),
        isEnableBackupLink_(isEnableBackupLink), retryBase_(retryBase), isRootRetryCtx_(false)
    {
        rankId_ = agentInfo.userRank;
        deviceLogicId_ = agentInfo.deviceLogicId;
        netDevCtx_ = agentInfo.netDevCtx;
        backUpNetDevCtx_ = agentInfo.backUpNetDevCtx;
        std::string dfxInfo = "deviceIP:" + std::string(agentInfo.deviceIP.GetReadableIP()) +
            ";hostIP:" + std::string(agentInfo.hostIP.GetReadableIP());
        EXCEPTION_THROW_IF_COND_ERR(memcpy_s(localRetryInfo_.dfxIpInfo, sizeof(localRetryInfo_.dfxIpInfo),
            dfxInfo.c_str(), dfxInfo.size()) != EOK, "memcpy_s dfxIpInfo failed.");
        localRetryInfo_.dfxIpInfo[dfxInfo.size()] = '\0';
    }

    // server状态机初始化
    RetryContext(std::map<u32, std::shared_ptr<HcclSocket> > &sockets,
        std::shared_ptr<OpRetryBase> retryBase, const OpRetryAgentInfo& agentInfo) :
        retryBase_(retryBase), isRootRetryCtx_(true)
    {
        for (auto it = sockets.begin(); it != sockets.end(); ++it) {
            HcclAgentRetryInfo tempAgentInfo;
            tempAgentInfo.socket = it->second;
            serverSockets_.insert(std::make_pair(it->first, std::move(tempAgentInfo)));
        }
        rankId_ = agentInfo.userRank;
        std::string dfxInfo = "deviceIP:" + std::string(agentInfo.deviceIP.GetReadableIP()) +
            ",hostIP:" + std::string(agentInfo.hostIP.GetReadableIP());
        EXCEPTION_THROW_IF_COND_ERR(memcpy_s(localRetryInfo_.dfxIpInfo, sizeof(localRetryInfo_.dfxIpInfo),
            dfxInfo.c_str(), dfxInfo.size()) != EOK, "memcpy_s dfxIpInfo failed.");
        localRetryInfo_.dfxIpInfo[dfxInfo.size()] = '\0';
    }

    RetryState GetRetryState() {
        return state_;
    }
    const char *GetReadableCtxState() const {
        return GetReadableState(state_);
    }

    void SetRetryState(RetryState nextState, std::shared_ptr<OpRetryBase> retryBase) {
        HCCL_RUN_INFO("[OpRetry][%s]State Transfer, cur state %s, next state %s",
            GetOpRetryMachineType(), GetReadableState(state_), GetReadableState(nextState));
        state_ = nextState;
        retryBase_ = retryBase;
        localRetryInfo_.retryState = state_;
    }

    // 外部接口调用Request()
    HcclResult Request() {
        CHK_SMART_PTR_NULL(retryBase_);
        return retryBase_->Handle(this);
    }

    u32 GetRankId() {
        return rankId_;
    }

    const char *GetOpRetryMachineType() const {
        std::string ctxType = isRootRetryCtx_ ? "Server" : "Agent";
        return ctxType.c_str();
    }

    const char *GetDfxIpInfo() const {
        return localRetryInfo_.dfxIpInfo;
    }

    std::string group_ = "";
    s32 deviceLogicId_ = INVALID_INT;
    u32 rankId_ = INVALID_UINT;

    // agent状态机储存信息
    std::shared_ptr<HcclSocket> agentSocket_ = nullptr;
    std::shared_ptr<HDCommunicate> h2dPtr_ = nullptr;
    std::shared_ptr<HDCommunicate> d2hPtr_ = nullptr;
    std::shared_ptr<HcclOpStreamRes> opStreamPtr_ = nullptr;
    OpRetryResetNotifyCallback notifyResetCallback_ = nullptr;
    OpRetrySetTransprotStatusCallback setTransprotStatusCallback_ = nullptr;
    bool isEnableBackupLink_ = false;
    RetryInfo localRetryInfo_;
    ChangeLinkInfo localChangeLinkInfo_;
    LinkPortStatus linkPortStatus_;
    bool isChangeLinkInfoInit_ = false;
    std::map<u32, bool> lastLinkPortStatus_;
    bool isUseDefaultPort_ = true;
    HcclNetDevCtx netDevCtx_ = nullptr;
    HcclNetDevCtx backUpNetDevCtx_ = nullptr;
    bool isBSRRdmaRecvError_ = false;
    bool isBSRRdmaSendError_ = false;
    HcclOpIdentifier RemainSendOpId_;
    HcclOpIdentifier RemainRecvOpId_;
    
    // server状态机储存信息
    std::map<u32, HcclAgentRetryInfo> serverSockets_;
    std::vector<u32> needRetryServerRanks_;
    HcclOpIdentifier curFaultOpId;
    std::map<u32, HcclOpIdentifier> errorRankList_;
    bool isRdmaError = false;
    bool isAlreadyChangeLink = false;
private:
    std::shared_ptr<OpRetryBase> retryBase_ = nullptr;
    RetryState state_ = RETRY_STATE_RESERVED;
    bool isRootRetryCtx_ = false;
};
}
#endif