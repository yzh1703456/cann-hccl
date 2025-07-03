/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RETRY_AGENT_H
#define HCCL_RETRY_AGENT_H
#include "opretry_base.h"
namespace hccl {
// agent状态机 response状态转移表
const std::map<RetryState, RetryState> RETRY_AGENT_RESP_STATE_LABEL {
    {RETRY_STATE_RESP_AICPU_ERR, RETRY_STATE_WAIT_CMD_STOP_AICPU},
    {RETRY_STATE_RESP_AICPU_STOPED, RETRY_STATE_WAIT_CMD_STOP_STREAM},
    {RETRY_STATE_RESP_STREAM_STOPED, RETRY_STATE_WAIT_CMD_CLEAR_STREAM},
    {RETRY_STATE_RESP_STREAM_CLEARED, RETRY_STATE_WAIT_CMD_STOP_TRANSPORT},
    {RETRY_STATE_RESP_STOP_TRANSPORT, RETRY_STATE_WAIT_CMD_CHECK_LINK},
    {RETRY_STATE_RESP_RESUME_TRANSPORT, RETRY_STATE_WAIT_CMD_RESET_NOTIFY},
    {RETRY_STATE_RESP_NOTIFY_RESETED, RETRY_STATE_WAIT_CMD_CHECK},
    {RETRY_STATE_RESP_CHECK_INFO, RETRY_STATE_WAIT_CMD_CAN_RETRY},
    {RETRY_STATE_RESP_AICPU_RETRYEND, RETRY_STATE_AGENT_RUNNING},
    {RETRY_STATE_RESP_RUNNING_ERR, RETRY_STATE_WAIT_CMD_RETRY_FAIL}
};

constexpr u32 RDMA_CQE_ERR_STATUS = 0x0C;

HcclResult CreateOpRetryAgentByState(RetryState state, RetryContext* retryCtx);

// RETRY_STATE_RESP_RUNNING_ERR 重执行异常状态处理
class OpRetryAgentBase : public OpRetryBase {
public:
    HcclResult ProcessError(RetryContext* retryCtx) override;
};

// RETRY_STATE_AGENT_RUNNING 正常运行状态
class OpRetryAgentRunning : public OpRetryAgentBase {
public:
    OpRetryAgentRunning();
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
private:
    HcclResult ParseKfcErr(RetryContext* retryCtx, RetryState &nextState); // 轮询aicpu状态
    HcclResult ParseRdmaErr(RetryContext* retryCtx, RetryState &nextState);// 轮询RDMA状态
    std::chrono::steady_clock::time_point lastRecvCmdTime_; // 上一次收到server端命令的时间
    std::chrono::steady_clock::time_point lastPollAicpuTime_; // 上一次轮询aicpu的时间
    std::chrono::steady_clock::time_point lastKeepTime_; // 上一次发保活数据的时间
    std::chrono::seconds pollTimeout_; // 轮询aicpu的超时时间
    std::chrono::seconds keepTimeout_; // 和server端保活通信的超时时间
    
    std::chrono::steady_clock::time_point lastPollRcTime_; // 上一次轮询Rdma cqe的时间
    std::chrono::seconds pollRcTimeout_; // 轮询Rdma Cqe的超时时间
};

// 公共状态-发送信息
class OpRetryAgentResponse : public OpRetryAgentBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

// 公共状态-接收命令
class OpRetryAgentWaitCmd : public OpRetryAgentBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
private:
    HcclResult ParseCommandWithOpId(RetryContext* retryCtx, RetryCommandInfo &commandinfo, RetryState &nextState);
    void ResetBatchSendRecvRdmaErr(RetryContext* retryCtx, u32 dstRank);
};

// 公共状态-轮询AicpuCtx中的Status字段
class OpRetryAgentPollAicpuStop : public OpRetryAgentBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class OpRetryAgentResponseLinkInfo : public OpRetryAgentBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};

class OpRetryAgentWaitChangeLinkInfo : public OpRetryAgentBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
private:
    void UpdateChangeLinkInfo(ChangeLinkInfo &localChangeLinkInfo, ChangeLinkInfo &recvChangeLinkInfo);
};

// RETRY_STATE_AGENT_RETRY_FAIL 重执行失败
class OpRetryAgentRetryFail : public OpRetryAgentBase {
public:
    HcclResult ProcessEvent(RetryContext* retryCtx) override;
};
}
#endif