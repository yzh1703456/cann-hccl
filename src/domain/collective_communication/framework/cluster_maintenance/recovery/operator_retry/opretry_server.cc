/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opretry_server.h"
#include "externalinput_pub.h"
#include "heartbeat.h"

namespace hccl {

HcclResult CreateOpRetryServerByState(RetryState state, RetryContext* retryCtx)
{
    HCCL_INFO("[OpRetry][Server]CreateOpRetryServerByState state[%s]", GetReadableState(state));
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    switch (state) {
        case RETRY_STATE_SERVER_RUNNING: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerRunning>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STETA_HANDLE_ALL_ERR: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerHandleError>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_SERVER_RETRY_FAIL: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerRetryFail>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_WAIT_LINK_CHECKED:
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerWaitLinkInfo>()), return HCCL_E_PTR);
            break;
        case RETRY_STATE_WAIT_AICPU_STOPED:
        case RETRY_STATE_WAIT_STREAM_STOPED:
        case RETRY_STATE_WAIT_STREAM_CLEARED:
        case RETRY_STATE_WAIT_STOP_TRANSPORT:
        case RETRY_STATE_WAIT_NOTIFY_RESETED:
        case RETRY_STATE_WAIT_RESUME_TRANSPORT:
        case RETRY_STATE_WAIT_CHECK_INFO:
        case RETRY_STATE_WAIT_CAN_RETRY: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerWaitResp>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_CHECK_ALL_LINK:
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerCheckAllLink>()), return HCCL_E_PTR);
            break;
        case RETRY_STATE_CMD_RESUME_TRANSPORT: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerIssueChangeLinkAndResume>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_CMD_STOP_AICPU:
        case RETRY_STATE_CMD_STOP_STREAM:
        case RETRY_STATE_CMD_CLEAR_STREAM:
        case RETRY_STATE_CMD_STOP_TRANSPORT:
        case RETRY_STATE_CMD_CHECK_LINK:
        case RETRY_STATE_CMD_RESET_NOTIFY:
        case RETRY_STATE_CMD_CHECK:
        case RETRY_STATE_CMD_CAN_RETRY: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerIssueCmd>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_CHECK_OP: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerCheckOp>()), return HCCL_E_PTR);
            break;
        }
        default: {
            HCCL_ERROR("[OpRetry][Server]CreateOpRetryServerByState failed, state[%s] is invalid",
                GetReadableState(state));
            return HCCL_E_NOT_SUPPORT;
        }
    }
    retryCtx->SetRetryState(state, retryPtr);
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerBase::ProcessError(RetryContext* retryCtx)
{
    HCCL_ERROR("[%s]OpRetryServer run fail, rankId[%u], state[%s]", __func__, retryCtx->rankId_,
        retryCtx->GetReadableCtxState());
    // 状态切换至RETRY_STATE_SERVER_RETRY_FAIL
    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RETRY_FAIL, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerRunning::ProcessEvent(RetryContext* retryCtx)
{
    if (retryCtx->errorRankList_.size() > 0) {
        // 若当前errorRankList_中有未处理的errorRank，则先进行处理
        HCCL_RUN_INFO("[OpRetry][Server]deal rank from errorRankList_, size[%d]", retryCtx->errorRankList_.size());
        CHK_RET(CreateOpRetryServerByState(RETRY_STETA_HANDLE_ALL_ERR, retryCtx));
        return HCCL_SUCCESS;
    }

    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_KEEP_INTERVAL);
    // 轮询接收agent信息
    for (auto &it : retryCtx->serverSockets_) {
        const u32 &agentId = it.first;
        // 若对端已经关闭, 则不再轮询
        if (disableAgent_.find(agentId) != disableAgent_.end()) {
            continue;
        }

        // 记录时间, 检测和对端上一次通信时间是否超过保活时间
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        if (lastRecvTimes_.find(agentId) == lastRecvTimes_.end()) {
            lastRecvTimes_.insert(std::make_pair(agentId, curTime));
        }

        // 轮询接收agent状态机信息
        HcclResult ret = WaitResponse(it.second.socket, it.second.retryInfo);
        if (ret == HCCL_SUCCESS) { // 成功接收到数据
            RetryState nextState = RETRY_STATE_SERVER_RUNNING;
            CHK_RET(ParaseErrorCode(retryCtx, it.second, nextState));
            if (nextState != RETRY_STATE_SERVER_RUNNING) {
                // 收到第一个报错后加入errorRankList_中，并切换到RETRY_STETA_HANDLE_ALL_ERR状态
                HCCL_RUN_INFO("[OpRetry][Server]agent[%u] tag[%s] index[%u] find error, insert to errorRankList_", 
                    agentId, it.second.retryInfo.opInfo.opId.tag, it.second.retryInfo.opInfo.opId.index);
                retryCtx->errorRankList_.insert(std::make_pair(agentId, it.second.retryInfo.opInfo.opId));
                CHK_RET(CreateOpRetryServerByState(RETRY_STETA_HANDLE_ALL_ERR, retryCtx));
                return HCCL_SUCCESS;
            }
            lastRecvTimes_[agentId] = curTime;
        } else if (ret == HCCL_E_AGAIN) { // 未接收到数据
            // 校验是否超时
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastRecvTimes_[agentId]);
            if (elapsed > timeout) {
                HCCL_WARNING("[OpRetry][Server]OpRetryServerRunning recv Retry Frame from agentId[%u] timeout",
                    agentId);
                lastRecvTimes_[agentId] = curTime;
            }
        } else { // 接收数据失败
            disableAgent_.insert(agentId);
            HCCL_RUN_INFO("[OpRetry][Server]WaitResponse from agentId[%u] fail, ret[%u]", agentId, ret);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerHandleError::ProcessEvent(RetryContext* retryCtx)
{
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_WAIT_CAN_RETRY_RANK);
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    u32 waitTime = GetExternalInputRetryHoldTime();
    while (true) {
        // 判断是否超时
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        if (elapsed > timeout) {
            HCCL_ERROR("[OpRetry][Server]aicpu can not retry, opname is inconsistent");
            for (auto &it : retryCtx->serverSockets_) {
                auto tag = std::string(reinterpret_cast<const char*>(it.second.retryInfo.opInfo.opId.tag));
                HCCL_ERROR("[OpRetry][Server]OpRetryHandle retryinfo rank[%u] tag[%s] index[%u] IpInfo[%s]", it.first,
                    tag.c_str(), it.second.retryInfo.opInfo.opId.index, it.second.retryInfo.dfxIpInfo);
            }
            return HCCL_E_TIMEOUT;
        }

        // 轮询接收agent信息,只期望收上开故障信息
        for (auto &it : retryCtx->serverSockets_) {
            const u32 &agentId = it.first;
            if (retryCtx->errorRankList_.find(agentId) != retryCtx->errorRankList_.end()) {
                // 若当前rank已在errorRankList_，则不进行轮训
                continue;
            }
            // 轮询接收agent状态机信息
            HcclResult ret = WaitResponse(it.second.socket, it.second.retryInfo);
            if (ret == HCCL_SUCCESS) { // 成功接收到数据
                RetryState nextState = RETRY_STATE_SERVER_RUNNING;
                CHK_RET(ParaseErrorCode(retryCtx, it.second, nextState));
                if (nextState != RETRY_STATE_SERVER_RUNNING) {
                    // 当前rank报错，收集到errorRankList_后统一处理
                    HCCL_RUN_INFO("[OpRetry][Server]agent[%u] tag[%s] index[%u] find error, insert to errorRankList_", 
                        agentId, it.second.retryInfo.opInfo.opId.tag, it.second.retryInfo.opInfo.opId.index);
                    retryCtx->errorRankList_.insert(std::make_pair(agentId, it.second.retryInfo.opInfo.opId));
                    continue;
                }
            } else if (ret == HCCL_E_AGAIN) {
                // 未收到数据，则发送一个保活数据给agent
                RetryCommandInfo commandInfo;
                commandInfo.command= RETRY_CMD_RUNNING;
                CHK_RET(IssueCommandWithOpId(it.second.socket, commandInfo));
            }
        }

        bool isFoundSendRecv = false;
        std::set <u32> errorRank;
        for (auto iter= retryCtx->errorRankList_.begin(); iter!= retryCtx->errorRankList_.end();++iter) {
            errorRank.insert(iter->first);
        }
        // 对errorRankList_中rank进行遍历
        for (auto rank:errorRank){
            if (retryCtx->errorRankList_[rank].isSendRecv) {
                // 当前报错rank中存在send/recv算子，优先处理send/recv算子
                isFoundSendRecv = true;
                auto curOpId = retryCtx->errorRankList_[rank];
                uint32_t remoteRank = (rank==curOpId.detRank) ? curOpId.srcRank : curOpId.detRank;
                auto &remoteOpId = retryCtx->serverSockets_[remoteRank].retryInfo.opInfo.opId;
                std::string curTag = std::string(reinterpret_cast<const char*>(curOpId.tag));
                std::string remoteTag = std::string(reinterpret_cast<const char*>(remoteOpId.tag));
                //sendrecv没有下边那两字段
                auto remoteSendTag = std::string(reinterpret_cast<const char*>(remoteOpId.bsrInfo[HCCL_SEND].bsrTag));
                auto remoteRecvTag = std::string(reinterpret_cast<const char*>(remoteOpId.bsrInfo[HCCL_RECV].bsrTag));
                HCCL_RUN_INFO("[OpRetry][Server]curRank[%u], tag[%s], index[%u], startTaskComplete[%d]"\
                               "remoteRank[%u], remotetag[%s], remoteindex[%u], remoteStartTaskComplete[%d]"\
                               "Sendtag[%s], sendindex[%u]"\
                               "Recvtag[%s], recvindex[%u]",
                    rank, curTag.c_str(), curOpId.index, curOpId.isBsrTaskStart,
                    remoteRank, remoteTag.c_str(), remoteOpId.index, remoteOpId.isBsrTaskStart,
                    remoteSendTag.c_str(), remoteOpId.bsrInfo[HCCL_SEND].index,
                    remoteRecvTag.c_str(), remoteOpId.bsrInfo[HCCL_RECV].index);
                if (curOpId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && !remoteOpId.isBsrTaskStart) {
                	continue;
                }
                // 如果对端也停在同一个send/recv算子，则触发该算子的重执行
                if ((curTag == remoteSendTag && curOpId.index == remoteOpId.bsrInfo[HCCL_SEND].index) || 
                    (curTag == remoteRecvTag && curOpId.index == remoteOpId.bsrInfo[HCCL_RECV].index) ||
                    (curTag == remoteTag && curOpId.index == remoteOpId.index)) {
                    // 从errorRankList_中清除本端和对端rank
                    retryCtx->errorRankList_.erase(rank);
                    if (retryCtx->errorRankList_.find(remoteRank) != retryCtx->errorRankList_.end()) {
                        if (curTag == remoteTag && curOpId.index == remoteOpId.index)
                        {
                            HCCL_RUN_INFO("[OpRetry][Server]delete remoteRank[%u] from errorRankList_", remoteRank);
                            retryCtx->errorRankList_.erase(remoteRank);
                        }
                    }
                    // 触发重执行
                    HCCL_RUN_INFO("[OpRetry][Server]begin to exec retry of tag[%s] from rank[%u] and rank[%u]",
                        curTag.c_str(), rank, remoteRank);
                    retryCtx->needRetryServerRanks_.clear();
                    CHK_PRT(SetNeedRetryServerRank(retryCtx, curOpId));
                    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_STOP_AICPU, retryCtx));
                    return HCCL_SUCCESS;
                }
            }
        }

        if (!isFoundSendRecv) {
            // 如果没有找到send/recv算子，判断所有rank是否停在同一个算子
            bool isAllTagSame = true;
            u32 firstErrorRank = *(errorRank.begin());
            auto curOpId = retryCtx->errorRankList_[firstErrorRank];
            auto curTag = std::string(reinterpret_cast<const char*>(curOpId.tag));
            for (auto &it : retryCtx->serverSockets_) {
                auto remoteTag = std::string(reinterpret_cast<const char*>(it.second.retryInfo.opInfo.opId.tag));
                if (curTag != remoteTag) {
                    isAllTagSame = false;
                    break;
                }
            }
            if (isAllTagSame) {
                retryCtx->errorRankList_.clear();
                // 所有rank停在同一个算子，开始重执行
                HCCL_RUN_INFO("[OpRetry][Server]begin to exec retry of tag[%s] from rank[%u]",
                    curTag.c_str(), firstErrorRank);
                retryCtx->needRetryServerRanks_.clear();
                CHK_PRT(SetNeedRetryServerRank(retryCtx, curOpId));
                CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_STOP_AICPU, retryCtx));
                return HCCL_SUCCESS;
            }
        }
        errorRank.clear();
        SaluSleep(waitTime * TIME_MS_TO_US);
        HCCL_INFO("[OpRetry][Server]no rank can retry, wait for [%u]ms for collect all error rank", waitTime);
    }
}

HcclResult OpRetryServerHandleError::SetNeedRetryServerRank(RetryContext* retryCtx, const HcclOpIdentifier &opId)
{
    if (opId.isSendRecv) {
        // 在send/recv场景下，仅需对本端和对端进行重执行即可
        if (retryCtx->serverSockets_.find(opId.srcRank) == retryCtx->serverSockets_.end() ||
            retryCtx->serverSockets_.find(opId.detRank) == retryCtx->serverSockets_.end()) {
            HCCL_ERROR("[OpRetry][Server]srcRank[%u] or detRank[%u] isn't in serverSockets_", 
                opId.srcRank, opId.detRank);
            return HCCL_E_INTERNAL;
        }
        retryCtx->needRetryServerRanks_.push_back(opId.srcRank);
        retryCtx->needRetryServerRanks_.push_back(opId.detRank);
        retryCtx->curFaultOpId = opId;
        HCCL_INFO("[OpRetry][Server]set needRetryServerRank[%u] for send/recv success: srcRank=[%u],detRank=[%u],"
            "tag =[%s], streamid =%u",
            retryCtx->needRetryServerRanks_.size(), opId.srcRank, opId.detRank, opId.tag, opId.streamId);
    } else {
        // 其余场景下需要对所有rank进行重执行
        for (auto &it : retryCtx->serverSockets_) {
            retryCtx->curFaultOpId = opId;
            retryCtx->needRetryServerRanks_.push_back(it.first);
        }
        HCCL_DEBUG("[OpRetry][Server]set needRetryServerRank[%u] success", retryCtx->needRetryServerRanks_.size());
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerRunning::ParaseErrorCode(RetryContext* retryCtx, HcclAgentRetryInfo &agentInfo, RetryState &nextState)
{
    // 处理接收到的数据
    KfcError errorCode = agentInfo.retryInfo.opInfo.execStatus.kfcError;
    switch (errorCode) {
        case KfcError::kNone: { // 发送保活数据
            //保活数据携带一个空的opid
            RetryCommandInfo commandInfo;
            commandInfo.command= RETRY_CMD_RUNNING;
            CHK_RET(IssueCommandWithOpId(agentInfo.socket, commandInfo));
            break;
        }
        case KfcError::kRdma:
            retryCtx->isRdmaError = true;
        case KfcError::kSdma: { // 处理ERROR
            nextState = RETRY_STATE_CMD_STOP_AICPU;
            HCCL_RUN_INFO("[OpRetry][Server]OpRetryServerRunning recv ErrorCode[%d] from rank[%u]",
                errorCode, agentInfo.retryInfo.rankId);
            break;
        }
        default: { // 不支持的ErrorCode
            HCCL_ERROR("[OpRetry][Server]OpRetryServerRunning recv invalid ErrorCode[%d] from rank[%u]",
                errorCode, agentInfo.retryInfo.rankId);
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerIssueCmd::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = HCCL_SUCCESS;
    RetryState curState = retryCtx->GetRetryState();
    // 获取下一个状态
    auto itState = RETRY_SERVER_STATE_TRANSFER_LABEL.find(curState);
    CHK_PRT_RET(itState == RETRY_SERVER_STATE_TRANSFER_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueCmd fail, state[%s] is not in RETRY_SERVER_STATE_TRANSFER_LABEL",
            GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryState nextState = itState->second;

    // 发送命令
    auto itCommand = RETRY_SERVER_STATE_TO_CMD_LABEL.find(curState);
    CHK_PRT_RET(itCommand == RETRY_SERVER_STATE_TO_CMD_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueCmd fail, state[%s] is not in RETRY_SERVER_STATE_TO_CMD_LABEL",
            GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryCommand command = itCommand->second;
    HCCL_INFO("[OpRetry][Server]OpRetryServerIssueCmd curState[%s], command[%s]", GetReadableState(curState),
        GetReadableCmd(command));

    for (auto rank : retryCtx->needRetryServerRanks_) {
        RetryCommandInfo commandInfo;
        commandInfo.command = command;
        commandInfo.opId = retryCtx->curFaultOpId;
        HCCL_INFO("[OpRetry][Server]IssueCommandWithOpId tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d]," 
            "streamid[%u]",
            commandInfo.opId.tag, commandInfo.opId.index, commandInfo.opId.srcRank, 
            commandInfo.opId.detRank,commandInfo.opId.isSendRecv, commandInfo.opId.streamId);
        ret = IssueCommandWithOpId(retryCtx->serverSockets_[rank].socket, commandInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueCmd IssueCommand fail, curState[%s], command[%s]",
            GetReadableState(curState), GetReadableCmd(command)), ret);
    }
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerWaitResp::ProcessEvent(RetryContext* retryCtx)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_SEND_RECV_TIMEOUT);
    RetryState curState = retryCtx->GetRetryState();

    // 获取预期的下一个server状态
    auto serverTransferIt = RETRY_SERVER_STATE_TRANSFER_LABEL.find(curState);
    CHK_PRT_RET(serverTransferIt == RETRY_SERVER_STATE_TRANSFER_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp fail, state[%s] is not in RETRY_SERVER_STATE_TRANSFER_LABEL",
        GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryState expectNextState = serverTransferIt->second;

    // 获取预期的对端agent状态
    auto agentStateIt = RETRY_SERVER_WAIT_AGENT_STATE_LABEL.find(curState);
    CHK_PRT_RET(agentStateIt == RETRY_SERVER_WAIT_AGENT_STATE_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp fail, state[%s] is not in RETRY_SERVER_WAIT_AGENT_STATE_LABEL",
        GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryState expectagentState = agentStateIt->second;
    HCCL_DEBUG("[OpRetry][Server]OpRetryServerWaitResp state[%s], expect next state[%s], expect peer state[%s]",
        GetReadableState(curState), GetReadableState(expectNextState), GetReadableState(expectagentState));

    std::set<u32> recvVaild;
    while (recvVaild.size() < retryCtx->needRetryServerRanks_.size()) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp timeout"), HCCL_E_TIMEOUT);

        for (auto rank : retryCtx->needRetryServerRanks_) {
            if (recvVaild.find(rank) != recvVaild.end()) {
                continue;
            }
            auto &agentRetryInfo = retryCtx->serverSockets_[rank];
            // 接收agent信息
            HcclResult ret = WaitResponse(agentRetryInfo.socket, agentRetryInfo.retryInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
                HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp WaitResponse fail, ret[%u]", ret), ret);

            RetryState dstState = agentRetryInfo.retryInfo.retryState;
            if (ret == HCCL_SUCCESS && dstState == expectagentState) { // 接收到对端信息且状态有效
                recvVaild.insert(rank);
                HCCL_INFO("[OpRetry][Server]OpRetryServerWaitResp recv success from dst[%u], state[%s]",
                    rank, GetReadableState(dstState));
            } else if (ret == HCCL_SUCCESS && dstState == RETRY_STATE_RESP_RUNNING_ERR) { // 对端重执行失败
                recvVaild.insert(rank);
                PrintAgentInfoAfterFail(retryCtx->serverSockets_, recvVaild);
                HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp dst rank[%u] with IpInfo[%s] retry fail, " \
                    "command all rank retry fail", rank, agentRetryInfo.retryInfo.dfxIpInfo);
                CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RETRY_FAIL, retryCtx));
                return HCCL_SUCCESS;
            }
        }
    }

    CHK_RET(CreateOpRetryServerByState(expectNextState, retryCtx));
    return HCCL_SUCCESS;
}

void OpRetryServerWaitResp::PrintAgentInfoAfterFail(std::map<u32, HcclAgentRetryInfo> &serverSockets,
    std::set<u32> &recvVaild)
{
    for (auto it = serverSockets.begin(); it != serverSockets.end(); ++it) {
        if (recvVaild.find(it->first) == recvVaild.end()) { // 未接收到有效数据
            continue;
        }
        auto &opInfo = it->second.retryInfo.opInfo;
        const char* tag = reinterpret_cast<const char*>(opInfo.opId.tag);
        u32 index = opInfo.opId.index;
        const KfcStatus &aicpuState = opInfo.execStatus.kfcStatus;
        if (aicpuState == KfcStatus::kEnd) { // 该rank未下发算子，或算子已执行结束
            HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp dst[%u] with IpInfo[%s], hccl op not launch or "\
                "is complete, hccl aicpu can not retry", it->first, it->second.retryInfo.dfxIpInfo);
        }
        HCCL_RUN_INFO("[OpRetry][Server]Print rank[%u], tag[%s], index[%u], aicpuStatus[%d]",
            it->first, tag, index, aicpuState);
    }
}

HcclResult OpRetryServerCheckOp::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = CheckRetryInfo(*retryCtx);
    RetryState nextState = (ret == HCCL_SUCCESS) ? RETRY_STATE_CMD_CAN_RETRY : RETRY_STATE_SERVER_RETRY_FAIL;
    HCCL_INFO("[OpRetry][Server]check op ret[%d], nextState[%s]", ret, GetReadableState(nextState));
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerWaitLinkInfo::ProcessEvent(RetryContext* retryCtx)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_SEND_RECV_TIMEOUT);
    // 下一个server状态
    RetryState nextState = RETRY_STATE_CHECK_ALL_LINK;

    std::set<u32> recvVaild;
    while (recvVaild.size() < retryCtx->needRetryServerRanks_.size()) {
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitLinkInfo timeout"), HCCL_E_TIMEOUT);

        for (auto rank : retryCtx->needRetryServerRanks_) {
            if (recvVaild.find(rank) != recvVaild.end()) {
                continue;
            }
            auto &agentRetryInfo = retryCtx->serverSockets_[rank];
            // 接收agent信息
            HcclResult ret = WaitLinkPortCheckResult(agentRetryInfo.socket, agentRetryInfo.linkPortStatus);
            CHK_PRT_RET(ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
                HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitLinkCheckResult fail, ret[%u]", ret), ret);
            if (ret == HCCL_SUCCESS) {
                recvVaild.insert(rank);
                HCCL_INFO("[OpRetry][Server]OpRetryServerWaitLinkCheckResult recv success from dst[%u], ", rank);
            }
        }
    }
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    HCCL_INFO("[OpRetry][Server]OpRetryServerWaitLinkInfo success");
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerCheckAllLink::ProcessEvent(RetryContext* retryCtx)
{
    // 收集所有rank的主备网口信息
    std::map<u32, std::pair<bool, bool>> allLinkInfo;
    for (auto rank: retryCtx->needRetryServerRanks_) {
        auto &linkPortStatus = retryCtx->serverSockets_[rank].linkPortStatus;
        allLinkInfo.insert({rank, std::make_pair(linkPortStatus.defaultPort, linkPortStatus.backupPort)});
    }

    // 对所有rank依次遍历
    for (auto rank: retryCtx->needRetryServerRanks_) {
        u32 remoteRankIndex = 0;
        auto &linkPortStatus = retryCtx->serverSockets_[rank].linkPortStatus;
        // 对rank的所有对端进行遍历
        for (u32 i = 0; i < linkPortStatus.rankSize; i++) {
            u32 remoteRank = linkPortStatus.rankList[i];
            retryCtx->serverSockets_[rank].changeLinkInfo.remoteRankList[remoteRankIndex] = remoteRank;
            if (allLinkInfo[rank].first && allLinkInfo[remoteRank].first) {
                // 本端和对端的主网口均up，则使用主网口
                retryCtx->serverSockets_[rank].changeLinkInfo.isUseDefaultPort[remoteRankIndex] = true;
            } else if (allLinkInfo[rank].second && allLinkInfo[remoteRank].second) {
                // 本端和对端的备网口均up，则使用备网口
                retryCtx->serverSockets_[rank].changeLinkInfo.isUseDefaultPort[remoteRankIndex] = false;
            } else {
                // 本端和对端无可用的网口，重执行失败
                HCCL_ERROR("[OpRetry][Server]rank[%u]:default[%d], backup[%d], IpInfo[%s]; rank[%u]:default[%d], "
                    "backup[%d], can not find same port, can not retry", rank, allLinkInfo[rank].first,
                    allLinkInfo[rank].second, retryCtx->serverSockets_[rank].retryInfo.dfxIpInfo, remoteRank,
                    allLinkInfo[remoteRank].first, allLinkInfo[remoteRank].second);
                CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RETRY_FAIL, retryCtx));
                return HCCL_SUCCESS;
            }
            remoteRankIndex += 1;
        }
        retryCtx->serverSockets_[rank].changeLinkInfo.remoteRankNum = remoteRankIndex;
    }

    // 打印所有rank的借轨信息
    for (auto rank: retryCtx->needRetryServerRanks_) {
        auto &changeLinkInfo = retryCtx->serverSockets_[rank].changeLinkInfo;
        std::string changeLinkInfoStr = "rank[" + std::to_string(rank) + "]";
        for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
            changeLinkInfoStr += (std::to_string(changeLinkInfo.remoteRankList[i]) + ":" + 
                std::to_string(changeLinkInfo.isUseDefaultPort[i]) + "; ");
        }
        HCCL_INFO("[OpRetry][Server]changeLinkInfoStr:%s", changeLinkInfoStr.c_str());
    }

    // 所有rank网口确认成功，切换到给agent发借轨命令状态
    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_RESUME_TRANSPORT, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerIssueChangeLinkAndResume::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = HCCL_SUCCESS;
    RetryState curState = retryCtx->GetRetryState();
    // 先将每个rank的changeLinkInfo发送至对应agent
    for (auto rank : retryCtx->needRetryServerRanks_) {
        ret = IssueChangeLink(retryCtx->serverSockets_[rank].socket, retryCtx->serverSockets_[rank].changeLinkInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueChangeLink fail, curState[%s]",
            GetReadableState(curState)), ret);
        HCCL_INFO("[OpRetry][Server]OpRetryServerIssueChangeLink send to rank[%u] success", rank);
    }
    // 再发送resume transport命令至每个agent
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerIssueCmd>()), return HCCL_E_PTR);
    RetryState nextState = RETRY_STATE_CMD_RESUME_TRANSPORT;
    retryCtx->SetRetryState(nextState, retryPtr);
    HCCL_INFO("[OpRetry][Server]OpRetryServerIssueChangeLinkAndResume success");
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerRetryFail::ProcessEvent(RetryContext* retryCtx)
{
    RetryCommandInfo commandInfo;
    commandInfo.command = RETRY_CMD_RETRY_FAIL;
    HCCL_INFO("[OpRetry][Server]retry fail, command all rank %s", GetReadableCmd(commandInfo.command));
    for (auto rank : retryCtx->needRetryServerRanks_) {
        HcclResult ret = IssueCommandWithOpId(retryCtx->serverSockets_[rank].socket, commandInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[OpRetry][Server]OpRetryServerRetryFail IssueCommandWithOpId AgentId[%u] fail", rank),
            ret);
    }

    // 重执行异常，通知心跳存在异常 -> 广播异常给整个集群
    Heartbeat::GetInstance(retryCtx->deviceLogicId_).SetOpretryErr();

    RetryState nextState = RETRY_STATE_SERVER_RUNNING;
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}
}