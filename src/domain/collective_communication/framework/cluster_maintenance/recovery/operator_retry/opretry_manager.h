/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPRETRY_MANAGER_H
#define HCCL_OPRETRY_MANAGER_H
#include <thread>
#include <mutex>
#include "opretry_base.h"

namespace hccl {
struct RetryCtrl {
    std::unique_ptr<std::thread> thread;
    std::shared_ptr<RetryContext> retryCtx;
    bool startExec = false;
};

class OpRetryManager
{
public:
    OpRetryManager() = default;
    ~OpRetryManager();
    HcclResult RegisterOpRetryMachine(const std::string &group, u32 rankSize, bool isRoot,
        std::shared_ptr<HcclSocket> agentConnection, std::map<u32, std::shared_ptr<HcclSocket> > &serverConnections,
        std::shared_ptr<HDCommunicate> h2dPtr, std::shared_ptr<HDCommunicate> d2hPtr,
        std::shared_ptr<HcclOpStreamRes> opStreamPtr, OpRetryResetNotifyCallback notifyResetCallback,
        OpRetrySetTransprotStatusCallback setTransprotStatusCallback, bool isEnableBackupLink,
        const OpRetryServerInfo& serverInfo, const OpRetryAgentInfo& agentInfo);
    HcclResult UnRegisterOpRetryManager(const std::string& group);

    static HcclResult AddLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList, bool incre = false);
    static HcclResult GetLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList);
    static HcclResult DeleteLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier);

private:
    HcclResult Init();
    HcclResult DeInit();
    HcclResult RegisterAgentRetryMachine(const std::string& group, std::shared_ptr<HcclSocket> socket,
        std::shared_ptr<HDCommunicate> h2dPtr, std::shared_ptr<HDCommunicate> d2hPtr,
        std::shared_ptr<HcclOpStreamRes> opStreamPtr, OpRetryResetNotifyCallback notifyResetCallback,
        OpRetrySetTransprotStatusCallback setTransprotStatusCallback, bool isEnableBackupLink,
        const OpRetryAgentInfo& agentInfo);
    HcclResult RegisterServerRetryMachine(const std::string& group,
        std::map<u32, std::shared_ptr<HcclSocket>> &serverConnections, const OpRetryAgentInfo& agentInfo);
    void RetryStateMonitor(const std::string &group, std::shared_ptr<RetryContext> retryCtx, const bool &startExec,
        HcclRtContext rtCtx_);

private:
    std::map<std::string, RetryCtrl> serverOpRetry;
    std::map<std::string, RetryCtrl> agentOpRetry_;
    bool initialized_ = false;
    std::mutex ProcessLock_;
};
}
#endif