/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opretry_manager.h"
#include "opretry_link_manage.h"
#include "opretry_connection_pub.h"
#include "opretry_agent.h"
#include "opretry_server.h"
#include "adapter_rts_common.h"
#include "sal_pub.h"

namespace hccl {
OpRetryManager::~OpRetryManager()
{
    HCCL_DEBUG("Destory OpRetryManager");
    (void)DeInit();
}

HcclResult OpRetryManager::Init()
{
    CHK_PRT_RET(initialized_ == true, HCCL_WARNING("OpRetryManager has already initialized"), HCCL_SUCCESS);
    initialized_ = true;
    HCCL_INFO("OpRetryManager Init success");
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::DeInit()
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    if (initialized_) {
        initialized_ = false;
        for (auto it = agentOpRetry_.begin(); it != agentOpRetry_.end(); ++it) {
            if (it->second.thread != nullptr && it->second.thread->joinable()) {
                it->second.thread->join();
            }
        }
        agentOpRetry_.clear();

        for (auto it = serverOpRetry.begin(); it != serverOpRetry.end(); ++it) {
            if (it->second.thread != nullptr && it->second.thread->joinable()) {
                it->second.thread->join();
            }
        }
        serverOpRetry.clear();
        HCCL_INFO("OpRetryManager DeInit success");
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::RegisterOpRetryMachine(const std::string& group, u32 rankSize, bool isRoot,
    std::shared_ptr<HcclSocket> agentConnection, std::map<u32, std::shared_ptr<HcclSocket> > &serverConnections,
    std::shared_ptr<HDCommunicate> h2dPtr, std::shared_ptr<HDCommunicate> d2hPtr,
    std::shared_ptr<HcclOpStreamRes> opStreamPtr, OpRetryResetNotifyCallback notifyResetCallback,
    OpRetrySetTransprotStatusCallback setTransprotStatusCallback, bool isEnableBackupLink,
    const OpRetryServerInfo& serverInfo, const OpRetryAgentInfo& agentInfo)
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    CHK_SMART_PTR_NULL(h2dPtr);
    CHK_SMART_PTR_NULL(d2hPtr);
    CHK_SMART_PTR_NULL(opStreamPtr);
    CHK_PRT_RET(group.empty(),
        HCCL_ERROR("[OpRetryManager][RegisterOpRetryMachine]params invalid, group is empty"), HCCL_E_PARA);
    if (agentConnection == nullptr && serverConnections.empty()) {
        CHK_RET(OpRetryConnectionPub::Init(group, rankSize, serverInfo, agentInfo));
        CHK_RET(OpRetryConnectionPub::GetConns(group, isRoot, agentConnection, serverConnections));
    }
    // 初始化
    if (initialized_ == false) {
        CHK_RET(Init());
    }

    // 注册agent状态机
    CHK_RET(RegisterAgentRetryMachine(group, agentConnection, h2dPtr, d2hPtr,
        opStreamPtr, notifyResetCallback, setTransprotStatusCallback, isEnableBackupLink, agentInfo));

    // 注册server状态机
    if (isRoot) {
        CHK_RET(RegisterServerRetryMachine(group, serverConnections, agentInfo));
    }
    HCCL_INFO("[Register][RetryMachine]group[%s] register success", group.c_str());
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::RegisterAgentRetryMachine(const std::string& group, std::shared_ptr<HcclSocket> socket,
    std::shared_ptr<HDCommunicate> h2dPtr, std::shared_ptr<HDCommunicate> d2hPtr,
    std::shared_ptr<HcclOpStreamRes> opStreamPtr, OpRetryResetNotifyCallback notifyResetCallback,
    OpRetrySetTransprotStatusCallback setTransprotStatusCallback, bool isEnableBackupLink,
    const OpRetryAgentInfo& agentInfo)
{
    if (agentOpRetry_.find(group) != agentOpRetry_.end()) {
        HCCL_INFO("[Register][AgentRetryMachine]group[%s] has Registered to agentOpRetry, skip", group.c_str());
        return HCCL_SUCCESS;
    }

    RetryCtrl retryCtrl;
    agentOpRetry_.insert(std::make_pair(group, std::move(retryCtrl)));
    std::shared_ptr<OpRetryBase> retryPtr;
    EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryAgentRunning>()), return HCCL_E_PTR);
    EXECEPTION_CATCH((agentOpRetry_[group].retryCtx =
        std::make_shared<RetryContext>(group, socket, h2dPtr, d2hPtr, opStreamPtr, notifyResetCallback, retryPtr,
        setTransprotStatusCallback, isEnableBackupLink, agentInfo)), return HCCL_E_PTR);
    agentOpRetry_[group].startExec = true;

    HcclRtContext ctx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&ctx));
    agentOpRetry_[group].thread.reset(new (std::nothrow) std::thread(&OpRetryManager::RetryStateMonitor, this,
        group, agentOpRetry_[group].retryCtx, std::ref(agentOpRetry_[group].startExec), ctx));
    CHK_SMART_PTR_NULL(agentOpRetry_[group].thread);
    HCCL_INFO("[%s]group[%s] rank[%u], register to agentOpRetry success", __func__, group.c_str(), agentInfo.userRank);
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::RegisterServerRetryMachine(const std::string& group,
    std::map<u32, std::shared_ptr<HcclSocket>> &serverConnections, const OpRetryAgentInfo& agentInfo)
{
    if (serverOpRetry.find(group) != serverOpRetry.end()) {
        HCCL_INFO("[Register][ServerRetryMachine]group[%s] has Registered to serverOpRetry, skip", group.c_str());
        return HCCL_SUCCESS;
    }
    for (auto it = serverConnections.begin(); it != serverConnections.end(); ++it) {
        CHK_SMART_PTR_NULL(it->second);
    }

    RetryCtrl retryCtrl;
    serverOpRetry.insert(std::make_pair(group, std::move(retryCtrl)));
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerRunning>()), return HCCL_E_PTR);

    EXECEPTION_CATCH((serverOpRetry[group].retryCtx =
        std::make_shared<RetryContext>(serverConnections, retryPtr, agentInfo)), return HCCL_E_PTR);
    serverOpRetry[group].startExec = true;

    HcclRtContext ctx = nullptr;
    CHK_RET(hrtCtxGetCurrent(&ctx));
    serverOpRetry[group].thread.reset(new (std::nothrow) std::thread(&OpRetryManager::RetryStateMonitor, this,
        group, serverOpRetry[group].retryCtx, std::ref(serverOpRetry[group].startExec), ctx));
    CHK_SMART_PTR_NULL(serverOpRetry[group].thread);
    HCCL_INFO("[%s]group[%s] rank[%u], register to serverOpRetry success", __func__, group.c_str(), agentInfo.userRank);
    return HCCL_SUCCESS;
}

HcclResult OpRetryManager::UnRegisterOpRetryManager(const std::string& group)
{
    std::unique_lock<std::mutex> lock(ProcessLock_);
    CHK_PRT_RET(group.empty(),
        HCCL_ERROR("[OpRetryManager][UnRegisterOpRetryManager]params invalid, group is empty"), HCCL_E_PARA);
    HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister start", group.c_str());
    CHK_PRT_RET(initialized_ == false, HCCL_WARNING("OpRetryManager has been destroyed"), HCCL_SUCCESS);

    if (agentOpRetry_.find(group) != agentOpRetry_.end()) {
        agentOpRetry_[group].startExec = false;
        if (agentOpRetry_[group].thread != nullptr && agentOpRetry_[group].thread->joinable()) {
            agentOpRetry_[group].thread->join();
        }
        agentOpRetry_.erase(group);
        HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister agentOpRetry success", group.c_str());
    }

    if (serverOpRetry.find(group) != serverOpRetry.end()) {
        serverOpRetry[group].startExec = false;
        if (serverOpRetry[group].thread != nullptr && serverOpRetry[group].thread->joinable()) {
            serverOpRetry[group].thread->join();
        }
        serverOpRetry.erase(group);
        HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister serverOpRetry success", group.c_str());
    }
    HCCL_INFO("[UnRegister][OpRetryManager]group[%s] unregister success", group.c_str());
    OpRetryConnectionPub::DeInit(group);
    return HCCL_SUCCESS;
}

void OpRetryManager::RetryStateMonitor(const std::string &group, std::shared_ptr<RetryContext> retryCtx,
    const bool &startExec, HcclRtContext rtCtx)
{
    CHK_SMART_PTR_RET_NULL(retryCtx);
    CHK_SMART_PTR_RET_NULL(rtCtx);
    CHK_RET_NULL(hrtCtxSetCurrent(rtCtx));

    // 给当前线程添加名字
    SetThreadName("Hccl_OpRetry");

    HCCL_RUN_INFO("[%s]%s start, group[%s], rankId[%u], IpInfo[%s]", __func__, retryCtx->GetOpRetryMachineType(),
        group.c_str(), retryCtx->GetRankId(), retryCtx->GetDfxIpInfo());

    HcclResult ret = HCCL_SUCCESS;
    while(initialized_ && startExec) {
        ret = retryCtx->Request();
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("RetryStateMonitor group[%s] exec fail", group.c_str()), );
    }
    HCCL_INFO("RetryStateMonitor group[%s] exit, ret[%d], initialized_[%d], startExec[%d]",
        group.c_str(), ret, initialized_, startExec);
}

HcclResult OpRetryManager::AddLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList, bool incre)
{
    return OpretryLinkManage::GetInstance(deviceLogicID).AddLinkInfoByIdentifier(identifier, newTag, remoteRankList, incre);
}
 
HcclResult OpRetryManager::GetLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList)
{
    return OpretryLinkManage::GetInstance(deviceLogicID).GetLinkInfoByIdentifier(identifier, newTag, remoteRankList);
}
 
HcclResult OpRetryManager::DeleteLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier)
{
    return OpretryLinkManage::GetInstance(deviceLogicID).DeleteLinkInfoByIdentifier(identifier);
}

}