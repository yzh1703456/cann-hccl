/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "i_hccl_one_sided_service.h"

namespace hccl {
using namespace std;

IHcclOneSidedService::IHcclOneSidedService(unique_ptr<HcclSocketManager> &socketManager,
    unique_ptr<NotifyPool> &notifyPool)
    : socketManager_(socketManager), notifyPool_(notifyPool)
{
}

HcclResult IHcclOneSidedService::Config(const HcclDispatcher &dispatcher,
    const HcclRankLinkInfo &localRankInfo, const RankTable_t *rankTable)
{
    CHK_PTR_NULL(dispatcher);
    CHK_PTR_NULL(rankTable);

    dispatcher_ = dispatcher;
    localRankInfo_ = localRankInfo;
    localRankVnicInfo_ = localRankInfo;
    rankTable_ = rankTable;

    return HCCL_SUCCESS;
}

HcclResult IHcclOneSidedService::SetNetDevCtx(const HcclNetDevCtx &netDevCtx, bool useRdma)
{
    if (useRdma) {
        netDevRdmaCtx_ = netDevCtx;
        CHK_PTR_NULL(netDevRdmaCtx_);
    } else {
        netDevIpcCtx_ = netDevCtx;
        CHK_PTR_NULL(netDevIpcCtx_);
    }
    return HCCL_SUCCESS;
}

HcclResult IHcclOneSidedService::GetNetDevCtx(HcclNetDevCtx &netDevCtx, bool useRdma)
{
    if (useRdma) {
        netDevCtx = netDevRdmaCtx_;
    } else {
        netDevCtx = netDevIpcCtx_;
    }
    return HCCL_SUCCESS;
}

}
