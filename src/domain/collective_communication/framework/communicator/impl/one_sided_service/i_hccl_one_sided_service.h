/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef I_HCCL_ONE_SIDED_SERVICE_H
#define I_HCCL_ONE_SIDED_SERVICE_H

#include <memory>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "hccl_network_pub.h"
#include "hccl_socket.h"
#include "topoinfo_struct.h"
#include "hccl_socket_manager.h"
#include "notify_pool.h"

namespace hccl {
class IHcclOneSidedService {
public:
    IHcclOneSidedService(std::unique_ptr<HcclSocketManager> &socketManager,
        std::unique_ptr<NotifyPool> &notifyPool);

    virtual ~IHcclOneSidedService() = default;

    // 为了尽可能保障框架依赖兼容性，除了引用以外，参数不通过构造函数传递
    virtual HcclResult Config(const HcclDispatcher &dispatcher,
        const HcclRankLinkInfo &localRankInfo, const RankTable_t *rankTable);

    virtual HcclResult SetNetDevCtx(const HcclNetDevCtx &netDevCtx, bool useRdma);
    virtual HcclResult GetNetDevCtx(HcclNetDevCtx &netDevCtx, bool useRdma);

protected:
    HcclNetDevCtx netDevRdmaCtx_{};
    HcclNetDevCtx netDevIpcCtx_{};
    HcclDispatcher dispatcher_{};
    HcclRankLinkInfo localRankInfo_{};
    HcclRankLinkInfo localRankVnicInfo_{};
    const RankTable_t *rankTable_{};
    std::unique_ptr<HcclSocketManager> &socketManager_;
    std::unique_ptr<NotifyPool> &notifyPool_;
};
}

#endif