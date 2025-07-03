/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPRETRY_LINK_MANAGE_H
#define HCCL_OPRETRY_LINK_MANAGE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "log.h"
#include "aicpu_operator_pub.h"


namespace hccl {

constexpr u32 MAX_DEV_NUM = 16;

class OpretryLinkManage {
public:
    static OpretryLinkManage& GetInstance(s32 deviceLogicID);

    HcclResult AddLinkInfoByIdentifier(const std::string &identifier, const std::string &newTag, std::vector<u32> &remoteRankList, bool incre);
    HcclResult GetLinkInfoByIdentifier(const std::string &identifier, const std::string &newTag, std::vector<u32> &remoteRankList);
    HcclResult DeleteLinkInfoByIdentifier(const std::string &identifier);

private:
    OpretryLinkManage() = default;
    ~OpretryLinkManage();

    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<u32>>> allRemoteRankList_;

    std::mutex opretryLinkMutex_;
    bool isDeInit_ = false;
};

} // namespace hccl
#endif // HCCL_OPRETRY_RPING_H