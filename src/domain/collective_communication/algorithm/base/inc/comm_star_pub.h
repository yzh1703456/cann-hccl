/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_STAR_PUB_H
#define COMM_STAR_PUB_H

#include "comm_base_pub.h"

namespace hccl {
class CommStar : public CommBase {
public:
    explicit CommStar(const std::string &collectiveId, const u32 userRank,
                      const u32 userRankSize, const u32 rank, const u32 rankSize, const TopoType topoFlag,
                      const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool> &notifyPool,
                      std::map<HcclIpAddress, HcclNetDevCtx> &netDevCtxMap,
                      const IntraExchanger &exchanger, const std::vector<RankInfo> paraVector,
                      const DeviceMem& inputMem, const DeviceMem& outputMem, const bool isUsedRdmaLevel0,
                      const void* transportResourceInfoAddr, size_t transportResourceInfoSize,
                      const std::string &tag = "",
                      const NICDeployment nicDeployInner = NICDeployment::NIC_DEPLOYMENT_DEVICE,
                      const u32 subUserRankRoot = INVALID_VALUE_RANKID, bool isHaveCpuRank = false);

    ~CommStar() override;

protected:
    // 计算当前rank与其他rank之间的link个数:server/client两种角色,RING需要派生类实现
    HcclResult CalcLink() override;
    HcclResult MakeClientInfo(const u32 dstRank, RankInfo &dstRankInfo, bool isInterRdma, bool isInterHccs) override;
    HcclResult MakeServerInfo(const u32 dstRank, RankInfo &dstRankInfo, bool isInterRdma, bool isInterHccs) override;
    HcclResult CreateInterLinks() override;
    HcclResult SetMachinePara(MachineType machineType, const std::string &serverId, u32 dstRank,
        const std::vector<std::shared_ptr<HcclSocket> > &sockets, MachinePara &machinePara) override;
    void SetTransportParam(TransportPara &para, MachinePara &machinePara) override;
    HcclResult CreateExchangerNetwork() override; // server间HCCS通信模式下，创建节点间的建链关系
    HcclResult GetChipType(int64_t &devType, u32 devLogicId);

    HcclResult GetDevIP(const HcclIpAddress& hostIp, const u32& devicePhyId,
        HcclIpAddress& ip);
    HcclResult CreateLinksThread(
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> &serverSocketsMap,
        std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> &clientSocketsMap);
};
}  // namespace hccl
#endif /* COMM_STAR_PUB_H */
