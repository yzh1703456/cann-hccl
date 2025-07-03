/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_HEARTBEAT_H
#define HCCL_HEARTBEAT_H

#include <thread>
#include <map>
#include <mutex>

#include "hccl/hccl_types.h"
#include "log.h"
#include "reference_map.h"
#include "ring_buffer.h"
#include "common.h"
#include "sal_pub.h"
#include "hccl_socket_manager.h"
#include "transport_pub.h"
namespace hccl {
using RankId = u32;
constexpr u32 BROADCAST_INTERVAL = 50; // 背景线程执行周期为50 ms
constexpr u32 STUCK_INTERVAL = 300000; // 5min监控一次,默认 300000 ms
constexpr u32 STUCK_COUNT = STUCK_INTERVAL / BROADCAST_INTERVAL;
using UIDType = struct HcclHeartBeatUid {
    char id[512] = {0}; // ip[IP_ADDRESS_BUFFER_LEN] + ifname[MAX_INTERFACE_NAME_LEN] + devid 最大不超过512字节
    bool operator == (const HcclHeartBeatUid &that) const
    {
        return std::string(this->id) == std::string(that.id);
    }
    bool operator != (const HcclHeartBeatUid &that) const
    {
        return std::string(this->id) != std::string(that.id);
    }
    bool operator < (const HcclHeartBeatUid &that) const
    {
        return std::string(this->id) < std::string(that.id);
    }
};
}

namespace std {
template <> class hash<hccl::HcclHeartBeatUid> {
public:
    size_t operator () (const hccl::HcclHeartBeatUid &uid) const
    {
        return hash<string>()(string(uid.id));
    }
};
}

namespace hccl {
constexpr u8 HAS_CONN = 1;
constexpr u8 NO_CONN = 0;
constexpr u32 TIME_FROM_1900 = 1900;

enum class HeartBeatStatus {
    HEARTBEAT_OK,
    HEARTBEAT_LOST,
    HEARTBEAT_NOTIFY,
    HEARTBEAT_CQE_ERR,
    HEARTBEAT_OPRETRY_NOT_SUPPORT,
    HEARTBEAT_STUCK
};
const std::map<HeartBeatStatus, std::string> HEARTBEAT_STATUS_STR_MAP{
    {HeartBeatStatus::HEARTBEAT_OK, "OK"},
    {HeartBeatStatus::HEARTBEAT_LOST, "LOST"},
    {HeartBeatStatus::HEARTBEAT_NOTIFY, "NOTIFY"},
    {HeartBeatStatus::HEARTBEAT_CQE_ERR, "ERROR CQE"},
    {HeartBeatStatus::HEARTBEAT_OPRETRY_NOT_SUPPORT, "OPRETRY NOT SUPPORT"},
    {HeartBeatStatus::HEARTBEAT_STUCK, "STUCK"}
};
inline std::string GetHeartBeatStatusStr(HeartBeatStatus  status)
{
    auto iter = HEARTBEAT_STATUS_STR_MAP.find(status);
    if (iter == HEARTBEAT_STATUS_STR_MAP.end()) {
        return "Unknown";
    } else {
        return iter->second;
    }
}

struct CounterStat {
    std::pair<int32_t, int32_t> oldCounter{0, 0};
    std::pair<int32_t, int32_t> newCounter{0, 0};
    std::uint64_t issueCnt = 0;
    bool isNeedDetect = false;
    bool isFirst = true;
    std::uint64_t couterPrintInter = STUCK_COUNT;
    CounterStat() {};
};

struct HeartBeatFrame {
    UIDType src;
    UIDType dst;
    UIDType crimer;
    UIDType informer;
    HeartBeatStatus status = HeartBeatStatus::HEARTBEAT_OK;
    HcclUs TOARelative; // time of arrival (Relative)
    HcclSystemTime TOASystem; // time of arrival (System)
    HeartBeatFrame() {}
    HeartBeatFrame(UIDType &crimer, UIDType &informer, HeartBeatStatus status, HcclUs TOARelativeIn,
        HcclSystemTime TOASystemIn)
        : crimer(crimer), informer(informer), status(status), TOARelative(TOARelativeIn),
        TOASystem(TOASystemIn)
    {}
    HeartBeatFrame(UIDType &src, UIDType &dst, UIDType &crimer, UIDType &informer, HeartBeatStatus status)
        : src(src), dst(dst), crimer(crimer), informer(informer), status(status)
    {}
};

struct ConnInfo {
    std::shared_ptr<HcclSocket> socket = nullptr;
    std::queue<HeartBeatFrame> sendBuffer;
    u32 restSize = 0;
    RingBuffer recvBuffer;
    u32 lostNum = 0;
    bool newConn = false;
    std::vector<SocketWlistInfo> wlistInfosVec;
    ConnInfo() {}
    ConnInfo(bool newConn, std::shared_ptr<HcclSocket> &socket)
        : socket(socket), newConn(newConn)
    {}
};

using ErrQpnInfo = struct TagErrQpnInfo {
    CqeInfo cqeInfo;
    u32 qpn;
    TagErrQpnInfo() {}
    TagErrQpnInfo(const CqeInfo &cqeInfo, u32 qpn)
        : cqeInfo(cqeInfo), qpn(qpn)
    {}
    bool operator<(const TagErrQpnInfo& other) const {
        return qpn < other.qpn;
    }
};

using ErrCqeInfo = struct TagErrCqeInfo {
    CqeInfo cqeInfo;
    std::string identifier;
    RankId remoteRank;
    u32 qpn;
    TagErrCqeInfo() {}
    TagErrCqeInfo(CqeInfo &cqeInfo, const std::string &identifier, RankId remoteRank, u32 qpn)
        : cqeInfo(cqeInfo), identifier(identifier), remoteRank(remoteRank), qpn(qpn)
    {}
};

class Heartbeat {
public:
    static Heartbeat& GetInstance(s32 deviceLogicID);
    HcclResult RegisterToHeartBeat(u32 userRank, DevType devType, std::vector<RankInfo> &rankInfoList, const u32 port,
        const bool isNeedNic, const std::string &commIdentifier, bool useSuperPodMode, bool isUsedRdmaLevel0,
        bool retryEnable = false, bool backupEnable = false);
    HcclResult RegisterToHeartBeat(u32 userRank, DevType devType, std::vector<RankInfo> &rankInfoList, const u32 port,
        const bool isNeedNic, u32 peerRankId, const std::string &commIdentifier, const std::string& tag,
        bool useSuperPodMode, bool isUsedRdmaLevel0, bool retryEnable = false, bool backupEnable = false);
    HcclResult UnRegisterRanks(const std::string& group = HCCL_WORLD_GROUP);
    // 集合通信，解开注册
    void UnRegisterToHeartBeat(DevType devType, const std::string &commIdentifier);
    // 非点对点通信，解开注册
    void UnRegisterToHeartBeat(DevType devType, const std::string &commIdentifier, const std::string &tag);
    HcclResult CheckErrorCqe(const std::string &identifier, HcclResult &result);
    HcclResult SetRankPortInfo(bool isUseRankPort, std::vector<u32> &ranksPort, std::vector<u32> &vnicRanksPorts,
        bool devPortSwitchOn);
    std::vector<std::string> GetErrStatusVec();
    HcclResult GetQpnErr(const std::string &identifier, std::set<std::tuple<u32, u32, u32>> &qpErrSet);
    HcclResult BroadcastCqeErr(const std::string &identifier);
    HcclResult ClearAllCqeErr(const std::string &identifier);
    HcclResult ClearCqeErr(const std::string &identifier, u32 remoteRank, u32 qpn = 0);
    void SetOpretryErr();
 
private:
    Heartbeat() = default;
    ~Heartbeat();
    HcclResult Init(const RankInfo& locRank, const bool useSuperPodMode, const bool isNeedNic, const u32 port);
    HcclResult DeInit();
    HcclResult RegisterRanks(const RankInfo& locRank, std::vector<RankInfo>& rankInfos, const u32 port,
        const bool isNeedNic, const std::string& group = HCCL_WORLD_GROUP, bool isUsedRdmaLevel0 = false,
        bool isUsedRdma = false);
    std::string GetConnTag(HcclSocketRole role, UIDType &rem);
    HcclResult GetConnInfo(RankInfo& remRank, bool isUsedRdmaLevel0, HcclSocketRole role, HcclSocketType type,
        std::map<UIDType, ConnInfo>& needConnectRank);
    template <typename T> HcclResult GetSamePlaneConnInfo(HcclSocketType type, std::vector<std::pair<T, u32>>& connVec,
        T& locId, std::vector<RankInfo>& rankInfos, std::map<UIDType, ConnInfo>& needConnectRank,
        bool isUsedRdmaLevel0, u32 worldRank);
    HcclResult GetConnectRank(const RankInfo& locRank, std::vector<RankInfo>& rankInfos, std::map<UIDType,
        ConnInfo>& needConnectRank, bool isUsedRdmaLevel0, bool isUsedRdma = false);
    UIDType GetUId(const RankInfo& rankInfo) const;
    std::string FormatUId(const UIDType& uid) const;
    HcclResult SendFrame(UIDType &dst, UIDType &crimer, UIDType &informer, HeartBeatStatus status);
    HcclResult RecvFrame(UIDType &src);
    HcclResult ParseFrame(HeartBeatFrame& bf, UIDType &src);
    void SetStatus(UIDType &crimer, UIDType &informer, HeartBeatStatus status, bool needBroadcast = true);
    void HeartbeatStatusMonitor();
    void ProcessExceptionEvent();
    void ProcessCqeErrInfo();
    HcclResult CreateHeartConnect(const std::string &group, std::map<UIDType, ConnInfo> &needConnectRank, bool &isLoop,
        std::chrono::time_point<std::chrono::steady_clock> &startTime);
    void DelErrorSocket();
    bool IsKeyEvent(HeartBeatFrame &event, HcclUs curTime);
    void MakeErrMsg(std::queue<HeartBeatFrame> &keyEvents, std::vector<std::string> &errStatusVec);
    std::vector<std::string> PrintEvents(std::map<HeartBeatStatus, std::queue<HeartBeatFrame>> &keyEvents);
	void StuckDetection(uint64_t &cnt, CounterStat &counterStat);
    void InitStuckDetection(CounterStat &counterStat);
    void PrintAndBroadCastErrorCqe(const ErrCqeInfo &info);
    void SaveQpnForOpRetry(const ErrCqeInfo &info);
    void OpRetryCQEHandle(const HcclNetDevCtx netDevCtx);
    bool GetRetryEnable(const ErrCqeInfo &info);
    HcclResult ClearRetryEnableMapItem(const std::string &identifier);
    void ProcessCqeErrInfoByNetDevCtx(const HcclIpAddress &nicIp);
    bool IsEnableBackupLink();
    void RegisterRetryInfo(const std::string &commIdentifier, bool retryEnable, bool backupEnable);
    HcclResult InitNic(const NicType nicType, const s32 devicePhyId, const s32 deviceLogicId,
        const hccl::HcclIpAddress ip, const u32 port, const bool isBackUp = false);
    u32 GetPort(HcclSocketType type, u32 remoteUserRank, u32 remoteDeviceId);
    u32 GetHostPort(s32 devicePhyId);
    HcclResult PrepareConnect(ConnInfo &info);

    struct Status {
        HeartBeatStatus status = HeartBeatStatus::HEARTBEAT_OK;
        UIDType informer;
        bool needBroadcast = false;
        Status() {}
    };

    HcclIpAddress vnicIp_;
    HcclIpAddress nicIp_;
    HcclIpAddress backupNicIp_;
    u32 devicePhyId_;
    u32 deviceBackUpPhyId_;
    u32 superDeviceId_;
    NICDeployment nicDeploy_;
    UIDType uid_;
    bool initialized_ = false;
    u32 lostThreshold_ = 0;
    bool isDeInit_ = false;
    bool startSendRecvTask_ = false;
    std::map<std::string, std::map<UIDType, u8>> groupMap_;
    ReferenceMap<UIDType, ConnInfo> rankId2SocketMap_;
    ReferenceMap<UIDType, Status> rankId2StatusMap_;
    std::unique_ptr<std::thread> sendRecvThread_;
    std::queue<HeartBeatFrame> errStatusQueue_;
    std::queue<UIDType> errRankQueue_;
    std::mutex ProcessLock_;
    u32 deviceLogicId_;
    u32 deviceBackupLogicId_;
    std::map<std::string, std::set<HcclIpAddress>> remoteIpMap;
    std::set<u32> qpnDissociativeSet;
    std::mutex remoteIpMutex_;
    bool isUseRankPort_{ false };
    bool devPortSwitchOn_{ false };
    std::vector<u32> nicRanksPorts_;
    std::vector<u32> vnicRanksPorts_;
    std::vector<UIDType> errorSocket_;
    std::map<std::string, std::map<u32, std::set<ErrQpnInfo>>> rankMapForRetryAgent;
    std::mutex qpnMapMutexForRetry_;
    std::map<std::string, bool> retryEnableTable_;
    std::mutex retryEnableMutex_;
    std::set<std::string> backupEnableTable_;
    std::mutex backupEnableMutex_;
    std::mutex ctxMapMutex_;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;
    std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> listenSocketMap_;
};
} // namespace hccl

#endif // HCCL_HEARTBEAT_H