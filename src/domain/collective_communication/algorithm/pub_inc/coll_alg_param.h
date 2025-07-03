/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_COMM_H
#define COLL_ALG_COMM_H

#include <string>
#include <set>
#include <unordered_set>

#include "hccl_common.h"
#include "hccl_types.h"
#include "transport_pub.h"
#include "stream_pub.h"
#include "local_notify.h"
#include "hccl_trace_info.h"
#include "common.h"
#include "threadManage.h"
#include "transport_common.h"
#include "common_pub.h"

namespace hccl {
using RankId = u32;

enum OpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

enum DeviceMode {
    HOST = 0,
    AICPU = 1
};

using LevelNSubCommTransport = std::vector<SingleSubCommTransport>;
using OpCommTransport = std::vector<LevelNSubCommTransport>;

struct AlgResourceRequest {
    u64 scratchMemSize = 0;
    u32 streamNum = 0;
    u32 notifyNum = 0;
    bool needAivBuffer = false;
    DeviceMode mode = DeviceMode::HOST;     // 用于区分是host模式，还是aicpu模式
    OpCommTransport opTransport;
    void Describe()
    {
        HCCL_DEBUG("[AlgResourceRequest], scratchMemSize[%u], streamNum[%u], notifyNum[%u], needAivBuffer[%u], "
            "DeviceMode[%d].", scratchMemSize, streamNum, notifyNum, needAivBuffer, mode);
    };
};

struct AlgResourceResponse {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
    std::vector<Stream> slaveStreams;
    std::vector<Stream> slaveDevStreams;
    std::vector<std::shared_ptr<LocalNotify> > notifiesMain; // Main Signals, 与Aux成对使用，大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesAux; // Auxiliary Signals, 与Main成对使用, 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesDevMain; // 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesDevAux; // 大小等同于slaveStreams
    OpCommTransport opTransportResponse; // 默认的Transport资源
    OpCommTransport opTransportResponseBackUp;  // Transport备资源 (借轨场景使用)
    std::vector<std::shared_ptr<ThreadManage>> threadManage;
};

enum class BatchSendRecvCurMode {
    SEND = 0,
    RECV = 1,
    SEND_RECV = 2,
    SEND_RECV_RESERVED
};

struct OpParam {
    std::string tag = "";
    Stream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank = 0;
    RankId srcRank = 0;
    bool aicpuUnfoldMode = false;
    HcclTraceInfo* opBaseAtraceInfo = nullptr;
    union {
        struct {
            u64 count;
            HcclDataType dataType;
            u64 strideCount;
        } DataDes = {0, HCCL_DATA_TYPE_RESERVED, 0};
        struct {
            void* counts;
            void* displs;
            HcclDataType dataType;
        } VDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            u64 sendCount;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
            void* sendCountMatrix;
        } All2AllDataDes;
        struct {
            HcclSendRecvItem* sendRecvItemsPtr;
            u32 itemNum;
            u32 curIterNum;
            BatchSendRecvCurMode curMode;
        } BatchSendRecvDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isZeroCopy = false;
    u32 index = 0;
};
}   // namespace hccl
#endif