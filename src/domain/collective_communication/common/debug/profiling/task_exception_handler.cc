/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <iostream>
#include <cstdint>
#include <iomanip>
#include <array>
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "task_exception_handler.h"
#include "sal_pub.h"
#include "../../../algorithm/pub_inc/common.h"
#include "runtime/rt_error_codes.h"

using namespace hccl;
using namespace std;
GetErrStatusVecCallBack g_GetErrStatusVecCallBack = nullptr;
std::mutex g_communicatorCallbackMapMutex;
array<map<s32, GetAicpuTaskExceptionCallBack>, MAX_MODULE_DEVICE_NUM> g_communicatorCallbackMap;
std::mutex g_commHadCallbackArrayMutex;
array<bool, MAX_MODULE_DEVICE_NUM> g_commHadCallbackArray = {false};
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void RegisterGetErrStatusVecCallBack(GetErrStatusVecCallBack p1)
{
    g_GetErrStatusVecCallBack = p1;
    return;
}

void RegisterGetAicpuTaskExceptionCallBack(s32 streamId, u32 deviceLogicId, GetAicpuTaskExceptionCallBack p1)
{
    lock_guard<mutex> lock(g_communicatorCallbackMapMutex);
    g_communicatorCallbackMap[deviceLogicId].emplace(streamId, p1);
    return;
}
#ifdef __cplusplus
}
#endif // __cplusplus
namespace hccl {
    namespace hccl_alg {
        std::vector<std::string> GetErrStatusVec(s32 deviceLogicID)
        {
            if (g_GetErrStatusVecCallBack != nullptr) {
                return g_GetErrStatusVecCallBack(deviceLogicID);
            } else {
                HCCL_RUN_WARNING("[GetErrStatusVec]g_GetErrStatusVecCallBack is nullptr.");
            }
            return std::vector<std::string>();
        }
    }
}

static std::string g_kernelNameList[] = {
 "aiv_all_gather_91093_smalldata_graph.h",
 "aiv_all_gather_910b_bigdata.h",
 "aiv_all_gather_910b_graph.h",
 "aiv_all_gather_910B_rdma_graph.h",
 "aiv_all_gather_910B_rdma.h",
 "aiv_all_gather_910b_smalldata.h",
 "aiv_all_gather_v_910b_bigdata.h",
 "aiv_all_gather_v_910b_smalldata.h",
 "aiv_all_reduce_910b_bigdata_graph.h",
 "aiv_all_reduce_910b_bigdata.h",
 "aiv_all_reduce_910b_middata.h",
 "aiv_all_reduce_910b_rdma_middata_graph_step1",
 "aiv_all_reduce_910b_rdma_middata_step1",
 "aiv_all_reduce_910b_rdma_smalldata_graph_step1",
 "aiv_all_reduce_910b_rdma_smalldata_step1",
 "aiv_all_reduce_910b_smalldata_graph.h",
 "aiv_all_reduce_910b_smalldata.h",
 "aiv_all_to_all_91093_base.h",
 "aiv_all_to_all_91093_graph.h",
 "aiv_all_to_all_91093.h",
 "aiv_all_to_all_910b_smalldata.h",
 "aiv_all_to_all_rdma_910b.h",
 "aiv_all_to_all_v_91093_graph.h",
 "aiv_all_to_all_v_91093.h",
 "aiv_all_to_all_v_91093_single.h",
 "aiv_all_to_all_v_910b_graph.h",
 "aiv_all_to_all_v_910b.h",
 "aiv_all_to_all_vc_910b_graph.h",
 "aiv_all_to_all_vc_910b.h",
 "aiv_all_to_all_vc_910b_no_loop.h",
 "aiv_reduce_scatter_91093_smalldata_graph.h",
 "aiv_reduce_scatter_910b_bigdata.h",
 "aiv_reduce_scatter_910b_graph.h",
 "aiv_reduce_scatter_910b_middata.h",
 "aiv_reduce_scatter_910b_rdma_graph.h",
 "aiv_reduce_scatter_910b_rdma.h",
 "aiv_reduce_scatter_910b_smalldata.h",
 "aiv_reduce_scatter_v_910b_bigdata.h",
 "aiv_reduce_scatter_v_910b_middata.h",
 "aiv_reduce_scatter_v_910b_smalldata.h",
 "aiv_sync_910b.h",
 "aiv_all_gather_91093_smalldata.h",
 "aiv_reduce_scatter_91093_smalldata.h",
 "aiv_all_reduce_910b_rdma_middata_graph_step2",
 "aiv_all_reduce_910b_rdma_middata_step2",
 "aiv_all_reduce_910b_rdma_smalldata_graph_step2",
 "aiv_all_reduce_910b_rdma_smalldata_step2",
 "aiv_all_reduce_910b_rdma_smalldata_graph_step3",
 "aiv_all_reduce_910b_rdma_smalldata_step3",
};

std::string GetTaskName(TaskType taskType, bool isAlgInfo = false);
std::string GetLinkTypeName(LinkType linkInput);
std::string GetAlgTypeStr(AlgType algType);
std::string GetTaskBriefsName(TaskType taskType);

namespace {
constexpr u32 STREAM_COUNT_UPPER_LIMIT = 2048; // stream 数量最大值2048，防止内存占用量过大
constexpr u32 TASK_COUNT_UPPER_LIMIT = 2048; // task 数量最大值2048，防止内存占用量过大
constexpr u32 TASK_COUNT_UPPER_LIMIT_OP_BASE = 65535; // 单算子模式task数量最大值
constexpr u32 TASK_CONTEXT_SIZE = 50; // task 执行失败时打印前序task的数量
constexpr u32 TASK_CONTEXT_INFO_SIZE = LOG_TMPBUF_SIZE - 50; // task 执行失败时打印前序task信息的长度限制
constexpr u32 PRINT_TASK_AIV_INFO_COUNT = 10;
constexpr u32 TASK_AIV_KERNEL_NUM = 49;
constexpr u32 AIV_KERNEL_FLAG_SIZE_PER_OP = 6;
u32 maxStrCount = 0;
u32 maxTaskCount = 0;
}
array<map<int, shared_ptr<deque<TaskInfo>>>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::taskMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::taskMapMutex;
array<map<int, shared_ptr<deque<FFTSOpInfo>>>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opMapMutex;
array<std::map<int, shared_ptr<std::deque<std::pair<std::shared_ptr<FFTSOpInfo>, \
    std::shared_ptr<std::vector<CtxInfo>>>>>>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opCtxInfo;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::opCtxInfoMutex;
array<std::vector<CtxInfo>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::ctxInfoArray;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::ctxInfoVectorMutex;
array<std::map<const std::string, std::pair<const std::string, std::shared_ptr<GroupRankInfo>>>, \
    MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupRankMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupRankMapMutex;
array<std::map<const std::string, std::shared_ptr<std::queue<OpDataInfo>>>, \
    MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::tagOpDataMap;
array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::tagOpDataMapMutex;
std::array<std::map<const std::string, std::string>, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupUdiMap;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> TaskExceptionHandler::groupUdiMapMutex;
TaskInfo::TaskInfo(u32 &streamID, u32 &taskID, string &tag, TaskType &taskType, AlgType &algType, u32 &index,
    const TaskParaDMA &para) : streamID(streamID), taskID(taskID), tag(tag), taskType(taskType), isAlgInfo(false),
    algType(algType), index(index)
{
    taskPara.DMA.src = para.src;
    taskPara.DMA.dst = para.dst;
    taskPara.DMA.size = para.size;
    taskPara.DMA.notifyID = para.notifyID;
    taskPara.DMA.linkType = para.linkType;
    taskPara.DMA.remoteUserRank = para.remoteUserRank;
}
TaskInfo::TaskInfo(u32 &streamID, u32 &taskID, string &tag, TaskType &taskType, AlgType &algType, u32 &index,
    const TaskParaReduce &para) : streamID(streamID), taskID(taskID), tag(tag), taskType(taskType), isAlgInfo(false),
    algType(algType), index(index)
{
    taskPara.Reduce.src = para.src;
    taskPara.Reduce.dst = para.dst;
    taskPara.Reduce.size = para.size;
    taskPara.Reduce.op = para.op;
    taskPara.Reduce.dataType = para.dataType;
    taskPara.Reduce.linkType = para.linkType;
    taskPara.Reduce.remoteUserRank = para.remoteUserRank;
}
TaskInfo::TaskInfo(u32 &streamID, u32 &taskID, string &tag, TaskType &taskType, AlgType &algType, u32 &index,
    const TaskParaNotify &para) : streamID(streamID), taskID(taskID), tag(tag), taskType(taskType), isAlgInfo(false),
    algType(algType), index(index)
{
    taskPara.Notify.notifyID = para.notifyID;
    taskPara.Notify.stage = para.stage;
    taskPara.Notify.remoteUserRank = para.remoteUserRank;
}
TaskInfo::TaskInfo(u32 &streamID, u32 &taskID, string &tag, const TaskParaAiv& para) :
    streamID(streamID), taskID(taskID), tag(tag), isAlgInfo(true)
{
    taskPara.Aiv.cmdType = para.cmdType;
    taskPara.Aiv.tag = para.tag;
    taskPara.Aiv.size = para.size;
    taskPara.Aiv.blockDim = para.blockDim;
    taskPara.Aiv.rankSize = para.rankSize;
    taskPara.Aiv.flagMem = para.flagMem;
    taskPara.Aiv.aivRdmaStep = para.aivRdmaStep;
}
CtxInfo::CtxInfo(TaskType &taskType, const TaskParaDMA &para)
    : taskType(taskType)
{
    ctxPara.DMA.src = para.src;
    ctxPara.DMA.dst = para.dst;
    ctxPara.DMA.size = para.size;
    ctxPara.DMA.notifyID = para.notifyID;
    ctxPara.DMA.linkType = para.linkType;
    ctxPara.DMA.remoteUserRank = para.remoteUserRank;
}
CtxInfo::CtxInfo(TaskType &taskType, const TaskParaReduce &para)
    : taskType(taskType)
{
    ctxPara.Reduce.src = para.src;
    ctxPara.Reduce.dst = para.dst;
    ctxPara.Reduce.size = para.size;
    ctxPara.Reduce.op = para.op;
    ctxPara.Reduce.dataType = para.dataType;
    ctxPara.Reduce.linkType = para.linkType;
    ctxPara.Reduce.remoteUserRank = para.remoteUserRank;
}
CtxInfo::CtxInfo(TaskType &taskType, const TaskParaNotify &para)
    : taskType(taskType)
{
    ctxPara.Notify.notifyID = para.notifyID;
    ctxPara.Notify.stage = para.stage;
    ctxPara.Notify.remoteUserRank = para.remoteUserRank;
}

string TaskInfo::GetBaseInfoStr() // 防止tag字符串过长，base信息和para信息分开打印
{
    string taskContent;
    taskContent += "streamID:[";
    taskContent += std::to_string(streamID);
    taskContent += "], taskID[";
    taskContent += std::to_string(taskID);
    taskContent += "], taskType[";
    taskContent += GetTaskName(taskType, isAlgInfo);
    taskContent += "], tag[";
    taskContent += tag;
    taskContent += "], ";
    taskContent += GetAlgTypeStr(algType);
    return taskContent;
}

string TaskInfo::GetRankInfo()
{
    u32 remoteRank = INVALID_VALUE_RANKID;
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            remoteRank = taskPara.DMA.remoteUserRank;
            break;
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            remoteRank = taskPara.Reduce.remoteUserRank;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            remoteRank = taskPara.Notify.remoteUserRank;
            break;
        default:
            return "/";
    }
    return (remoteRank == INVALID_VALUE_RANKID) ? "/" : to_string(remoteRank);
}

string TaskInfo::GetNotifyInfo()
{
    u64 notifyInfo = INVALID_U64;
    switch (taskType) {
        case TaskType::TASK_RDMA:
            notifyInfo = taskPara.DMA.notifyID;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            notifyInfo = taskPara.Notify.notifyID;
            break;
        default:
            return "/";
    }
    if (notifyInfo == INVALID_U64) {
            return "/";
        } else {
            stringstream paraStr;
            // NotifyId取后八位16进制数进行打印
            paraStr << std::hex << static_cast<u32>(notifyInfo);
            return paraStr.str();
        }
}

string TaskInfo::GetParaInfoStr()
{
    if(isAlgInfo){
        return GetParaAiv();
    }
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            return GetParaDMA();
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            return GetParaReduce();
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            return GetParaNotify();
        default:
            return "unkown task";
    }
}

string TaskInfo::GetParaDMA()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(taskPara.DMA.src)) << "], dst:"
            << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(taskPara.DMA.dst)) << "], size:"
            << "[0x" << std::hex << static_cast<u64>(taskPara.DMA.size) << "], notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字符长度对齐
            << std::setfill('0') << taskPara.DMA.notifyID << "], link type:["
            << GetLinkTypeName(taskPara.DMA.linkType) << "], remote rank:["
            << ((taskPara.DMA.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                to_string(taskPara.DMA.remoteUserRank)) << "]";
    retStr += paraStr.str();
    return retStr;
}

string TaskInfo::GetParaNotify()
{
    string retStr;
    stringstream paraStr;
    paraStr << "notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字节长度对齐
            << std::setfill('0') << taskPara.Notify.notifyID << "], stage:[" << taskPara.Notify.stage
            << "], remote rank:[" << ((taskPara.Notify.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
            to_string(taskPara.Notify.remoteUserRank)) << "]";
    retStr += paraStr.str();
    return retStr;
}

string TaskInfo::GetParaReduce()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(taskPara.Reduce.src)) << "], dst:"
            << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(taskPara.Reduce.dst)) << "], size:"
            << "[0x"
            << std::hex << static_cast<u64>(taskPara.Reduce.size * ProfilerBase::sizeOf[taskPara.Reduce.dataType])
            << "], op:[" << std::to_string(ProfilerBase::opString[taskPara.Reduce.op]) << "], data type:["
            << std::to_string(ProfilerBase::dataTypeString[taskPara.Reduce.dataType]) << "], link type:["
            << GetLinkTypeName(taskPara.Reduce.linkType) << "], remote rank:["
            << ((taskPara.Reduce.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                to_string(taskPara.Reduce.remoteUserRank)) << "]";
    retStr += paraStr.str();
    return retStr;
}

string TaskInfo::GetParaAiv()
{
    string retStr;
    stringstream paraStr;
    paraStr << "cmdType:[" << static_cast<int>(taskPara.Aiv.cmdType) << "], "
            << "tag: [" << taskPara.Aiv.tag << "], " 
            << "size: [" << taskPara.Aiv.size << "], " 
            << "blockDim: [" << taskPara.Aiv.blockDim << "], "
            << "rankSize: [" << taskPara.Aiv.rankSize << "], "
            << "aivRdmaStep: [" << taskPara.Aiv.aivRdmaStep <<"], "
            << "flagMem: [0x" << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(taskPara.Aiv.flagMem))
            << "]";

    retStr += paraStr.str();
    return retStr;
}

u32 TaskInfo::GetRemoteUserRank()
{
    return taskPara.Notify.remoteUserRank;
}

string CtxInfo::GetCtxBaseInfoStr() // 防止tag字符串过长，base信息和para信息分开打印
{
    string taskContent;
    taskContent += "taskType[";
    taskContent += GetTaskName(taskType);
    taskContent += "].";
    return taskContent;
}

string CtxInfo::GetCtxRankInfo()
{
    u32 remoteRank = INVALID_VALUE_RANKID;
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            remoteRank = ctxPara.DMA.remoteUserRank;
            break;
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            remoteRank = ctxPara.Reduce.remoteUserRank;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            remoteRank = ctxPara.Notify.remoteUserRank;
            break;
        default:
            return "/";
    }
    return (remoteRank == INVALID_VALUE_RANKID) ? "/" : to_string(remoteRank);
}

string CtxInfo::GetCtxNotifyInfo()
{
    u64 notifyInfo = INVALID_U64;
    switch (taskType) {
        case TaskType::TASK_RDMA:
            notifyInfo = ctxPara.DMA.notifyID;
            break;
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            notifyInfo = ctxPara.Notify.notifyID;
            break;
        default:
            return "/";
    }
    if (notifyInfo == INVALID_U64) {
            return "/";
        } else {
            stringstream paraStr;
            // NotifyId取后八位16进制数进行打印
            paraStr << std::hex << static_cast<u32>(notifyInfo);
            return paraStr.str();
        }
}


string CtxInfo::GetCtxParaInfoStr()
{
    switch (taskType) {
        case TaskType::TASK_SDMA:
        case TaskType::TASK_RDMA:
            return GetCtxParaDMA();
        case TaskType::TASK_REDUCE_INLINE:
        case TaskType::TASK_REDUCE_TBE:
            return GetCtxParaReduce();
        case TaskType::TASK_NOTIFY_RECORD:
        case TaskType::TASK_NOTIFY_WAIT:
            return GetCtxParaNotify();
        default:
            return "unkown task";
    }
}

string CtxInfo::GetCtxParaDMA()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(ctxPara.DMA.src)) << "], dst:"
            << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(ctxPara.DMA.dst)) << "], size:"
            << "[0x" << std::hex << static_cast<u64>(ctxPara.DMA.size) << "], notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字符长度对齐
            << std::setfill('0') << ctxPara.DMA.notifyID << "], link type:["
            << GetLinkTypeName(ctxPara.DMA.linkType) << "], remote rank:["
            << ((ctxPara.DMA.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                to_string(ctxPara.DMA.remoteUserRank)) << "]";
    retStr += paraStr.str();
    return retStr;
}

string CtxInfo::GetCtxParaNotify()
{
    string retStr;
    stringstream paraStr;
    paraStr << "notify id:"
            << "[0x" << std::hex << std::setw(16) // 16字节长度对齐
            << std::setfill('0') << ctxPara.Notify.notifyID << "], stage:[" << ctxPara.Notify.stage
            << "], remote rank:[" << ((ctxPara.Notify.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
            to_string(ctxPara.Notify.remoteUserRank)) << "]";
    retStr += paraStr.str();
    return retStr;
}

string CtxInfo::GetCtxParaReduce()
{
    string retStr;
    stringstream paraStr;
    paraStr << "src:" << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(ctxPara.Reduce.src)) << "], dst:"
            << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(ctxPara.Reduce.dst)) << "], size:"
            << "[0x"
            << std::hex << static_cast<u64>(ctxPara.Reduce.size * ProfilerBase::sizeOf[ctxPara.Reduce.dataType])
            << "], op:[" << std::to_string(ProfilerBase::opString[ctxPara.Reduce.op]) << "], data type:["
            << std::to_string(ProfilerBase::dataTypeString[ctxPara.Reduce.dataType]) << "], link type:["
            << GetLinkTypeName(ctxPara.Reduce.linkType) << "], remote rank:["
            << ((ctxPara.Reduce.remoteUserRank == INVALID_VALUE_RANKID) ? "local" :
                to_string(ctxPara.Reduce.remoteUserRank)) << "]";
    retStr += paraStr.str();
    return retStr;
}

u32 CtxInfo::GetCtxRemoteUserRank()
{
    return ctxPara.Notify.remoteUserRank;
}

std::string GetTaskName(TaskType taskType, bool isAlgInfo)
{
    std::string taskName;

    if (isAlgInfo){
        taskName = "Task AIV";
        return taskName;
    }

    switch (taskType) {
        case TaskType::TASK_SDMA:
            taskName += "Memcpy";
            break;
        case TaskType::TASK_RDMA:
            taskName += "RDMASend";
            break;
        case TaskType::TASK_REDUCE_INLINE:
            taskName += "Reduce Inline";
            break;
        case TaskType::TASK_REDUCE_TBE:
            taskName += "Reduce TBE";
            break;
        case TaskType::TASK_NOTIFY_RECORD:
            taskName += "Notify Record";
            break;
        case TaskType::TASK_NOTIFY_WAIT:
            taskName += "Notify Wait";
            break;
        default:
            return "unkown task";
    }

    return taskName;
}
std::string GetTaskBriefsName(TaskType taskType)
{
    std::string taskName;
    switch (taskType) {
        case TaskType::TASK_SDMA:
            taskName += "M";
            break;
        case TaskType::TASK_RDMA:
            taskName += "RS";
            break;
        case TaskType::TASK_REDUCE_INLINE:
            taskName += "IR";
            break;
        case TaskType::TASK_REDUCE_TBE:
            taskName += "R";
            break;
        case TaskType::TASK_NOTIFY_RECORD:
            taskName += "NR";
            break;
        case TaskType::TASK_NOTIFY_WAIT:
            taskName += "NW";
            break;
        default:
            return "unkown task";
    }

    return taskName;
}
std::string GetLinkTypeName(LinkType linkInput)
{
    switch (linkInput) {
        case LinkType::LINK_ONCHIP:
            return "OnChip";
        case LinkType::LINK_HCCS:
            return "HCCS";
        case LinkType::LINK_PCIE:
            return "PCIe";
        case LinkType::LINK_ROCE:
            return "RoCE";
        case LinkType::LINK_SIO:
            return "SIO";
        case LinkType::LINK_HCCS_SW:
            return "HCCS_SW";
        default:
            return "OnChip";
    }
}

std::string GetAlgTypeStr(AlgType algType)
{
    std::string algTypeStr = "";
    algTypeStr += "AlgType(level 0-1-2):[";
    auto alg0It = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
    if (alg0It != HCCL_ALGO_LEVEL0_NAME_MAP.end()) {
        algTypeStr += alg0It->second;
    } else {
        algTypeStr += "null";
    }

    algTypeStr += "-";
    auto alg1It = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType.algoLevel1);
    if (alg1It != HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        algTypeStr += alg1It->second;
    } else {
        algTypeStr += "null";
    }

    algTypeStr += "-";
    auto alg2It = HCCL_ALGO_LEVEL2_NAME_MAP.find(algType.algoLevel2);
    if (alg2It != HCCL_ALGO_LEVEL2_NAME_MAP.end()) {
        algTypeStr += alg2It->second;
    } else {
        algTypeStr += "null";
    }
    algTypeStr += "].";
    return algTypeStr;
}

string FFTSOpInfo::GetBaseInfoStr() // 防止tag字符串过长，base信息和para信息分开打印
{
    string taskContent;
    taskContent += "streamID:[";
    taskContent += std::to_string(streamID);
    taskContent += "], taskID[";
    taskContent += std::to_string(taskID);
    taskContent += "], tag[";
    taskContent += std::string(tag);
    taskContent += "], ";
    taskContent += GetAlgTypeStr(algType);
    return taskContent;
}
TaskExceptionHandler::TaskExceptionHandler(u32 deviceLogicId) : ProfilerBase(deviceLogicId) {}
TaskExceptionHandler::~TaskExceptionHandler() {}
std::string GetAndPrintHeartbeatErr(rtExceptionInfo *exceptionInfo)
{
    auto errStatusVec = hccl_alg::GetErrStatusVec(exceptionInfo->deviceid);
    std::string errMsg = "";
    int errSize = errStatusVec.size();
    if (errSize > 0) {
        int maxListSize = 3;  // 放入errMsg中的异常事件最多只有3个
        if (errSize <= maxListSize) {
            errMsg = "\nthere are(is) " + std::to_string(errSize) + " abnormal device(s):\n";
        } else {
            errMsg = "\nthere are " + std::to_string(errSize) + " abnormal device(s), " +
                "only the first 3 devices are listed:\n";
        }

        for (int i = 0; i < errSize; i++) {
            HCCL_ERROR("%s", errStatusVec[i].c_str());
            if (i < maxListSize) {
                errMsg += ("\t" + errStatusVec[i] + "\n");
            }
        }
    }
    return errMsg;
}
void TaskExceptionHandler::PrintTaskContextInfo(const std::shared_ptr<std::vector<CtxInfo>> &taskList, u32 contextId)
{
    HCCL_ERROR("FFTS+ run failed, context sequence before error task is "
        "[NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), Memcpy:M(rank), Reduce: R(rank), "
        "InlineReduce:IR(rank), RDMASend:RS(rank,id)]:");
    std::string taskContextInfo = "";
    u32 startIndex = (contextId > TASK_CONTEXT_SIZE) ? (contextId - TASK_CONTEXT_SIZE) : 0;
    for (; startIndex < contextId; startIndex++) {
        auto curCtxInfo = taskList->at(startIndex);

        std::string taskStr = GetTaskBriefsName(curCtxInfo.taskType);
        taskStr += "(";
        taskStr += curCtxInfo.GetCtxRankInfo();
        if (curCtxInfo.taskType == TaskType::TASK_NOTIFY_RECORD || curCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT ||
            curCtxInfo.taskType == TaskType::TASK_RDMA) {
            taskStr += ("," + curCtxInfo.GetCtxNotifyInfo());
        }
        taskStr += "),";
        if (taskContextInfo.size() + taskStr.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s ...", taskContextInfo.c_str());
            taskContextInfo = "";
        }
        taskContextInfo += taskStr;
    }
    HCCL_ERROR("%s end.", taskContextInfo.c_str());
    return;
}

void TaskExceptionHandler::TimeStruct2Str(struct timeval &tv, std::string &opDataContent)
{
    const u32 length = 128;
    char timeStr[length] = { 0 };
    std::string timeStamp;
    const time_t sec =  tv.tv_sec;
    struct tm nowTime = {0};
    const struct tm *tmp = localtime_r(&sec, &nowTime);
    if (tmp == nullptr) {
        return;
    }

    int32_t err = snprintf_s(timeStr, length, length - 1, "%04d-%02d-%02d-%02d:%02d:%02d.%03ld.%03ld",
                             (nowTime.tm_year + 1900), nowTime.tm_mon + 1, nowTime.tm_mday, nowTime.tm_hour, nowTime.tm_min,
                             nowTime.tm_sec, tv.tv_usec / 1000, tv.tv_usec % 1000);
    if (err == -1) {
        timeStamp = "unknown time";
    } else {
        timeStamp = timeStr;
    }

    opDataContent += "timeStamp:[";
    opDataContent += timeStamp;
    opDataContent += "]";

    return;
}
void TaskExceptionHandler::PrintOpDataInfo(OpDataInfo &opDataInfo, bool isFftsPlus)
{
    stringstream opDataStr;
    opDataStr << "src" << "[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(opDataInfo.src)) << "], dst[0x"
            << std::hex << static_cast<const u64>(reinterpret_cast<const uintptr_t>(opDataInfo.dst)) << "], ";

    string opStr;
    if (opDataInfo.reduceType != HcclReduceOp::HCCL_REDUCE_RESERVED) {
        opStr += "reduceType[";
        opStr += GetReduceOpEnumStr(opDataInfo.reduceType);
        opStr += "], ";
    }

    string opDataContent;
    TimeStruct2Str(opDataInfo.tv, opDataContent);
    opDataContent += ", deviceId[";
    opDataContent += std::to_string(opDataInfo.deviceId);
    opDataContent += "], index[";
    opDataContent += std::to_string(opDataInfo.index);
    opDataContent += "], count[";
    opDataContent += std::to_string(opDataInfo.count);
    opDataContent += "], ";
    opDataContent += opStr;
    opDataContent += opDataStr.str();
    opDataContent += "dataType[";
    opDataContent += GetDataTypeEnumStr(opDataInfo.dataType);
    opDataContent += "].";

    std::string titleStr = isFftsPlus ? "[TaskExceptionHandler][Callback]FFTS+ run failed" :
        "[TaskExceptionHandler][Callback]Task run failed";
    HCCL_ERROR("%s, opData information is %s", titleStr.c_str(), opDataContent.c_str());
    return;
}

bool TaskExceptionHandler::DealExceptionOpData(rtExceptionInfo *exceptionInfo, std::string &tag, bool isFftsPlus,
    u32 index)
{
    if (GetExternalInputHcclDftLevel() == false) {
        return false;
    }
    bool opDataFound = false;
    std::unique_lock<std::mutex> lock(tagOpDataMapMutex[exceptionInfo->deviceid]);
    auto opDataIt = tagOpDataMap[exceptionInfo->deviceid].find(tag);
    CHK_PRT_RET(opDataIt == tagOpDataMap[exceptionInfo->deviceid].end(),
        HCCL_ERROR("tag not found. the fail tag is not from HCCL. tag[%s]", tag.c_str()), false);
    auto &opDataQueIt = opDataIt->second;
    CHK_PRT_RET(opDataQueIt->size() == 0, HCCL_ERROR("[TaskExceptionHandler][Callback] OpData queue size 0"), false);
    auto opDataInfo = opDataQueIt->front();
    while (opDataQueIt->size() > 0) {
        HCCL_DEBUG("[TaskExceptionHandler][Callback]index %u opData index %u size %u",
            index, opDataQueIt->front().index, opDataQueIt->size());
        if (index == opDataQueIt->front().index) {
            opDataInfo = opDataQueIt->front();
            opDataFound = true;   // 需要匹配最后下发的task，不能break
        }
        opDataQueIt->pop();
    }
    if (!opDataFound) {
        return false;
    }

    PrintOpDataInfo(opDataInfo, isFftsPlus);
    return true;
}

bool TaskExceptionHandler::DealExceptionGroupRank(rtExceptionInfo *exceptionInfo, std::string &tag,
    bool isFftsPlus, std::string &groupRankContentInfo)
{
    if (GetExternalInputHcclDftLevel() == false) {
        return false;
    }
    std::unique_lock<std::mutex> lock(groupRankMapMutex[exceptionInfo->deviceid]);
    auto groupRankIt = groupRankMap[exceptionInfo->deviceid].find(tag);
    CHK_PRT_RET(groupRankIt == groupRankMap[exceptionInfo->deviceid].end(),
        HCCL_INFO("tag not found. the fail tag is not from HCCL. tag[%s]", tag.c_str()), false);

    auto groupUdiIt = groupUdiMap[exceptionInfo->deviceid].find(groupRankIt->second.first);
    CHK_PRT_RET(groupUdiIt == groupUdiMap[exceptionInfo->deviceid].end(),
        HCCL_INFO("group not found. the fail group is not from HCCL. group[%s]",
        groupRankIt->second.first.c_str()), false);

    string peerRankStr;
    if ((groupRankIt->second.second)->remoteRankId != INVALID_VALUE_RANKSIZE) {
        peerRankStr += "], peerRankId[";
        peerRankStr += std::to_string((groupRankIt->second.second)->remoteRankId);
    }

    string groupRankContent;
    groupRankContent += "group:[";
    groupRankContent += groupRankIt->second.first;
    groupRankContent += "], user define information[";
    groupRankContent += groupUdiIt->second;
    groupRankContent += "], rankSize[";
    groupRankContent += std::to_string((groupRankIt->second.second)->rankSize);
    groupRankContent += "], rankId[";
    groupRankContent += std::to_string((groupRankIt->second.second)->rankId);
    groupRankContent += peerRankStr;
    groupRankContent += "].";
    groupRankContentInfo = groupRankContent;

    std::string titleStr = isFftsPlus ? "[TaskExceptionHandler][Callback]FFTS+ run failed" :
        "[TaskExceptionHandler][Callback]Task run failed";
    HCCL_ERROR("%s, groupRank information is %s",
        titleStr.c_str(), groupRankContent.c_str());
    return true;
}

bool TaskExceptionHandler::DealExceptionCtx(rtExceptionInfo *exceptionInfo)
{
    std::unique_lock<std::mutex> lock(opCtxInfoMutex[exceptionInfo->deviceid]);
    if (!FindAndValidateContext(exceptionInfo)) {
        return false;
    }
	
	auto mapIt = opCtxInfo[exceptionInfo->deviceid].find(exceptionInfo->streamid);
	auto &queIt = mapIt->second;
	auto fftsOpInfo = *(queIt->front().first);
    auto exceptionCtxInfo = (*(queIt->front().second))[0];

    if (!ProcessContext(exceptionInfo)) {
        return false;
    }

	u32 index = fftsOpInfo.index;
	std::string groupRankContentInfo = "";
    std::string tag(fftsOpInfo.tag);
	DealExceptionGroupRank(exceptionInfo, tag, true, groupRankContentInfo);
	DealExceptionOpData(exceptionInfo, tag, true, index);
	std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo);
	if (exceptionCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT) {
		RPT_INPUT_ERR(true,
			"EI0002",
			std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
			std::vector<std::string>({
				std::to_string(exceptionCtxInfo.GetCtxRemoteUserRank()),
				exceptionCtxInfo.GetCtxBaseInfoStr().c_str(), (exceptionCtxInfo.GetCtxParaInfoStr() + errMsg).c_str(),
				groupRankContentInfo.c_str()
			})
		);
	}
    return true;
}

bool TaskExceptionHandler::FindAndValidateContext(rtExceptionInfo *exceptionInfo)
{
    auto mapIt = opCtxInfo[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    if (mapIt == opCtxInfo[exceptionInfo->deviceid].end()) {
        HCCL_INFO("stream not found. the fail ctx is not from HCCL. streamid[%u]", exceptionInfo->streamid);
        return false;
    }

    auto &queIt = mapIt->second;
    if (queIt->size() == 0) {
        HCCL_ERROR("[TaskExceptionHandler][Callback] CtxOpInfo queue size 0");
        return false;
    }

    if ((*(queIt->front().second)).size() == 0) {
        HCCL_ERROR("[TaskExceptionHandler][Callback] CtxInfoVector size 0");
        return false;
    }

    return true;
}

bool TaskExceptionHandler::ProcessContext(rtExceptionInfo *exceptionInfo)
{
    auto mapIt = opCtxInfo[exceptionInfo->deviceid].find(exceptionInfo->streamid);
	auto &queIt = mapIt->second;
    auto fftsOpInfo = *(queIt->front().first);
    auto exceptionCtxInfo = (*(queIt->front().second))[0];
    uint16_t unvalidCtxid = 65535;
    bool ctxFound = false;

    while (queIt->size() > 0) {
        if (exceptionInfo->taskid == queIt->back().first->taskID) {
            fftsOpInfo = *(queIt->back().first);
            if (exceptionInfo->expandInfo.u.fftsPlusInfo.contextId == unvalidCtxid) {
                // 子图任务粒度下，RTS返回的异常task不包含contexId时的处理，约定contextId为65535。只记录算子信息
                HCCL_WARNING("[TaskExceptionHandler][Callback]FFTS+ ctx run failed, unvalid contexid," \
                    "base opInformation is %s", fftsOpInfo.GetBaseInfoStr().c_str());
            } else if (exceptionInfo->expandInfo.u.fftsPlusInfo.contextId < 0 ||
                exceptionInfo->expandInfo.u.fftsPlusInfo.contextId >= queIt->back().second->size()) {
                HCCL_ERROR("[TaskExceptionHandler][Callback]FFTS+ ctx run failed, contextId[%u] is out of vector "
                    "size[%zu], base opInformation is %s",
                    exceptionInfo->expandInfo.u.fftsPlusInfo.contextId, queIt->back().second->size(),
                    fftsOpInfo.GetBaseInfoStr().c_str());
            } else {
                exceptionCtxInfo = (*(queIt->back().second))[exceptionInfo->expandInfo.u.fftsPlusInfo.contextId];
                ctxFound = true;
            }
            break;
        } else {
            queIt->pop_back();
        }
    }

    if (!ctxFound) {
        return false;
    }

    if (exceptionCtxInfo.taskType == TaskType::TASK_NOTIFY_WAIT) { // 只在出错task为NotifyWait时打印前序task序列
        PrintTaskContextInfo(queIt->back().second, exceptionInfo->expandInfo.u.fftsPlusInfo.contextId);
    }

    queIt->clear();

    HCCL_ERROR("[TaskExceptionHandler][Callback]FFTS+ run failed, op information is %s",
        fftsOpInfo.GetBaseInfoStr().c_str());
    HCCL_ERROR("[TaskExceptionHandler][Callback]FFTS+ run failed, context base information is %s",
        exceptionCtxInfo.GetCtxBaseInfoStr().c_str());
    HCCL_ERROR("[TaskExceptionHandler][Callback]FFTS+ run failed, context para information is %s, tag[%s].",
        exceptionCtxInfo.GetCtxParaInfoStr().c_str(), fftsOpInfo.tag);

    return true;
}

bool TaskExceptionHandler::DealExceptionOp(rtExceptionInfo *exceptionInfo)
{
    std::unique_lock<std::mutex> lock(opMapMutex[exceptionInfo->deviceid]);
    bool taskFound = false;
    auto mapIt = opMap[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    CHK_PRT_RET(mapIt == opMap[exceptionInfo->deviceid].end(),
        HCCL_INFO("stream not found. the fail op is not from HCCL. streamid[%u]", exceptionInfo->streamid), false);
    auto &queIt = mapIt->second;
    CHK_PRT_RET(queIt->size() == 0, HCCL_ERROR("[TaskExceptionHandler][Callback] OpInfo queue size 0"), false);
    auto exceptionOpInfo = queIt->back();
    while (queIt->size() > 0) {
        if (exceptionInfo->taskid == queIt->back().taskID) {
            exceptionOpInfo = queIt->back();
            taskFound = true;   // 从后往前匹配最后下发的相同taskId
            break;
        }
        queIt->pop_back();
    }
    if (!taskFound) {
        return false;
    }
    queIt->clear();
    HCCL_ERROR("[TaskExceptionHandler][%s]FFTS+ run failed, base information is %s", __func__,
        exceptionOpInfo.GetBaseInfoStr().c_str());
    u32 index = exceptionOpInfo.index;
    std::string groupRankContentInfo = "";
    std::string tag(exceptionOpInfo.tag);
    DealExceptionGroupRank(exceptionInfo, tag, true, groupRankContentInfo);
    DealExceptionOpData(exceptionInfo, tag, true, index);
    std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo);
    if (exceptionInfo->retcode == ACL_ERROR_RT_FFTS_PLUS_TIMEOUT) {
        RPT_INPUT_ERR(true,
            "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>({
                "unknown", exceptionOpInfo.GetBaseInfoStr().c_str(), errMsg.c_str(), groupRankContentInfo.c_str()})
        );
    }
    return true;
}

void TaskExceptionHandler::PrintTaskContextInfo(const std::shared_ptr<std::deque<TaskInfo>> &taskQue)
{
    HCCL_ERROR("Task run failed, context sequence before error task is "
        "[NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), Memcpy:M(rank), Reduce: R(rank), "
        "InlineReduce:IR(rank), RDMASend:RS(rank,id)]:");
    std::string taskContextInfo = "";
    u32 startIndex = (taskQue->size() > TASK_CONTEXT_SIZE) ? (taskQue->size() - TASK_CONTEXT_SIZE) : 0;
    for (; startIndex < taskQue->size(); startIndex++) {
        auto taskInfo = taskQue->at(startIndex);

        std::string taskStr = GetTaskBriefsName(taskInfo.taskType);
        taskStr += "(";
        taskStr += taskInfo.GetRankInfo();
        if (taskInfo.taskType == TaskType::TASK_NOTIFY_RECORD || taskInfo.taskType == TaskType::TASK_NOTIFY_WAIT ||
            taskInfo.taskType == TaskType::TASK_RDMA) {
            taskStr += ("," + taskInfo.GetNotifyInfo());
        }
        taskStr += "),";
        if (taskContextInfo.size() + taskStr.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s ...", taskContextInfo.c_str());
            taskContextInfo = "";
        }
        taskContextInfo += taskStr;
    }
    HCCL_ERROR("%s end.", taskContextInfo.c_str());
    return;
}

void TaskExceptionHandler::PrintTaskAivBuffer(const std::shared_ptr<std::deque<TaskInfo>> &taskQue)
{
    if (taskQue->empty()) {
        return;
    }
    // width参考aiv_communication_base.cc的MAX_FLAG_SIZE_PER_KERNEL
    u32 width = AIV_KERNEL_FLAG_SIZE_PER_OP * 16 * 8;
    u32 flagMemSize = 1024*1024;
    auto& taskInfo = taskQue->back();
    u32 cnt = taskInfo.taskPara.Aiv.rankSize;
    s32* flagMem = reinterpret_cast<s32*>(malloc(flagMemSize));
    hrtMemSyncCopy(flagMem, flagMemSize, reinterpret_cast<u8 *>(taskInfo.taskPara.Aiv.flagMem), flagMemSize, 
                   HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);

    // 目前40个kernel, 后续新增时需要联动修改
    std::stringstream tmpSS;
    for (u32 kernel_idx = 0; kernel_idx < TASK_AIV_KERNEL_NUM; ++kernel_idx) {
        auto mem = flagMem + kernel_idx * width;
        tmpSS.str("");

        tmpSS << "[name:" << g_kernelNameList[kernel_idx];
        for (u32 i = 0; i < AIV_KERNEL_FLAG_SIZE_PER_OP; ++i) {
            tmpSS << " [";
            for (u32 j = i * cnt; j < (i + 1) * cnt; ++j) {
                tmpSS << std::dec << " " << mem[j * 8];
            }
            tmpSS << "],";
        }
        tmpSS << "]\n";
        HCCL_ERROR("%s ", tmpSS.str().c_str());
    }
    free(flagMem);
}

void TaskExceptionHandler::PrintTaskAivInfo(const std::shared_ptr<std::deque<TaskInfo>> &taskQue)
{
    HCCL_ERROR("Task run failed, context sequence before error task is "
        "[NotifyRecord:NR(rank,id), NotifyWait:NW(rank,id), Memcpy:M(rank), Reduce: R(rank), "
        "InlineReduce:IR(rank), RDMASend:RS(rank,id)]:");
    std::string taskContextInfo = "";
    // 从后往前遍历，最多打印PRINT_TASK_AIV_INFO_COUNT个taskAiv
    int cnt = PRINT_TASK_AIV_INFO_COUNT;
    for(auto it = taskQue->end()-1; it >= taskQue->begin(); --it){
        if(!it->isAlgInfo){
            continue;
        }        
        if(cnt <= 0){
            break;
        }
        auto taskInfo = *it;
        std::string taskStr = "[AIV]";
        taskStr += "(" + taskInfo.GetParaAiv() + "),";
        if (taskContextInfo.size() + taskStr.size() >= TASK_CONTEXT_INFO_SIZE) {
            HCCL_ERROR("%s ...", taskContextInfo.c_str());
            taskContextInfo = "";
        }
        taskContextInfo += taskStr;
        cnt--;
    }
    HCCL_ERROR("%s end.", taskContextInfo.c_str());
    return;
}

bool TaskExceptionHandler::DealExceptionTask(rtExceptionInfo *exceptionInfo)
{
    std::unique_lock<std::mutex> lock(taskMapMutex[exceptionInfo->deviceid]);
    bool taskFound = false;
    auto mapIt = taskMap[exceptionInfo->deviceid].find(exceptionInfo->streamid);
    CHK_PRT_RET(mapIt == taskMap[exceptionInfo->deviceid].end(),
        HCCL_INFO("stream not found. the fail task is not from HCCL. streamid[%u]", exceptionInfo->streamid), false);
    auto &queIt = mapIt->second;
    CHK_PRT_RET(queIt->size() == 0, HCCL_ERROR("[TaskExceptionHandler][Callback] TaskInfo queue size 0"), false);
    
    // 从后往前匹配最后下发的相同taskId
    auto exceptionTaskInfo = queIt->back();
    while (queIt->size() > 0) {
        if (exceptionInfo->taskid == queIt->back().taskID) {
            exceptionTaskInfo = queIt->back();
            taskFound = true;   
            break;
        }
        queIt->pop_back();
    }
    if (!taskFound) {
        return false;
    }

    if (exceptionTaskInfo.isAlgInfo){
        PrintTaskAivBuffer(queIt);
        PrintTaskAivInfo(queIt);
    }else if(exceptionTaskInfo.taskType == TaskType::TASK_NOTIFY_WAIT) { 
        // 只在出错task为NotifyWait时打印前序task序列
        PrintTaskContextInfo(queIt);
    }

    queIt->clear();
    HCCL_ERROR("[TaskExceptionHandler][%s]Task from HCCL run failed.", __func__);
    // 防止tag字符串过长， 信息分开打印
    HCCL_ERROR("[TaskExceptionHandler][%s]Task run failed, base information is %s", __func__,
        exceptionTaskInfo.GetBaseInfoStr().c_str());
    HCCL_ERROR("[TaskExceptionHandler][%s]Task run failed, para information is %s, tag[%s].", __func__,
        exceptionTaskInfo.GetParaInfoStr().c_str(), exceptionTaskInfo.tag.c_str());
    u32 index = exceptionTaskInfo.index;
    std::string groupRankContentInfo = "";
    DealExceptionGroupRank(exceptionInfo, exceptionTaskInfo.tag, false, groupRankContentInfo);
    DealExceptionOpData(exceptionInfo, exceptionTaskInfo.tag, false, index);
    std::string errMsg = GetAndPrintHeartbeatErr(exceptionInfo);
    if (exceptionTaskInfo.taskType == TaskType::TASK_NOTIFY_WAIT) {
        RPT_INPUT_ERR(true,
            "EI0002",
            std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
            std::vector<std::string>({
                std::to_string(exceptionTaskInfo.GetRemoteUserRank()),
                exceptionTaskInfo.GetBaseInfoStr().c_str(), (exceptionTaskInfo.GetParaInfoStr() + errMsg).c_str(),
                groupRankContentInfo.c_str()})
        );
    }
    return true;
}

void TaskExceptionHandler::PrintAicpuErrorMessage(rtExceptionInfo *exceptionInfo, bool &isExistAicpuError)
{
    ErrorMessageReport errorMessage;
    unique_lock<std::mutex> lock(g_commHadCallbackArrayMutex);
    if (g_commHadCallbackArray[exceptionInfo->deviceid]) {
        // 防止同一个device上出现通信主流和kernel流均出现task exception时runtime调用两次callback
        // HDC通道信息不是读清，防止aicpu task exception重复上报
        HCCL_WARNING("aicpu error message been reported. deviceid[%u]", exceptionInfo->deviceid);
        return;
    }
    lock.unlock();
    if (g_communicatorCallbackMap[exceptionInfo->deviceid].find(exceptionInfo->streamid) !=\
        g_communicatorCallbackMap[exceptionInfo->deviceid].end()) {
        // 找到对应的通信域，并调用回调函数从HDC通道获取AICPU异常信息
        errorMessage = (g_communicatorCallbackMap[exceptionInfo->deviceid])[exceptionInfo->streamid]();
        std::string groupUdi;
        std::string groupName = std::string(errorMessage.group);
        ProfilerBase::GetUdiByGroup(groupName, groupUdi);
        if (strlen(errorMessage.tag) > 0) {
            isExistAicpuError = true;
            string groupRankContent;
            groupRankContent += "group:[";
            groupRankContent += std::string(errorMessage.group);
            groupRankContent += "], user define information[";
            groupRankContent += groupUdi;
            groupRankContent += "], rankSize[";
            groupRankContent += std::to_string(errorMessage.rankSize);
            groupRankContent += "], rankId[";
            groupRankContent += std::to_string(errorMessage.rankId);
            groupRankContent += " ";
            groupRankContent += std::to_string(errorMessage.remoteUserRank);
            groupRankContent += "].";
            u32 streamId = static_cast<u32>(errorMessage.streamId);
            std::string tag = std::string(errorMessage.tag);
            u32 index = 0;
            TaskParaNotify para(static_cast<u64>(errorMessage.notifyId), errorMessage.stage, errorMessage.remoteUserRank);
            TaskInfo exceptionTaskInfo(streamId, errorMessage.taskId, tag, errorMessage.taskType, errorMessage.algType, index, para);
            HCCL_ERROR("[TaskExceptionHandler][Callback][HOST]Task from HCCL run failed.");
            // 防止tag字符串过长， 信息分开打印
            HCCL_ERROR("[TaskExceptionHandler][Callback][HOST]Task run failed, base information is %s",
                exceptionTaskInfo.GetBaseInfoStr().c_str());
            HCCL_ERROR("[TaskExceptionHandler][Callback][HOST]Task run failed, para information is %s, tag[%s].",
                exceptionTaskInfo.GetParaInfoStr().c_str(), exceptionTaskInfo.tag.c_str());
            HCCL_ERROR("[TaskExceptionHandler][Callback][HOST]Task run failed, group information is %s, tag[%s].",
                groupRankContent.c_str(), exceptionTaskInfo.tag.c_str());
            RPT_INPUT_ERR(true,
                "EI0002",
                std::vector<std::string>({"remote_rankid", "base_information", "task_information", "group_rank_content"}),
                std::vector<std::string>({
                    std::to_string(exceptionTaskInfo.GetRemoteUserRank()), exceptionTaskInfo.GetBaseInfoStr().c_str(),
                    exceptionTaskInfo.GetParaInfoStr().c_str(), groupRankContent.c_str()})
                );
            lock.lock();
            g_commHadCallbackArray[exceptionInfo->deviceid] = true;
        }
    } else {
        HCCL_INFO("PrintAicpuErrorMessage streamId[%d] is not found.", exceptionInfo->streamid);
    }
    return;
}

void TaskExceptionHandler::Callback(rtExceptionInfo *exceptionInfo)
{
    bool isExistAicpuError = false;
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[TaskExceptionHandler][Callback] exceptionInfo is nullptr.");
        return;
    }
    PrintAicpuErrorMessage(exceptionInfo, isExistAicpuError);
    if (isExistAicpuError) {
        // 如果已经有AICPU上报的task exception, 则host侧无需再次重复上报
        return;
    }
    CHK_PRT_RET(exceptionInfo->deviceid >= MAX_MODULE_DEVICE_NUM,
        HCCL_WARNING("deviceID[%u] from exceptionInfo is bigger than MAX_MODULE_DEVICE_NUM[%u]",
        exceptionInfo->deviceid, MAX_MODULE_DEVICE_NUM),);
    SaluSleep(ONE_MILLISECOND_OF_USLEEP); // sleep 1ms，等待task被存入数据结构
    HCCL_DEBUG("[TaskExceptionHandler][Callback]Task run failed, ffts+ task type:%d, TaskExceptionSwitch:%u",
        exceptionInfo->expandInfo.type, GetExternalInputTaskExceptionSwitch());
    if (exceptionInfo->expandInfo.type == RT_EXCEPTION_FFTS_PLUS) {
        if (GetExternalInputTaskExceptionSwitch() == 1) {
            DealExceptionCtx(exceptionInfo);     // 子任务粒度
        } else {
            DealExceptionOp(exceptionInfo);      // 算子粒度
        }
    } else {
        DealExceptionTask(exceptionInfo);
    }
    return;
}
HcclResult TaskExceptionHandler::Init()
{
    CHK_RET(hrtRegTaskFailCallbackByModule(Callback));

    CHK_RET(hrtGetMaxStreamAndTask(maxStrCount, maxTaskCount));

    maxStrCount = (maxStrCount < STREAM_COUNT_UPPER_LIMIT) ? maxStrCount : STREAM_COUNT_UPPER_LIMIT;
    maxTaskCount = (maxTaskCount < TASK_COUNT_UPPER_LIMIT) ? maxTaskCount : TASK_COUNT_UPPER_LIMIT;
    // 单算子模式task过多的特殊处理
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        maxTaskCount = TASK_COUNT_UPPER_LIMIT_OP_BASE;
    }
    HCCL_INFO("get from RTS the max stream count[%u] the max task count[%u]", maxStrCount, maxTaskCount);

    if (GetExternalInputHcclEnableFfts() &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetExternalInputTaskExceptionSwitch() == 1) {
        for (std::vector<CtxInfo> ctxInfoVector : ctxInfoArray) {
            ctxInfoVector.reserve(100); // vector预留100个ctxInfo空间
        }
    }

    // 对全局变量g_commHadCallbackArray进行初始化
    for (u32 i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        g_commHadCallbackArray[i] = false;
    }
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::DeInit()
{
    CHK_RET(hrtRegTaskFailCallbackByModule(nullptr));
    HCCL_INFO("deInit taskFailCallback");
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType &taskType, const TaskParaNotify &para)
{
    CHK_PRT_RET(deviceLogicId_ >= MAX_MODULE_DEVICE_NUM,
        HCCL_ERROR("[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than MAX_MODULE_DEVICE_NUM[%u]",
            deviceLogicId_, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_INFO("[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u], taskType[%d]", __func__,
        streamID, taskID, taskType);
    if (GetExternalInputHcclEnableFfts() &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetExternalInputTaskExceptionSwitch() == 1) {
        std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]);  // 防止存入和读取冲突
        CtxInfo tmpCtxInfo(taskType, para);
        ctxInfoArray[deviceLogicId_].insert(ctxInfoArray[deviceLogicId_].end(), tmpCtxInfo);
        return HCCL_SUCCESS;
    }

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    TaskInfo tmpTaskInfo(streamID, taskID, tag, taskType, algType, index, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));

    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 &streamID, u32 &taskID, TaskType &taskType, const TaskParaNotify &para)
{
    return Save(streamID, streamID, taskID, taskType, para);
}

HcclResult TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType &taskType, const TaskParaDMA &para)
{
    CHK_PRT_RET(deviceLogicId_ >= MAX_MODULE_DEVICE_NUM,
        HCCL_ERROR("[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than MAX_MODULE_DEVICE_NUM[%u]",
            deviceLogicId_, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_INFO("[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u], taskType[%d]", __func__,
        streamID, taskID, taskType);
    if (GetExternalInputHcclEnableFfts() &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetExternalInputTaskExceptionSwitch() == 1) {
        std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]);  // 防止存入和读取冲突
        CtxInfo tmpCtxInfo(taskType, para);
        ctxInfoArray[deviceLogicId_].insert(ctxInfoArray[deviceLogicId_].end(), tmpCtxInfo);
        return HCCL_SUCCESS;
    }

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    TaskInfo tmpTaskInfo(streamID, taskID, tag, taskType, algType, index, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 &streamID, u32 &taskID, TaskType &taskType, const TaskParaDMA &para)
{
    return Save(streamID, streamID, taskID, taskType, para);
}

HcclResult TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID, TaskType &taskType, const TaskParaReduce &para)
{
    CHK_PRT_RET(deviceLogicId_ >= MAX_MODULE_DEVICE_NUM,
        HCCL_ERROR("[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than MAX_MODULE_DEVICE_NUM[%u]",
            deviceLogicId_, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_INFO("[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u], taskType[%d]", __func__,
        streamID, taskID, taskType);
    if (GetExternalInputHcclEnableFfts() &&
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE &&
        GetExternalInputTaskExceptionSwitch() == 1) {
        std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]);  // 防止存入和读取冲突
        CtxInfo tmpCtxInfo(taskType, para);
        ctxInfoArray[deviceLogicId_].insert(ctxInfoArray[deviceLogicId_].end(), tmpCtxInfo);
        return HCCL_SUCCESS;
    }

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    TaskInfo tmpTaskInfo(streamID, taskID, tag, taskType, algType, index, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}


HcclResult TaskExceptionHandler::Save(u32 &streamID, u32 &taskID, const TaskParaAiv &para)
{
    CHK_PRT_RET(deviceLogicId_ >= MAX_MODULE_DEVICE_NUM,
        HCCL_ERROR("[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than MAX_MODULE_DEVICE_NUM[%u]",
            deviceLogicId_, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);

    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(streamID, tag));
    TaskInfo tmpTaskInfo(streamID, taskID, tag, para);
    CHK_RET(InsertTaskMap(streamID, tmpTaskInfo));
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 &streamID, u32 &taskID, TaskType &taskType, const TaskParaReduce &para)
{
    return Save(streamID, streamID, taskID, taskType, para);
}

HcclResult TaskExceptionHandler::Save(u32 captureStreamID, u32 streamID, u32 taskID)
{
    CHK_PRT_RET(deviceLogicId_ >= MAX_MODULE_DEVICE_NUM,
        HCCL_ERROR("[TaskExceptionHandler][Save]deviceLogicId_[%u] is bigger than MAX_MODULE_DEVICE_NUM[%u]",
            deviceLogicId_, MAX_MODULE_DEVICE_NUM), HCCL_E_INTERNAL);
    HCCL_INFO("[TaskExceptionHandler][%s]Save task info, streamId[%u], taskId[%u]", __func__, streamID, taskID);
    std::string tag;
    CHK_RET(ProfilerBase::GetTagByStream(captureStreamID, tag));
    AlgType algType = AlgType::Reserved();
    CHK_RET(ProfilerBase::GetAlgTypeByStream(captureStreamID, algType));
    u32 index = 0;
    ProfilerBase::GetSubmittedOpCnt(index);

    if (GetExternalInputTaskExceptionSwitch() == 1) {
        CHK_RET(InsertOpCtxInfo(streamID, taskID, tag, algType, index));
    } else {
        CHK_RET(InsertOpMap(streamID, taskID, tag, algType, index));
    }
    CHK_RET(InsertRankInfo(tag));
    CHK_RET(InsertOpData(tag));
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Save(u32 &streamID, u32 &taskID)
{
    return Save(streamID, streamID, taskID);
}

HcclResult TaskExceptionHandler::SaveToLog(const TaskParaHost &paraHost)
{
    (void)paraHost;
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::InsertTaskMap(u32 &streamID, TaskInfo &tmpTaskInfo) const
{
    std::unique_lock<std::mutex> lock(taskMapMutex[deviceLogicId_]);
    auto it = taskMap[deviceLogicId_].find(streamID);
    if (it == taskMap[deviceLogicId_].end()) {
        // streamID 复用且不会超过最大stream数量，因此Map的size超过最大stream数量属于异常场景
        CHK_PRT_RET(taskMap[deviceLogicId_].size() >= maxStrCount, HCCL_ERROR("[Insert][TaskMap]taskMap size is "
            "bigger than max stream count[%u]. stream add fail", maxStrCount), HCCL_E_INTERNAL);
        std::shared_ptr<deque<TaskInfo>> tmpTaskInfoQue = nullptr;
        EXECEPTION_CATCH((tmpTaskInfoQue = make_shared<deque<TaskInfo>>()), return HCCL_E_PTR);
        tmpTaskInfoQue->push_back(tmpTaskInfo);
        taskMap[deviceLogicId_].insert({ streamID, tmpTaskInfoQue });
    } else { // 由于不允许多线程对同一stream操作，因此此处不需要保留锁，并且此处访问量最多，性能考虑也最好不要加锁
        lock.unlock();
        it->second->push_back(tmpTaskInfo);
        if (it->second->size() > maxTaskCount) {
            it->second->pop_front();
        }
    }
    return HCCL_SUCCESS;
}
HcclResult TaskExceptionHandler::InsertOpMap(u32 &streamID, u32 &taskID, string &tag, AlgType &algType,
    u32 &index) const
{
    FFTSOpInfo tmpOpPara;
    CHK_SAFETY_FUNC_RET(memcpy_s(tmpOpPara.tag, sizeof(tmpOpPara.tag), tag.c_str(), tag.size()));
    tmpOpPara.streamID = streamID;
    tmpOpPara.taskID = taskID;
    tmpOpPara.algType = algType;
    tmpOpPara.index = index;
    std::unique_lock<std::mutex> lock(opMapMutex[deviceLogicId_]); // 防止存入和读取冲突
    auto it = opMap[deviceLogicId_].find(streamID);
    if (it == opMap[deviceLogicId_].end()) {
        CHK_PRT_RET(opMap[deviceLogicId_].size() >= maxStrCount, HCCL_ERROR("[Insert][OpMap]Map size is "
            "bigger than max stream count[%u]. stream add fail", maxStrCount), HCCL_E_INTERNAL);
        std::shared_ptr<deque<FFTSOpInfo>> tmpOpInfoQue = nullptr;
        EXECEPTION_CATCH((tmpOpInfoQue = make_shared<deque<FFTSOpInfo>>()), return HCCL_E_PTR);
        tmpOpInfoQue->push_back(tmpOpPara);
        opMap[deviceLogicId_].insert({ streamID, tmpOpInfoQue });
    } else {
        it->second->push_back(tmpOpPara);
        if (it->second->size() > maxTaskCount) {
            it->second->pop_front();
        }
    }
    return HCCL_SUCCESS;
}
HcclResult TaskExceptionHandler::InsertOpCtxInfo(u32 &streamID, u32 &taskID, string &tag,
    AlgType &algType, u32 &index) const
{
    FFTSOpInfo tmpOpInfo;
    CHK_SAFETY_FUNC_RET(memcpy_s(tmpOpInfo.tag, sizeof(tmpOpInfo.tag), tag.c_str(), tag.size()));
    tmpOpInfo.streamID = streamID;
    tmpOpInfo.taskID = taskID;
    tmpOpInfo.algType = algType;
    tmpOpInfo.index = index;
    std::shared_ptr<FFTSOpInfo> tmpOpInfoPtr = nullptr;
    EXECEPTION_CATCH((tmpOpInfoPtr = std::make_shared<FFTSOpInfo>()), return HCCL_E_PTR);
    *tmpOpInfoPtr = tmpOpInfo;
    std::shared_ptr<vector<CtxInfo>> tempCtxVectorPtr = nullptr;
    EXECEPTION_CATCH((tempCtxVectorPtr = std::make_shared<vector<CtxInfo>>()), return HCCL_E_PTR);
    std::unique_lock<std::mutex> lock(ctxInfoVectorMutex[deviceLogicId_]);  // 防止存入和读取冲突
    *tempCtxVectorPtr = ctxInfoArray[deviceLogicId_];
    auto tempPair = std::make_pair(tmpOpInfoPtr, tempCtxVectorPtr);
    std::unique_lock<std::mutex> infoLock(opCtxInfoMutex[deviceLogicId_]); // 防止存入和读取冲突
    auto tempDeque = opCtxInfo[deviceLogicId_].find(streamID);
    if (tempDeque == opCtxInfo[deviceLogicId_].end()) {
        CHK_PRT_RET(opCtxInfo[deviceLogicId_].size() >= maxStrCount, HCCL_ERROR("[Insert][opCtxInfo]Map size is "
            "bigger than max stream count[%u]. stream add fail", maxStrCount), HCCL_E_INTERNAL);
        std::shared_ptr<std::deque<std::pair<std::shared_ptr<FFTSOpInfo>,
            std::shared_ptr<std::vector<CtxInfo>>>>> tmpOpInfoQue = nullptr;
        EXECEPTION_CATCH((tmpOpInfoQue = std::make_shared<std::deque<std::pair<std::shared_ptr<FFTSOpInfo>,
            std::shared_ptr<std::vector<CtxInfo>>>>>()), return HCCL_E_PTR);
        tmpOpInfoQue->push_back(tempPair);
        opCtxInfo[deviceLogicId_].insert({ streamID, tmpOpInfoQue });
    } else {
        tempDeque->second->push_back(tempPair);
        if (tempDeque->second->size() > maxTaskCount) {
            tempDeque->second->pop_front();
        }
    }
    ctxInfoArray[deviceLogicId_].clear();
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::InsertRankInfo(std::string &tag) const
{
    if (GetExternalInputHcclDftLevel() == false) {
        return HCCL_SUCCESS;
    }
    std::string groupName;
    CHK_RET(ProfilerBase::GetGroupNameByTag(tag, groupName));
    GroupRankInfo groupRankInfo;
    CHK_RET(ProfilerBase::GetRankInfoByGroup(groupName, groupRankInfo));
    std::string groupUdi;
    CHK_RET(ProfilerBase::GetUdiByGroup(groupName, groupUdi));

    HCCL_DEBUG("[TaskExceptionHandler][Callback]InsertRankInfo tag %s group %s",
        tag.c_str(), groupName.c_str());
    {
        std::unique_lock<std::mutex> groupRankMapLock(groupRankMapMutex[deviceLogicId_]);
        std::shared_ptr<GroupRankInfo> tmpRankInfo = nullptr;
        EXECEPTION_CATCH((tmpRankInfo = std::make_shared<GroupRankInfo>()), return HCCL_E_PTR);
        *tmpRankInfo = groupRankInfo;
        auto groupRankIt = groupRankMap[deviceLogicId_].find(tag);
        if (groupRankIt == groupRankMap[deviceLogicId_].end()) {
            auto tempPair = std::make_pair(groupName, tmpRankInfo);
            groupRankMap[deviceLogicId_].insert({ tag, tempPair });
        } else {
            groupRankIt->second.second = tmpRankInfo;
        }
    }

    {
        std::lock_guard<std::mutex> groupUdiMapLock(groupUdiMapMutex[deviceLogicId_]);
        auto groupUdiIt = groupUdiMap[deviceLogicId_].find(groupName);
        if (groupUdiIt == groupUdiMap[deviceLogicId_].end()) {
            groupUdiMap[deviceLogicId_].insert({ groupName, groupUdi });
        } else {
            groupUdiIt->second = groupUdi;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::InsertOpData(std::string &tag) const
{
    if (GetExternalInputHcclDftLevel() == false) {
        return HCCL_SUCCESS;
    }
    OpDataInfo opDataInfo;
    CHK_RET(ProfilerBase::GetOpDataInfoByTag(tag, opDataInfo));
    std::unique_lock<std::mutex> lock(tagOpDataMapMutex[deviceLogicId_]);
    auto tempDeque = tagOpDataMap[deviceLogicId_].find(tag);
    if (tempDeque == tagOpDataMap[deviceLogicId_].end()) {
        std::shared_ptr<queue<OpDataInfo>> tmpOpDataInfo = nullptr;
        EXECEPTION_CATCH((tmpOpDataInfo = std::make_shared<queue<OpDataInfo>>()), return HCCL_E_PTR);
        tmpOpDataInfo->push(opDataInfo);
        tagOpDataMap[deviceLogicId_].insert({ tag, tmpOpDataInfo });
        HCCL_DEBUG("[TaskExceptionHandler][Callback]InsertOpData index %u tag %s",
            opDataInfo.index, tag.c_str());
    } else {
        HCCL_DEBUG("[TaskExceptionHandler][Callback]InsertOpData index %u opData index %u size %u tag %s",
            opDataInfo.index, tempDeque->second->back().index, (tempDeque->second)->size(), tag.c_str());
        if (tempDeque->second->back().index != opDataInfo.index) { // 需要去重，taskid不同时可能是同一个
            tempDeque->second->push(opDataInfo);
        }
        if ((tempDeque->second)->size() > 3000) { // 队列深度大于3000则老化
            HCCL_DEBUG("[Insert][opDataMap]Map size is [%u], need to pop head data.", (tempDeque->second)->size());
            tempDeque->second->pop();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::Flush()
{
    return HCCL_SUCCESS;
}

HcclResult TaskExceptionHandler::TaskExceptionHandler::Run(const StepData &stepData)
{
    (void)stepData;
    return HCCL_SUCCESS;
}
