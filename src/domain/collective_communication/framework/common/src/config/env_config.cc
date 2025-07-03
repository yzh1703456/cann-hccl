/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config.h"
#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include "adapter_error_manager_pub.h"
#include "log.h"
#include "sal_pub.h"
#include "mmpa_api.h"

using namespace hccl;

static std::mutex g_envConfigMutex;
static EnvConfig g_envConfig;

constexpr char ENV_EMPTY_STRING[] = "EmptyString";

constexpr char HCCL_AUTO_PORT_CONFIG[] = "auto"; // 端口范围配置为auto时，由OS分配浮动监听端口
constexpr u32 MAX_PORT_NUMBER = 65535; // 合法端口号的上限
constexpr u32 HCCL_SOCKET_PORT_RANGE_AUTO = 0; // 需要保留的

HcclResult InitEnvConfig()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    if (g_envConfig.initialized) {
        return HCCL_SUCCESS;
    }
    // 初始化环境变量
    CHK_RET(InitEnvParam());

    g_envConfig.initialized = true;

    return HCCL_SUCCESS;
}

bool GetExternalInputHostPortSwitch()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.hostSocketPortSwitch;
}

bool GetExternalInputNpuPortSwitch()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.npuSocketPortSwitch;
}

const u32& EnvConfig::GetExternalInputRdmaTrafficClass()
{
    return g_envConfig.rdmaTrafficClass;
}

const u32& EnvConfig::GetExternalInputRdmaServerLevel()
{
    return g_envConfig.rdmaServerLevel;
}

const std::vector<HcclSocketPortRange> &GetExternalInputHostSocketPortRange()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.hostSocketPortRange;
}

const std::vector<HcclSocketPortRange> &GetExternalInputNpuSocketPortRange()
{
    std::lock_guard<std::mutex> lock(g_envConfigMutex);
    return g_envConfig.npuSocketPortRange;
}
HcclResult InitEnvParam()
{
    HcclResult ret = ParseHostSocketPortRange();
    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"env", "tips"}),
        std::vector<std::string>({"HCCL_HOST_SOCKET_PORT_RANGE", "Please check whether the port range is valid."}));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InitEnvParam]errNo[0x%016llx] In init environtment param, parse "
            "HCCL_HOST_SOCKET_PORT_RANGE failed. errorno[%d]", HCCL_ERROR_CODE(ret), ret), ret);

    ret = ParseNpuSocketPortRange();
    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"env", "tips"}),
        std::vector<std::string>({"HCCL_NPU_SOCKET_PORT_RANGE", "Please check whether the port range is valid."}));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InitEnvParam]errNo[0x%016llx] In init environtment param, parse "
            "HCCL_NPU_SOCKET_PORT_RANGE failed. errorno[%d]", HCCL_ERROR_CODE(ret), ret), ret);

    ret = g_envConfig.ParseRDMATrafficClass();
    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"env", "tips"}),
        std::vector<std::string>({"HCCL_RDMA_TC", "Value range[0, 255], Must be a multiple of 4"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InitEnvParam]errNo[0x%016llx] In init environtment param, parse "
            "HCCL_RDMA_TC failed. errorno[%d]", HCCL_ERROR_CODE(ret), ret), ret);

    ret = g_envConfig.ParseRDMAServerLevel();
    RPT_ENV_ERR(ret != HCCL_SUCCESS, "EI0001", std::vector<std::string>({"env", "tips"}),
        std::vector<std::string>({"HCCL_RDMA_SL", "Value range[0, 7]"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InitEnvParam]errNo[0x%016llx] In init environtment param, parse "
            "HCCL_RDMA_SL failed. errorno[%d]", HCCL_ERROR_CODE(ret), ret), ret);

    return HCCL_SUCCESS;
}

bool EnvConfig::CheckEnvLen(const char *envStr, u32 envMaxLen)
{
    // 校验环境变量长度
    u32 envLen = strnlen(envStr, envMaxLen + 1);
    if (envLen == (envMaxLen + 1)) {
        HCCL_ERROR("[CheckEnvLen] errNo[0x%016llx] env len is invalid, len is %u", HCCL_ERROR_CODE(HCCL_E_PARA), envLen);
        return false;
    }
    return true;
}

HcclResult SetDefaultSocketPortRange(const SocketLocation &socketLoc, std::vector<HcclSocketPortRange> &portRangeVec)
{
    if (socketLoc == SOCKET_HOST) {
        g_envConfig.hostSocketPortSwitch = false;
        portRangeVec.clear();
        HCCL_RUN_WARNING("HCCL_HOST_SOCKET_PORT_RANGE is not set. Multi-process will not be supported!");
    } else if (socketLoc == SOCKET_NPU) {
        g_envConfig.npuSocketPortSwitch = false;
        portRangeVec.clear();
        HCCL_RUN_WARNING("HCCL_NPU_SOCKET_PORT_RANGE is not set. Multi-process will not be supported!");
    } else {
        HCCL_ERROR("[SetDefaultSocketPortRange] undefined socket location, fail to init socket port range by default.");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckSocketPortRangeValid(const std::string &envName, const std::vector<HcclSocketPortRange> &portRanges)
{
    std::vector<HcclSocketPortRange> rangeVec(portRanges.begin(), portRanges.end());
    std::sort(rangeVec.begin(), rangeVec.end(), [](auto &a, auto &b) {
        return a.min == b.min ? a.max < b.max : a.min < b.min;
    });
    for (size_t i = 0; i < rangeVec.size(); ++i) {
        // the socket range should not be inverted
        CHK_PRT_RET(rangeVec[i].min > rangeVec[i].max,
            HCCL_ERROR("[Check][PortRangeValid] In %s, in socket port range [%u, %u], "
                "the lower bound is greater than the upper bound.",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max),
            HCCL_E_PARA);

        // the socket range should not include the reserved port for auto listening.
        CHK_PRT_RET(rangeVec[i].min <= HCCL_SOCKET_PORT_RANGE_AUTO && rangeVec[i].max >=  HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_ERROR("[Check][PortRangeValid] In %s, socket port range [%u, %u] includes "
                "the reserved port number [%u]. please do not use port [%u] in socket port range.",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, HCCL_SOCKET_PORT_RANGE_AUTO,
                HCCL_SOCKET_PORT_RANGE_AUTO),
            HCCL_E_PARA);

        // the socket range should not exceed the maximum port number
        CHK_PRT_RET(rangeVec[i].max > MAX_PORT_NUMBER,
            HCCL_ERROR("[Check][PortRangeValid] In %s, in socket port range [%u, %u], "
                "the upper bound exceed max port number[%u].",
                envName.c_str(), rangeVec[i].min, rangeVec[i].max, MAX_PORT_NUMBER),
            HCCL_E_PARA);

        // the socket range should not be overlapped
        CHK_PRT_RET(i != 0 && rangeVec[i - 1].max >= rangeVec[i].min,
            HCCL_ERROR("[Check][PortRangeValid] In %s, "
                "socket port range [%u, %u] is conflict with socket port range [%u, %u].",
                envName.c_str(), rangeVec[i - 1].min, rangeVec[i - 1].max, rangeVec[i].min, rangeVec[i].max),
            HCCL_E_PARA);
    }

    return HCCL_SUCCESS;
}

HcclResult GetUIntFromStr(const std::string &digitStr, u32 &val)
{
    HcclResult ret = IsAllDigit(digitStr.c_str());
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetUIntFromStr] str[%s] is not all digit.",
        digitStr.c_str()), ret);
    ret = SalStrToULong(digitStr.c_str(), HCCL_BASE_DECIMAL, val);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetUIntFromStr] str[%s] is a invalid number.",
        digitStr.c_str()), ret);
    return HCCL_SUCCESS;
}

bool SplitString(std::string &totalStr, std::string &prefixStr, const std::string &delim)
{
    std::size_t found = totalStr.find(delim);
    if (found == std::string::npos) {
        return false;
    }
    prefixStr = totalStr.substr(0, found);
    totalStr = totalStr.substr(found + 1);
    return true;
}

HcclResult SplitSinglePortRange(const std::string &envName, std::string &rangeStr, HcclSocketPortRange &portRange)
{
    std::string rangeMin{};
    const std::string delim = "-";
    if (SplitString(rangeStr, rangeMin, delim)) {
        CHK_RET(GetUIntFromStr(rangeMin, portRange.min));
        CHK_RET(GetUIntFromStr(rangeStr, portRange.max));
    } else {
        CHK_RET(GetUIntFromStr(rangeStr, portRange.min));
        portRange.max = portRange.min;
    }
    HCCL_INFO("[Split][SinglePortRange] Load hccl socket port range [%u, %u] from %s",
        portRange.min, portRange.max, envName.c_str());
    return HCCL_SUCCESS;
}

HcclResult SplitHcclSocketPortRange(const std::string &envName, std::string &portRangeConfig,
    std::vector<HcclSocketPortRange> &portRangeVec)
{
    std::string rangeStr{};
    const std::string delim = ",";
    while (SplitString(portRangeConfig, rangeStr, delim)) {
        HcclSocketPortRange portRange = {};
        CHK_RET(SplitSinglePortRange(envName, rangeStr, portRange));
        portRangeVec.emplace_back(portRange);
    }
    HcclSocketPortRange portRange = {};
    CHK_RET(SplitSinglePortRange(envName, portRangeConfig, portRange));
    portRangeVec.emplace_back(portRange);

    CHK_RET(CheckSocketPortRangeValid(envName, portRangeVec));
    return HCCL_SUCCESS;
}

HcclResult PortRangeSwitchOn(const SocketLocation &socketLoc)
{
    if (socketLoc == SOCKET_HOST) {
        g_envConfig.hostSocketPortSwitch = true;
        HCCL_INFO("HCCL_HOST_SOCKET_PORT_RANGE is set, switch on.");
    } else if (socketLoc == SOCKET_NPU) {
        g_envConfig.npuSocketPortSwitch = true;
        HCCL_INFO("HCCL_NPU_SOCKET_PORT_RANGE is set, switch on.");
    } else {
        HCCL_ERROR("[PortRangeSwitchOn] undefined socket location, fail to init socket port range by default.");
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

void PrintSocketPortRange(const std::string &envName, const std::vector<HcclSocketPortRange> &portRangeVec)
{
    // assemble port ranges into a string to print the result range
    std::ostringstream portRangeOss;
    for (auto range : portRangeVec) {
        portRangeOss << " [" << std::to_string(range.min) << ", " << std::to_string(range.max) << "]";
    }
    HCCL_RUN_INFO("%s is set to%s.", envName.c_str(), portRangeOss.str().c_str());
}

HcclResult SetSocketPortRange(const std::string &envName, const std::string &socketPortRange,
    const SocketLocation &socketLoc, std::vector<HcclSocketPortRange> &portRangeVec)
{
    portRangeVec.clear();

    // the environment variable is not set
    if (!socketPortRange.compare(ENV_EMPTY_STRING)) {
        CHK_RET(SetDefaultSocketPortRange(socketLoc, portRangeVec));
        return HCCL_SUCCESS;
    }

    // the socket port range is set to auto, then the os will listen on the ports dymamically and automatically.
    if (!socketPortRange.compare(HCCL_AUTO_PORT_CONFIG)) {
        HcclSocketPortRange autoSocketPortRange = {
            HCCL_SOCKET_PORT_RANGE_AUTO,
            HCCL_SOCKET_PORT_RANGE_AUTO
        };
        portRangeVec.emplace_back(autoSocketPortRange);
        CHK_RET(PortRangeSwitchOn(socketLoc));
        HCCL_RUN_INFO("%s is set to %s as [%u, %u].", envName.c_str(), HCCL_AUTO_PORT_CONFIG,
            autoSocketPortRange.min, autoSocketPortRange.max);
        return HCCL_SUCCESS;
    }

    std::string portRangeConfig = socketPortRange;
    // the environment variable is set to an empty string
    portRangeConfig.erase(std::remove(portRangeConfig.begin(), portRangeConfig.end(), ' '), portRangeConfig.end());
    if (portRangeConfig.empty()) {
        CHK_RET(SetDefaultSocketPortRange(socketLoc, portRangeVec));
        return HCCL_SUCCESS;
    }
    // load ranges from string
    CHK_RET(SplitHcclSocketPortRange(envName, portRangeConfig, portRangeVec));
    if (portRangeVec.size() == 0) {
        HCCL_ERROR("Load empty port range from %s, please check.", envName.c_str());
        return HCCL_E_PARA;
    }
    CHK_RET(PortRangeSwitchOn(socketLoc));
    (void) PrintSocketPortRange(envName, portRangeVec);
    return HCCL_SUCCESS;
}

HcclResult ParseHostSocketPortRange()
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_HOST_SOCKET_PORT_RANGE, mmSysGetEnvValue);
    std::string hostSocketPortRangeEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    CHK_RET(SetSocketPortRange("HCCL_HOST_SOCKET_PORT_RANGE", hostSocketPortRangeEnv, SOCKET_HOST,
        g_envConfig.hostSocketPortRange));
    return HCCL_SUCCESS;
}

HcclResult ParseNpuSocketPortRange()
{
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_NPU_SOCKET_PORT_RANGE, mmSysGetEnvValue);
    std::string npuSocketPortRangeEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    CHK_RET(SetSocketPortRange("HCCL_NPU_SOCKET_PORT_RANGE", npuSocketPortRangeEnv, SOCKET_NPU,
        g_envConfig.npuSocketPortRange));
    return HCCL_SUCCESS;
}

// 通用的环境变量解析函数
HcclResult ParseEnvConfig(const EnvConfigParam& param, std::string& envValue, u32& resultValue)
{
    if (!envValue.compare(ENV_EMPTY_STRING)) {
        HCCL_RUN_INFO("%s set by default to [%u]", param.envName.c_str(), param.defaultValue);
        resultValue = param.defaultValue;
        return HCCL_SUCCESS;
    }

    // 校验环境变量长度
    bool isEnvLenValid = g_envConfig.CheckEnvLen(envValue.c_str(), g_envConfig.MAX_LEN_OF_DIGIT_ENV);
    CHK_PRT_RET(!isEnvLenValid,
        HCCL_ERROR("[Parse][%s] errNo[0x%016llx] Invalid %s env len, len is bigger than [%u], errorno[%d]",
        param.envName.c_str(), HCCL_ERROR_CODE(HCCL_E_PARA), param.envName.c_str(), g_envConfig.MAX_LEN_OF_DIGIT_ENV, HCCL_E_PARA),
        HCCL_E_PARA);
    
    CHK_RET(IsAllDigit(envValue.c_str()));
    
    HcclResult ret = SalStrToULong(envValue.c_str(), HCCL_BASE_DECIMAL, resultValue);
    // 若转换出错或者设置的值不在有效范围内，报错
    CHK_PRT_RET((ret != HCCL_SUCCESS || resultValue < param.minValue || resultValue > param.maxValue),
        HCCL_ERROR("[Parse][%s] is invalid. except: [%u, %u], actual: [%u]", param.envName.c_str(), param.minValue, param.maxValue, resultValue),
        HCCL_E_PARA);
    
    // 如果提供了baseValue，检查是否是baseValue的整数倍
    if (param.baseValue != 0 && resultValue % param.baseValue != 0) {
        HCCL_ERROR("[Parse] %s[%u] is not a multiple of [%u]", param.envName.c_str(), resultValue, param.baseValue);
        return HCCL_E_PARA;
    }

    HCCL_RUN_INFO("%s set by environment to [%u]", param.envName.c_str(), resultValue);
    return HCCL_SUCCESS;
}

HcclResult EnvConfig::ParseRDMATrafficClass()
{
    EnvConfigParam param = {
        "HCCL_RDMA_TC",
        HCCL_RDMA_TC_DEFAULT,
        HCCL_RDMA_TC_MIN,
        HCCL_RDMA_TC_MAX,
        HCCL_RDMA_TC_BASE
    };
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_TC, mmSysGetEnvValue);
    std::string envValue = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    return ParseEnvConfig(param, envValue, g_envConfig.rdmaTrafficClass);
}

HcclResult EnvConfig::ParseRDMAServerLevel()
{
    EnvConfigParam param = {
        "HCCL_RDMA_SL",
        HCCL_RDMA_SL_DEFAULT,
        HCCL_RDMA_SL_MIN,
        HCCL_RDMA_SL_MAX,
        0
    };
    char* mmSysGetEnvValue = nullptr;
    MM_SYS_GET_ENV(MM_ENV_HCCL_RDMA_SL, mmSysGetEnvValue);
    std::string envValue = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
    return ParseEnvConfig(param, envValue, g_envConfig.rdmaServerLevel);
}