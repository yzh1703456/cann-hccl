/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_CONFIG_PUB_H
#define HCCL_COMM_CONFIG_PUB_H

constexpr u32 COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0;  // Magic word值，用于校验传入的配置结构体是否已经被初始化

enum CommConfigVersion {
    COMM_CONFIG_VERSION_ONE = 1,
    COMM_CONFIG_VERSION_TWO = 2,
    COMM_CONFIG_VERSION_THREE = 3,
    COMM_CONFIG_VERSION_FOUR = 4,
    COMM_CONFIG_VERSION_FIVE = 5                     // 当前支持的最高版本
};

enum CommConfigOpExpansion {
    COMM_CONFIG_OPEXPANSION_DEFAULT = 0,                // Config配置默认模式
    COMM_CONFIG_OPEXPANSION_HOST = 1,                   // Config配置HOST模式
    COMM_CONFIG_OPEXPANSION_AICPU = 2,                  // Config配置AICPU模式
    COMM_CONFIG_OPEXPANSION_AIV = 3                     // Config配置AIV模式
};

// 通信域级别配置参数结构体 - 内部信息
typedef struct CommConfigInfoDef {
    size_t configSize;  // 配置结构体大小
    u32 magicWord;      // Magic word
    u32 version;        // HCCL版本
    char reserved[8];   // 8 byte 保留字段
} CommConfigInfo;

// 通信域级别配置参数结构体 - 外部配置项
typedef struct CommConfigHandleDef {
    CommConfigInfo info;
    u32 bufferSize;     // ccl buffer 大小配置
    u32 deterministic;   // 确定性计算配置
    char commName[COMM_NAME_MAX_LENGTH];  // 通信域名称
    char udi[UDI_MAX_LENGTH];   // user define information
    u32 opExpansionMode;    // 0：默认值  1：host  2：aicpu  3:aiv
    u32 trafficClass;
    u32 serviceLevel;
} CommConfigHandle;

namespace hccl {
class CommConfig {
public:
    CommConfig(const std::string &commName);  // 构造函数需传入默认的通信域ID
    ~CommConfig();

    HcclResult Load(const HcclCommConfig *userConfig); // 读取通信域配置
    u64 GetConfigBufferSize() const;               // 获取CCL buffer大小配置
    u8 GetConfigDeterministic() const;             // 获取确定性计算配置
    const std::string& GetConfigCommName() const;  // 获取通信域名称
    const std::string& GetConfigUdi() const;  // 获取UDI
    bool GetConfigAivMode() const;         // 获取AIV配置
    bool GetConfigAicpuUnfold() const;         // 获取AICPU配置, 在310P和A3中AICPU展开
    u32 GetConfigTrafficClass() const;
    u32 GetConfigServiceLevel() const;

private:
    HcclResult CheckMagicWord(const CommConfigHandle& config);      // 检查Magic Word是否合法
    HcclResult SetConfigByVersion(const CommConfigHandle& config);  // 根据版本号读取配置，保证兼容性

    HcclResult SetConfigBufferSize(const CommConfigHandle& config);     // 设置通信Buffer配置
    HcclResult SetConfigDeterministic(const CommConfigHandle& config);  // 设置确定性计算配置
    HcclResult SetConfigCommName(const CommConfigHandle& config);       // 设置通信域名称
    HcclResult SetConfigUdi(const CommConfigHandle& config);  // 设置UDI
    HcclResult SetConfigOpExpansionMode(const CommConfigHandle& config);  // 设置AIV和AICPU, 在310P和A3中AICPU展开

    u64 bufferSize_;        // CCL buffer大小配置，单位B
    u8 deterministic_;      // 确定性计算配置：0-关闭，1-开启，其他数字暂时保留
    std::string commName_;  // 通信域名称
    std::string udi_;       // user define information，用于在报错日志中定位错误通信域
    bool aivMode_;
    bool aicpuUnfold_;
    u32 trafficClass_;
    u32 serviceLevel_;
};
}
#endif /* HCCL_COMM_CONFIG_PUB_H */