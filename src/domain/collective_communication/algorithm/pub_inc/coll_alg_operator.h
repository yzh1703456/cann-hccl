/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_OPERATOR_BASE_H
#define ALG_OPERATOR_BASE_H

#include <vector>
#include "hccl_impl.h"
#include "parallel_task_loader.h"
#include "dispatcher.h"
#include "ccl_buffer_manager.h"
#include "hccl_trace_info.h"
#include "device_capacity.h"
#include "topo_matcher.h"

#include "coll_alg_param.h"
#include "coll_executor_base.h"
#include "coll_alg_utils.h"
#include "alg_configurator.h"

namespace hccl {
struct PreProcessMetaInfo {
    HcclCMDType opType;
    std::vector<u64> inputData;
    u64 inputSize;
    u64 outputSize;
};

class CollAlgOperator {
public:
    CollAlgOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
                    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher, HcclCMDType opType);
    virtual ~CollAlgOperator() = default;

    virtual HcclResult SelectAlg(const std::string& tag,
        const OpParam& param, std::string& algName, std::string& newTag);
    virtual HcclResult CalcResRequest(const std::string& algName,
        const OpParam& param, AlgResourceRequest& resourceRequest);
    virtual HcclResult Orchestrate(const std::string& algName,
        OpParam& param, AlgResourceResponse& algResource);
    // batchsendrecv判断是否需要增量建链
    HcclResult CalcIncreLinkRequest(const std::string& algName, const OpParam& param,
        AlgResourceRequest& resourceRequest);
    AlgType GetAlgType();
    void SetLegacyHcclImpl(std::unique_ptr<hcclImpl> &hcclImpl);
    HcclResult SetAlgOpContext(AlgOpContext algOpContext);
    HcclResult SetRetryEnable(bool retryEnable);
    HcclResult SetAivClearEnable(bool aivClearEnable);
    bool SupportRetryWithInplaceCheck(
        const HcclCMDType &opType, OpParam &param, std::string& algName, u8 &isInplaceStatus,
        InplaceSupportRetryStatus &inPlaceSupportRetryStatus);
    HcclResult GetBlockDim(u32& blcckDim);
protected:
    std::string GenerateNewTagByAlgTypeLevel1(std::string tag, std::string algTypeLevel1Tag) const;
    u32 CalcContextNumForPipeline(HcclCMDType hcclCMDType);
    HcclResult  AutoSelectAlgTypeLevel1(HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize,
                                        std::string &algTypeLevel1Tag, bool isInlineReduce = false,
                                        bool isRdmaReduce = false, bool isAivMode = false);
    bool SingleMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op);
    bool Is2U2PInfer();
    bool IsMultiMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op);
    bool Is910BSingleMesh();
    bool NeedCreateSingleMeshPlane(const bool isInlineReduce);
    virtual HcclResult SetExecutorAttr(const OpParam& param);

    AlgType algType_;    // 算法类型
    TopoType topoType_;
    bool isAlgoLevel1Default_ = false;
    bool isHaveCpuRank_;
    bool inlineReduceSwitchOn_;
    std::string identifier_;
    OpMode opMode;

    AlgConfigurator* algConfigurator_ = nullptr;
    CCLBufferManager &cclBufferManager_;

    u32 serverNum_;
    u32 moduleNum_;
    u32 superPodNum_;
    u32 deviceNumPerServer_;
    u32 deviceNumPerAggregation_;
    bool multiModuleDiffDeviceNumMode_;
    bool multiSuperPodDiffServerNumMode_;
    u32 meshAggregationRankSize_;
    bool isDiffDeviceModule_;
    bool isDiffDeviceType_;
    u32 gcdDeviceNumPerAggregation_;
    bool isSingleMeshAggregation_ = false;
    bool meshSinglePlane_ = false;
    bool isAllRankSamePlane_ = false;
    bool is310PDuoCard_;
    bool isCommon310P3DUO_;
    s32 hccsPortNum_;
    bool isSupportRdmaLite_ = false;    // 是否支持rdma lite
    bool useSuperPodMode_ = false;
    u32 userRank_; // 本group中的userrank
    u32 realUserRank_; // world group中的userrank
    u32 userRankSize_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    DevType deviceType_;
    std::vector<u32> nicList_;
    std::unordered_map<u32, u32> pairLinkCounter_; // server内所有device间的链路类型计数
    hcclImpl* hcclImpl_ = nullptr;
    std::unique_ptr<CollExecutorBase> executor_;
    HcclDispatcher dispatcher_;
    std::unique_ptr<TopoMatcher> &topoMatcher_;
    HcclWorkflowMode workflowMode_;
    bool retryEnable_ = false;
    bool aivClearEnable_ = false;
    AlgOpContext algOpContext_;
private:
    virtual HcclResult SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);

    HcclResult AppendTag(const AlgTypeLevel1 &algTypeLevel1, std::string &tag);
    HcclResult SelectAlgoForComm(HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth,
        AlgTypeLevel1 &algType);
    HcclResult GetDefaultAlgoLevel1V2(HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize,
        AlgTypeLevel1 &algType, bool isInlineReduce = false, bool isRdmaReduce = false, bool isAivMode = false);
    void SetAlgoAttr(AlgConfigurator* algConfigurator);
    void SetTopoAttr(AlgConfigurator* algConfigurator);

    std::map<HcclCMDType, std::function<HcclResult(float, u64, float, AlgTypeLevel1 &)>> selectFuncMap_ = {
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
            std::bind(&CollAlgOperator::SelectAlgoTypeForReduceScatter, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)},
        {HcclCMDType::HCCL_CMD_ALLGATHER,
            std::bind(&CollAlgOperator::SelectAlgoTypeForAllGather, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)},
        {HcclCMDType::HCCL_CMD_ALLREDUCE,
            std::bind(&CollAlgOperator::SelectAlgoTypeForAllReduce, this,
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)},
    };
};
}   // namespace hccl

#endif /** __ALG_OPERATOR_BASE_H__ */