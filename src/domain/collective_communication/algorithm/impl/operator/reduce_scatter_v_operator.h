/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_V_OPERATOR_H
#define REDUCE_SCATTER_V_OPERATOR_H

#include "coll_alg_operator.h"
#include "coll_alg_op_registry.h"

namespace hccl {
class ReduceScatterVOperator : public CollAlgOperator {
public:
    ReduceScatterVOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
        HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~ReduceScatterVOperator();
    HcclResult SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag);

private:
    HcclResult SelectAlgfor910B(const OpParam& param, std::string& algName);
    HcclResult SelectAlgfor310P3(const OpParam& param, std::string& algName);
};

}

#endif /** __REDUCE_SCATTER_V_OPERATOR_H__ */