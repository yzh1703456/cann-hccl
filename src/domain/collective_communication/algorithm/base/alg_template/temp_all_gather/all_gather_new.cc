/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_new.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherNew::AllGatherNew(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher)
{
}

AllGatherNew::~AllGatherNew()
{
}

HcclResult AllGatherNew::TxVector(const LINK &link, const std::vector<Slice> &txSlices)
{
    std::vector<TxMemoryInfo> txMems;
    for (const Slice &txSlice : txSlices) {
        DeviceMem srcMem = outputMem_.range(txSlice.offset, txSlice.size);
        HCCL_DEBUG("tx srcMem[%p] range[%llu] size[%llu] ", srcMem.ptr(), txSlice.offset, txSlice.size);
        txMems.emplace_back(TxMemoryInfo{UserMemType::OUTPUT_MEM, txSlice.offset + baseOffset_,
            srcMem.ptr(), txSlice.size});
    }
    CHK_RET(link->TxAsync(txMems, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllGatherNew::RxVector(const LINK &link, const std::vector<Slice> &rxSlices)
{
    std::vector<RxMemoryInfo> rxMems;
    for (const Slice &rxSlice : rxSlices) {
        DeviceMem dstMem = outputMem_.range(rxSlice.offset, rxSlice.size);
        HCCL_DEBUG("rx dstMem[%p] range[%llu], size[%llu] ",  dstMem.ptr(),
            rxSlice.offset, rxSlice.size);
        rxMems.emplace_back(RxMemoryInfo{UserMemType::OUTPUT_MEM, rxSlice.offset + baseOffset_,
            dstMem.ptr(), rxSlice.size});
    }
    CHK_RET(link->RxAsync(rxMems, stream_));
    return HCCL_SUCCESS;
}

// 服务器间allgather的入口函数
HcclResult AllGatherNew::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherNew run_async rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    // 获取ring algorithm所需的通信连接
    u32 ringPrevRank = (rank + rankSize - 1) % rankSize;
    u32 ringNextRank = (rank + 1) % rankSize;

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherNew][RunAsync]rank[%u] linkSize is less than rankSize", rank);
        return HCCL_E_INTERNAL;
    }

    linkLeft_ = links[ringPrevRank];
    CHK_SMART_PTR_NULL(linkLeft_);

    linkRight_ = links[ringNextRank];
    CHK_SMART_PTR_NULL(linkRight_);

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherNew][RunAsync]unitSize is zero");
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices.resize(rankSize);

        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = (inputMem_.size() < outputMem_.size()) ? 0 : (sliceSize * i);
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", \
                       rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }

    // 双buffer下, 先将input拷贝到output的合适位置
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(inputSlices[rank].offset, inputSlices[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    CHK_RET(RunAllGather(rank, rankSize, slices_));

    if (barrierSwitchOn_) {
        // 执行barrier，保证数据收发完成
        CHK_RET(ExecuteBarrier(linkLeft_, linkRight_));
    }

    HCCL_INFO("AllGatherNew finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

HcclResult AllGatherNew::RunAllGather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[Run][AllGather]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret = HCCL_SUCCESS;

    // 首次传输，将本rank的数据发送到下游
    u32 sliceSize = outputSlices.size() / rankSize;
    u32 rxSliceIndex = ForwordRank(rank, rankSize, 1);
    u32 txSliceIndex = rank;
    for (u32 i = 0; i < rankSize - 1; i++) {
        HCCL_DEBUG("rank[%u] round[%u] will tx_ack  outputslice[%u].offset is[%llu] size[%llu]",
            rank, i, rxSliceIndex, outputSlices[rxSliceIndex].offset, outputSlices[rxSliceIndex].size);
        CHK_RET(linkLeft_->TxAck(stream_));

        // reduce目的操作
        HCCL_DEBUG("rank[%u] round[%u] will rx ack because outputSlices[%u] size[%llu] ", rank, \
            i, txSliceIndex, outputSlices[txSliceIndex].size);
        CHK_RET(linkRight_->RxAck(stream_));

        std::vector<Slice> txSegsSlice;
        std::vector<Slice> rxSegsSlice;
        for (u32 j = 0; j < sliceSize; j++) {
            txSegsSlice.push_back(outputSlices[txSliceIndex * sliceSize + j]);
            rxSegsSlice.push_back(outputSlices[rxSliceIndex * sliceSize + j]);
        }
        ret = TxVector(linkRight_, txSegsSlice);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] round[%u] Right Link tx outputSlices[%u] "\
                "Failed", rank, i, txSliceIndex), ret);

        // reduce源操作
        HCCL_DEBUG("rank[%u]  round[%u] rx data outputSlices[%u] offset[%llu] size[%llu]", \
            rank, i, rxSliceIndex, outputSlices[rxSliceIndex].offset, outputSlices[rxSliceIndex].size);
        ret = RxVector(linkLeft_, rxSegsSlice);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[Run][AllGather]rank[%u] round[%u]  Left Link rx outputSlices[%u] "\
                "Failed", rank, i, rxSliceIndex), ret);

        // 末尾传输, 只接收一次, 不用再次发送
        txSliceIndex = ForwordRank(txSliceIndex, rankSize, 1);
        rxSliceIndex = ForwordRank(rxSliceIndex, rankSize, 1);

        ret = linkLeft_->RxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]RxWaitDone failed"), ret);
        ret = linkRight_->TxWaitDone(stream_);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Run][ReduceScatter]TxWaitDone failed"), ret);
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_NEW, AllGatherNew);
}  // namespace hccl
