// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "mori/core/core.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "mori/utils/env_utils.hpp"
#include "mori/utils/hip_helper.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori {
namespace moe {

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

static constexpr int32_t EP_CONFIG_I32_VERSION = 1;

// 56 → block_elems = 7168/56 = 128, matching the AccumNum=8 + VecBytes=8 dequant specialization.
static constexpr int kDefaultFp8BlockwiseScaleDim = 56;
static constexpr const char* kFp8BlockwiseScaleDimEnv = "MORI_FP8_COMBINE_SCALE_DIM";

namespace {

constexpr size_t kV2LLControlBarrierBaseWords = 16;

const char* V2PhaseName(EpDispatchCombineHandle::V2LifecyclePhase phase) {
  switch (phase) {
    case EpDispatchCombineHandle::V2LifecyclePhase::ReadyDispatch:
      return "ready-dispatch";
    case EpDispatchCombineHandle::V2LifecyclePhase::LaunchingDispatch:
      return "launching-dispatch";
    case EpDispatchCombineHandle::V2LifecyclePhase::ReadyCombine:
      return "ready-combine";
    case EpDispatchCombineHandle::V2LifecyclePhase::LaunchingCombine:
      return "launching-combine";
    case EpDispatchCombineHandle::V2LifecyclePhase::Poisoned:
      return "poisoned-after-launch-failure";
  }
  return "unknown";
}

}  // namespace

void EpDispatchCombineHandle::BeginV2Dispatch(hipStream_t stream) {
  if (!IsInterNodeV2DirectType(config.kernelType)) return;
  std::lock_guard<std::mutex> lock(v2LifecycleMutex);
  if (v2LifecyclePhase != V2LifecyclePhase::ReadyDispatch) {
    throw std::runtime_error(std::string("V2 lifecycle rejects Dispatch while phase is ") +
                             V2PhaseName(v2LifecyclePhase));
  }
  if (v2LifecycleStreamBound && v2LifecycleStream != stream) {
    throw std::runtime_error("V2 lifecycle rejects cross-stream use of the same handle");
  }
  v2LifecycleStream = stream;
  v2LifecycleStreamBound = true;
  switch (inputType) {
    case HIP_R_32F:
      v2DispatchElemSize = sizeof(float);
      break;
    case HIP_R_16BF:
      v2DispatchElemSize = 2;
      break;
    case HIP_R_8F_E4M3:
    case HIP_R_8F_E4M3_FNUZ:
      v2DispatchElemSize = 1;
      break;
    default:
      throw std::runtime_error("V2 Dispatch does not support the prepared input type");
  }
  v2LifecyclePhase = V2LifecyclePhase::LaunchingDispatch;
}

void EpDispatchCombineHandle::CommitV2Dispatch() {
  if (!IsInterNodeV2DirectType(config.kernelType)) return;
  std::lock_guard<std::mutex> lock(v2LifecycleMutex);
  if (v2LifecyclePhase != V2LifecyclePhase::LaunchingDispatch)
    throw std::runtime_error("V2 lifecycle Dispatch commit without begin");
  v2LifecyclePhase = V2LifecyclePhase::ReadyCombine;
}

void EpDispatchCombineHandle::AbortV2Dispatch() {
  if (!IsInterNodeV2DirectType(config.kernelType)) return;
  std::lock_guard<std::mutex> lock(v2LifecycleMutex);
  // A launch failure may occur after one or more kernels were already enqueued. Rolling the
  // host phase back would allow a new generation to reuse transport/counter storage while those
  // kernels are still live. Fail closed: the handle must be destroyed after an abort.
  if (v2LifecyclePhase == V2LifecyclePhase::LaunchingDispatch)
    v2LifecyclePhase = V2LifecyclePhase::Poisoned;
}

void EpDispatchCombineHandle::BeginV2Combine(hipStream_t stream) {
  if (!IsInterNodeV2DirectType(config.kernelType)) return;
  std::lock_guard<std::mutex> lock(v2LifecycleMutex);
  if (v2LifecyclePhase != V2LifecyclePhase::ReadyCombine) {
    throw std::runtime_error(std::string("V2 lifecycle rejects Combine while phase is ") +
                             V2PhaseName(v2LifecyclePhase));
  }
  if (!v2LifecycleStreamBound || v2LifecycleStream != stream) {
    throw std::runtime_error("V2 lifecycle rejects cross-stream use of the same handle");
  }
  v2LifecyclePhase = V2LifecyclePhase::LaunchingCombine;
}

void EpDispatchCombineHandle::CommitV2Combine() {
  if (!IsInterNodeV2DirectType(config.kernelType)) return;
  std::lock_guard<std::mutex> lock(v2LifecycleMutex);
  if (v2LifecyclePhase != V2LifecyclePhase::LaunchingCombine)
    throw std::runtime_error("V2 lifecycle Combine commit without begin");
  v2LifecyclePhase = V2LifecyclePhase::ReadyDispatch;
}

void EpDispatchCombineHandle::AbortV2Combine() {
  if (!IsInterNodeV2DirectType(config.kernelType)) return;
  std::lock_guard<std::mutex> lock(v2LifecycleMutex);
  if (v2LifecyclePhase == V2LifecyclePhase::LaunchingCombine)
    v2LifecyclePhase = V2LifecyclePhase::Poisoned;
}

std::vector<int32_t> EpDispatchCombineConfig::ToPackedI32Array() const {
  return {
      EP_CONFIG_I32_VERSION,
      rank,
      worldSize,
      hiddenDim,
      scaleDim,
      scaleTypeSize,
      maxTokenTypeSize,
      maxNumInpTokenPerRank,
      numExpertPerRank,
      numExpertPerToken,
      maxTotalRecvTokens,
      warpNumPerBlock,
      blockNum,
      static_cast<int32_t>(useExternalInpBuffer),
      static_cast<int32_t>(kernelType),
      gpuPerNode,
      rdmaBlockNum,
      numQpPerPe,
      static_cast<int32_t>(quantType),
      static_cast<int32_t>(enableSdma),
  };
}

EpDispatchCombineConfig EpDispatchCombineConfig::FromPackedI32Array(const int32_t* packed,
                                                                    size_t size) {
  // Runtime check to ensure the size of the packed array is correct
  if (size - 1 != kPackedI32Len) {
    throw std::runtime_error("EpDispatchCombineConfig i32 decode failed: invalid size");
  }
  if (packed == nullptr || packed[0] != EP_CONFIG_I32_VERSION) {
    throw std::runtime_error("EpDispatchCombineConfig i32 decode failed: unsupported version");
  }

  EpDispatchCombineConfig cfg;
  cfg.rank = packed[1];
  cfg.worldSize = packed[2];
  cfg.hiddenDim = packed[3];
  cfg.scaleDim = packed[4];
  cfg.scaleTypeSize = packed[5];
  cfg.maxTokenTypeSize = packed[6];
  cfg.maxNumInpTokenPerRank = packed[7];
  cfg.numExpertPerRank = packed[8];
  cfg.numExpertPerToken = packed[9];
  cfg.maxTotalRecvTokens = packed[10];
  cfg.warpNumPerBlock = packed[11];
  cfg.blockNum = packed[12];
  cfg.useExternalInpBuffer = (packed[13] != 0);
  cfg.kernelType = static_cast<KernelType>(packed[14]);
  cfg.gpuPerNode = packed[15];
  cfg.rdmaBlockNum = packed[16];
  cfg.numQpPerPe = packed[17];
  cfg.quantType = static_cast<QuantType>(packed[18]);
  cfg.enableSdma = (packed[19] != 0);
  return cfg;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                     EpDispatchCombineHandle                                    */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineHandle::EpDispatchCombineHandle(EpDispatchCombineConfig config_)
    : config(config_) {
  if (!IsPowerOf2(config.gpuPerNode) || config.worldSize % config.gpuPerNode != 0) {
    throw std::invalid_argument(
        "gpuPerNode must be a power of two and divide worldSize");
  }
  if (IsInterNodeV2DirectType(config.kernelType) &&
      (config.numQpPerPe < 1 || config.numQpPerPe > 8)) {
    throw std::invalid_argument("InterNodeV2LL numQpPerPe must be in [1, 8]");
  }
  if (IsInterNodeV2DirectType(config.kernelType) && config.maxTokenTypeSize < 2) {
    throw std::invalid_argument(
        "InterNodeV2LL maxTokenTypeSize must be at least 2 for BF16 Combine buffers");
  }
  int shmemNumQpPerPe = ShmemNumQpPerPe();
  if (config.numQpPerPe > shmemNumQpPerPe) {
    const int requestedNumQpPerPe = config.numQpPerPe;
    config.numQpPerPe = shmemNumQpPerPe;
    MORI_OPS_INFO("numQpPerPe {} larger than shmem numQpPerPe {}, set to {}",
                  requestedNumQpPerPe, shmemNumQpPerPe, shmemNumQpPerPe);
  }
  if (IsInterNodeV2DirectType(config.kernelType)) {
    if (config.warpNumPerBlock < config.numQpPerPe) {
      throw std::invalid_argument(
          "InterNodeV2LL warpNumPerBlock must be >= numQpPerPe");
    }
  }

  if (IsBlockwiseCombineQuant(config.quantType)) {
    fp8BlockwiseCombineScaleDim =
        env::GetPositiveIntOr(kFp8BlockwiseScaleDimEnv, kDefaultFp8BlockwiseScaleDim);
    fp8BlockwiseCombineScaleTypeSize = static_cast<int>(sizeof(float));
    if (config.rank == 0) {
      MORI_OPS_INFO("Blockwise combine ({}) scale_dim={} (override via {})",
                    config.quantType == QuantType::Fp4BlockwiseQuant ? "FP4" : "FP8",
                    fp8BlockwiseCombineScaleDim, kFp8BlockwiseScaleDimEnv);
    }
  }

  // Read the SDMA flag from the Context-cached snapshot (set once at Context
  // construction). Reading getenv directly here would race with the
  // SymmMemManager / Context decisions made at shmem init time -- the symptom
  // was tests that set MORI_ENABLE_SDMA inside the test function deadlocking
  // because Malloc started returning uncached buffers while Context still
  // believed the transport was P2P.
  config.enableSdma = ShmemSdmaEnabled();
  MORI_OPS_INFO("EpDispatchCombine SDMA {} (currently only effective for AsyncLL kernel type)",
                config.enableSdma ? "enabled" : "disabled");
  if (config.kernelType == KernelType::AsyncLL && !config.enableSdma && config.rank == 0) {
    MORI_OPS_WARN(
        "Mori AsyncLL is selected but SDMA is disabled. AsyncLL without SDMA uses compute units "
        "for communication, which provides little overlap benefit and can severely degrade "
        "performance. Use a non-AsyncLL kernel path or set MORI_ENABLE_SDMA=1.");
  }
  if (config.maxTotalRecvTokens > 0) {
    int worstCase = config.worldSize * config.maxNumInpTokenPerRank;
    if (config.maxTotalRecvTokens > worstCase) {
      MORI_OPS_INFO("maxTotalRecvTokens={} exceeds worst case {}, clamping to worst case",
                    config.maxTotalRecvTokens, worstCase);
      config.maxTotalRecvTokens = worstCase;
    }
    MORI_OPS_INFO(
        "maxTotalRecvTokens={}, effective MaxNumTokensToRecvPerRank={}, "
        "buffer MaxNumTokensToRecv={} (original worst case={})",
        config.maxTotalRecvTokens, config.MaxNumTokensToRecvPerRank(), config.MaxNumTokensToRecv(),
        worstCase);
  }
  InitializeShmemBuf();
  InitializeTokenNumSignalBuf();
  InitializeOrderMapBuf();
  InitializeBarrier();

  this->multiProcessorCount = GetCurDeviceMultiProcessorCount();
  this->maxThreads = std::min(GetCurDeviceMaxThreads(), 1024);
  MORI_OPS_INFO("Device capability: multiProcessorCount={}, maxThreads={}",
                static_cast<int>(this->multiProcessorCount), static_cast<int>(this->maxThreads));
}

EpDispatchCombineHandle::~EpDispatchCombineHandle() {
  auto* states = mori::shmem::ShmemStatesSingleton::GetInstance();
  if (states->status != mori::shmem::ShmemStatesStatus::Initialized) {
    return;
  }
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  FinalizeShmemBuf();
  FinalizeTokenNumSignalBuf();
  FinalizeOrderMapBuf();
  FinalizeBarrier();
}

mori::application::SymmMemObjPtr ShmemMallocAndReturnMemObjPtr(size_t size, unsigned int flags) {
  void* buf = ShmemExtMallocWithFlags(size, flags);
  HIP_RUNTIME_CHECK(hipMemset(buf, 0, size));
  mori::application::SymmMemObjPtr obj = ShmemQueryMemObjPtr(buf);
  assert(obj.IsValid());
  return obj;
}

void EpDispatchCombineHandle::InitializeShmemBuf() {
  size_t combineOutSize = static_cast<ssize_t>(config.MaxNumTokensToSendPerRank()) *
                          config.HiddenDimSz() * config.maxTokenTypeSize;
  size_t dispatchOutSize = static_cast<ssize_t>(config.MaxNumTokensToRecv()) *
                           config.HiddenDimSz() * config.maxTokenTypeSize;
  size_t maxStagingSize =
      static_cast<ssize_t>(config.MaxNumTokensToRecv()) * config.MaxXferBytesPerToken();
  if (config.kernelType == KernelType::IntraNode && IsBlockwiseCombineQuant(config.quantType)) {
    size_t blockwiseScaleBytes =
        (fp8BlockwiseCombineScaleDim > 0)
            ? static_cast<size_t>(fp8BlockwiseCombineScaleDim) * fp8BlockwiseCombineScaleTypeSize
            : 0;
    // FP4 packs the token region at 0.5 byte/elem (CombineTokenRegionBytes()), so its staging slot
    // is half the FP8 one -- no FP8-sized over-allocation for FP4.
    maxStagingSize = static_cast<size_t>(config.MaxNumTokensToRecv()) *
                     (config.CombineTokenRegionBytes() + config.IndexBytes() +
                      config.WeightBytes() + config.SrcTokenIdBytes() + blockwiseScaleBytes);
  }

  if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL) {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsIntraNode>();
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  } else if (IsInterNodeV1BufferType(config.kernelType)) {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNodeV1>();
    const int nNodes = config.worldSize / config.gpuPerNode;
    // V2LL uses generation-parity transport slots.  The sender may start packing the next
    // dispatch after the previous combine has advanced the epoch, while the peer is still
    // consuming the preceding RDMA payload. Keep the original single-slot layout for V1 and
    // allocate two complete transport generations for V2LL.
    const size_t transportRingSize =
        config.kernelType == KernelType::InterNodeV2LL ? size_t{2} : size_t{1};
    size_t dispatchInpSize = static_cast<ssize_t>(nNodes) * config.MaxNumTokensToSendPerRank() *
                             config.MaxXferBytesPerToken() * transportRingSize;
    size_t stagingSize = static_cast<ssize_t>(2 * nNodes) * config.MaxNumTokensToSendPerRank() *
                         config.MaxXferBytesPerToken() * transportRingSize;
    size_t dispatchStagingSize =
        static_cast<ssize_t>(config.MaxNumTokensToSendPerRank()) *
        config.MaxXferBytesPerToken() * transportRingSize;
    bufs.dispatchInp = ShmemMallocAndReturnMemObjPtr(dispatchInpSize, hipDeviceMallocUncached);
    const size_t v2PackedTokenBytes =
        config.V2PackedTokenSlots() * config.HiddenDimSz() * config.maxTokenTypeSize;
    const size_t combineInpSize = IsInterNodeV2DirectType(config.kernelType)
                                      ? v2PackedTokenBytes
                                      : maxStagingSize;
    const size_t actualDispatchOutSize = IsInterNodeV2DirectType(config.kernelType)
                                             ? v2PackedTokenBytes
                                             : dispatchOutSize;
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(combineInpSize, hipDeviceMallocUncached);
    bufs.staging = ShmemMallocAndReturnMemObjPtr(stagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut =
        ShmemMallocAndReturnMemObjPtr(actualDispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
    bufs.dispatchStaging =
        ShmemMallocAndReturnMemObjPtr(dispatchStagingSize, hipDeviceMallocUncached);
  } else {
    auto& bufs = shmemTokBufs.emplace<ShmemBufsInterNode>();
    // NOTE(ditian12): no overflow protection for dispatchInp/combinInp/staging in async kernel,
    // hence have to allocate to max size we need to either implement compact layout or add
    // pre-assertion to prevent silent memory access fault
    size_t maxStagingSize =
        static_cast<ssize_t>(config.MaxNumTokensToSend()) * config.MaxXferBytesPerToken();
    bufs.dispatchInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.combineInp = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.staging = ShmemMallocAndReturnMemObjPtr(maxStagingSize, hipDeviceMallocUncached);
    bufs.dispatchOut = ShmemMallocAndReturnMemObjPtr(dispatchOutSize, hipDeviceMallocUncached);
    bufs.combineOut = ShmemMallocAndReturnMemObjPtr(combineOutSize, hipDeviceMallocUncached);
  }

  size_t maxWeightSize =
      static_cast<size_t>(config.MaxNumTokensToRecv()) * config.numExpertPerToken * sizeof(float);
  if (IsInterNodeV2DirectType(config.kernelType)) {
    // One routing weight per expert-major slot; no rank-partial top-k array is materialized.
    maxWeightSize = config.V2PackedTokenSlots() * sizeof(float);
  }
  shmemInpWeightsMemObj = ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemDispatchOutWeightsMemObj =
      ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);
  shmemCombineOutWeightsMemObj =
      ShmemMallocAndReturnMemObjPtr(maxWeightSize, hipDeviceMallocUncached);

  size_t userScaleSize = 0;
  if (config.scaleDim > 0 && config.scaleTypeSize > 0) {
    userScaleSize =
        static_cast<size_t>(config.MaxNumTokensToRecv()) * config.scaleDim * config.scaleTypeSize;
    if (IsInterNodeV2DirectType(config.kernelType)) {
      userScaleSize = config.V2PackedTokenSlots() * config.scaleDim * config.scaleTypeSize;
    }
  }
  size_t fp8BlockwiseScaleSize = 0;
  if (IsBlockwiseCombineQuant(config.quantType) && fp8BlockwiseCombineScaleDim > 0) {
    fp8BlockwiseScaleSize = static_cast<size_t>(config.MaxNumTokensToRecv()) *
                            fp8BlockwiseCombineScaleDim * fp8BlockwiseCombineScaleTypeSize;
  }
  size_t inpScaleSize = std::max(userScaleSize, fp8BlockwiseScaleSize);
  if (inpScaleSize > 0) {
    shmemInpScalesMemObj = ShmemMallocAndReturnMemObjPtr(inpScaleSize, hipDeviceMallocUncached);
  }
  if (userScaleSize > 0) {
    shmemOutScalesMemObj = ShmemMallocAndReturnMemObjPtr(userScaleSize, hipDeviceMallocUncached);
  }

  size_t maxIndicesSize =
      static_cast<size_t>(config.MaxNumTokensToRecv()) * config.numExpertPerToken * sizeof(index_t);
  shmemInpIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);
  shmemOutIndicesMemObj = ShmemMallocAndReturnMemObjPtr(maxIndicesSize, hipDeviceMallocUncached);

#ifdef ENABLE_PROFILER
  size_t debugBufSize = MAX_DEBUG_TIME_SLOTS * sizeof(int64_t);
  HIP_RUNTIME_CHECK(hipMalloc(&profilerConfig.debugTimeBuf, debugBufSize));
  HIP_RUNTIME_CHECK(hipMemset(profilerConfig.debugTimeBuf, 0, debugBufSize));

  size_t offsetBufSize = PROFILER_WARPS_PER_RANK * sizeof(unsigned int);
  HIP_RUNTIME_CHECK(hipMalloc(&profilerConfig.debugTimeOffset, offsetBufSize));
  HIP_RUNTIME_CHECK(hipMemset(profilerConfig.debugTimeOffset, 0, offsetBufSize));
#endif
}

void EpDispatchCombineHandle::FinalizeShmemBuf() {
  if (config.kernelType == KernelType::IntraNode || config.kernelType == KernelType::IntraNodeLL) {
    auto& bufs = std::get<ShmemBufsIntraNode>(shmemTokBufs);
    ShmemFree(bufs.dispatchOut->localPtr);
    ShmemFree(bufs.combineInp->localPtr);
    ShmemFree(bufs.combineOut->localPtr);
  } else if (IsInterNodeV1BufferType(config.kernelType)) {
    auto& bufs = std::get<ShmemBufsInterNodeV1>(shmemTokBufs);
    ShmemFree(bufs.dispatchInp->localPtr);
    ShmemFree(bufs.combineInp->localPtr);
    ShmemFree(bufs.dispatchOut->localPtr);
    ShmemFree(bufs.combineOut->localPtr);
    ShmemFree(bufs.staging->localPtr);
    ShmemFree(bufs.dispatchStaging->localPtr);
  } else {
    auto& bufs = std::get<ShmemBufsInterNode>(shmemTokBufs);
    ShmemFree(bufs.dispatchInp->localPtr);
    ShmemFree(bufs.combineInp->localPtr);
    ShmemFree(bufs.dispatchOut->localPtr);
    ShmemFree(bufs.combineOut->localPtr);
    ShmemFree(bufs.staging->localPtr);
  }
  ShmemFree(shmemInpWeightsMemObj->localPtr);
  ShmemFree(shmemDispatchOutWeightsMemObj->localPtr);
  ShmemFree(shmemCombineOutWeightsMemObj->localPtr);
  if (shmemInpScalesMemObj.IsValid()) ShmemFree(shmemInpScalesMemObj->localPtr);
  if (shmemOutScalesMemObj.IsValid()) ShmemFree(shmemOutScalesMemObj->localPtr);
  ShmemFree(shmemInpIndicesMemObj->localPtr);
  ShmemFree(shmemOutIndicesMemObj->localPtr);
#ifdef ENABLE_PROFILER
  HIP_RUNTIME_CHECK(hipFree(profilerConfig.debugTimeBuf));
  HIP_RUNTIME_CHECK(hipFree(profilerConfig.debugTimeOffset));
#endif
}

void EpDispatchCombineHandle::InitializeTokenNumSignalBuf() {
  size_t tokenNumSignalSize = config.worldSize * sizeof(index_t) * 2 * config.numQpPerPe;
  recvTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendTokenNumMemObj = ShmemMallocAndReturnMemObjPtr(tokenNumSignalSize, hipDeviceMallocUncached);
  sendAtomicSignalMemObj = ShmemMallocAndReturnMemObjPtr(
      (config.worldSize * 2) * sizeof(int64_t) * 2, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&totalRecvTokenNum, sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(totalRecvTokenNum, 0, sizeof(index_t)));

  size_t nodeTokenNumSignalSize = static_cast<size_t>(config.worldSize / config.gpuPerNode) *
                                  config.numQpPerPe * sizeof(uint64_t);
  if (config.kernelType == KernelType::InterNodeV2LL) {
    // Dispatch and Combine use independent monotonic-generation signal slots.
    nodeTokenNumSignalSize *= 2;
  }
  nodeRecvTokenNumMemObj =
      ShmemMallocAndReturnMemObjPtr(nodeTokenNumSignalSize, hipDeviceMallocUncached);
}

void EpDispatchCombineHandle::FinalizeTokenNumSignalBuf() {
  ShmemFree(recvTokenNumMemObj->localPtr);
  ShmemFree(sendTokenNumMemObj->localPtr);
  ShmemFree(sendAtomicSignalMemObj->localPtr);
  ShmemFree(nodeRecvTokenNumMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(totalRecvTokenNum));
}

void EpDispatchCombineHandle::InitializeOrderMapBuf() {
  size_t maxNumOutToken =
      static_cast<size_t>(config.MaxNumTokensToSend()) * config.numExpertPerRank;
  HIP_RUNTIME_CHECK(hipMalloc(&dispReceiverIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispReceiverIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&dispSenderIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispSenderIdxMap, 0, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&destPeTokenIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenIdxMap, -1, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&srcPeTokenIdxMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(srcPeTokenIdxMap, -1, maxNumOutToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&destPeTokenCounter, config.worldSize * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(destPeTokenCounter, 0, config.worldSize * sizeof(index_t)));

  HIP_RUNTIME_CHECK(
      hipMalloc(&destNodeTokenCounter, config.worldSize / config.gpuPerNode * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(destNodeTokenCounter, 0, config.worldSize / config.gpuPerNode * sizeof(index_t)));

  HIP_RUNTIME_CHECK(hipMalloc(&localPeTokenCounter, config.worldSize * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(localPeTokenCounter, 0, config.worldSize * sizeof(index_t)));

  size_t dispTokOffsetSize = sizeof(index_t);
  if (IsInterNodeV2DirectType(config.kernelType)) {
    dispTokOffsetSize = static_cast<size_t>(config.numExpertPerRank) * sizeof(index_t);
  }
  dispTokOffsetMemObj =
      ShmemMallocAndReturnMemObjPtr(dispTokOffsetSize, hipDeviceMallocUncached);
  dispTokIdToSrcTokIdMemObj =
      ShmemMallocAndReturnMemObjPtr(maxNumOutToken * sizeof(index_t), hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&dispDestTokIdMap, maxNumOutToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(dispDestTokIdMap, 0, maxNumOutToken * sizeof(index_t)));

  size_t maxNumInterNodeToken = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                config.MaxNumTokensToSendPerRank() * config.numExpertPerToken;
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispDestTokIdMap, maxNumInterNodeToken * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(interNodeDispDestTokIdMap, 0, maxNumInterNodeToken * sizeof(index_t)));

  HIP_RUNTIME_CHECK(
      hipMalloc(&blockFlagCounter, config.worldSize / config.gpuPerNode * sizeof(index_t)));
  HIP_RUNTIME_CHECK(
      hipMemset(blockFlagCounter, 0, config.worldSize / config.gpuPerNode * sizeof(index_t)));

  size_t interNodeDispSendMapSize = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                    config.MaxNumTokensToSendPerRank() * sizeof(index_t);
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeDispSendMap, interNodeDispSendMapSize));
  HIP_RUNTIME_CHECK(hipMemset(interNodeDispSendMap, 0, interNodeDispSendMapSize));

#ifdef ENABLE_STANDARD_MOE_ADAPT
  const size_t maxDispatchTokens = static_cast<size_t>(config.MaxNumTokensToRecv());
  const size_t mapSize = maxDispatchTokens * config.numExpertPerToken * sizeof(uint64_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispTokToEpSlotMap, mapSize));
  HIP_RUNTIME_CHECK(hipMemset(dispTokToEpSlotMap, 0, mapSize));

  if (IsInterNodeV2DirectType(config.kernelType)) {
    standardPackedRecvCount = dispTokOffsetMemObj->template GetAs<int*>();
  } else {
    HIP_RUNTIME_CHECK(hipMalloc(&standardPackedRecvCount, config.numExpertPerRank * sizeof(int)));
    HIP_RUNTIME_CHECK(hipMemset(standardPackedRecvCount, 0, config.numExpertPerRank * sizeof(int)));
  }
#endif
}

void EpDispatchCombineHandle::FinalizeOrderMapBuf() {
  HIP_RUNTIME_CHECK(hipFree(dispReceiverIdxMap));
  HIP_RUNTIME_CHECK(hipFree(dispSenderIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(srcPeTokenIdxMap));
  HIP_RUNTIME_CHECK(hipFree(destPeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(destNodeTokenCounter));
  HIP_RUNTIME_CHECK(hipFree(localPeTokenCounter));
  ShmemFree(dispTokOffsetMemObj->localPtr);
  ShmemFree(dispTokIdToSrcTokIdMemObj->localPtr);
  HIP_RUNTIME_CHECK(hipFree(dispDestTokIdMap));
  HIP_RUNTIME_CHECK(hipFree(interNodeDispDestTokIdMap));
  HIP_RUNTIME_CHECK(hipFree(blockFlagCounter));
  HIP_RUNTIME_CHECK(hipFree(interNodeDispSendMap));
#ifdef ENABLE_STANDARD_MOE_ADAPT
  HIP_RUNTIME_CHECK(hipFree(dispTokToEpSlotMap));
  if (!IsInterNodeV2DirectType(config.kernelType))
    HIP_RUNTIME_CHECK(hipFree(standardPackedRecvCount));
#endif
}

void EpDispatchCombineHandle::InitializeBarrier() {
  const size_t nNodes =
      static_cast<size_t>(config.worldSize / config.gpuPerNode);
  const size_t v2llControlWords =
      kV2LLControlBarrierBaseWords +
      nNodes * static_cast<size_t>(config.numQpPerPe);
  const size_t barrierWords =
      std::max({static_cast<size_t>(config.worldSize),
                v2llControlWords});
  const size_t barrierSize = barrierWords * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMalloc(&dispatchGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(dispatchGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&combineGridBarrier, barrierSize));
  HIP_RUNTIME_CHECK(hipMemset(combineGridBarrier, 0, barrierSize));
  HIP_RUNTIME_CHECK(hipMalloc(&crossDeviceBarrierFlag, sizeof(uint64_t)));
  crossDeviceBarrierFlag[0] = (IsInterNodeV1BufferType(config.kernelType) ||
                               (config.kernelType == KernelType::AsyncLL))
                                  ? 0
                                  : 1;
  crossDeviceBarrierMemObj =
      ShmemMallocAndReturnMemObjPtr(barrierSize * 2 * sizeof(uint64_t), hipDeviceMallocUncached);

  size_t interNodeChunkFlagSize = static_cast<size_t>(config.worldSize) / config.gpuPerNode *
                                  config.MaxNumTokensToSendPerRank() * sizeof(uint64_t);
  interNodeChunkFlagMemObj =
      ShmemMallocAndReturnMemObjPtr(interNodeChunkFlagSize, hipDeviceMallocUncached);

  HIP_RUNTIME_CHECK(hipMalloc(&interNodeChunkFlagCombine, interNodeChunkFlagSize));
  HIP_RUNTIME_CHECK(hipMemset(interNodeChunkFlagCombine, 0, interNodeChunkFlagSize));

  constexpr size_t kInterNodeBarrierWords = 2;
  HIP_RUNTIME_CHECK(hipMalloc(&interNodeBlocksBarrier,
                              kInterNodeBarrierWords * sizeof(index_t)));
  HIP_RUNTIME_CHECK(hipMemset(interNodeBlocksBarrier, 0,
                              kInterNodeBarrierWords * sizeof(index_t)));
}

void EpDispatchCombineHandle::FinalizeBarrier() {
  HIP_RUNTIME_CHECK(hipFree(dispatchGridBarrier));
  HIP_RUNTIME_CHECK(hipFree(combineGridBarrier));
  HIP_RUNTIME_CHECK(hipFree(crossDeviceBarrierFlag));
  HIP_RUNTIME_CHECK(hipFree(interNodeChunkFlagCombine));
  HIP_RUNTIME_CHECK(hipFree(interNodeBlocksBarrier));
  ShmemFree(crossDeviceBarrierMemObj->localPtr);
  ShmemFree(interNodeChunkFlagMemObj->localPtr);
}

void EpDispatchCombineHandle::LaunchReset(hipStream_t stream) {}

/* ---------------------------------------------------------------------------------------------- */
/*                              Args construction for Python launch                               */
/* ---------------------------------------------------------------------------------------------- */
EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum) {
  EpDispatchCombineArgsRaw args;
  args.config = handle.config;
  args.fp8BlockwiseCombineScaleDim = handle.fp8BlockwiseCombineScaleDim;
  args.v2DispatchElemSize = handle.v2DispatchElemSize;
  args.rdmaBlockNum = rdmaBlockNum;
  args.curRankNumToken = handle.curRankNumToken;
  args.tokenIndices = handle.tokenIndices;
  args.inpTokenBuf = handle.inpTokenBuf;
  args.outTokenBuf = handle.outTokenBuf;
  args.weightsBuf = handle.weightsBuf;
  args.scalesBuf = handle.scalesBuf;
  args.destPeTokenCounter = handle.destPeTokenCounter;
  args.localPeTokenCounter = handle.localPeTokenCounter;
  if (handle.config.kernelType == KernelType::IntraNode ||
      handle.config.kernelType == KernelType::IntraNodeLL) {
    args.intraNodeTokBufs = std::get<ShmemBufsIntraNode>(handle.shmemTokBufs);
  } else if (IsInterNodeV1BufferType(handle.config.kernelType)) {
    args.interNodeV1TokBufs = std::get<ShmemBufsInterNodeV1>(handle.shmemTokBufs);
  } else {
    args.interNodeTokBufs = std::get<ShmemBufsInterNode>(handle.shmemTokBufs);
  }
  args.shmemInpWeightsMemObj = handle.shmemInpWeightsMemObj;
  args.shmemDispatchOutWeightsMemObj = handle.shmemDispatchOutWeightsMemObj;
  args.shmemCombineOutWeightsMemObj = handle.shmemCombineOutWeightsMemObj;
  args.shmemInpScalesMemObj = handle.shmemInpScalesMemObj;
  args.shmemOutScalesMemObj = handle.shmemOutScalesMemObj;
  args.shmemInpIndicesMemObj = handle.shmemInpIndicesMemObj;
  args.shmemOutIndicesMemObj = handle.shmemOutIndicesMemObj;
  args.recvTokenNumMemObj = handle.recvTokenNumMemObj;
  args.sendTokenNumMemObj = handle.sendTokenNumMemObj;
  args.sendAtomicSignalMemObj = handle.sendAtomicSignalMemObj;
  args.dispatchGridBarrier = handle.dispatchGridBarrier;
  args.combineGridBarrier = handle.combineGridBarrier;
  args.dispReceiverIdxMap = handle.dispReceiverIdxMap;
  args.dispSenderIdxMap = handle.dispSenderIdxMap;
  args.destPeTokenIdxMap = handle.destPeTokenIdxMap;
  // V2 direct kernels do not use the legacy srcPeTokenIdxMap. Reuse that existing ABI slot for
  // the host-side local address of the symmetric packed-count allocation; the GPU SymmMemObj's
  // implicit localPtr alias is not reliable on nonzero nodes.
  args.srcPeTokenIdxMap = IsInterNodeV2DirectType(handle.config.kernelType)
                              ? handle.dispTokOffsetMemObj->template GetAs<index_t*>()
                              : handle.srcPeTokenIdxMap;
  args.dispTokOffsetMemObj = handle.dispTokOffsetMemObj;
  args.dispTokIdToSrcTokIdMemObj = handle.dispTokIdToSrcTokIdMemObj;
  args.dispDestTokIdMap = handle.dispDestTokIdMap;
  args.totalRecvTokenNum = handle.totalRecvTokenNum;
  args.crossDeviceBarrierMemObj = handle.crossDeviceBarrierMemObj;
  args.crossDeviceBarrierFlag = handle.crossDeviceBarrierFlag;
  args.interNodeChunkFlagMemObj = handle.interNodeChunkFlagMemObj;
  args.destNodeTokenCounter = handle.destNodeTokenCounter;
  args.nodeRecvTokenNumMemObj = handle.nodeRecvTokenNumMemObj;
  args.blockFlagCounter = handle.blockFlagCounter;
  args.interNodeBlocksBarrier = handle.interNodeBlocksBarrier;
  args.interNodeDispDestTokIdMap = handle.interNodeDispDestTokIdMap;
  args.interNodeChunkFlagCombine = handle.interNodeChunkFlagCombine;
  args.interNodeDispSendMap = handle.interNodeDispSendMap;
#ifdef ENABLE_PROFILER
  args.profilerConfig = handle.profilerConfig;
#endif
#ifdef ENABLE_STANDARD_MOE_ADAPT
  args.enableStandardMoeOutput = handle.enableStandardMoeOutput;
  args.standardPackedRecvX = handle.standardPackedRecvX;
  args.standardPackedRecvCount = handle.standardPackedRecvCount;
  args.standardPackedRecvSrcInfo = handle.standardPackedRecvSrcInfo;
  args.standardPackedRecvLayoutRange = handle.standardPackedRecvLayoutRange;
  args.dispTokToEpSlotMap = handle.dispTokToEpSlotMap;
#endif
  return args;
}

void EpDispatchCombineRoutingPtrs::Validate() const {
  if (IsValid()) return;
  std::string missing;
  auto append = [&](const char* name, const index_t* ptr) {
    if (ptr == nullptr) {
      if (!missing.empty()) missing += ", ";
      missing += name;
    }
  };
  append("dispDestTokIdMap", dispDestTokIdMap);
  append("interNodeDispDestTokIdMap", interNodeDispDestTokIdMap);
  append("interNodeDispSendMap", interNodeDispSendMap);
  append("totalRecvTokenNum", totalRecvTokenNum);
  append("dispTokIdToSrcTokIdLocal", dispTokIdToSrcTokIdLocal);
  throw std::invalid_argument(
      "EpDispatchCombineRoutingPtrs: missing required routing pointer(s): " + missing);
}

EpDispatchCombineArgsRaw GetEpDispatchCombineArgsRaw(const EpDispatchCombineHandle& handle,
                                                     int rdmaBlockNum,
                                                     const EpDispatchCombineRoutingPtrs* routing,
                                                     bool replayMode) {
  EpDispatchCombineArgsRaw args = GetEpDispatchCombineArgsRaw(handle, rdmaBlockNum);
  args.replayMode = replayMode;
  if (routing != nullptr) {
    routing->Validate();
    args.dispDestTokIdMap = routing->dispDestTokIdMap;
    args.interNodeDispDestTokIdMap = routing->interNodeDispDestTokIdMap;
    args.interNodeDispSendMap = routing->interNodeDispSendMap;
    args.totalRecvTokenNum = routing->totalRecvTokenNum;
    args.dispTokIdToSrcTokIdLocal = routing->dispTokIdToSrcTokIdLocal;
  }
  return args;
}

}  // namespace moe
}  // namespace mori
