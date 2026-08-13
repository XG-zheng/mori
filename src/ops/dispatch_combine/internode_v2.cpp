// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License

#include "src/ops/dispatch_combine/internode_v2.hpp"

#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"

namespace mori {
namespace moe {
namespace v2 {

constexpr size_t kRdmaSplitAlignment = 2048;
constexpr int kQpCounterBaseSlot = 4;
constexpr int kResetReleaseSlot = 15;
constexpr int kQpEarlyCounterBaseSlot = 16;

inline __host__ __device__ int QpEarlyCounterSlot(
    const EpDispatchCombineConfig& config, int sourceNode, int qpId) {
  return kQpEarlyCounterBaseSlot + sourceNode * config.numQpPerPe + qpId;
}

inline __host__ __device__ int QpEarlyControlWords(
    const EpDispatchCombineConfig& config) {
  const int nNodes = config.worldSize / config.gpuPerNode;
  return kQpEarlyCounterBaseSlot + nNodes * config.numQpPerPe;
}

inline __device__ uint32_t* QpEarlyCounter(
    const EpDispatchCombineConfig& config, uint32_t* barrier,
    int sourceNode, int qpId) {
  return barrier + QpEarlyCounterSlot(config, sourceNode, qpId);
}

inline __device__ bool UseV2LLEpochSignals(const EpDispatchCombineConfig& config) {
  return config.kernelType == KernelType::InterNodeV2LL;
}

inline __device__ size_t SignalPhaseStride(const EpDispatchCombineConfig& config) {
  return static_cast<size_t>(config.worldSize / config.gpuPerNode) * config.numQpPerPe;
}

inline __device__ size_t DispatchSignalSlot(const EpDispatchCombineConfig& config, int node,
                                           int qpId) {
  return static_cast<size_t>(node) * config.numQpPerPe + qpId;
}

inline __device__ size_t CombineSignalSlot(const EpDispatchCombineConfig& config, int node,
                                          int qpId) {
  const size_t base = UseV2LLEpochSignals(config) ? SignalPhaseStride(config) : 0;
  return base + static_cast<size_t>(node) * config.numQpPerPe + qpId;
}

inline __device__ size_t LocalResetArrivalSlot(const EpDispatchCombineConfig& config, int pe) {
  return 2 * static_cast<size_t>(config.worldSize) + pe;
}

inline __device__ uint64_t EncodeReadySignal(const EpDispatchCombineConfig& config,
                                            uint64_t generation, uint32_t countPlusOne) {
  if (!UseV2LLEpochSignals(config)) return countPlusOne;
  return (generation << 32) | countPlusOne;
}

inline __device__ uint64_t ReadyGeneration(const EpDispatchCombineConfig& config,
                                           uint64_t signal) {
  return UseV2LLEpochSignals(config) ? (signal >> 32) : (signal != 0);
}

inline __device__ uint32_t ReadyCountPlusOne(const EpDispatchCombineConfig& config,
                                            uint64_t signal) {
  return UseV2LLEpochSignals(config) ? static_cast<uint32_t>(signal) : signal;
}

inline __device__ size_t TransportRingSlot(const EpDispatchCombineConfig& config,
                                          uint64_t generation) {
  return UseV2LLEpochSignals(config) ? (generation & 1u) : 0;
}

inline __device__ size_t DispatchSendBaseBytes(const EpDispatchCombineConfig& config,
                                              uint64_t generation) {
  return TransportRingSlot(config, generation) * config.MaxNumTokensToSendPerRank() *
         config.MaxXferBytesPerToken();
}

inline __device__ size_t DispatchRecvBaseBytes(const EpDispatchCombineConfig& config,
                                              uint64_t generation, int sourceNode) {
  const size_t nodeSlots = config.worldSize / config.gpuPerNode;
  return (TransportRingSlot(config, generation) * nodeSlots + sourceNode) *
         config.MaxNumTokensToSendPerRank() * config.MaxXferBytesPerToken();
}

template <typename T>
inline __device__ void SyncLocalResetCompletion(EpDispatchCombineArgs<T>& args,
                                                uint64_t generation) {
  DEF_COMMON_VARS;
  if (!UseV2LLEpochSignals(config)) return;
  const int nodePeOffset = myNode * config.gpuPerNode;
  // ResetAfterCombine is executed by the whole CTA. Publish every thread's counter stores before
  // announcing this PE. The matching wait is deliberately deferred to the next Dispatch, where
  // it overlaps staging pack/RDMA instead of extending the Combine epilogue.
  __threadfence_system();
  __syncthreads();
  if (warpId == 0) {
    if (laneId < config.gpuPerNode) {
      const int destPe = nodePeOffset + laneId;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(destPe) +
              LocalResetArrivalSlot(config, myPe),
          generation);
    }
  }
  __syncthreads();
}

template <typename T>
inline __device__ void WaitPreviousLocalReset(EpDispatchCombineArgs<T>& args,
                                              int scatterBlockId) {
  DEF_COMMON_VARS;
  if (!UseV2LLEpochSignals(config)) return;
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  // Generation zero is the constructor state before the first Dispatch. Starting with the
  // second Dispatch, wait for every local PE to publish the preceding Combine reset.
  if (generation == 0) return;

  uint32_t* release = args.dispatchGridBarrier + kResetReleaseSlot;
  if (scatterBlockId == 0 && warpId == 0) {
    const int nodePeOffset = myNode * config.gpuPerNode;
    const uint64_t* localFlags =
        args.crossDeviceBarrierMemObj->template GetAs<const uint64_t*>();
    if (laneId < config.gpuPerNode) {
      const int sourcePe = nodePeOffset + laneId;
      while (core::AtomicLoadRelaxedSystem(
                 localFlags + LocalResetArrivalSlot(config, sourcePe)) < generation) {
      }
    }
    __syncwarp();
    if (laneId == 0)
      core::AtomicStoreRelaxed(release, static_cast<uint32_t>(generation));
  }
  if (threadIdx.x == 0) {
    while (core::AtomicLoadRelaxed(release) < static_cast<uint32_t>(generation)) {
    }
  }
  __syncthreads();
}

inline __device__ void AlignedQpSlice(size_t totalBytes, int qpCount, int qpId,
                                     size_t& byteOffset, size_t& bytes) {
  const size_t unalignedChunk = core::CeilDiv(totalBytes, static_cast<size_t>(qpCount));
  const size_t chunkBytes =
      core::CeilDiv(unalignedChunk, kRdmaSplitAlignment) * kRdmaSplitAlignment;
  byteOffset = static_cast<size_t>(qpId) * chunkBytes;
  bytes = byteOffset < totalBytes ? min(chunkBytes, totalBytes - byteOffset) : 0;
}

inline __device__ index_t PackExpertSlot(const EpDispatchCombineConfig& config, int pe,
                                         int linearSlot) {
  return static_cast<index_t>(static_cast<size_t>(pe) * config.V2PackedTokenSlots() +
                              static_cast<size_t>(linearSlot));
}

inline __device__ int PeFromPackedExpertSlot(const EpDispatchCombineConfig& config,
                                             index_t packed) {
  return static_cast<int>(static_cast<size_t>(packed) / config.V2PackedTokenSlots());
}

inline __device__ int LocalSlotFromPackedExpertSlot(const EpDispatchCombineConfig& config,
                                                    index_t packed) {
  return static_cast<int>(static_cast<size_t>(packed) % config.V2PackedTokenSlots());
}

inline __device__ index_t NullPackedExpertSlot(const EpDispatchCombineConfig& config) {
  return static_cast<index_t>(static_cast<size_t>(config.worldSize) *
                              config.V2PackedTokenSlots());
}

inline __device__ size_t V2RouteMapOffset(const EpDispatchCombineConfig& config, int sourceNode,
                                          int tokenId, int expertSlot) {
  return (static_cast<size_t>(sourceNode) * config.MaxNumTokensToSendPerRank() + tokenId) *
             config.numExpertPerToken +
         expertSlot;
}

inline __device__ size_t CombineNodeSlotOffset(const EpDispatchCombineConfig& config, int slot,
                                               int tokenId) {
  return (static_cast<size_t>(slot) * config.MaxNumTokensToSendPerRank() + tokenId) *
         config.HiddenDimSz();
}

inline __device__ size_t CombineNodeSlotOffsetForGeneration(
    const EpDispatchCombineConfig& config, uint64_t generation, int slot, int tokenId) {
  const int nNodes = config.worldSize / config.gpuPerNode;
  const size_t epochBase = TransportRingSlot(config, generation) * 2 * nNodes;
  return (epochBase + slot) * config.MaxNumTokensToSendPerRank() * config.HiddenDimSz() +
         static_cast<size_t>(tokenId) * config.HiddenDimSz();
}

inline __device__ void TokenQpSlice(int tokenCount, int qpNum, int qpId, int& tokenBegin,
                                    int& qpTokenCount);

template <typename T>
inline __device__ const uint8_t* NodeInputBase(EpDispatchCombineArgs<T>& args, int node);

template <typename T>
inline __device__ void DispatchOneShotSendFromCurrentBlock(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  if (warpId >= config.numQpPerPe || laneId >= nNodes || laneId == myNode) return;

  const int remoteNode = laneId;
  const int qpId = warpId;
  const int proxyPe = remoteNode * config.gpuPerNode + (config.rank % config.gpuPerNode);
  const size_t totalBytes = static_cast<size_t>(args.curRankNumToken) * xferBytes;
  size_t byteOffset = 0;
  size_t bytes = 0;
  AlignedQpSlice(totalBytes, config.numQpPerPe, qpId, byteOffset, bytes);
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  const size_t remoteOffset = DispatchRecvBaseBytes(config, generation, myNode) + byteOffset;
  const size_t localOffset = DispatchSendBaseBytes(config, generation) + byteOffset;
  const uint64_t signal =
      EncodeReadySignal(config, generation, static_cast<uint32_t>(args.curRankNumToken) + 1);
  const size_t signalOffset = DispatchSignalSlot(config, myNode, qpId) * sizeof(uint64_t);

  if (bytes > 0) {
    shmem::ShmemPutMemNbiSignalThread(
        args.interNodeV1TokBufs.dispatchInp, remoteOffset,
        args.interNodeV1TokBufs.dispatchStaging, localOffset, bytes, args.nodeRecvTokenNumMemObj,
        signalOffset, signal, core::atomicType::AMO_SET, proxyPe, qpId);
  } else {
    shmem::ShmemPutTypeImmNbiThread<uint64_t>(args.nodeRecvTokenNumMemObj, signalOffset, signal,
                                              proxyPe, qpId);
  }
}

template <typename T>
inline __device__ void DispatchOneShotSendTokenQpFromCurrentBlock(
    EpDispatchCombineArgs<T>& args, int qpId) {
  DEF_COMMON_VARS;
  if (warpId != 0 || laneId >= nNodes || laneId == myNode) return;

  const int remoteNode = laneId;
  const int proxyPe = remoteNode * config.gpuPerNode + (config.rank % config.gpuPerNode);
  int tokenBegin = 0;
  int qpTokenCount = 0;
  TokenQpSlice(args.curRankNumToken, config.numQpPerPe, qpId, tokenBegin, qpTokenCount);
  const size_t byteOffset = static_cast<size_t>(tokenBegin) * xferBytes;
  const size_t bytes = static_cast<size_t>(qpTokenCount) * xferBytes;
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  const size_t remoteOffset = DispatchRecvBaseBytes(config, generation, myNode) + byteOffset;
  const size_t localOffset = DispatchSendBaseBytes(config, generation) + byteOffset;
  const uint64_t signal =
      EncodeReadySignal(config, generation, static_cast<uint32_t>(args.curRankNumToken) + 1);
  const size_t signalOffset = DispatchSignalSlot(config, myNode, qpId) * sizeof(uint64_t);

  if (bytes > 0) {
    shmem::ShmemPutMemNbiSignalThread(
        args.interNodeV1TokBufs.dispatchInp, remoteOffset,
        args.interNodeV1TokBufs.dispatchStaging, localOffset, bytes, args.nodeRecvTokenNumMemObj,
        signalOffset, signal, core::atomicType::AMO_SET, proxyPe, qpId);
  } else {
    shmem::ShmemPutTypeImmNbiThread<uint64_t>(args.nodeRecvTokenNumMemObj, signalOffset, signal,
                                              proxyPe, qpId);
  }
}

inline __device__ void WarpCopyScaleRow(uint8_t* __restrict__ dest,
                                        const uint8_t* __restrict__ source,
                                        size_t scaleBytes) {
  // Scale rows are normally arrays of FP32 values (32 or 128 bytes). The generic WarpCopy falls
  // through to independent one-byte stores for buffers smaller than a wave-wide 16-byte vector.
  // Use 4-byte transactions: packet rows are always 4-byte aligned, but their FP8 stride is not
  // necessarily 16-byte aligned.
  constexpr size_t kVecBytes = 4;
  const int laneId = threadIdx.x & (warpSize - 1);
  size_t offset = static_cast<size_t>(laneId) * kVecBytes;
  while (offset + kVecBytes <= scaleBytes) {
    const auto value = core::load<kVecBytes>(source + offset);
    core::store<kVecBytes>(dest + offset, value);
    offset += static_cast<size_t>(warpSize) * kVecBytes;
  }
  const size_t vectorBytes = (scaleBytes / kVecBytes) * kVecBytes;
  offset = vectorBytes + laneId;
  while (offset < scaleBytes) {
    dest[offset] = source[offset];
    offset += warpSize;
  }
}

template <typename T>
inline __device__ void ScatterOneRoute(EpDispatchCombineArgs<T>& args, const uint8_t* source,
                                       int sourceNode, int tokenId, int expertSlot) {
  DEF_COMMON_VARS;
  const size_t routeMapOffset = V2RouteMapOffset(config, sourceNode, tokenId, expertSlot);
  if (laneId == 0) {
    args.interNodeDispDestTokIdMap[routeMapOffset] = NullPackedExpertSlot(config);
  }

  const index_t* indices = reinterpret_cast<const index_t*>(source + hiddenBytes);
  const index_t destExpert = indices[expertSlot];
  if (destExpert < 0) return;

  const int destPe = destExpert / config.numExpertPerRank;
  if (destPe < 0 || destPe >= config.worldSize ||
      (destPe / config.gpuPerNode) != myNode) {
    return;
  }

  const int localExpert = destExpert % config.numExpertPerRank;
  int packedSlot = 0;
  if (laneId == 0) {
    packedSlot = core::AtomicAddRelaxedSystem(
        args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe) + localExpert, index_t{1});
    // destPeTokenCounter is ordinary hipMalloc memory owned by this source GPU. Only the
    // expert-slot allocator above targets a peer GPU and needs system scope.
    if (UseV2LLEpochSignals(config))
      core::AtomicAddRelaxed(args.destPeTokenCounter + destPe, index_t{1});
    else
      core::AtomicAddRelaxedSystem(args.destPeTokenCounter + destPe, index_t{1});
  }
  packedSlot = __shfl(packedSlot, 0);
  assert(packedSlot < config.V2MaxTokensPerExpert());

  const size_t linearSlot = static_cast<size_t>(localExpert) * config.V2MaxTokensPerExpert() +
                            static_cast<size_t>(packedSlot);
  core::WarpCopy<uint8_t, 8>(
      args.interNodeV1TokBufs.dispatchOut->template GetAs<uint8_t*>(destPe) +
          linearSlot * hiddenBytes,
      source, hiddenBytes);

  if (scaleBytes > 0) {
    WarpCopyScaleRow(
        args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + linearSlot * scaleBytes,
        source + hiddenBytes + indexBytes + weightBytes, scaleBytes);
  }

  if (laneId == 0) {
    const float* weights = reinterpret_cast<const float*>(source + hiddenBytes + indexBytes);
    args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe)[linearSlot] =
        weights[expertSlot];
    const index_t srcTokenId =
        reinterpret_cast<const index_t*>(source + hiddenBytes + indexBytes + weightBytes +
                                         scaleBytes)[0];
    args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[linearSlot] = srcTokenId;
    args.interNodeDispDestTokIdMap[routeMapOffset] =
        PackExpertSlot(config, destPe, static_cast<int>(linearSlot));
  }
}

template <typename T>
inline __device__ void ScatterOneLocalRoute(EpDispatchCombineArgs<T>& args, int tokenId,
                                            int expertSlot) {
  DEF_COMMON_VARS;
  const size_t routeMapOffset = V2RouteMapOffset(config, myNode, tokenId, expertSlot);
  if (laneId == 0) args.interNodeDispDestTokIdMap[routeMapOffset] = NullPackedExpertSlot(config);

  const index_t destExpert = args.tokenIndices[tokenId * config.numExpertPerToken + expertSlot];
  if (destExpert < 0) return;
  const int destPe = destExpert / config.numExpertPerRank;
  if (destPe < 0 || destPe >= config.worldSize ||
      (destPe / config.gpuPerNode) != myNode) {
    return;
  }

  const int localExpert = destExpert % config.numExpertPerRank;
  int packedSlot = 0;
  if (laneId == 0) {
    packedSlot = core::AtomicAddRelaxedSystem(
        args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe) + localExpert, index_t{1});
    if (UseV2LLEpochSignals(config))
      core::AtomicAddRelaxed(args.destPeTokenCounter + destPe, index_t{1});
    else
      core::AtomicAddRelaxedSystem(args.destPeTokenCounter + destPe, index_t{1});
  }
  packedSlot = __shfl(packedSlot, 0);
  assert(packedSlot < config.V2MaxTokensPerExpert());

  const size_t linearSlot = static_cast<size_t>(localExpert) * config.V2MaxTokensPerExpert() +
                            static_cast<size_t>(packedSlot);
  core::WarpCopy<T, 8>(
      args.interNodeV1TokBufs.dispatchOut->template GetAs<T*>(destPe) + linearSlot * hiddenDim,
      args.inpTokenBuf + static_cast<size_t>(tokenId) * hiddenDim, hiddenDim);
  if (scaleBytes > 0) {
    WarpCopyScaleRow(
        args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + linearSlot * scaleBytes,
        args.scalesBuf + static_cast<size_t>(tokenId) * scaleBytes, scaleBytes);
  }
  if (laneId == 0) {
    args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe)[linearSlot] =
        args.weightsBuf[tokenId * config.numExpertPerToken + expertSlot];
    args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[linearSlot] =
        static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId));
    args.interNodeDispDestTokIdMap[routeMapOffset] =
        PackExpertSlot(config, destPe, static_cast<int>(linearSlot));
  }
}

template <typename T>
inline __device__ void ScatterOneTokenMajorRoute(EpDispatchCombineArgs<T>& args,
                                                 const uint8_t* source, int sourceNode,
                                                 int tokenId, int expertSlot) {
  DEF_COMMON_VARS;
  const size_t routeMapOffset = V2RouteMapOffset(config, sourceNode, tokenId, expertSlot);
  if (laneId == 0)
    args.interNodeDispDestTokIdMap[routeMapOffset] = NullPackedExpertSlot(config);

  const index_t* indices = reinterpret_cast<const index_t*>(source + hiddenBytes);
  const index_t destExpert = indices[expertSlot];
  if (destExpert < 0) return;
  const int destPe = destExpert / config.numExpertPerRank;
  if (destPe < 0 || destPe >= config.worldSize ||
      (destPe / config.gpuPerNode) != myNode)
    return;

  // V1LL token-major semantics keep one row per (source token, destination PE), even when the
  // token selects multiple experts on that PE. The row retains the complete top-k metadata.
  for (int previous = 0; previous < expertSlot; ++previous) {
    const index_t previousExpert = indices[previous];
    if (previousExpert >= 0 && previousExpert / config.numExpertPerRank == destPe) return;
  }

  int packedSlot = 0;
  if (laneId == 0) {
    packedSlot = core::AtomicAddRelaxedSystem(
        args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), index_t{1});
    core::AtomicAddRelaxed(args.destPeTokenCounter + destPe, index_t{1});
  }
  packedSlot = __shfl(packedSlot, 0);
  assert(packedSlot < config.MaxNumTokensToRecv());

  core::WarpCopy<uint8_t, 8>(
      args.interNodeV1TokBufs.dispatchOut->template GetAs<uint8_t*>(destPe) +
          static_cast<size_t>(packedSlot) * hiddenBytes,
      source, hiddenBytes);
  core::WarpCopy<uint8_t, 4>(
      args.shmemOutIndicesMemObj->template GetAs<uint8_t*>(destPe) +
          static_cast<size_t>(packedSlot) * indexBytes,
      source + hiddenBytes, indexBytes);
  core::WarpCopy<uint8_t, 4>(
      args.shmemDispatchOutWeightsMemObj->template GetAs<uint8_t*>(destPe) +
          static_cast<size_t>(packedSlot) * weightBytes,
      source + hiddenBytes + indexBytes, weightBytes);
  if (scaleBytes > 0) {
    WarpCopyScaleRow(
        args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) +
            static_cast<size_t>(packedSlot) * scaleBytes,
        source + hiddenBytes + indexBytes + weightBytes, scaleBytes);
  }
  if (laneId == 0) {
    const index_t srcTokenId =
        reinterpret_cast<const index_t*>(source + hiddenBytes + indexBytes + weightBytes +
                                         scaleBytes)[0];
    args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[packedSlot] = srcTokenId;
    args.interNodeDispDestTokIdMap[routeMapOffset] =
        PackExpertSlot(config, destPe, packedSlot);
  }
}

template <typename T>
inline __device__ void ScatterOneLocalTokenMajorRoute(EpDispatchCombineArgs<T>& args,
                                                      int tokenId, int expertSlot) {
  DEF_COMMON_VARS;
  const size_t routeMapOffset = V2RouteMapOffset(config, myNode, tokenId, expertSlot);
  if (laneId == 0)
    args.interNodeDispDestTokIdMap[routeMapOffset] = NullPackedExpertSlot(config);

  const index_t destExpert = args.tokenIndices[tokenId * config.numExpertPerToken + expertSlot];
  if (destExpert < 0) return;
  const int destPe = destExpert / config.numExpertPerRank;
  if (destPe < 0 || destPe >= config.worldSize ||
      (destPe / config.gpuPerNode) != myNode)
    return;
  for (int previous = 0; previous < expertSlot; ++previous) {
    const index_t previousExpert =
        args.tokenIndices[tokenId * config.numExpertPerToken + previous];
    if (previousExpert >= 0 && previousExpert / config.numExpertPerRank == destPe) return;
  }

  int packedSlot = 0;
  if (laneId == 0) {
    packedSlot = core::AtomicAddRelaxedSystem(
        args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), index_t{1});
    core::AtomicAddRelaxed(args.destPeTokenCounter + destPe, index_t{1});
  }
  packedSlot = __shfl(packedSlot, 0);
  assert(packedSlot < config.MaxNumTokensToRecv());

  core::WarpCopy<T, 8>(
      args.interNodeV1TokBufs.dispatchOut->template GetAs<T*>(destPe) +
          static_cast<size_t>(packedSlot) * hiddenDim,
      args.inpTokenBuf + static_cast<size_t>(tokenId) * hiddenDim, hiddenDim);
  core::WarpCopy<index_t, 4>(
      args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe) +
          static_cast<size_t>(packedSlot) * config.numExpertPerToken,
      args.tokenIndices + static_cast<size_t>(tokenId) * config.numExpertPerToken,
      config.numExpertPerToken);
  core::WarpCopy<float, 4>(
      args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe) +
          static_cast<size_t>(packedSlot) * config.numExpertPerToken,
      args.weightsBuf + static_cast<size_t>(tokenId) * config.numExpertPerToken,
      config.numExpertPerToken);
  if (scaleBytes > 0) {
    WarpCopyScaleRow(
        args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) +
            static_cast<size_t>(packedSlot) * scaleBytes,
        args.scalesBuf + static_cast<size_t>(tokenId) * scaleBytes, scaleBytes);
  }
  if (laneId == 0) {
    args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[packedSlot] =
        static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId));
    args.interNodeDispDestTokIdMap[routeMapOffset] =
        PackExpertSlot(config, destPe, packedSlot);
  }
}

template <typename T>
inline __device__ void CopyTokenToStaging(EpDispatchCombineArgs<T>& args, uint8_t* staging,
                                          int tokenId) {
  DEF_COMMON_VARS;
  uint8_t* dest = staging + static_cast<size_t>(tokenId) * xferBytes;
  core::WarpCopy<uint8_t, 8>(dest, reinterpret_cast<const uint8_t*>(args.inpTokenBuf) +
                                      static_cast<size_t>(tokenId) * hiddenBytes,
                             hiddenBytes);
  core::WarpCopy<uint8_t, 4>(dest + hiddenBytes,
                             reinterpret_cast<const uint8_t*>(args.tokenIndices) +
                                 static_cast<size_t>(tokenId) * indexBytes,
                             indexBytes);
  core::WarpCopy<uint8_t, 4>(dest + hiddenBytes + indexBytes,
                             reinterpret_cast<const uint8_t*>(args.weightsBuf) +
                                 static_cast<size_t>(tokenId) * weightBytes,
                             weightBytes);
  if (scaleBytes > 0) {
    WarpCopyScaleRow(dest + hiddenBytes + indexBytes + weightBytes,
                     args.scalesBuf + static_cast<size_t>(tokenId) * scaleBytes, scaleBytes);
  }
  if (laneId == 0) {
    reinterpret_cast<index_t*>(dest + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
        static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId));
  }
}

template <typename T>
inline __device__ void CopyTokenPartToStaging(
    EpDispatchCombineArgs<T>& args, uint8_t* staging, int tokenId,
    int tokenPart, int warpsPerToken) {
  DEF_COMMON_VARS;
  uint8_t* dest = staging + static_cast<size_t>(tokenId) * xferBytes;
  const uint8_t* source =
      reinterpret_cast<const uint8_t*>(args.inpTokenBuf) +
      static_cast<size_t>(tokenId) * hiddenBytes;
  const size_t hiddenBytesPerWarp =
      core::CeilDiv(core::CeilDiv(hiddenBytes,
                                  static_cast<size_t>(warpsPerToken)),
                    size_t{16}) *
      size_t{16};
  const size_t hiddenOffset =
      static_cast<size_t>(tokenPart) * hiddenBytesPerWarp;
  const size_t hiddenChunk =
      hiddenOffset < hiddenBytes
          ? min(hiddenBytes - hiddenOffset, hiddenBytesPerWarp)
          : 0;
  if (hiddenChunk > 0)
    core::WarpCopy<uint8_t, 8>(dest + hiddenOffset, source + hiddenOffset,
                               hiddenChunk);

  if (tokenPart != 0) return;
  core::WarpCopy<uint8_t, 4>(
      dest + hiddenBytes,
      reinterpret_cast<const uint8_t*>(args.tokenIndices) +
          static_cast<size_t>(tokenId) * indexBytes,
      indexBytes);
  core::WarpCopy<uint8_t, 4>(
      dest + hiddenBytes + indexBytes,
      reinterpret_cast<const uint8_t*>(args.weightsBuf) +
          static_cast<size_t>(tokenId) * weightBytes,
      weightBytes);
  if (scaleBytes > 0) {
    WarpCopyScaleRow(
        dest + hiddenBytes + indexBytes + weightBytes,
        args.scalesBuf + static_cast<size_t>(tokenId) * scaleBytes,
        scaleBytes);
  }
  if (laneId == 0) {
    reinterpret_cast<index_t*>(
        dest + hiddenBytes + indexBytes + weightBytes + scaleBytes)[0] =
        static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId));
  }
}

template <typename T>
inline __device__ uint8_t* DispatchStagingBase(EpDispatchCombineArgs<T>& args,
                                               uint64_t generation) {
  return args.interNodeV1TokBufs.dispatchStaging->template GetAs<uint8_t*>() +
         DispatchSendBaseBytes(args.config, generation);
}

template <typename T>
inline __device__ void CopyToStagingGroup(EpDispatchCombineArgs<T>& args, int groupBlockId,
                                         int groupBlockNum) {
  DEF_COMMON_VARS;
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  const int groupWarpId = groupBlockId * warpNum + warpId;
  const int groupWarpNum = groupBlockNum * warpNum;
  uint8_t* staging = DispatchStagingBase(args, generation);
  for (int tokenId = groupWarpId; tokenId < args.curRankNumToken; tokenId += groupWarpNum) {
    CopyTokenToStaging(args, staging, tokenId);
  }
}

template <typename T>
inline __device__ void CopyToStagingGroupMultiWarp(
    EpDispatchCombineArgs<T>& args, int groupBlockId, int groupBlockNum) {
  DEF_COMMON_VARS;
  const uint64_t generation =
      core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  const int groupWarpId = groupBlockId * warpNum + warpId;
  const int groupWarpNum = groupBlockNum * warpNum;
  uint8_t* staging = DispatchStagingBase(args, generation);
  const int warpsPerToken =
      args.curRankNumToken > 0
          ? core::CeilDiv(groupWarpNum,
                          static_cast<int>(args.curRankNumToken))
          : 1;
  const int workCount = args.curRankNumToken * warpsPerToken;
  for (int work = groupWarpId; work < workCount; work += groupWarpNum) {
    const int tokenId = work / warpsPerToken;
    const int tokenPart = work % warpsPerToken;
    CopyTokenPartToStaging(args, staging, tokenId, tokenPart,
                           warpsPerToken);
  }
}

inline __device__ void TokenQpSlice(int tokenCount, int qpNum, int qpId, int& tokenBegin,
                                    int& qpTokenCount) {
  const int base = tokenCount / qpNum;
  const int remainder = tokenCount % qpNum;
  qpTokenCount = base + (qpId < remainder ? 1 : 0);
  tokenBegin = qpId * base + min(qpId, remainder);
}

template <typename T>
inline __device__ void CopyToStagingQpGroup(EpDispatchCombineArgs<T>& args, int qpId,
                                           int groupBlockId, int groupBlockNum) {
  DEF_COMMON_VARS;
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  int tokenBegin = 0;
  int qpTokenCount = 0;
  TokenQpSlice(args.curRankNumToken, config.numQpPerPe, qpId, tokenBegin, qpTokenCount);
  const int groupWarpId = groupBlockId * warpNum + warpId;
  const int groupWarpNum = groupBlockNum * warpNum;
  uint8_t* staging = DispatchStagingBase(args, generation);
  for (int qpToken = groupWarpId; qpToken < qpTokenCount; qpToken += groupWarpNum) {
    CopyTokenToStaging(args, staging, tokenBegin + qpToken);
  }
}

template <typename T>
inline __device__ void CopyToStagingQpGroupMultiWarp(
    EpDispatchCombineArgs<T>& args, int qpId, int groupBlockId,
    int groupBlockNum) {
  DEF_COMMON_VARS;
  const uint64_t generation =
      core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  int tokenBegin = 0;
  int qpTokenCount = 0;
  TokenQpSlice(args.curRankNumToken, config.numQpPerPe, qpId,
               tokenBegin, qpTokenCount);
  const int groupWarpId = groupBlockId * warpNum + warpId;
  const int groupWarpNum = groupBlockNum * warpNum;
  const int warpsPerToken =
      qpTokenCount > 0
          ? core::CeilDiv(groupWarpNum, qpTokenCount)
          : 1;
  const int workCount = qpTokenCount * warpsPerToken;
  uint8_t* staging = DispatchStagingBase(args, generation);
  for (int work = groupWarpId; work < workCount; work += groupWarpNum) {
    const int tokenOffset = work / warpsPerToken;
    const int tokenPart = work % warpsPerToken;
    CopyTokenPartToStaging(args, staging, tokenBegin + tokenOffset,
                           tokenPart, warpsPerToken);
  }
}

template <typename T>
inline __device__ void ScatterLocalGroup(EpDispatchCombineArgs<T>& args, int groupWarpId,
                                         int groupWarpNum) {
  DEF_COMMON_VARS;
  if (groupWarpId == 0 && laneId == 0)
    core::AtomicStoreRelaxed(args.blockFlagCounter + myNode, args.curRankNumToken);
  const size_t routeCount =
      static_cast<size_t>(args.curRankNumToken) * config.numExpertPerToken;
  for (size_t route = groupWarpId; route < routeCount; route += groupWarpNum) {
    ScatterOneLocalRoute(args, route / config.numExpertPerToken,
                         route % config.numExpertPerToken);
  }
}

template <typename T>
inline __device__ void ScatterLocalTokenMajorGroup(EpDispatchCombineArgs<T>& args,
                                                   int groupWarpId, int groupWarpNum) {
  DEF_COMMON_VARS;
  if (groupWarpId == 0 && laneId == 0)
    core::AtomicStoreRelaxed(args.blockFlagCounter + myNode, args.curRankNumToken);
  const size_t routeCount =
      static_cast<size_t>(args.curRankNumToken) * config.numExpertPerToken;
  for (size_t route = groupWarpId; route < routeCount; route += groupWarpNum) {
    ScatterOneLocalTokenMajorRoute(args, route / config.numExpertPerToken,
                                   route % config.numExpertPerToken);
  }
}

template <typename T>
inline __device__ uint64_t WaitNodeInputDirect(EpDispatchCombineArgs<T>& args, int node) {
  DEF_COMMON_VARS;
  if (node == myNode) return static_cast<uint64_t>(args.curRankNumToken);

  uint64_t tokenCount = 0;
  if (laneId < config.numQpPerPe) {
    uint64_t* signals = args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
    const size_t signalSlot = DispatchSignalSlot(config, node, laneId);
    const uint64_t expectedGeneration =
        UseV2LLEpochSignals(config)
            ? core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1
            : 1;
    uint64_t ready = 0;
    while (ReadyGeneration(config, ready) < expectedGeneration)
      ready = core::AtomicLoadRelaxedSystem(signals + signalSlot);
    assert(ReadyGeneration(config, ready) == expectedGeneration);
    tokenCount = ReadyCountPlusOne(config, ready) - 1;
  }
  __syncwarp();
  return __shfl(tokenCount, 0);
}

template <typename T>
inline __device__ uint64_t WaitNodeInput(EpDispatchCombineArgs<T>& args, int node,
                                         int nodeWarpId) {
  DEF_COMMON_VARS;
  if (node == myNode) return static_cast<uint64_t>(args.curRankNumToken);

  // Wait on all QP signals with separate lanes, but share the result only within a CTA.  A single
  // grid leader serialized release behind its scheduling slot, while making every consumer warp
  // hammer the uncached system signal generated hundreds of duplicate polls.  One waiter warp per
  // CTA retains broad receive concurrency and parallel per-QP waits without the polling storm.
  __shared__ uint64_t ctaTokenCount;
  if (warpId == 0) {
    const uint64_t count = WaitNodeInputDirect(args, node);
    if (laneId == 0) ctaTokenCount = count;
  }
  __syncthreads();
  return ctaTokenCount;
}

template <typename T>
inline __device__ const uint8_t* NodeInputBase(EpDispatchCombineArgs<T>& args, int node) {
  DEF_COMMON_VARS;
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
  if (node == myNode) {
    return args.interNodeV1TokBufs.dispatchStaging->template GetAs<const uint8_t*>() +
           DispatchSendBaseBytes(config, generation);
  }
  return args.interNodeV1TokBufs.dispatchInp->template GetAs<const uint8_t*>() +
         DispatchRecvBaseBytes(config, generation, node);
}

template <typename T>
inline __device__ float CombineRouteWeight(EpDispatchCombineArgs<T>& args, int sourceNode,
                                           uint64_t generation, int tokenId, int expertSlot) {
  DEF_COMMON_VARS;
  // Dispatch keeps the raw token packet alive until Combine releases this generation.  Read the
  // weight from that node-local staging/receive packet instead of fetching the copied scalar from
  // the destination expert GPU.  Hundreds of warps otherwise issue dependent 4-byte XGMI loads
  // before they can start the vector payload, reducing useful inflight bytes.
  const uint8_t* base = nullptr;
  if (sourceNode == myNode) {
    base = args.interNodeV1TokBufs.dispatchStaging->template GetAs<const uint8_t*>() +
           DispatchSendBaseBytes(config, generation);
  } else {
    base = args.interNodeV1TokBufs.dispatchInp->template GetAs<const uint8_t*>() +
           DispatchRecvBaseBytes(config, generation, sourceNode);
  }
  // The retained packet has the Dispatch element type, which can differ from T (the Group-GEMM
  // output / Combine type). FP8 Dispatch + BF16 Combine previously doubled both offsets here.
  const size_t dispatchElemSize = args.v2DispatchElemSize;
  assert(dispatchElemSize > 0 && dispatchElemSize <= config.maxTokenTypeSize);
  const size_t dispatchHiddenBytes = config.HiddenBytes(dispatchElemSize);
  const size_t dispatchXferBytes = config.XferBytesPerToken(dispatchElemSize);
  const uint8_t* token = base + static_cast<size_t>(tokenId) * dispatchXferBytes;
  return reinterpret_cast<const float*>(token + dispatchHiddenBytes + indexBytes)[expertSlot];
}

template <typename T>
inline __device__ void ScatterNode(EpDispatchCombineArgs<T>& args, int node, int nodeWarpId,
                                   int nodeWarpNum) {
  DEF_COMMON_VARS;
  const uint64_t tokenCount = WaitNodeInput(args, node, nodeWarpId);
  if (laneId == 0) args.blockFlagCounter[node] = static_cast<index_t>(tokenCount);
  const uint8_t* sourceBase = NodeInputBase(args, node);
  const size_t routeCount = static_cast<size_t>(tokenCount) * config.numExpertPerToken;

  for (size_t route = nodeWarpId; route < routeCount; route += nodeWarpNum) {
    const int tokenId = route / config.numExpertPerToken;
    const int expertSlot = route % config.numExpertPerToken;
    ScatterOneRoute(args, sourceBase + static_cast<size_t>(tokenId) * xferBytes, node, tokenId,
                    expertSlot);
  }
}

template <typename T>
inline __device__ void ScatterNodeTokenMajor(EpDispatchCombineArgs<T>& args, int node,
                                             int nodeWarpId, int nodeWarpNum) {
  DEF_COMMON_VARS;
  const uint64_t tokenCount = WaitNodeInput(args, node, nodeWarpId);
  if (laneId == 0) args.blockFlagCounter[node] = static_cast<index_t>(tokenCount);
  const uint8_t* sourceBase = NodeInputBase(args, node);
  const size_t routeCount = static_cast<size_t>(tokenCount) * config.numExpertPerToken;
  for (size_t route = nodeWarpId; route < routeCount; route += nodeWarpNum) {
    const int tokenId = route / config.numExpertPerToken;
    ScatterOneTokenMajorRoute(args, sourceBase + static_cast<size_t>(tokenId) * xferBytes, node,
                              tokenId, route % config.numExpertPerToken);
  }
}

template <typename T, bool TokenMajor = false>
inline __device__ void DispatchSyncCta(EpDispatchCombineArgs<T>& args, int expectedBlocks) {
  DEF_COMMON_VARS;
  __shared__ int isLastBlock;
  // Every lane participates in expert-major X copies. A lane-0-only fence does not publish
  // writes issued by the other lanes before the CTA completion counter becomes visible.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    isLastBlock =
        (atomicAdd(args.dispatchGridBarrier + 1, 1u) + 1u == static_cast<uint32_t>(expectedBlocks));
  }
  __syncthreads();
  if (!isLastBlock || warpId != 0) return;

  const int nodePeOffset = myNode * config.gpuPerNode;
  if (UseV2LLEpochSignals(config)) {
    const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag) + 1;
    if (laneId < config.gpuPerNode) {
      const int destPe = nodePeOffset + laneId;
      const uint32_t countPlusOne =
          static_cast<uint32_t>(
              core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe)) +
          1;
      uint64_t* signal = args.recvTokenNumMemObj->template GetAs<uint64_t*>(destPe) + myPe;
      core::AtomicStoreSeqCstSystem(signal,
                                    EncodeReadySignal(config, generation, countPlusOne));
      core::AtomicStoreRelaxed(args.destPeTokenCounter + destPe, index_t{0});
    }
    __syncwarp();

    const uint64_t* recvTokenNums =
        args.recvTokenNumMemObj->template GetAs<const uint64_t*>();
    for (int srcPe = nodePeOffset + laneId; srcPe < nodePeOffset + config.gpuPerNode;
         srcPe += warpSize) {
      uint64_t ready = 0;
      while (ReadyGeneration(config, ready) < generation)
        ready = core::AtomicLoadRelaxedSystem(recvTokenNums + srcPe);
      assert(ReadyGeneration(config, ready) == generation);
      atomicAdd(args.totalRecvTokenNum,
                static_cast<index_t>(ReadyCountPlusOne(config, ready) - 1));
    }
  } else {
    if (laneId < config.gpuPerNode) {
      const int destPe = nodePeOffset + laneId;
      const index_t count = core::AtomicLoadSeqCstSystem(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      core::AtomicStoreSeqCstSystem(signal, count);
      core::AtomicStoreSeqCstSystem(args.destPeTokenCounter + destPe, index_t{0});
    }
    __syncwarp();

    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    for (int srcPe = nodePeOffset + laneId; srcPe < nodePeOffset + config.gpuPerNode;
         srcPe += warpSize) {
      index_t* signal = recvTokenNums + srcPe;
      const index_t recvRoutes = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      atomicAdd(args.totalRecvTokenNum, recvRoutes);
      core::AtomicStoreSeqCstSystem(signal, 0);
    }
  }
  __syncwarp();
  if constexpr (TokenMajor) {
    // Peer slot allocation uses the SymmMemObj's per-PE P2P address, while the public tensor view
    // uses the allocation's host-side local address.  These are normally aliases, but static-heap
    // registration on a nonzero node must not rely on the GPU metadata's localPtr alias. Publish
    // the already validated per-source total through the explicit raw local address carried in
    // the args ABI.
    if (laneId == 0) {
      const index_t count = core::AtomicLoadSeqCst(args.totalRecvTokenNum);
      core::AtomicStoreSeqCstSystem(args.srcPeTokenIdxMap, count);
    }
  }
  __syncwarp();
  if (laneId < config.numQpPerPe)
    core::AtomicStoreRelaxed(args.combineGridBarrier + 4 + laneId, 0u);
  __syncwarp();
  if (laneId == 0) {
    core::AtomicStoreRelaxed(args.dispatchGridBarrier + 1, 0u);
    atomicAdd(args.crossDeviceBarrierFlag, 1);
  }
}

// Finalize only needs one CTA to observe that every producer CTA has completed before clearing
// shared counters. Releasing all CTAs through a reusable generation barrier adds a second global
// polling wave even though nonzero CTAs immediately exit. Collapse this edge to one arrival per
// CTA and a block-zero-only waiter.
inline __device__ bool CompleteGridForBlockZero(uint32_t* counter, int blockNum) {
  // The block-zero epilogue does not consume producer output; it only resets
  // control state after every CTA has arrived. combineOut is exposed after
  // kernel completion on the caller's stream, which is also the reuse edge for
  // the next generation, so no producer-side memory fence is required here.
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(counter, 1u);
  if (blockIdx.x != 0) return false;
  if (threadIdx.x == 0) {
    while (core::AtomicLoadRelaxed(counter) < static_cast<uint32_t>(blockNum)) {
    }
    core::AtomicStoreRelaxed(counter, 0u);
  }
  __syncthreads();
  return true;
}

template <typename T>
inline __device__ void WaitForLocalGroupGemmAllToAll(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  uint32_t* ready = args.combineGridBarrier + 2;
  const int nodePeOffset = myNode * config.gpuPerNode;

  if (globalWarpId == 0) {
    __threadfence_system();
    // Publish this rank's stream-ready generation directly to every local peer. This replaces
    // the old rank->coordinator->rank two-hop release with one parallel XGMI hop.
    if (laneId < config.gpuPerNode) {
      const int destPe = nodePeOffset + laneId;
      core::AtomicStoreRelaxedSystem(
          args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(destPe) + myPe,
          generation);
    }
    __syncwarp();

    const uint64_t* localFlags =
        args.crossDeviceBarrierMemObj->template GetAs<const uint64_t*>();
    if (laneId < config.gpuPerNode) {
      const int sourcePe = nodePeOffset + laneId;
      while (core::AtomicLoadRelaxedSystem(localFlags + sourcePe) < generation) {
      }
    }
    __syncwarp();
    if (laneId == 0) core::AtomicStoreRelaxed(ready, static_cast<uint32_t>(generation));
  } else if (laneId == 0) {
    while (core::AtomicLoadRelaxed(ready) != static_cast<uint32_t>(generation)) {
    }
  }
  __syncwarp();
}

template <typename T, int AccumUnroll = 2, int VecBytes = 8>
inline __device__ void ComputeNodePartialGroup(EpDispatchCombineArgs<T>& args, int sourceNode,
                                               int groupBlockId, int groupBlockNum) {
  DEF_COMMON_VARS;
  const int sharedSlots = max(nNodes, config.numExpertPerToken);
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * sharedSlots;
  float* routeWeights =
      reinterpret_cast<float*>(reinterpret_cast<T**>(sharedMem) + warpNum * sharedSlots) +
      warpId * sharedSlots;
  T* staging = args.interNodeV1TokBufs.staging->template GetAs<T*>();
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);

  const int tokenCount = core::AtomicLoadRelaxed(args.blockFlagCounter + sourceNode);
  // Allocate the whole resident grid from the actual source-node token count.  Using configured
  // capacity (128) left half of the warps idle at the common 64-token point and reduced a 1-token
  // partial to a single warp, despite hundreds of resident warps being available.
  const int workTokens = max(tokenCount, 1);
  const int groupWarpId = groupBlockId * warpNum + warpId;
  const int groupWarpNum = groupBlockNum * warpNum;
  const int warpsPerToken = max(groupWarpNum / workTokens, 1);
  const size_t hiddenPerWarp = core::CeilDiv(hiddenDim, static_cast<size_t>(warpsPerToken));
  for (int work = groupWarpId; work < tokenCount * warpsPerToken; work += groupWarpNum) {
    const int tokenId = work / warpsPerToken;
    const int tokenPart = work % warpsPerToken;
    const size_t hiddenOffset = static_cast<size_t>(tokenPart) * hiddenPerWarp;
    const size_t hiddenSize =
        (hiddenOffset < hiddenDim) ? min(hiddenDim - hiddenOffset, hiddenPerWarp) : 0;
    if (laneId < config.numExpertPerToken) {
      srcPtrs[laneId] = nullptr;
      routeWeights[laneId] = 0.0f;
      const index_t packed = args.interNodeDispDestTokIdMap[
          V2RouteMapOffset(config, sourceNode, tokenId, laneId)];
      if (packed != NullPackedExpertSlot(config)) {
        const int destPe = PeFromPackedExpertSlot(config, packed);
        const int linearSlot = LocalSlotFromPackedExpertSlot(config, packed);
        srcPtrs[laneId] =
            args.interNodeV1TokBufs.combineInp->template GetAs<T*>(destPe) +
            static_cast<size_t>(linearSlot) * hiddenDim + hiddenOffset;
        routeWeights[laneId] =
            CombineRouteWeight(args, sourceNode, generation, tokenId, laneId);
      }
    }
    __syncwarp();

    T* partial = nullptr;
    if (sourceNode == myNode) {
      // The local node partial is already part of this rank's final output. Write it directly to
      // combineOut so finalize only has to add the remote node partial in place.
      partial = args.interNodeV1TokBufs.combineOut->template GetAs<T*>() +
                static_cast<size_t>(tokenId) * hiddenDim + hiddenOffset;
    } else {
      partial = staging +
                CombineNodeSlotOffsetForGeneration(config, generation, nNodes + sourceNode,
                                                   tokenId) +
                hiddenOffset;
    }
#define V2_WARP_ACCUM_CASE(AccumNum)                                                   \
  case AccumNum:                                                                      \
    core::WarpAccum<T, VecBytes, AccumNum, AccumUnroll>(partial, srcPtrs, routeWeights, \
                                                  hiddenSize);                         \
    break
      switch (config.numExpertPerToken) {
        V2_WARP_ACCUM_CASE(1);
        V2_WARP_ACCUM_CASE(2);
        V2_WARP_ACCUM_CASE(4);
        V2_WARP_ACCUM_CASE(6);
        V2_WARP_ACCUM_CASE(8);
        default:
          core::WarpAccum<T, 8>(partial, srcPtrs, routeWeights, config.numExpertPerToken,
                                hiddenSize);
          break;
      }
#undef V2_WARP_ACCUM_CASE
  }
}

template <typename T, int AccumUnroll = 2, int VecBytes = 8>
inline __device__ void ComputeNodePartial(EpDispatchCombineArgs<T>& args, int sourceNode) {
  ComputeNodePartialGroup<T, AccumUnroll, VecBytes>(args, sourceNode, blockIdx.x, gridDim.x);
}

template <typename T, bool TokenMajor, int AccumUnroll = 2,
          int VecBytes = 8>
inline __device__ void ComputeNodePartialQpGroup(
    EpDispatchCombineArgs<T>& args, int sourceNode, int qpId,
    int groupBlockId, int groupBlockNum) {
  DEF_COMMON_VARS;
  const int sharedSlots = max(nNodes, config.numExpertPerToken);
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * sharedSlots;
  float* routeWeights = reinterpret_cast<float*>(
      reinterpret_cast<T**>(sharedMem) + warpNum * sharedSlots) +
      warpId * sharedSlots;
  T* staging = args.interNodeV1TokBufs.staging->template GetAs<T*>();
  const uint64_t generation =
      core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  const int sourceTokenCount =
      core::AtomicLoadRelaxed(args.blockFlagCounter + sourceNode);
  int tokenBegin = 0;
  int tokenCount = 0;
  TokenQpSlice(sourceTokenCount, config.numQpPerPe, qpId, tokenBegin,
               tokenCount);
  const int groupWarpId = groupBlockId * warpNum + warpId;
  const int groupWarpNum = groupBlockNum * warpNum;
  const int warpsPerToken =
      max(groupWarpNum / max(tokenCount, 1), 1);
  const size_t hiddenPerWarp =
      core::CeilDiv(hiddenDim, static_cast<size_t>(warpsPerToken));

  for (int work = groupWarpId; work < tokenCount * warpsPerToken;
       work += groupWarpNum) {
    const int tokenOffset = work / warpsPerToken;
    const int tokenId = tokenBegin + tokenOffset;
    const int tokenPart = work % warpsPerToken;
    const size_t hiddenOffset =
        static_cast<size_t>(tokenPart) * hiddenPerWarp;
    const size_t hiddenSize =
        hiddenOffset < hiddenDim
            ? min(hiddenDim - hiddenOffset, hiddenPerWarp)
            : 0;
    T* routeSrc = nullptr;
    if (laneId < config.numExpertPerToken) {
      const index_t packed = args.interNodeDispDestTokIdMap[
          V2RouteMapOffset(config, sourceNode, tokenId, laneId)];
      if (packed != NullPackedExpertSlot(config)) {
        const int destPe = PeFromPackedExpertSlot(config, packed);
        const int linearSlot = LocalSlotFromPackedExpertSlot(config, packed);
        routeSrc =
            args.interNodeV1TokBufs.combineInp->template GetAs<T*>(destPe) +
            static_cast<size_t>(linearSlot) * hiddenDim + hiddenOffset;
      }
      srcPtrs[laneId] = routeSrc;
      if constexpr (!TokenMajor) {
        routeWeights[laneId] =
            routeSrc == nullptr
                ? 0.0f
                : CombineRouteWeight(args, sourceNode, generation, tokenId,
                                     laneId);
      }
    }
    __syncwarp();

    T* partial =
        staging + CombineNodeSlotOffsetForGeneration(
                      config, generation, nNodes + sourceNode, tokenId) +
        hiddenOffset;
#define V2_QP_ACCUM_CASE(AccumNum)                                             \
  case AccumNum:                                                              \
    core::WarpAccum<T, VecBytes, AccumNum, AccumUnroll>(                      \
        partial, srcPtrs, TokenMajor ? nullptr : routeWeights, hiddenSize);   \
    break
    switch (config.numExpertPerToken) {
      V2_QP_ACCUM_CASE(1);
      V2_QP_ACCUM_CASE(2);
      V2_QP_ACCUM_CASE(4);
      V2_QP_ACCUM_CASE(6);
      V2_QP_ACCUM_CASE(8);
      default:
        core::WarpAccum<T, VecBytes>(
            partial, srcPtrs, TokenMajor ? nullptr : routeWeights,
            config.numExpertPerToken, hiddenSize);
        break;
    }
#undef V2_QP_ACCUM_CASE
  }
}

template <typename T, int AccumUnroll = 2, int VecBytes = 8>
inline __device__ void ComputeNodePartialTokenMajor(EpDispatchCombineArgs<T>& args,
                                                    int sourceNode) {
  DEF_COMMON_VARS;
  const int sharedSlots = max(nNodes, config.numExpertPerToken);
  extern __shared__ char sharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(sharedMem) + warpId * sharedSlots;
  T* staging = args.interNodeV1TokBufs.staging->template GetAs<T*>();
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  const int tokenCount = core::AtomicLoadRelaxed(args.blockFlagCounter + sourceNode);
  const int workTokens = max(tokenCount, 1);
  const int warpsPerToken = max(globalWarpNum / workTokens, 1);
  const size_t hiddenPerWarp = core::CeilDiv(hiddenDim, static_cast<size_t>(warpsPerToken));

  for (int work = globalWarpId; work < tokenCount * warpsPerToken; work += globalWarpNum) {
    const int tokenId = work / warpsPerToken;
    const int tokenPart = work % warpsPerToken;
    const size_t hiddenOffset = static_cast<size_t>(tokenPart) * hiddenPerWarp;
    const size_t hiddenSize =
        hiddenOffset < hiddenDim ? min(hiddenDim - hiddenOffset, hiddenPerWarp) : 0;
    T* routeSrc = nullptr;
    if (laneId < config.numExpertPerToken) {
      const index_t packed = args.interNodeDispDestTokIdMap[
          V2RouteMapOffset(config, sourceNode, tokenId, laneId)];
      if (packed != NullPackedExpertSlot(config)) {
        const int destPe = PeFromPackedExpertSlot(config, packed);
        const int tokenSlot = LocalSlotFromPackedExpertSlot(config, packed);
        routeSrc =
            args.interNodeV1TokBufs.combineInp->template GetAs<T*>(destPe) +
            static_cast<size_t>(tokenSlot) * hiddenDim + hiddenOffset;
      }
    }
    const unsigned long long validMask = __ballot(routeSrc != nullptr);
    const int activeAccumNum = __popcll(validMask);
    if (routeSrc != nullptr) {
      const unsigned long long lowerLanes =
          (1ull << static_cast<unsigned>(laneId)) - 1ull;
      const int compactSlot = __popcll(validMask & lowerLanes);
      srcPtrs[compactSlot] = routeSrc;
    }
    __syncwarp();

    T* partial = nullptr;
    if (sourceNode == myNode) {
      partial = args.interNodeV1TokBufs.combineOut->template GetAs<T*>() +
                static_cast<size_t>(tokenId) * hiddenDim + hiddenOffset;
    } else {
      partial = staging +
                CombineNodeSlotOffsetForGeneration(config, generation, nNodes + sourceNode,
                                                   tokenId) +
                hiddenOffset;
    }
    // Token-major preprocessing has already reduced and weighted every expert hosted by one
    // destination PE. Duplicate-PE routes are null in the route map, so compact the live rows.
    // Keep the worker's requested unroll: the generic dynamic helper hard-codes unroll=2 and
    // regresses the production V16/U1 path when all eight destination PEs are active.
#define V2_TOKEN_ACTIVE_ACCUM_CASE(AccumNum)                                  \
  case AccumNum:                                                              \
    core::WarpAccum<T, VecBytes, AccumNum, AccumUnroll>(                      \
        partial, srcPtrs, nullptr, hiddenSize);                               \
    break
    switch (activeAccumNum) {
      V2_TOKEN_ACTIVE_ACCUM_CASE(1);
      V2_TOKEN_ACTIVE_ACCUM_CASE(2);
      V2_TOKEN_ACTIVE_ACCUM_CASE(4);
      V2_TOKEN_ACTIVE_ACCUM_CASE(6);
      V2_TOKEN_ACTIVE_ACCUM_CASE(8);
      default:
        core::WarpAccumDynamic<T, VecBytes>(
            partial, srcPtrs, nullptr, activeAccumNum, hiddenSize);
        break;
    }
#undef V2_TOKEN_ACTIVE_ACCUM_CASE
  }
}

template <typename T>
inline __device__ void CombineOneShotSend(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  if (blockId != 0 || warpId >= config.numQpPerPe || laneId >= nNodes || laneId == myNode) return;

  const int sourceNode = laneId;
  const int qpId = warpId;
  const int proxyPe = sourceNode * config.gpuPerNode + (config.rank % config.gpuPerNode);
  const size_t tokenCount = core::AtomicLoadRelaxed(args.blockFlagCounter + sourceNode);
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  size_t localOffset =
      CombineNodeSlotOffsetForGeneration(config, generation, nNodes + sourceNode, 0) * sizeof(T);
  size_t remoteOffset =
      CombineNodeSlotOffsetForGeneration(config, generation, myNode, 0) * sizeof(T);
  const size_t totalBytes = tokenCount * hiddenBytes;
  size_t byteOffset = 0;
  size_t bytes = 0;
  AlignedQpSlice(totalBytes, config.numQpPerPe, qpId, byteOffset, bytes);
  localOffset += byteOffset;
  remoteOffset += byteOffset;
  const uint64_t signal =
      EncodeReadySignal(config, generation, static_cast<uint32_t>(tokenCount) + 1);
  const size_t signalOffset = CombineSignalSlot(config, myNode, qpId) * sizeof(uint64_t);

  if (bytes > 0) {
    shmem::ShmemPutMemNbiSignalThread(
        args.interNodeV1TokBufs.staging, remoteOffset,
        args.interNodeV1TokBufs.staging, localOffset, bytes,
        args.nodeRecvTokenNumMemObj, signalOffset, signal,
        core::atomicType::AMO_SET, proxyPe, qpId);
  } else {
    shmem::ShmemPutTypeImmNbiThread<uint64_t>(args.nodeRecvTokenNumMemObj, signalOffset, signal,
                                              proxyPe, qpId);
  }
}

template <typename T>
inline __device__ void CombineOneShotSendSourceQp(
    EpDispatchCombineArgs<T>& args, int sourceNode, int qpId) {
  DEF_COMMON_VARS;
  if (qpId >= config.numQpPerPe || sourceNode == myNode) return;

  const int proxyPe = sourceNode * config.gpuPerNode +
                      (config.rank % config.gpuPerNode);
  const int tokenCount =
      core::AtomicLoadRelaxed(args.blockFlagCounter + sourceNode);
  const uint64_t generation =
      core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  int tokenBegin = 0;
  int qpTokenCount = 0;
  TokenQpSlice(tokenCount, config.numQpPerPe, qpId, tokenBegin,
               qpTokenCount);
  const size_t byteOffset =
      static_cast<size_t>(tokenBegin) * hiddenBytes;
  const size_t bytes =
      static_cast<size_t>(qpTokenCount) * hiddenBytes;
  const size_t localOffset =
      CombineNodeSlotOffsetForGeneration(config, generation,
                                         nNodes + sourceNode, 0) *
          sizeof(T) +
      byteOffset;
  const size_t remoteOffset =
      CombineNodeSlotOffsetForGeneration(config, generation, myNode, 0) *
          sizeof(T) +
      byteOffset;
  const uint64_t signal = EncodeReadySignal(
      config, generation, static_cast<uint32_t>(tokenCount) + 1);
  const size_t signalOffset =
      CombineSignalSlot(config, myNode, qpId) * sizeof(uint64_t);
  if (bytes > 0) {
    shmem::ShmemPutMemNbiSignalThread(
        args.interNodeV1TokBufs.staging, remoteOffset,
        args.interNodeV1TokBufs.staging, localOffset, bytes,
        args.nodeRecvTokenNumMemObj, signalOffset, signal,
        core::atomicType::AMO_SET, proxyPe, qpId);
  } else {
    shmem::ShmemPutTypeImmNbiThread<uint64_t>(
        args.nodeRecvTokenNumMemObj, signalOffset, signal, proxyPe, qpId);
  }
}

template <typename T>
inline __device__ void CombineSendQuiet(EpDispatchCombineArgs<T>& args) {
  // The attached remote signal is the completion edge consumed by the destination. A blocking
  // local CQ drain here serializes the inverse XGMI/RDMA pipeline and is not required before the
  // source staging slot is reused (the next combine has a fresh local-ready generation).
}

template <typename T>
inline __device__ void WaitRemoteNodePartial(EpDispatchCombineArgs<T>& args,
                                              int node) {
  DEF_COMMON_VARS;
  if (node == myNode) return;
  for (int qpId = laneId; qpId < config.numQpPerPe; qpId += warpSize) {
    uint64_t* signals =
        args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>();
    const size_t signalSlot = CombineSignalSlot(config, node, qpId);
    const uint64_t expectedGeneration =
        UseV2LLEpochSignals(config)
            ? core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag)
            : 1;
    uint64_t ready = 0;
    while (ReadyGeneration(config, ready) < expectedGeneration)
      ready = core::AtomicLoadRelaxedSystem(signals + signalSlot);
    assert(ReadyGeneration(config, ready) == expectedGeneration);
    assert(ReadyCountPlusOne(config, ready) - 1 ==
           static_cast<uint64_t>(args.curRankNumToken));
  }
  __syncwarp();
}

template <typename T>
inline __device__ void WaitRemoteNodePartials(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  for (int step = 1; step < nNodes; ++step)
    WaitRemoteNodePartial(args, (myNode + step) % nNodes);
}

template <typename T>
inline __device__ void FinalizeNodePartials(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;
  T* staging = args.interNodeV1TokBufs.staging->template GetAs<T*>();
  const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
  const int outputTokens = max(static_cast<int>(args.curRankNumToken), 1);
  const int warpsPerToken = max(globalWarpNum / outputTokens, 1);
  const size_t hiddenPerWarp =
      core::CeilDiv(hiddenDim, static_cast<size_t>(warpsPerToken));

  for (int work = globalWarpId;
       work < args.curRankNumToken * warpsPerToken;
       work += globalWarpNum) {
    const int tokenId = work / warpsPerToken;
    const int tokenPart = work % warpsPerToken;
    const size_t hiddenOffset =
        static_cast<size_t>(tokenPart) * hiddenPerWarp;
    const size_t hiddenSize = hiddenOffset < hiddenDim
                                  ? min(hiddenDim - hiddenOffset, hiddenPerWarp)
                                  : 0;
    T* output = args.interNodeV1TokBufs.combineOut->template GetAs<T*>() +
                static_cast<size_t>(tokenId) * hiddenDim + hiddenOffset;
    // Contributor pointers are uniform across the warp. Compute them directly
    // in ring order instead of staging them through shared memory for every
    // output slice. The accumulation order and generic N-node behavior remain
    // unchanged.
    for (int step = 1; step < nNodes; ++step) {
      const int contributorNode = (myNode + step) % nNodes;
      T* remote = staging +
                  CombineNodeSlotOffsetForGeneration(
                      config, generation, contributorNode, tokenId) +
                  hiddenOffset;
      core::WarpAccum(output, remote, hiddenSize);
    }
  }
}

template <typename T>
inline __device__ void ResetAfterCombineGroup(EpDispatchCombineArgs<T>& args,
                                               int groupThreadId,
                                               int groupThreadNum) {
  DEF_COMMON_VARS;
  index_t* expertCounts =
      args.dispTokOffsetMemObj->template GetAs<index_t*>(myPe);
  for (int expert = groupThreadId; expert < config.numExpertPerRank;
       expert += groupThreadNum) {
    core::AtomicStoreSeqCstSystem(expertCounts + expert, index_t{0});
  }
  if (groupThreadId == 0 && args.srcPeTokenIdxMap != expertCounts)
    core::AtomicStoreSeqCstSystem(args.srcPeTokenIdxMap, index_t{0});
  for (int node = groupThreadId; node < nNodes; node += groupThreadNum)
    core::AtomicStoreRelaxed(args.blockFlagCounter + node, index_t{0});
  for (int control = kQpEarlyCounterBaseSlot + groupThreadId;
       control < QpEarlyControlWords(config); control += groupThreadNum)
    core::AtomicStoreRelaxed(args.combineGridBarrier + control, uint32_t{0});
  if (!UseV2LLEpochSignals(config)) {
    for (int signal = groupThreadId; signal < nNodes * config.numQpPerPe;
         signal += groupThreadNum) {
      core::AtomicStoreSeqCstSystem(
          args.nodeRecvTokenNumMemObj->template GetAs<uint64_t*>() + signal, uint64_t{0});
    }
  }
  if (groupThreadId == 0)
    core::AtomicStoreRelaxed(args.totalRecvTokenNum, index_t{0});
}

template <typename T>
inline __device__ void ResetAfterCombine(EpDispatchCombineArgs<T>& args) {
  ResetAfterCombineGroup(args, blockIdx.x * blockDim.x + threadIdx.x,
                         gridDim.x * blockDim.x);
}

template <typename T>
inline __device__ void ResetAfterCombineBlock(EpDispatchCombineArgs<T>& args) {
  ResetAfterCombineGroup(args, threadIdx.x, blockDim.x);
}

}  // namespace v2

template <typename T, bool TokenMajor, bool MultiWarpCopy = false>
inline __device__ void EpDispatchInterNodeV2LLKernelImpl(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  const int copyBlockNum = args.dispatchCopyBlockNum;
  const int scatterBlockNum = blockNum - copyBlockNum;
  assert(copyBlockNum > 0 && scatterBlockNum >= nNodes);

  if (blockId < copyBlockNum) {
    int qpId = 0;
    int qpBlockNum = copyBlockNum;
    if constexpr (TokenMajor) {
      qpId = blockId % config.numQpPerPe;
      const int qpBlockId = blockId / config.numQpPerPe;
      qpBlockNum = core::CeilDiv(copyBlockNum - qpId,
                                static_cast<int>(config.numQpPerPe));
      if constexpr (MultiWarpCopy)
        v2::CopyToStagingQpGroupMultiWarp(
            args, qpId, qpBlockId, qpBlockNum);
      else
        v2::CopyToStagingQpGroup(args, qpId, qpBlockId, qpBlockNum);
    } else {
      if constexpr (MultiWarpCopy)
        v2::CopyToStagingGroupMultiWarp(args, blockId, copyBlockNum);
      else
        v2::CopyToStagingGroup(args, blockId, copyBlockNum);
    }
    // All vector-copy lanes must publish staging before the last producer posts RDMA.
    __threadfence_system();
    __syncthreads();
    __shared__ int lastCopyBlock;
    if (threadIdx.x == 0) {
      uint32_t* copyCounter = args.dispatchGridBarrier;
      if constexpr (TokenMajor) copyCounter += v2::kQpCounterBaseSlot + qpId;
      lastCopyBlock =
          (atomicAdd(copyCounter, 1u) + 1u == static_cast<uint32_t>(qpBlockNum));
      if (lastCopyBlock && TokenMajor) core::AtomicStoreRelaxed(copyCounter, 0u);
    }
    __syncthreads();
    if (lastCopyBlock) {
      if constexpr (TokenMajor)
        v2::DispatchOneShotSendTokenQpFromCurrentBlock(args, qpId);
      else
        v2::DispatchOneShotSendFromCurrentBlock(args);
      if constexpr (!TokenMajor)
        if (threadIdx.x == 0)
          core::AtomicStoreRelaxed(args.dispatchGridBarrier, 0u);
    }
    return;
  }

  const int scatterBlockId = blockId - copyBlockNum;
  v2::WaitPreviousLocalReset(args, scatterBlockId);
  const int receiverWarpId = scatterBlockId * warpNum + warpId;
  const int receiverWarpNum = scatterBlockNum * warpNum;
  for (int step = 0; step < nNodes; ++step) {
    const int node = (myNode + step) % nNodes;
    if constexpr (TokenMajor) {
      if (node == myNode)
        v2::ScatterLocalTokenMajorGroup(args, receiverWarpId, receiverWarpNum);
      else
        v2::ScatterNodeTokenMajor(args, node, receiverWarpId, receiverWarpNum);
    } else {
      if (node == myNode)
        v2::ScatterLocalGroup(args, receiverWarpId, receiverWarpNum);
      else
        v2::ScatterNode(args, node, receiverWarpId, receiverWarpNum);
    }
  }
  v2::DispatchSyncCta<T, TokenMajor>(args, scatterBlockNum);
}

template <typename T, bool RingReceiver>
__device__ void EpDispatchInterNodeV2LLKernel_body(EpDispatchCombineArgs<T> args) {
  static_assert(RingReceiver, "production V2LL uses the generic ring receiver");
  EpDispatchInterNodeV2LLKernelImpl<T, false>(args);
}

template <typename T, bool RingReceiver>
__device__ void EpDispatchInterNodeV2LLTokenMajor_body(EpDispatchCombineArgs<T> args) {
  static_assert(RingReceiver, "production V2LL uses the generic ring receiver");
  EpDispatchInterNodeV2LLKernelImpl<T, true>(args);
}

template <typename T, bool RingReceiver>
__device__ void EpDispatchInterNodeV2LLMultiWarpCopy_body(
    EpDispatchCombineArgs<T> args) {
  static_assert(RingReceiver, "production V2LL uses the generic ring receiver");
  EpDispatchInterNodeV2LLKernelImpl<T, false, true>(args);
}

template <typename T, bool RingReceiver>
__device__ void EpDispatchInterNodeV2LLTokenMajorMultiWarpCopy_body(
    EpDispatchCombineArgs<T> args) {
  static_assert(RingReceiver, "production V2LL uses the generic ring receiver");
  EpDispatchInterNodeV2LLKernelImpl<T, true, true>(args);
}

template <typename T, int AccumUnroll, int VecBytes = 8>
inline __device__ void EpCombineInterNodeV2LLRemotePartialImpl(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v2::WaitForLocalGroupGemmAllToAll(args);
  for (int step = 1; step < nNodes; ++step) {
    v2::ComputeNodePartial<T, AccumUnroll, VecBytes>(args, (myNode + step) % nNodes);
  }
}

template <typename T>
__device__ void EpCombineInterNodeV2LLRemotePartialV16_body(EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLRemotePartialImpl<T, 1, 16>(args);
}

template <typename T, int AccumUnroll, int VecBytes = 8>
inline __device__ void EpCombineInterNodeV2LLRemotePartialTokenMajorImpl(
    EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v2::WaitForLocalGroupGemmAllToAll(args);
  for (int step = 1; step < nNodes; ++step)
    v2::ComputeNodePartialTokenMajor<T, AccumUnroll, VecBytes>(
        args, (myNode + step) % nNodes);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLRemotePartialTokenMajorV16_body(
    EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLRemotePartialTokenMajorImpl<T, 1, 16>(args);
}

template <typename T, bool TokenMajor>
inline __device__ void EpCombineInterNodeV2LLRemotePartialQpEarlyImpl(
    EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v2::WaitForLocalGroupGemmAllToAll(args);

  const int qpId = blockId % config.numQpPerPe;
  const int qpBlockId = blockId / config.numQpPerPe;
  const int qpBlockNum = core::CeilDiv(
      blockNum - qpId, static_cast<int>(config.numQpPerPe));
  __shared__ int lastQpProducer;
#pragma clang loop unroll(disable)
  for (int step = 1; step < nNodes; ++step) {
    const int sourceNode = (myNode + step) % nNodes;
    v2::ComputeNodePartialQpGroup<T, TokenMajor, 1, 16>(
        args, sourceNode, qpId, qpBlockId, qpBlockNum);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t* counter = v2::QpEarlyCounter(
          config, args.combineGridBarrier, sourceNode, qpId);
      lastQpProducer =
          atomicAdd(counter, 1u) + 1u ==
          static_cast<uint32_t>(qpBlockNum);
      if (lastQpProducer) core::AtomicStoreRelaxed(counter, 0u);
    }
    __syncthreads();
    // Send from the last producer on a separate QP stripe immediately after
    // this source node's XGMI partial is complete. Other CTAs can start the
    // next ring source while the NIC progresses independently.
    if (lastQpProducer && threadIdx.x == 0)
      v2::CombineOneShotSendSourceQp(args, sourceNode, qpId);
  }
}

template <typename T>
__device__ void EpCombineInterNodeV2LLRemotePartialQpEarlyV16_body(
    EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLRemotePartialQpEarlyImpl<T, false>(args);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLRemotePartialTokenMajorQpEarlyV16_body(
    EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLRemotePartialQpEarlyImpl<T, true>(args);
}

template <typename T, int AccumUnroll, int VecBytes = 8>
inline __device__ void EpCombineInterNodeV2LLSendLocalImpl(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v2::CombineOneShotSend(args);
  // NIC transfer progresses independently while the full compute grid builds the local partial.
  v2::ComputeNodePartial<T, AccumUnroll, VecBytes>(args, myNode);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLSendLocalV16_body(EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLSendLocalImpl<T, 1, 16>(args);
}

template <typename T, int AccumUnroll, int VecBytes = 8>
inline __device__ void EpCombineInterNodeV2LLSendLocalTokenMajorImpl(
    EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v2::CombineOneShotSend(args);
  v2::ComputeNodePartialTokenMajor<T, AccumUnroll, VecBytes>(args, myNode);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLSendLocalTokenMajorV16_body(
    EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLSendLocalTokenMajorImpl<T, 1, 16>(args);
}

template <typename T, bool TokenMajor>
inline __device__ void EpCombineInterNodeV2LLLocalQpEarlyImpl(
    EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  if constexpr (TokenMajor)
    v2::ComputeNodePartialTokenMajor<T, 1, 16>(args, myNode);
  else
    v2::ComputeNodePartial<T, 1, 16>(args, myNode);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLLocalQpEarlyV16_body(
    EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLLocalQpEarlyImpl<T, false>(args);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLLocalTokenMajorQpEarlyV16_body(
    EpDispatchCombineArgs<T> args) {
  EpCombineInterNodeV2LLLocalQpEarlyImpl<T, true>(args);
}

template <typename T>
__device__ void EpCombineInterNodeV2LLFinalize_body(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;
  v2::WaitRemoteNodePartials(args);
  v2::FinalizeNodePartials(args);
  // Reset consumes the expert-major buffers only after every finalize CTA is done. Keeping this
  // epilogue in the same launch removes the dedicated Reset kernel from the critical path.
  if (v2::CompleteGridForBlockZero(args.combineGridBarrier, blockNum)) {
    const uint64_t generation = core::AtomicLoadRelaxed(args.crossDeviceBarrierFlag);
    // One CTA resets the transient counters after all finalize CTAs have completed.
    v2::ResetAfterCombineBlock(args);
    v2::SyncLocalResetCompletion(args, generation);
    v2::CombineSendQuiet(args);
  }
}


}  // namespace moe
}  // namespace mori
