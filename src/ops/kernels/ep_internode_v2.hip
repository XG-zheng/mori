// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License

// Production V2LL: generic N-node ring Dispatch and multi-stage BF16 Combine.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

WRAP_BOOL(EpDispatchInterNodeV2LLKernel, bf16_ring, hip_bfloat16, true)
MORI_FP8_FNUZ(
    WRAP_BOOL(EpDispatchInterNodeV2LLKernel, fp8_fnuz_ring, __hip_fp8_e4m3_fnuz, true))
MORI_FP8_OCP(WRAP_BOOL(EpDispatchInterNodeV2LLKernel, fp8_ocp_ring, __hip_fp8_e4m3, true))

WRAP_BOOL(EpDispatchInterNodeV2LLTokenMajor, bf16_ring, hip_bfloat16, true)
MORI_FP8_FNUZ(WRAP_BOOL(EpDispatchInterNodeV2LLTokenMajor, fp8_fnuz_ring,
                        __hip_fp8_e4m3_fnuz, true))
MORI_FP8_OCP(
    WRAP_BOOL(EpDispatchInterNodeV2LLTokenMajor, fp8_ocp_ring, __hip_fp8_e4m3, true))

WRAP(EpCombineInterNodeV2LLRemotePartialV16, bf16, hip_bfloat16)
WRAP(EpCombineInterNodeV2LLRemotePartialTokenMajorV16, bf16, hip_bfloat16)
WRAP(EpCombineInterNodeV2LLSendLocalV16, bf16, hip_bfloat16)
WRAP(EpCombineInterNodeV2LLSendLocalTokenMajorV16, bf16, hip_bfloat16)
WRAP(EpCombineInterNodeV2LLFinalize, bf16, hip_bfloat16)
