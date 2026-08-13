# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from mori import cpp as mori_cpp
from mori.tensor_utils import from_gpu_ptr, dtype_to_int
import logging
import os
from dataclasses import dataclass
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

TOPK_IDX_DTYPE = torch.int32
WARP_SIZE = 64


class EpDispatchCombineKernelType(mori_cpp.EpDispatchCombineKernelType):
    def __str__(self):
        return self.name


class EpDispatchCombineQuantType(mori_cpp.EpDispatchCombineQuantType):
    def __str__(self):
        return self.name


_QUANT_TYPE_MAP = {
    "none": EpDispatchCombineQuantType.None_,
    "fp8_direct_cast": EpDispatchCombineQuantType.Fp8DirectCast,
    "fp8_blockwise": EpDispatchCombineQuantType.Fp8BlockwiseQuant,
    # Blockwise FP4 (E2M1) combine is its own quant type. It shares the blockwise staging/scale
    # layout with FP8 but transports packed FP4 (0.5 byte/elem) and uses half-sized staging slots.
    "fp4_blockwise": EpDispatchCombineQuantType.Fp4BlockwiseQuant,
}

# Blockwise combine quant types share the staging/scale layout and kernel launch config; only the
# element codec (and staging slot size) differ, so kernel selection treats them together and then
# swaps the codec token (fp8bwq <-> fp4bwq) in the kernel name.
_BLOCKWISE_COMBINE_QUANT_TYPES = (
    EpDispatchCombineQuantType.Fp8BlockwiseQuant,
    EpDispatchCombineQuantType.Fp4BlockwiseQuant,
)

# The FP4 blockwise combine kernels registered in ep_intranode.hip. Kernel-name selection derives
# an fp4bwq name from the fp8bwq one; the result is asserted against this set so a mismatch fails
# loudly instead of launching a non-existent symbol.
_FP4_COMBINE_KERNELS = frozenset(
    {
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4bwq",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4bwq_noweight_block128_vec8",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4bwq_noweight_block256_vec8",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4bwq_noweight_block128_vec8_top9",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp4bwq_noweight_block256_vec8_top9",
    }
)


def _normalize_quant_type(quant_type):
    if isinstance(quant_type, EpDispatchCombineQuantType):
        return quant_type
    if isinstance(quant_type, str):
        key = quant_type.strip().lower()
        if key in _QUANT_TYPE_MAP:
            return _QUANT_TYPE_MAP[key]
    raise ValueError(
        f"invalid quant_type '{quant_type}', expected one of {list(_QUANT_TYPE_MAP.keys())}"
    )


def _current_stream():
    return torch.cuda.current_stream().cuda_stream


@dataclass
class EpDispatchRoutingHandle:
    """Per-call routing snapshot from cache-routing dispatch, replayed by combine / replay-routing dispatch."""

    disp_dest_tok_id_map: torch.Tensor
    inter_node_disp_dest_tok_id_map: torch.Tensor
    inter_node_disp_send_map: torch.Tensor
    total_recv_token_num: torch.Tensor
    disp_tok_id_to_src_tok_id_local: torch.Tensor
    cur_rank_num_token: int = 0

    def tensors(self):
        return (
            self.disp_dest_tok_id_map,
            self.inter_node_disp_dest_tok_id_map,
            self.inter_node_disp_send_map,
            self.total_recv_token_num,
            self.disp_tok_id_to_src_tok_id_local,
        )

    @classmethod
    def from_tensors(cls, tensors, cur_rank_num_token=0):
        return cls(*tensors, cur_rank_num_token=cur_rank_num_token)


@dataclass
class EpDispatchCombineConfig:
    """Configuration for :class:`EpDispatchCombineOp`.

    Args:
        data_type: Deprecated. Tensor dtype kept only for backward
            compatibility with tests and examples. Kernel launch dtype is
            inferred from the runtime input tensor instead of this field.
        rank: Rank of the current process in the expert-parallel group.
        world_size: Total number of ranks participating in the dispatch/combine
            operation.
        hidden_dim: Hidden dimension of each token embedding.
        scale_dim: Number of scale values stored per token for quantized paths.
            Describes caller-provided dispatch scales (e.g. FP4 input).
            ``quant_type="fp8_blockwise"`` combine uses its own internal
            scale_dim driven by ``MORI_FP8_COMBINE_SCALE_DIM`` (default 56).
        scale_type_size: Size in bytes of each scale element.
        max_token_type_size: Maximum size in bytes for the token element type.
        max_num_inp_token_per_rank: Maximum number of input tokens each rank
            can process.
        num_experts_per_rank: Number of local experts hosted on each rank.
        num_experts_per_token: Number of experts selected for each token.
        warp_num_per_block: Number of warps per GPU block for the kernel launch.
        block_num: Number of GPU blocks to launch for the main kernel.
        max_total_recv_tokens: Optional cap used to derive the maximum number
            of tokens a rank can receive, which also affects memory
            consumption. A value of ``0`` disables the cap. If the actual
            received token count exceeds the derived limit, the kernel
            currently asserts.
        use_external_inp_buf: Whether the operator expects the input buffer to
            be managed externally.
        kernel_type: Dispatch/combine kernel implementation to use.
        gpu_per_node: Number of GPUs per node. This affects all kernel types.
        rdma_block_num: Number of RDMA blocks for inter-node kernels.
        num_qp_per_pe: Number of queue pairs per processing element. ``0`` selects
            the resolved default (2 for InterNodeV2LL, 1 otherwise). Explicit
            values from 1 through 8 are supported.
        v2_layout: Output layout for V2LL. ``"expert_major"`` writes the Group
            GEMM input grouped by local expert; ``"token_major"`` matches the
            V1LL token-major contract.
        quant_type: Quantization mode. Supported string values are ``"none"``,
            ``"fp8_direct_cast"``, and ``"fp8_blockwise"``.
        v2_copy_block_num: Number of fused Dispatch staging-copy CTAs. Zero
            derives a topology-independent default from the launch geometry.
    """

    data_type: (
        torch.dtype
    )  # Deprecated for kernel launch (runtime dtype inferred from input tensor); retained for test/example compatibility
    rank: int
    world_size: int
    hidden_dim: int
    scale_dim: int
    scale_type_size: int
    max_token_type_size: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    warp_num_per_block: int = 8
    block_num: int = 80
    max_total_recv_tokens: int = 0
    use_external_inp_buf: bool = True
    kernel_type: EpDispatchCombineKernelType = EpDispatchCombineKernelType.IntraNode
    gpu_per_node: int = 8
    rdma_block_num: int = 0
    num_qp_per_pe: int = 0
    quant_type: str = "none"
    v2_layout: str = "expert_major"
    v2_copy_block_num: int = 0


def _cpp_dispatch_combine_factory(entity_name, allow_missing=False):
    if allow_missing:
        return getattr(mori_cpp, entity_name, None)
    return getattr(mori_cpp, entity_name)


# ---------------------------------------------------------------------------
# Kernel type → .hsaco compilation unit mapping
# ---------------------------------------------------------------------------
_KERNEL_TYPE_TO_HIP = {
    EpDispatchCombineKernelType.IntraNode: "ep_intranode",
    EpDispatchCombineKernelType.IntraNodeLL: "ep_intranode",
    EpDispatchCombineKernelType.InterNode: "ep_internode",
    EpDispatchCombineKernelType.InterNodeV1: "ep_internode_v1",
    EpDispatchCombineKernelType.InterNodeV1LL: "ep_internode_v1ll",
    EpDispatchCombineKernelType.InterNodeV2LL: "ep_internode_v2",
    EpDispatchCombineKernelType.AsyncLL: "ep_async_ll",
}


def _is_v2_direct_type(kernel_type):
    return kernel_type.value == EpDispatchCombineKernelType.InterNodeV2LL.value


_V2LL_DEFAULT_LAUNCH_SCHEDULES = {
    # (device, world, GPUs/node, hidden, topk, dtype):
    # (max_tokens, dispatch_blocks, dispatch_warps, copy_blocks,
    #  combine_blocks, combine_warps)
    # These are production acceptance points for BF16 EP16 on MI355X/MI350X.
    # Explicit launch arguments and config copy-block overrides stay authoritative.
    ("mi355x", 16, 8, 7168, 8, torch.bfloat16): (
        (32, 128, 4, 32, 56, 8),
        (64, 160, 4, 32, 112, 4),
        (128, 160, 4, 32, 112, 4),
    ),
    ("mi350x", 16, 8, 7168, 8, torch.bfloat16): (
        (32, 128, 4, 32, 56, 8),
        (64, 160, 4, 32, 112, 4),
        (128, 160, 4, 32, 112, 4),
    ),
}


def _v2ll_default_launch(config, num_tokens, dtype, *, is_dispatch):
    """Return a validated production launch bucket, or ``None``."""
    if config.kernel_type != EpDispatchCombineKernelType.InterNodeV2LL:
        return None
    from mori.ops import utils as gpu_utils

    key = (
        gpu_utils.detect_model(),
        config.world_size,
        config.gpu_per_node,
        config.hidden_dim,
        config.num_experts_per_token,
        dtype,
    )
    schedule = _V2LL_DEFAULT_LAUNCH_SCHEDULES.get(key)
    if schedule is None:
        return None
    for (
        max_tokens,
        dispatch_blocks,
        dispatch_warps,
        copy_blocks,
        combine_blocks,
        combine_warps,
    ) in schedule:
        if num_tokens <= max_tokens:
            return (
                (dispatch_blocks, dispatch_warps, copy_blocks)
                if is_dispatch
                else (combine_blocks, combine_warps, 0)
            )
    return None


# dtype → kernel name suffix
_DTYPE_SUFFIX = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
}
try:
    _DTYPE_SUFFIX[torch.float8_e4m3fn] = "fp8_ocp"
except AttributeError:
    pass
try:
    _DTYPE_SUFFIX[torch.float8_e4m3fnuz] = "fp8_fnuz"
except AttributeError:
    pass
try:
    _DTYPE_SUFFIX[torch.float4_e2m1fn_x2] = "fp4"
except AttributeError:
    pass

# pointer size on device for shared memory calculation (sizeof(T**) and sizeof(float**))
_PTR_SIZE = 8


def warmup_jit_kernels(kernel_type):
    """Pre-compile kernels for a kernel_type. Call from main process before spawning workers."""
    from mori.ops._jit_loader import ensure_compiled, _compiled_hsaco

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    hip_name = _KERNEL_TYPE_TO_HIP[kernel_type]
    ensure_compiled(hip_name)
    return _compiled_hsaco.get(hip_name)


def _ensure_jit_kernels(kernel_type):
    """Ensure the required kernels for this kernel_type are JIT-compiled."""
    from mori.ops._jit_loader import ensure_compiled

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    ensure_compiled(_KERNEL_TYPE_TO_HIP[kernel_type])


def _load_hip_modules(kernel_type):
    """Load HipModule for the given kernel_type and init shmem gpu states."""
    from mori.ops._jit_loader import load_hip_module

    if kernel_type not in _KERNEL_TYPE_TO_HIP:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    return load_hip_module(_KERNEL_TYPE_TO_HIP[kernel_type], init_shmem=True)


class EpDispatchCombineOp:
    def __init__(self, config):
        self.config = config
        if not 0 <= config.num_qp_per_pe <= 8:
            raise ValueError(
                "num_qp_per_pe must be 0 (auto) or an explicit value in [1, 8], "
                f"got {config.num_qp_per_pe}"
            )
        if config.num_qp_per_pe == 0:
            config.num_qp_per_pe = 2 if _is_v2_direct_type(config.kernel_type) else 1
        if config.v2_layout not in ("expert_major", "token_major"):
            raise ValueError(
                "v2_layout must be expert_major or token_major, got "
                f"{config.v2_layout!r}"
            )
        if config.v2_copy_block_num < 0:
            raise ValueError("v2_copy_block_num must be non-negative")
        if _is_v2_direct_type(config.kernel_type) and config.max_token_type_size < 2:
            raise ValueError(
                "InterNodeV2LL max_token_type_size must be at least 2 for "
                "BF16 Combine buffers"
            )
        _ensure_jit_kernels(config.kernel_type)

        if dist.is_initialized():
            dist.barrier()

        handle_class = _cpp_dispatch_combine_factory("EpDispatchCombineHandle")
        self._cpp_config = mori_cpp.EpDispatchCombineConfig(
            rank=config.rank,
            world_size=config.world_size,
            hidden_dim=config.hidden_dim,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            max_token_type_size=config.max_token_type_size,
            max_num_inp_token_per_rank=config.max_num_inp_token_per_rank,
            num_experts_per_rank=config.num_experts_per_rank,
            num_experts_per_token=config.num_experts_per_token,
            warp_num_per_block=config.warp_num_per_block,
            block_num=config.block_num,
            use_external_inp_buf=config.use_external_inp_buf,
            kernel_type=config.kernel_type,
            gpu_per_node=config.gpu_per_node,
            rdma_block_num=config.rdma_block_num,
            num_qp_per_pe=config.num_qp_per_pe,
            quant_type=_normalize_quant_type(config.quant_type),
            max_total_recv_tokens=config.max_total_recv_tokens,
        )
        # Some SHMEM allocation/registration paths temporarily select devices by global PE.
        # Preserve the framework's node-local device across construction; otherwise ranks on
        # nonzero nodes can leave HIP on an ordinal outside the process-visible 0..N-1 range and
        # later zero-copy tensor views bind to an invalid device.
        framework_device = torch.cuda.current_device()
        try:
            self._handle = handle_class(self._cpp_config)
            self._hip_module = _load_hip_modules(config.kernel_type)
        finally:
            torch.cuda.set_device(framework_device)
        self._handle_info = mori_cpp.get_handle_info(self._handle)

        self._fp8_blockwise_combine_scale_dim = self._handle_info[
            "fp8_blockwise_combine_scale_dim"
        ]
        self._fp8_blockwise_combine_scale_type_size = self._handle_info[
            "fp8_blockwise_combine_scale_type_size"
        ]
        # Detect the fp4_blockwise combine so we can fail fast on unsupported archs at construction.
        # (Kernel selection keys off the Fp4BlockwiseQuant enum directly.)
        self._combine_is_fp4 = (
            isinstance(config.quant_type, str)
            and config.quant_type.strip().lower() == "fp4_blockwise"
        )
        if self._combine_is_fp4:
            # The packed-FP4 combine relies on the gfx950 OCP FP4 conversion instructions
            # (cvt_scalef32_pk_f32_fp4). On other archs there is no hardware path, so fail fast
            # instead of silently selecting an fp4 kernel that would fall back to slow software.
            from mori.jit.config import detect_gpu_arch

            _arch = str(detect_gpu_arch())
            if "gfx950" not in _arch:
                raise ValueError(
                    f"quant_type='fp4_blockwise' combine requires a gfx950 GPU (OCP FP4 "
                    f"conversion instructions); detected arch '{_arch}'. Use 'fp8_blockwise' "
                    f"instead on this device."
                )

        self._dispatch_out_ptrs = mori_cpp.get_dispatch_output_ptrs(self._handle, True)
        self._combine_out_ptrs = mori_cpp.get_combine_output_ptrs(self._handle, True)

        self.local_expert_count = torch.zeros(
            config.num_experts_per_rank, dtype=torch.int32, device="cuda"
        )

        self._reset_func = _cpp_dispatch_combine_factory("launch_reset")
        self._get_dispatch_src_token_pos_func = _cpp_dispatch_combine_factory(
            "get_dispatch_src_token_pos"
        )
        self._get_cur_rank_num_token = _cpp_dispatch_combine_factory(
            "get_cur_rank_num_token"
        )
        self._get_dispatch_sender_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_sender_token_idx_map"
        )
        self._get_dispatch_receiver_token_idx_map_func = _cpp_dispatch_combine_factory(
            "get_dispatch_receiver_token_idx_map"
        )
        self._get_registered_combine_input_buffer = _cpp_dispatch_combine_factory(
            "get_registered_combine_input_buffer"
        )

        self.launch_config_mode = os.environ.get("MORI_EP_LAUNCH_CONFIG_MODE", "MANUAL")
        if self.launch_config_mode == "AUTO":
            self._dispatch_rules = None
            self._combine_rules = None
            self._qt_str = "none"
            try:
                from mori.ops.tuning_config import (
                    TuningConfigManager,
                    kernel_type_to_config_str,
                    quant_type_to_config_str,
                    detect_gpu_model,
                )
                from mori.jit.config import detect_gpu_arch

                gpu_arch = detect_gpu_arch()
                gpu_model = detect_gpu_model()
                kt_str = kernel_type_to_config_str(config.kernel_type)
                self._qt_str = quant_type_to_config_str(config.quant_type)
                mgr = TuningConfigManager.get_instance(
                    gpu_arch,
                    kt_str,
                    config.world_size,
                    gpu_model,
                )
                self._dispatch_rules = mgr.dispatch_rules or None
                self._combine_rules = mgr.combine_rules or None
                if logger.isEnabledFor(logging.DEBUG):
                    if self._dispatch_rules is None and self._combine_rules is None:
                        logger.debug(
                            "AUTO tuning: no config for %s_%s_%s_ep%d; "
                            "using hard-coded fallback.",
                            gpu_arch,
                            gpu_model,
                            kt_str,
                            config.world_size,
                        )
                    else:
                        d_dtypes = sorted(
                            {r["dtype"] for r in (self._dispatch_rules or [])}
                        )
                        c_dtypes = sorted(
                            {r["dtype"] for r in (self._combine_rules or [])}
                        )
                        logger.debug(
                            "AUTO tuning: %s_%s_%s_ep%d — "
                            "dispatch(%d rules, dtypes=%s) combine(%d rules, dtypes=%s)",
                            gpu_arch,
                            gpu_model,
                            kt_str,
                            config.world_size,
                            len(self._dispatch_rules or []),
                            d_dtypes,
                            len(self._combine_rules or []),
                            c_dtypes,
                        )
            except Exception as exc:
                logger.warning(
                    "AUTO tuning: failed to load config (%s); "
                    "using hard-coded fallback.",
                    exc,
                )

            if (
                config.kernel_type.value
                == EpDispatchCombineKernelType.InterNodeV1.value
            ):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (96, 64, 8)
            elif (
                config.kernel_type.value
                == EpDispatchCombineKernelType.InterNodeV1LL.value
            ):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (256, 128, 8)
            elif _is_v2_direct_type(config.kernel_type):
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (
                    min(config.block_num, self._handle_info["multi_processor_count"]),
                    config.rdma_block_num,
                    config.warp_num_per_block,
                )
            else:
                (
                    self.auto_block_num,
                    self.auto_rdma_block_num,
                    self.auto_warp_per_block,
                ) = (128, 0, 16)
        elif self.launch_config_mode == "MANUAL":
            self._dispatch_rules = None
            self._combine_rules = None
            self._qt_str = "none"
            self.auto_block_num, self.auto_rdma_block_num, self.auto_warp_per_block = (
                None,
                None,
                None,
            )
        else:
            raise ValueError(
                f"invalid MORI_EP_LAUNCH_CONFIG_MODE, must be ['MANUAL', 'AUTO'], got '{self.launch_config_mode}'"
            )

    # ------------------------------------------------------------------
    # Kernel launch helpers
    # ------------------------------------------------------------------
    def _resolve_launch_params(
        self,
        block_num,
        rdma_block_num,
        warp_per_block,
        *,
        num_tokens=0,
        hidden_dim=0,
        dtype=None,
        tuning_rules=None,
        zero_copy=None,
        quant_type=None,
    ):
        if tuning_rules and dtype is not None:
            from mori.ops.tuning_config import TuningConfigManager

            params = TuningConfigManager.lookup(
                tuning_rules,
                dtype,
                num_tokens,
                hidden_dim,
                zero_copy,
                quant_type,
                topk=self.config.num_experts_per_token,
            )
            if params is not None:
                return params.block_num, params.rdma_block_num, params.warp_per_block
        bn = self.auto_block_num if self.auto_block_num else block_num
        rbn = self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num
        wpb = self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block
        actual_bn = self.config.block_num if bn <= 0 else bn
        actual_rbn = self.config.rdma_block_num if rbn <= 0 else rbn
        actual_wpb = self.config.warp_num_per_block if wpb <= 0 else wpb
        return actual_bn, actual_rbn, actual_wpb

    def _get_func(self, name):
        return self._hip_module.get_function(name)

    def _get_adapter_func(self, name):
        """Load standalone Standard-MoE adapter kernels from ep_intranode.

        These kernels are transport-independent and are emitted only by the
        ep_intranode JIT source, even when the owning EP op uses an inter-node
        implementation.
        """
        if not hasattr(self, "_adapter_hip_module"):
            from mori.ops._jit_loader import ensure_compiled, load_hip_module

            ensure_compiled("ep_intranode")
            self._adapter_hip_module = load_hip_module("ep_intranode", init_shmem=True)
        return self._adapter_hip_module.get_function(name)

    def _launch_adapter(self, func_name, grid, block, shared_mem, stream, args_ptr):
        func = self._get_adapter_func(func_name)
        func.launch_struct(grid, block, shared_mem, stream, args_ptr)

    def _dispatch_shared_mem(self, warp_per_block):
        """Shared memory for dispatch kernels (worldSize + numExpertPerRank per warp + numExpertPerRank) * sizeof(index_t)."""
        return (
            self.config.world_size * warp_per_block
            + self.config.num_experts_per_rank * warp_per_block
            + self.config.num_experts_per_rank
        ) * 4  # sizeof(index_t)

    def _combine_shared_mem(self, warp_per_block, use_weights=True):
        """Shared memory for combine kernels."""
        quant_type = _normalize_quant_type(self.config.quant_type)
        shared_slots = self.config.num_experts_per_token
        if _is_v2_direct_type(self.config.kernel_type):
            shared_slots = max(
                shared_slots,
                self.config.world_size // self.config.gpu_per_node,
            )
        num_ptr_arrays = 1 + int(bool(use_weights))
        if quant_type in _BLOCKWISE_COMBINE_QUANT_TYPES:
            num_ptr_arrays += 1
        return warp_per_block * shared_slots * num_ptr_arrays * _PTR_SIZE

    def _launch(self, func_name, grid, block, shared_mem, stream, args_ptr):
        func = self._get_func(func_name)
        func.launch_struct(grid, block, shared_mem, stream, args_ptr)

    def _launch_multi(self, func_names, grids, blocks, shared_mems, stream, args_ptr):
        from mori.jit.hip_driver import launch_multi

        funcs = [self._get_func(name)._func for name in func_names]
        launch_multi(funcs, grids, blocks, shared_mems, stream, args_ptr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_launch_config(
        self, is_dispatch=True, block_num=-1, rdma_block_num=-1, warp_per_block=-1
    ):
        rules = self._dispatch_rules if is_dispatch else self._combine_rules
        if rules:
            from mori.ops.tuning_config import TuningConfigManager

            zc = not self.config.use_external_inp_buf if not is_dispatch else None
            qt = self._qt_str if not is_dispatch else None
            params = TuningConfigManager.lookup(
                rules,
                self.config.data_type,
                self.config.max_num_inp_token_per_rank,
                self.config.hidden_dim,
                zero_copy=zc,
                quant_type=qt,
                topk=self.config.num_experts_per_token,
            )
            if params is not None:
                return params.block_num, params.rdma_block_num, params.warp_per_block
        return (
            self.auto_block_num if self.auto_block_num else block_num,
            self.auto_rdma_block_num if self.auto_rdma_block_num else rdma_block_num,
            self.auto_warp_per_block if self.auto_warp_per_block else warp_per_block,
        )

    def max_num_tokens_to_recv(self):
        return self._cpp_config.max_num_tokens_to_recv()

    def max_num_tokens_to_recv_per_rank(self):
        return self._cpp_config.max_num_tokens_to_recv_per_rank()

    def max_num_tokens_to_send(self):
        return self._cpp_config.max_num_tokens_to_send()

    def max_num_tokens_to_send_per_rank(self):
        return self._cpp_config.max_num_tokens_to_send_per_rank()

    def decode_send_flat_idx(self, flat_idx):
        """Decode a flat send index into (rank, local_token_id)."""
        stride = self.max_num_tokens_to_send()
        return int(flat_idx) // stride, int(flat_idx) % stride

    def get_registered_combine_input_buffer(
        self, dtype: torch.dtype, hidden_dim: int = -1
    ):
        ptr, shape0, shape1 = self._get_registered_combine_input_buffer(
            self._handle, hidden_dim
        )
        return from_gpu_ptr(ptr, (shape0, shape1), dtype)

    def _alloc_routing_handle(self) -> EpDispatchRoutingHandle:
        cfg = self.config
        world_size = cfg.world_size
        m_send = cfg.max_num_inp_token_per_rank
        e = cfg.num_experts_per_token
        r = cfg.num_experts_per_rank
        n_nodes = max(1, world_size // cfg.gpu_per_node)

        device = "cuda"
        return EpDispatchRoutingHandle(
            disp_dest_tok_id_map=torch.zeros(
                world_size * m_send * r, dtype=torch.int32, device=device
            ),
            inter_node_disp_dest_tok_id_map=torch.zeros(
                n_nodes * m_send * e, dtype=torch.int32, device=device
            ),
            inter_node_disp_send_map=torch.zeros(
                n_nodes * m_send, dtype=torch.int32, device=device
            ),
            total_recv_token_num=torch.zeros(1, dtype=torch.int32, device=device),
            disp_tok_id_to_src_tok_id_local=torch.zeros(
                world_size * m_send * r, dtype=torch.int32, device=device
            ),
        )

    def _supports_routing_handle(self) -> bool:
        kt = self.config.kernel_type.value
        return kt in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
        )

    @staticmethod
    def _routing_source_token_count(routing: EpDispatchRoutingHandle) -> int:
        """Return the cache-routing source token count stored in a routing handle."""
        n = int(routing.cur_rank_num_token)
        if n <= 0:
            raise ValueError(
                "routing handle has cur_rank_num_token <= 0; use a handle from "
                "dispatch(..., return_routing=True) or pass a positive "
                "cur_rank_num_token to EpDispatchRoutingHandle.from_tensors(...)"
            )
        return n

    def _build_args_routing(
        self,
        routing,
        *,
        rdma_block_num,
        hidden_dim,
        replay_mode,
        use_external_inp_buf=-1,
    ):
        return mori_cpp.build_args_with_routing(
            self._handle,
            rdma_block_num=rdma_block_num,
            hidden_dim=hidden_dim,
            use_external_inp_buf=use_external_inp_buf,
            replay_mode=replay_mode,
            disp_dest_tok_id_map_ptr=routing.disp_dest_tok_id_map.data_ptr(),
            inter_node_disp_dest_tok_id_map_ptr=routing.inter_node_disp_dest_tok_id_map.data_ptr(),
            inter_node_disp_send_map_ptr=routing.inter_node_disp_send_map.data_ptr(),
            total_recv_token_num_ptr=routing.total_recv_token_num.data_ptr(),
            disp_tok_id_to_src_tok_id_local_ptr=routing.disp_tok_id_to_src_tok_id_local.data_ptr(),
        )

    def dispatch(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
        *,
        routing: "EpDispatchRoutingHandle | None" = None,
        return_routing: bool = False,
    ):
        if _is_v2_direct_type(self.config.kernel_type):
            raise ValueError(
                "InterNodeV2LL uses a registered Group GEMM layout; use "
                "dispatch_v2_standard_moe()"
            )
        if routing is not None and return_routing:
            raise ValueError(
                "pass either `routing=` (replay) or `return_routing=True` "
                "(new layout), not both"
            )
        use_routing_handle = routing is not None or return_routing
        is_replay = routing is not None
        if use_routing_handle and not self._supports_routing_handle():
            raise NotImplementedError(
                f"routing handle path not supported for kernel_type="
                f"{self.config.kernel_type}; only IntraNode and InterNodeV1 "
                "expose cache/replay routing dispatch."
            )
        if return_routing:
            routing = self._alloc_routing_handle()

        num_tokens = int(input.size(0))
        if is_replay:
            replay_n = self._routing_source_token_count(routing)
            if num_tokens != replay_n:
                raise ValueError(
                    f"replay dispatch input has {num_tokens} tokens but "
                    f"routing.cur_rank_num_token={replay_n}"
                )
            if int(indices.size(0)) != replay_n:
                raise ValueError(
                    f"replay dispatch indices has {int(indices.size(0))} tokens but "
                    f"routing.cur_rank_num_token={replay_n}"
                )

        hidden_dim = input.size(1)
        weight_ptr = weights.data_ptr() if weights is not None else 0
        has_scales = scales is not None and self.config.scale_dim > 0
        scale_ptr = scales.data_ptr() if has_scales else 0
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
        )
        self._cached_dispatch_launch = (actual_bn, actual_rbn, actual_wpb)
        stream = _current_stream()
        self._dispatch_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weight_ptr,
            scale_ptr=scale_ptr,
            indices_ptr=indices.data_ptr(),
        )
        if use_routing_handle:
            args_ptr = self._build_args_routing(
                routing,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                replay_mode=is_replay,
            )
        else:
            args_ptr = mori_cpp.build_args(
                self._handle,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
            )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNode.value:
            self._launch(
                f"EpDispatchInterNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch_multi(
                [
                    f"EpDispatchCopyToStaging_{sfx}",
                    f"EpDispatchInterNodeV1Kernel_{sfx}",
                ],
                [mp, actual_bn],
                [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                [0, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch_multi(
                [
                    f"EpDispatchCopyToStaging_{sfx}",
                    f"EpDispatchInterNodeV1KernelLowLatency_{sfx}",
                ],
                [mp, actual_bn],
                [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                [0, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpDispatchIntraNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNodeLL.value:
            self._launch(
                f"EpDispatchIntraNodeLLKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            mb_block = WARP_SIZE * 16
            self._launch_multi(
                [
                    f"EpDispatchLowLatencyAsyncSendCopySlotAssign_{sfx}",
                    f"EpDispatchLowLatencyAsyncSendCopyMultiBlock_{sfx}",
                    f"EpDispatchLowLatencyAsyncSendTransfer_{sfx}",
                ],
                [mp_aligned, mp_aligned, self.config.world_size],
                [mb_block, mb_block, WARP_SIZE * actual_wpb],
                [0, 0, 0],
                stream,
                args_ptr,
            )
        else:
            raise ValueError(f"Unsupported dispatch kernel_type: {kt}")

        if call_local_expert_count and kt != EpDispatchCombineKernelType.AsyncLL.value:
            from mori.ops.local_expert_count import launch_local_expert_count

            _, _, _, outI_ptr, total_ptr = self._dispatch_out_ptrs
            if use_routing_handle:
                total_ptr = routing.total_recv_token_num.data_ptr()
            launch_local_expert_count(
                self._cpp_config,
                outI_ptr,
                total_ptr,
                self.local_expert_count.data_ptr(),
                stream=stream,
            )

        out_ptr, outW_ptr, outS_ptr, outI_ptr, total_ptr = self._dispatch_out_ptrs
        max_recv = self._cpp_config.max_num_tokens_to_recv()
        out = from_gpu_ptr(out_ptr, (max_recv, hidden_dim), input.dtype)
        out_weights = from_gpu_ptr(
            outW_ptr, (max_recv, self.config.num_experts_per_token), torch.float32
        )
        out_scales = None
        if has_scales and outS_ptr:
            out_scales = from_gpu_ptr(
                outS_ptr, (max_recv, self.config.scale_dim), scales.dtype
            )
        out_indices = from_gpu_ptr(
            outI_ptr, (max_recv, self.config.num_experts_per_token), TOPK_IDX_DTYPE
        )
        total_recv = (
            routing.total_recv_token_num
            if use_routing_handle
            else from_gpu_ptr(total_ptr, (1,), TOPK_IDX_DTYPE)
        )

        if return_routing:
            mori_cpp.snapshot_disp_tok_id_to_src_tok_id_local(
                self._handle,
                routing.disp_tok_id_to_src_tok_id_local.data_ptr(),
                stream=stream,
            )
            routing.cur_rank_num_token = int(input.size(0))

        base = (out, out_weights, out_scales, out_indices, total_recv)
        if return_routing:
            return base + (routing,)
        return base

    def dispatch_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        return self.dispatch(
            input,
            weights,
            scales,
            indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

    def dispatch_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
        call_local_expert_count: bool = False,
    ):
        if hasattr(self, "_cached_dispatch_launch"):
            _, _, actual_wpb = self._cached_dispatch_launch
        else:
            _, _, actual_wpb = self._resolve_launch_params(block_num, 0, warp_per_block)
        stream = _current_stream()
        assert hasattr(
            self, "_dispatch_dtype"
        ), "dispatch_recv requires a prior dispatch/dispatch_send call"
        sfx = _DTYPE_SUFFIX[self._dispatch_dtype]
        kt = self.config.kernel_type.value

        # Recv kernels must reuse the handle's inference pointers/state prepared by the
        # preceding dispatch_send/dispatch call, so only rebuild raw args here.
        args_ptr = mori_cpp.build_args(self._handle, rdma_block_num=0)
        if kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            mb_block = WARP_SIZE * 16
            self._launch_multi(
                [
                    f"EpDispatchLowLatencyAsyncRecvTransfer_{sfx}",
                    f"EpDispatchLowLatencyAsyncRecvCopyMultiBlock_{sfx}",
                ],
                [self.config.world_size, mp_aligned],
                [WARP_SIZE * actual_wpb, mb_block],
                [0, 0],
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                f"dispatch_recv only supports AsyncLL, got kernel_type={kt}"
            )

        if call_local_expert_count:
            from mori.ops.local_expert_count import launch_local_expert_count

            _, _, _, outI_ptr, total_ptr = self._dispatch_out_ptrs
            launch_local_expert_count(
                self._cpp_config,
                outI_ptr,
                total_ptr,
                self.local_expert_count.data_ptr(),
                stream=stream,
            )

    def combine(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        use_external_inp_buf: int = -1,
        call_reset: bool = False,
        *,
        routing: "EpDispatchRoutingHandle | None" = None,
    ):
        if _is_v2_direct_type(self.config.kernel_type):
            raise ValueError(
                "InterNodeV2LL consumes a registered Group GEMM output; use "
                "combine_standard_moe() with the registered V2 combine buffer"
            )
        if routing is not None and not self._supports_routing_handle():
            raise NotImplementedError(
                f"routing handle path not supported for kernel_type="
                f"{self.config.kernel_type}; only IntraNode and InterNodeV1 "
                "currently consume routing handles in combine."
            )

        if routing is not None:
            routing_n = self._routing_source_token_count(routing)
            if int(indices.size(0)) != routing_n:
                raise ValueError(
                    f"combine indices has {int(indices.size(0))} tokens but "
                    f"routing.cur_rank_num_token={routing_n}"
                )

        hidden_dim = input.size(1)
        weight_ptr = (
            weights.data_ptr() if weights is not None and weights.size(0) != 0 else 0
        )
        actual_use_ext = (
            use_external_inp_buf
            if use_external_inp_buf >= 0
            else int(self.config.use_external_inp_buf)
        )
        is_zero_copy = not actual_use_ext
        cur_n = (
            self._routing_source_token_count(routing)
            if routing is not None
            else self._get_cur_rank_num_token(self._handle)
        )
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=cur_n,
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._combine_rules,
            zero_copy=is_zero_copy,
            quant_type=self._qt_str,
        )
        self._cached_combine_launch = (actual_bn, actual_rbn, actual_wpb)
        stream = _current_stream()
        self._combine_dtype = input.dtype
        sfx = _DTYPE_SUFFIX[input.dtype]

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=cur_n,
            weight_ptr=weight_ptr,
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        if routing is not None:
            args_ptr = self._build_args_routing(
                routing,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                replay_mode=False,
                use_external_inp_buf=use_external_inp_buf,
            )
        else:
            args_ptr = mori_cpp.build_args(
                self._handle,
                rdma_block_num=actual_rbn,
                hidden_dim=hidden_dim,
                use_external_inp_buf=use_external_inp_buf,
            )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        kt = self.config.kernel_type.value
        quant_type = _normalize_quant_type(self.config.quant_type)
        shared_mem = self._combine_shared_mem(actual_wpb)

        if quant_type in _BLOCKWISE_COMBINE_QUANT_TYPES:
            label = (
                "fp4_blockwise"
                if quant_type == EpDispatchCombineQuantType.Fp4BlockwiseQuant
                else "fp8_blockwise"
            )
            # fp8 blockwise also runs on AsyncLL; fp4 blockwise is IntraNode-only.
            allowed_kts = [
                EpDispatchCombineKernelType.IntraNode.value,
                EpDispatchCombineKernelType.IntraNodeLL.value,
            ]
            if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                allowed_kts.append(EpDispatchCombineKernelType.AsyncLL.value)
            if kt not in allowed_kts:
                supported = (
                    "IntraNode/IntraNodeLL/AsyncLL"
                    if quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant
                    else "IntraNode/IntraNodeLL"
                )
                raise ValueError(
                    f"{label} combine currently only supports {supported} combine"
                )
            if sfx != "bf16":
                raise ValueError(f"{label} combine only supports bf16, got {sfx}")
            if not actual_use_ext:
                raise ValueError(
                    f"{label} combine currently requires --zero-copy 0 "
                    "(useExternalInpBuffer=True). P2P read path not yet implemented."
                )
            if self._fp8_blockwise_combine_scale_dim <= 0:
                raise ValueError(
                    f"{label} combine requires internal combine scale_dim > 0"
                )

        if kt == EpDispatchCombineKernelType.InterNode.value:
            self._launch(
                f"EpCombineInterNodeKernel_{sfx}",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1.value:
            mp = self._handle_info["multi_processor_count"]
            bsz = WARP_SIZE * actual_wpb
            self._launch_multi(
                [
                    f"EpCombineSync_{sfx}",
                    f"EpCombineSyncBarrier_{sfx}",
                    f"EpCombineInterNodeV1Kernel_{sfx}",
                    f"EpCombineAll_{sfx}",
                ],
                [mp, 1, actual_bn, mp],
                [bsz, WARP_SIZE, bsz, bsz],
                [0, 0, shared_mem, shared_mem],
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            bsz = WARP_SIZE * actual_wpb
            self._launch_multi(
                [
                    f"EpCombineSync_{sfx}",
                    f"EpCombineSyncBarrier_{sfx}",
                    f"EpCombineInterNodeV1KernelLowLatency_{sfx}",
                    f"EpCombineAll_{sfx}",
                ],
                [mp, 1, actual_bn, mp],
                [bsz, WARP_SIZE, bsz, bsz],
                [0, 0, shared_mem, shared_mem],
                stream,
                args_ptr,
            )
        elif kt in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.IntraNodeLL.value,
        ):
            if quant_type in _BLOCKWISE_COMBINE_QUANT_TYPES:
                # Mirror of the AccumNum=8/9 + VecBytes=8 specialization gating in
                # LaunchCombine() / launch.cpp. top-k==9 covers shared-expert fusion
                # (8 routed + 1 fused shared). Keep in sync.
                fp8_scale_dim = self._fp8_blockwise_combine_scale_dim
                block_elems = (hidden_dim + fp8_scale_dim - 1) // fp8_scale_dim
                base_vec8_top8_eligible = (
                    weight_ptr == 0
                    and (hidden_dim % 512) == 0
                    and self.config.num_experts_per_token in (8, 9)
                    and self.config.world_size > 4
                )
                top9 = self.config.num_experts_per_token == 9
                kernel_name = "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq"
                use_vec8_top8 = False
                if base_vec8_top8_eligible:
                    if block_elems == 128:
                        kernel_name = (
                            "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block128_vec8_top9"
                            if top9
                            else "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block128_vec8"
                        )
                        use_vec8_top8 = True
                    elif block_elems == 256:
                        kernel_name = (
                            "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block256_vec8_top9"
                            if top9
                            else "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block256_vec8"
                        )
                        use_vec8_top8 = True
                # Blockwise FP4: select the packed-FP4 kernel variants (identical launch config to
                # the fp8bwq variants; only the in-kernel quant/dequant math differs). Assert the
                # derived name is a registered fp4bwq symbol so a naming mismatch fails loudly.
                if quant_type == EpDispatchCombineQuantType.Fp4BlockwiseQuant:
                    kernel_name = kernel_name.replace("_fp8bwq", "_fp4bwq")
                    assert (
                        kernel_name in _FP4_COMBINE_KERNELS
                    ), f"fp4_blockwise combine selected unregistered kernel '{kernel_name}'"
                shared_mem = self._combine_shared_mem(
                    actual_wpb, use_weights=not use_vec8_top8
                )
                self._last_combine_kernel_name = kernel_name
                self._launch(
                    kernel_name,
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
            elif actual_use_ext:
                if (
                    sfx == "bf16"
                    and quant_type == EpDispatchCombineQuantType.Fp8DirectCast
                ):
                    self._launch(
                        "EpCombineIntraNodeKernel_bf16_nop2p_fp8cast",
                        grid,
                        block,
                        shared_mem,
                        stream,
                        args_ptr,
                    )
                else:
                    self._launch(
                        f"EpCombineIntraNodeKernel_{sfx}_nop2p",
                        grid,
                        block,
                        shared_mem,
                        stream,
                        args_ptr,
                    )
            else:
                self._launch(
                    f"EpCombineIntraNodeKernel_{sfx}_p2p",
                    grid,
                    block,
                    shared_mem,
                    stream,
                    args_ptr,
                )
        elif kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            if sfx == "bf16" and quant_type == EpDispatchCombineQuantType.Fp8DirectCast:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncSendCopy_bf16_fp8cast",
                        "EpCombineLowLatencyAsyncSendTransfer_bf16_fp8cast",
                    ],
                    [mp_aligned, self.config.world_size],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
            elif quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncSendCopy_bf16_fp8bwq",
                        "EpCombineLowLatencyAsyncSendTransfer_bf16_fp8bwq",
                    ],
                    [mp_aligned, self.config.world_size],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
            else:
                self._launch_multi(
                    [
                        f"EpCombineLowLatencyAsyncSendCopy_{sfx}",
                        f"EpCombineLowLatencyAsyncSendTransfer_{sfx}",
                    ],
                    [mp_aligned, self.config.world_size],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, 0],
                    stream,
                    args_ptr,
                )
        else:
            raise ValueError(f"Unsupported combine kernel_type: {kt}")

        out_ptr, outW_ptr = self._combine_out_ptrs
        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None
        if weight_ptr and outW_ptr:
            out_weights = from_gpu_ptr(
                outW_ptr,
                (
                    self.config.max_num_inp_token_per_rank,
                    self.config.num_experts_per_token,
                ),
                weights.dtype,
            )

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def combine_send(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        return self.combine(
            input,
            weights,
            indices,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )

    def combine_recv(
        self,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        if hasattr(self, "_cached_combine_launch"):
            _, _, actual_wpb = self._cached_combine_launch
        else:
            _, _, actual_wpb = self._resolve_launch_params(block_num, 0, warp_per_block)
        stream = _current_stream()
        assert hasattr(
            self, "_combine_dtype"
        ), "combine_recv requires a prior combine/combine_send call"
        sfx = _DTYPE_SUFFIX[self._combine_dtype]
        kt = self.config.kernel_type.value

        # Recv kernels must reuse the handle's inference pointers/state prepared by the
        # preceding combine_send/combine call, so only rebuild raw args here.
        args_ptr = mori_cpp.build_args(self._handle, rdma_block_num=0)
        shared_mem = self._combine_shared_mem(actual_wpb)
        if kt == EpDispatchCombineKernelType.AsyncLL.value:
            mp = self._handle_info["multi_processor_count"]
            mp_aligned = mp // self.config.world_size * self.config.world_size
            quant_type = _normalize_quant_type(self.config.quant_type)
            if sfx == "bf16" and quant_type == EpDispatchCombineQuantType.Fp8DirectCast:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncRecvTransfer_bf16_fp8cast",
                        "EpCombineLowLatencyAsyncRecvCopy_bf16_fp8cast",
                    ],
                    [self.config.world_size, mp_aligned],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
            elif quant_type == EpDispatchCombineQuantType.Fp8BlockwiseQuant:
                self._launch_multi(
                    [
                        "EpCombineLowLatencyAsyncRecvTransfer_bf16",
                        "EpCombineLowLatencyAsyncRecvCopy_bf16_fp8bwq",
                    ],
                    [self.config.world_size, mp_aligned],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
            else:
                self._launch_multi(
                    [
                        f"EpCombineLowLatencyAsyncRecvTransfer_{sfx}",
                        f"EpCombineLowLatencyAsyncRecvCopy_{sfx}",
                    ],
                    [self.config.world_size, mp_aligned],
                    [WARP_SIZE * actual_wpb, WARP_SIZE * actual_wpb],
                    [0, shared_mem],
                    stream,
                    args_ptr,
                )
        else:
            raise ValueError(
                f"combine_recv only supports AsyncLL, got kernel_type={kt}"
            )

    def dispatch_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
    ):
        set_fn = _cpp_dispatch_combine_factory(
            "set_standard_moe_output_buffers", allow_missing=True
        )
        if set_fn is None:
            raise RuntimeError(
                "dispatch_standard_moe is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )
        hidden_dim = input.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        packed_recv_x = torch.empty(
            (num_local_experts, max_tokens_per_expert, hidden_dim),
            dtype=input.dtype,
            device=input.device,
        )
        packed_recv_src_info = torch.empty(
            (num_local_experts, max_tokens_per_expert),
            dtype=torch.int32,
            device=input.device,
        )
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=input.device
        )

        set_fn(self._handle, packed_recv_x.data_ptr(), packed_recv_src_info.data_ptr())

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=(weights.data_ptr() if weights is not None else 0),
            scale_ptr=(
                scales.data_ptr()
                if scales is not None and self.config.scale_dim > 0
                else 0
            ),
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._dispatch_shared_mem(actual_wpb)
        kt = self.config.kernel_type.value

        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch(
                f"EpDispatchCopyToStaging_{sfx}", (mp,), block, 0, stream, args_ptr
            )
            self._launch(
                f"EpDispatchInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpDispatchIntraNodeKernel_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "dispatch_standard_moe only supports IntraNode/InterNodeV1LL"
            )

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    def dispatch_v2_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        """Dispatch directly into registered V2 Group-GEMM buffers.

        Every ``(token, expert)`` route receives its own expert slot. The returned
        X/scales/weights can be consumed directly by Group GEMM without
        ``ConvertDispatchOutput``. ``indices`` is part of the Standard-MoE
        contract: it must be int32, use ``-1`` only for masked routes, and contain
        no duplicate valid expert id within one token. The caller is responsible
        for providing in-range, unique expert ids.
        """
        if self.config.kernel_type != EpDispatchCombineKernelType.InterNodeV2LL:
            raise ValueError("dispatch_v2_standard_moe requires InterNodeV2LL")
        if input.ndim != 2:
            raise ValueError(
                f"V2LL input must be 2D [tokens, hidden], got shape={tuple(input.shape)}"
            )
        if input.size(0) > self.config.max_num_inp_token_per_rank:
            raise ValueError(
                "V2LL input exceeds max_num_inp_token_per_rank: "
                f"{input.size(0)} > {self.config.max_num_inp_token_per_rank}"
            )
        expected_indices_shape = (
            int(input.size(0)),
            self.config.num_experts_per_token,
        )
        if input.dtype not in _DTYPE_SUFFIX or _DTYPE_SUFFIX[input.dtype] not in (
            "bf16",
            "fp8_fnuz",
            "fp8_ocp",
        ):
            raise ValueError(
                f"V2LL Dispatch supports BF16 or FP8 input, got {input.dtype}"
            )
        if (
            tuple(weights.shape) != expected_indices_shape
            or weights.dtype != torch.float32
        ):
            raise ValueError(
                "V2 Standard-MoE weights must have shape "
                f"{expected_indices_shape} and dtype torch.float32, got "
                f"shape={tuple(weights.shape)}, dtype={weights.dtype}"
            )
        if tuple(indices.shape) != expected_indices_shape:
            raise ValueError(
                "V2 Standard-MoE indices must have shape "
                f"{expected_indices_shape}, got {tuple(indices.shape)}"
            )
        if indices.dtype != torch.int32:
            raise ValueError(
                f"V2 Standard-MoE indices must be torch.int32, got {indices.dtype}"
            )
        if input.device != weights.device or input.device != indices.device:
            raise ValueError(
                "V2LL input, weights, and indices must be on the same device"
            )
        if (
            not input.is_contiguous()
            or not weights.is_contiguous()
            or not indices.is_contiguous()
        ):
            raise ValueError("V2LL input, weights, and indices must be contiguous")
        if scales is not None:
            expected_scale_shape = (int(input.size(0)), self.config.scale_dim)
            if (
                tuple(scales.shape) != expected_scale_shape
                or scales.device != input.device
                or not scales.is_contiguous()
                or scales.element_size() != self.config.scale_type_size
            ):
                raise ValueError(
                    "V2LL scales must be contiguous, on the input device, and have shape "
                    f"{expected_scale_shape}; got shape={tuple(scales.shape)}, "
                    f"device={scales.device}, element_size={scales.element_size()}"
                )
        hidden_dim = input.size(1)
        if hidden_dim > self.config.hidden_dim:
            raise ValueError(
                "V2LL input hidden dimension exceeds configured buffer capacity: "
                f"{hidden_dim} > {self.config.hidden_dim}"
            )
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num,
            1,
            warp_per_block,
            num_tokens=input.size(0),
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._dispatch_rules,
        )
        use_default_launch = block_num <= 0 and warp_per_block <= 0
        default_copy_blocks = 0
        use_tuned_default = False
        if use_default_launch:
            v2ll_default = _v2ll_default_launch(
                self.config, int(input.size(0)), input.dtype, is_dispatch=True
            )
            if v2ll_default is not None:
                actual_bn, actual_wpb, default_copy_blocks = v2ll_default
                use_tuned_default = True
        num_nodes = self.config.world_size // self.config.gpu_per_node
        if actual_wpb < self.config.num_qp_per_pe:
            raise ValueError(
                "V2LL Dispatch requires at least one sender warp per QP: "
                f"warps={actual_wpb}, qps={self.config.num_qp_per_pe}"
            )
        copy_blocks = self.config.v2_copy_block_num or default_copy_blocks or max(
            self.config.num_qp_per_pe, actual_bn // 4
        )
        if copy_blocks >= actual_bn or actual_bn - copy_blocks < num_nodes:
            raise ValueError(
                "V2LL Dispatch must leave at least one scatter CTA per node: "
                f"blocks={actual_bn}, copy_blocks={copy_blocks}, nodes={num_nodes}"
            )
        mp = self._handle_info["multi_processor_count"]
        if actual_bn > mp:
            raise ValueError(
                "V2LL fused Dispatch requires a resident grid: "
                f"block_num={actual_bn} exceeds device CUs={mp}"
            )
        if (
            self.config.v2_layout == "token_major"
            and copy_blocks < self.config.num_qp_per_pe
        ):
            raise ValueError(
                "token-major V2LL requires at least one copy CTA per QP: "
                f"copy_blocks={copy_blocks}, qps={self.config.num_qp_per_pe}"
            )

        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]
        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=input.size(0),
            weight_ptr=weights.data_ptr(),
            scale_ptr=(scales.data_ptr() if scales is not None else 0),
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=0,
            dispatch_copy_block_num=copy_blocks,
            hidden_dim=hidden_dim,
        )
        block = (WARP_SIZE * actual_wpb,)
        use_multi_warp_copy = use_tuned_default and int(input.size(0)) == 64
        if self.config.v2_layout == "token_major":
            dispatch_kernel = (
                "EpDispatchInterNodeV2LLTokenMajorMultiWarpCopy"
                if use_multi_warp_copy
                else "EpDispatchInterNodeV2LLTokenMajor"
            )
        else:
            dispatch_kernel = (
                "EpDispatchInterNodeV2LLMultiWarpCopy"
                if use_multi_warp_copy
                else "EpDispatchInterNodeV2LLKernel"
            )
        mori_cpp.begin_v2_dispatch(self._handle, stream)
        try:
            self._launch(
                f"{dispatch_kernel}_{sfx}_ring",
                (actual_bn,),
                block,
                0,
                stream,
                args_ptr,
            )
        except Exception:
            mori_cpp.abort_v2_dispatch(self._handle)
            raise
        mori_cpp.commit_v2_dispatch(self._handle)

        (
            x_ptr,
            count_ptr,
            weight_ptr,
            scale_ptr,
            src_ptr,
            max_per_expert,
        ) = mori_cpp.get_v2_dispatch_output_ptrs(self._handle)
        if self.config.v2_layout == "token_major":
            max_recv_tokens = (
                self.config.world_size * self.config.max_num_inp_token_per_rank
            )
            token_x = from_gpu_ptr(x_ptr, (max_recv_tokens, hidden_dim), input.dtype)
            # Token-major publishes its V1LL-compatible total row count into symmetric counter
            # slot 0 at the Dispatch generation-completion edge.
            token_count = from_gpu_ptr(
                count_ptr,
                (1,),
                torch.int32,
            )
            token_weights = from_gpu_ptr(
                weight_ptr,
                (max_recv_tokens, self.config.num_experts_per_token),
                torch.float32,
            )
            token_indices = from_gpu_ptr(
                mori_cpp.get_v2_token_major_indices_ptr(self._handle),
                (max_recv_tokens, self.config.num_experts_per_token),
                torch.int32,
            )
            token_scales = None
            if scale_ptr and scales is not None:
                token_scales = from_gpu_ptr(
                    scale_ptr,
                    (max_recv_tokens, scales.size(1)),
                    scales.dtype,
                )
            token_src_info = from_gpu_ptr(src_ptr, (max_recv_tokens,), torch.int32)
            return (
                token_x,
                token_count,
                token_weights,
                token_indices,
                token_scales,
                token_src_info,
            )
        experts = self.config.num_experts_per_rank
        packed_x = from_gpu_ptr(
            x_ptr, (experts, max_per_expert, hidden_dim), input.dtype
        )
        packed_count = from_gpu_ptr(count_ptr, (experts,), torch.int32)
        packed_weights = from_gpu_ptr(
            weight_ptr, (experts, max_per_expert), torch.float32
        )
        packed_scales = None
        if scale_ptr and scales is not None:
            packed_scales = from_gpu_ptr(
                scale_ptr,
                (experts, max_per_expert, scales.size(1)),
                scales.dtype,
            )
        packed_src_info = from_gpu_ptr(src_ptr, (experts, max_per_expert), torch.int32)
        return (
            packed_x,
            packed_count,
            packed_weights,
            packed_scales,
            packed_src_info,
        )

    def get_v2_registered_combine_input_buffer(
        self, dtype: torch.dtype, hidden_dim: int = -1
    ):
        if self.config.kernel_type != EpDispatchCombineKernelType.InterNodeV2LL:
            raise ValueError("V2 combine input buffer requires InterNodeV2LL")
        if dtype != torch.bfloat16:
            raise ValueError("V2 combine input buffer is BF16")
        if hidden_dim > self.config.hidden_dim:
            raise ValueError(
                "V2LL combine hidden dimension exceeds configured buffer capacity: "
                f"{hidden_dim} > {self.config.hidden_dim}"
            )
        (
            ptr,
            experts,
            max_per_expert,
            actual_hidden,
        ) = mori_cpp.get_v2_combine_input_buffer(self._handle, hidden_dim)
        if self.config.v2_layout == "token_major":
            max_recv_tokens = (
                self.config.world_size * self.config.max_num_inp_token_per_rank
            )
            return from_gpu_ptr(ptr, (max_recv_tokens, actual_hidden), dtype)
        return from_gpu_ptr(ptr, (experts, max_per_expert, actual_hidden), dtype)

    def combine_standard_moe(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        block_num: int = -1,
        rdma_block_num: int = -1,
        warp_per_block: int = -1,
        call_reset: bool = False,
    ):
        kt = self.config.kernel_type.value
        set_fn = None
        if kt != EpDispatchCombineKernelType.InterNodeV2LL.value:
            set_fn = _cpp_dispatch_combine_factory(
                "set_standard_moe_output_buffers", allow_missing=True
            )
            if set_fn is None:
                raise RuntimeError(
                    "combine_standard_moe is not available. "
                    "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
                )
        hidden_dim = input.size(-1)
        cur_n = self._get_cur_rank_num_token(self._handle)
        actual_bn, actual_rbn, actual_wpb = self._resolve_launch_params(
            block_num,
            rdma_block_num,
            warp_per_block,
            num_tokens=cur_n,
            hidden_dim=hidden_dim,
            dtype=input.dtype,
            tuning_rules=self._combine_rules,
            zero_copy=False,
            quant_type=self._qt_str,
        )
        use_default_launch = block_num <= 0 and warp_per_block <= 0
        use_tuned_default = False
        if use_default_launch:
            v2ll_default = _v2ll_default_launch(
                self.config, int(cur_n), input.dtype, is_dispatch=False
            )
            if v2ll_default is not None:
                actual_bn, actual_wpb, _ = v2ll_default
                use_tuned_default = True
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[input.dtype]

        if set_fn is not None:
            set_fn(self._handle, input.data_ptr(), 0)

        mori_cpp.prepare_inference_args(
            self._handle,
            inp_ptr=input.data_ptr(),
            dtype=dtype_to_int(input.dtype),
            num_tokens=cur_n,
            weight_ptr=(
                weights.data_ptr()
                if weights is not None and weights.size(0) != 0
                else 0
            ),
            scale_ptr=0,
            indices_ptr=indices.data_ptr(),
        )
        args_ptr = mori_cpp.build_args(
            self._handle,
            rdma_block_num=actual_rbn,
            dispatch_copy_block_num=0,
            hidden_dim=hidden_dim,
        )

        grid = (actual_bn,)
        block = (WARP_SIZE * actual_wpb,)
        shared_mem = self._combine_shared_mem(actual_wpb)
        if kt == EpDispatchCombineKernelType.InterNodeV1LL.value:
            mp = self._handle_info["multi_processor_count"]
            self._launch(f"EpCombineSync_{sfx}", (mp,), block, 0, stream, args_ptr)
            self._launch(
                f"EpCombineSyncBarrier_{sfx}", (1,), (WARP_SIZE,), 0, stream, args_ptr
            )
            self._launch(
                f"EpCombineInterNodeV1KernelLowLatency_{sfx}_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
            self._launch(
                f"EpCombineAll_{sfx}", (mp,), block, shared_mem, stream, args_ptr
            )
        elif kt == EpDispatchCombineKernelType.InterNodeV2LL.value:
            if input.dtype != torch.bfloat16:
                raise ValueError(
                    "InterNodeV2LL Combine requires BF16 Group GEMM output"
                )
            if weights is not None:
                raise ValueError(
                    "InterNodeV2LL Combine consumes pre-weighted Group GEMM output; "
                    "weights must be None"
                )
            expected_indices_shape = (
                int(cur_n),
                self.config.num_experts_per_token,
            )
            if (
                tuple(indices.shape) != expected_indices_shape
                or indices.dtype != torch.int32
                or indices.device != input.device
                or not indices.is_contiguous()
            ):
                raise ValueError(
                    "InterNodeV2LL Combine indices must be contiguous int32 on the "
                    f"input device with shape={expected_indices_shape}; got "
                    f"shape={tuple(indices.shape)}, dtype={indices.dtype}, "
                    f"device={indices.device}"
                )
            if actual_wpb < self.config.num_qp_per_pe:
                raise ValueError(
                    "V2LL Combine requires at least one sender/waiter warp per QP: "
                    f"warps={actual_wpb}, qps={self.config.num_qp_per_pe}"
                )
            (
                registered_ptr,
                registered_experts,
                registered_capacity,
                registered_hidden,
            ) = mori_cpp.get_v2_combine_input_buffer(self._handle, hidden_dim)
            expected_shape = (
                (
                    self.config.world_size * self.config.max_num_inp_token_per_rank,
                    registered_hidden,
                )
                if self.config.v2_layout == "token_major"
                else (
                    registered_experts,
                    registered_capacity,
                    registered_hidden,
                )
            )
            if tuple(input.shape) != expected_shape or not input.is_contiguous():
                raise ValueError(
                    "InterNodeV2LL Combine requires the full contiguous registered "
                    f"buffer with shape={expected_shape}; got shape={tuple(input.shape)}"
                )
            if input.data_ptr() != registered_ptr:
                raise ValueError(
                    "InterNodeV2LL Group GEMM output must be written directly into "
                    "get_v2_registered_combine_input_buffer()"
                )
            if hidden_dim != registered_hidden:
                raise ValueError(
                    f"V2LL combine hidden_dim mismatch: input={hidden_dim}, "
                    f"registered={registered_hidden}"
                )
            use_qp_early = (
                use_tuned_default
                and int(cur_n) == 64
                and self.config.num_qp_per_pe > 1
                and (hidden_dim * input.element_size()) % 2048 == 0
            )
            if self.config.v2_layout == "token_major":
                remote_kernel = (
                    "EpCombineInterNodeV2LLRemotePartialTokenMajorQpEarlyV16"
                    if use_qp_early
                    else "EpCombineInterNodeV2LLRemotePartialTokenMajorV16"
                )
                local_kernel = (
                    "EpCombineInterNodeV2LLLocalTokenMajorQpEarlyV16"
                    if use_qp_early
                    else "EpCombineInterNodeV2LLSendLocalTokenMajorV16"
                )
            else:
                remote_kernel = (
                    "EpCombineInterNodeV2LLRemotePartialQpEarlyV16"
                    if use_qp_early
                    else "EpCombineInterNodeV2LLRemotePartialV16"
                )
                local_kernel = (
                    "EpCombineInterNodeV2LLLocalQpEarlyV16"
                    if use_qp_early
                    else "EpCombineInterNodeV2LLSendLocalV16"
                )
            mori_cpp.begin_v2_combine(self._handle, stream)
            try:
                self._launch_multi(
                    [
                        f"{remote_kernel}_bf16",
                        f"{local_kernel}_bf16",
                        "EpCombineInterNodeV2LLFinalize_bf16",
                    ],
                    [actual_bn, actual_bn, actual_bn],
                    [block[0], block[0], block[0]],
                    [shared_mem, shared_mem, shared_mem],
                    stream,
                    args_ptr,
                )
            except Exception:
                mori_cpp.abort_v2_combine(self._handle)
                raise
            mori_cpp.commit_v2_combine(self._handle)
        elif kt == EpDispatchCombineKernelType.IntraNode.value:
            self._launch(
                f"EpCombineIntraNodeKernel_{sfx}_p2p_stdmoe",
                grid,
                block,
                shared_mem,
                stream,
                args_ptr,
            )
        else:
            raise ValueError(
                "combine_standard_moe only supports "
                "IntraNode/InterNodeV1LL/InterNodeV2LL"
            )

        out_ptr = self._combine_out_ptrs[0]
        out = from_gpu_ptr(
            out_ptr,
            (self.config.max_num_inp_token_per_rank, hidden_dim),
            input.dtype,
        )
        out_weights = None

        if call_reset:
            self._reset_func(self._handle, _current_stream())
        return (out, out_weights)

    def convert_dispatch_output(
        self,
        dispatch_out_x: torch.Tensor,
        dispatch_out_topk_idx: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
        packed_recv_x: torch.Tensor = None,
        packed_recv_src_info: torch.Tensor = None,
        dispatch_out_scales: torch.Tensor = None,
        packed_recv_scales: torch.Tensor = None,
    ):
        build_fn = _cpp_dispatch_combine_factory(
            "build_convert_dispatch_output_args", allow_missing=True
        )
        if build_fn is None:
            raise RuntimeError(
                "convert_dispatch_output is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = dispatch_out_x.size(1)
        num_local_experts = self.config.num_experts_per_rank
        max_tokens_per_expert = (
            self.config.world_size * self.config.max_num_inp_token_per_rank
        )
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num, 0, warp_per_block
        )
        stream = _current_stream()

        expected_x_shape = (num_local_experts, max_tokens_per_expert, hidden_dim)
        expected_src_shape = (num_local_experts, max_tokens_per_expert)
        if packed_recv_x is None:
            packed_recv_x = torch.empty(
                expected_x_shape,
                dtype=dispatch_out_x.dtype,
                device=dispatch_out_x.device,
            )
        elif (
            tuple(packed_recv_x.shape) != expected_x_shape
            or packed_recv_x.dtype != dispatch_out_x.dtype
            or packed_recv_x.device != dispatch_out_x.device
        ):
            raise ValueError(
                "packed_recv_x must match "
                f"shape={expected_x_shape}, dtype={dispatch_out_x.dtype}, "
                f"device={dispatch_out_x.device}; got shape={tuple(packed_recv_x.shape)}, "
                f"dtype={packed_recv_x.dtype}, device={packed_recv_x.device}"
            )
        if packed_recv_src_info is None:
            packed_recv_src_info = torch.empty(
                expected_src_shape,
                dtype=torch.int32,
                device=dispatch_out_x.device,
            )
        elif (
            tuple(packed_recv_src_info.shape) != expected_src_shape
            or packed_recv_src_info.dtype != torch.int32
            or packed_recv_src_info.device != dispatch_out_x.device
        ):
            raise ValueError(
                "packed_recv_src_info must match "
                f"shape={expected_src_shape}, dtype=torch.int32, "
                f"device={dispatch_out_x.device}; got "
                f"shape={tuple(packed_recv_src_info.shape)}, "
                f"dtype={packed_recv_src_info.dtype}, "
                f"device={packed_recv_src_info.device}"
            )
        scale_bytes_per_token = 0
        if dispatch_out_scales is not None:
            if dispatch_out_scales.ndim != 2:
                raise ValueError(
                    "dispatch_out_scales must be 2D [tokens, scale_dim], got "
                    f"shape={tuple(dispatch_out_scales.shape)}"
                )
            expected_scale_shape = (
                num_local_experts,
                max_tokens_per_expert,
                dispatch_out_scales.size(1),
            )
            if packed_recv_scales is None:
                packed_recv_scales = torch.empty(
                    expected_scale_shape,
                    dtype=dispatch_out_scales.dtype,
                    device=dispatch_out_scales.device,
                )
            elif (
                tuple(packed_recv_scales.shape) != expected_scale_shape
                or packed_recv_scales.dtype != dispatch_out_scales.dtype
                or packed_recv_scales.device != dispatch_out_scales.device
            ):
                raise ValueError(
                    "packed_recv_scales must match "
                    f"shape={expected_scale_shape}, dtype={dispatch_out_scales.dtype}, "
                    f"device={dispatch_out_scales.device}; got "
                    f"shape={tuple(packed_recv_scales.shape)}, "
                    f"dtype={packed_recv_scales.dtype}, "
                    f"device={packed_recv_scales.device}"
                )
            scale_bytes_per_token = (
                dispatch_out_scales.size(1) * dispatch_out_scales.element_size()
            )
        elif packed_recv_scales is not None:
            raise ValueError("packed_recv_scales requires dispatch_out_scales")
        packed_recv_layout_range = torch.empty(
            0, dtype=torch.int64, device=dispatch_out_x.device
        )

        args_ptr = build_fn(
            self._handle,
            dispatch_out_x.data_ptr(),
            dispatch_out_topk_idx.data_ptr(),
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
            dispatch_out_x.element_size(),
            dispatch_out_scales.data_ptr() if dispatch_out_scales is not None else 0,
            packed_recv_scales.data_ptr() if packed_recv_scales is not None else 0,
            scale_bytes_per_token,
        )
        try:
            grid = (actual_bn,)
            block = (WARP_SIZE * actual_wpb,)
            self._launch_adapter(
                "mori_ConvertDispatchOutputKernel", grid, block, 0, stream, args_ptr
            )
        finally:
            mori_cpp.free_convert_args(args_ptr)

        packed_recv_count_ptr = mori_cpp.get_standard_moe_packed_recv_count_ptr(
            self._handle
        )
        packed_recv_count = from_gpu_ptr(
            packed_recv_count_ptr, (num_local_experts,), torch.int32
        )

        return (
            packed_recv_x,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
        )

    def convert_combine_input(
        self,
        packed_recv_x: torch.Tensor,
        packed_recv_src_info: torch.Tensor,
        packed_recv_layout_range: torch.Tensor,
        block_num: int = -1,
        warp_per_block: int = -1,
    ):
        build_fn = _cpp_dispatch_combine_factory(
            "build_convert_combine_input_args", allow_missing=True
        )
        if build_fn is None:
            raise RuntimeError(
                "convert_combine_input is not available. "
                "Rebuild with ENABLE_STANDARD_MOE_ADAPT=ON."
            )

        hidden_dim = packed_recv_x.size(2)
        actual_bn, _, actual_wpb = self._resolve_launch_params(
            block_num, 0, warp_per_block
        )
        stream = _current_stream()
        sfx = _DTYPE_SUFFIX[packed_recv_x.dtype]

        args_ptr = build_fn(
            self._handle,
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr(),
            hidden_dim,
        )
        try:
            grid = (actual_bn,)
            block = (WARP_SIZE * actual_wpb,)
            self._launch_adapter(
                f"ConvertCombineInputKernel_{sfx}", grid, block, 0, stream, args_ptr
            )
        finally:
            mori_cpp.free_convert_args(args_ptr)

        max_recv = self._cpp_config.max_num_tokens_to_recv()
        combine_input_ptr = mori_cpp.get_combine_input_ptr(self._handle)
        return from_gpu_ptr(
            combine_input_ptr, (max_recv, hidden_dim), packed_recv_x.dtype
        )

    def reset(self):
        self._reset_func(self._handle, _current_stream())

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.config.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()

        if self.config.kernel_type.value in (
            EpDispatchCombineKernelType.IntraNode.value,
            EpDispatchCombineKernelType.IntraNodeLL.value,
            EpDispatchCombineKernelType.InterNodeV1.value,
            EpDispatchCombineKernelType.InterNodeV1LL.value,
            EpDispatchCombineKernelType.AsyncLL.value,
        ):
            ptr, size = self._get_dispatch_src_token_pos_func(self._handle)
            return from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_sender_token_idx_map_func(self._handle)
        dispatch_sender_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        ptr, size = self._get_dispatch_receiver_token_idx_map_func(self._handle)
        dispatch_receiver_token_id_map = from_gpu_ptr(ptr, (size,), TOPK_IDX_DTYPE)

        max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        all_rank_sender_map = self._allgather_with_token_num_padding(
            dispatch_sender_token_id_map.cpu().to(torch.int64),
            self.config.max_num_inp_token_per_rank * self.config.num_experts_per_token,
        )

        cur_rank_num_token = self._get_cur_rank_num_token(self._handle)
        all_rank_num_token = [torch.empty(1) for i in range(self.config.world_size)]
        dist.all_gather(all_rank_num_token, torch.Tensor([cur_rank_num_token]))

        reverse_sender_token_id_map = {}
        for r in range(self.config.world_size):
            for i, mapped_id in enumerate(
                all_rank_sender_map[r].tolist()[
                    : int(all_rank_num_token[r][0].item())
                    * self.config.num_experts_per_token
                ]
            ):
                dest_pe = mapped_id // max_num_token_to_send_per_rank
                if dest_pe != self.config.rank:
                    continue
                mapped_id = (
                    mapped_id
                    - dest_pe * max_num_token_to_send_per_rank
                    + r * max_num_token_to_send_per_rank
                )
                reverse_sender_token_id_map[mapped_id] = (
                    i // self.config.num_experts_per_token
                )
        src_token_pos = []
        for i, recv_mapped_id in enumerate(dispatch_receiver_token_id_map.tolist()):
            src_pe = recv_mapped_id // max_num_token_to_send_per_rank
            if recv_mapped_id not in reverse_sender_token_id_map:
                print(
                    f"Warning: rank {self.config.rank} src_pe {src_pe} max_num_token_to_send_per_rank {max_num_token_to_send_per_rank} recv_mapped_id {recv_mapped_id} not in reverse_sender_token_id_map"
                )
                raise
            src_tok_id = reverse_sender_token_id_map[recv_mapped_id]
            # Match FlatTokenIndex used by the V1/LL kernels so callers can
            # decode every implementation with decode_send_flat_idx().  The
            # legacy reconstruction previously used the send-buffer slot
            # stride, which made EP>1 source token IDs decode out of bounds.
            src_token_pos.append(src_pe * self.max_num_tokens_to_send() + src_tok_id)

        return torch.tensor(src_token_pos, dtype=torch.int)

    def get_debug_time_buf(self):
        """Get the debug time buffer as a torch.Tensor (int64)."""
        ptr, size = mori_cpp.get_debug_time_buf(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int64)

    def get_debug_time_offset(self):
        """Get the debug time offset buffer as a torch.Tensor (int32)."""
        ptr, size = mori_cpp.get_debug_time_offset(self._handle)
        return from_gpu_ptr(ptr, (size,), torch.int32)
