# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Continuous multi-node bitwise validation for production InterNodeV2LL."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mori
import torch
import torch.distributed as dist


def _fp8_dtype():
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def _dispatch_dtype(name):
    return torch.bfloat16 if name == "bf16" else _fp8_dtype()


def _all_gather(tensor, world_size):
    original_device = tensor.device
    original_dtype = tensor.dtype
    original_shape = tensor.shape
    host_bytes = tensor.cpu().contiguous().view(torch.uint8)
    gathered_bytes = [torch.empty_like(host_bytes) for _ in range(world_size)]
    dist.all_gather(gathered_bytes, host_bytes)
    gathered = [
        value.view(original_dtype).view(original_shape) for value in gathered_bytes
    ]
    if original_device.type == "cuda":
        return [value.to(original_device) for value in gathered]
    return gathered


def _sha256_file(path):
    path = Path(path)
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_code_sha256():
    from mori.ops._jit_loader import _compiled_hsaco

    repo_root = Path(__file__).resolve().parents[3]
    return {
        "internode_v2.cpp": _sha256_file(
            repo_root / "src/ops/dispatch_combine/internode_v2.cpp"
        ),
        "dispatch_combine.py": _sha256_file(
            repo_root / "python/mori/ops/dispatch_combine.py"
        ),
        "ep_internode_v2.hsaco": _sha256_file(
            _compiled_hsaco.get("ep_internode_v2", "")
        ),
    }


def _stress_token_count(round_id, rank, max_tokens):
    """Make every rank visit every count in [0, max_tokens] over max_tokens + 1 rounds."""
    return (round_id * 37 + rank * 17) % (max_tokens + 1)


def _stress_source_x(round_id, rank, max_tokens, hidden_dim, dtype, device):
    token = torch.arange(max_tokens, dtype=torch.int32, device=device)[:, None]
    hidden = torch.arange(hidden_dim, dtype=torch.int32, device=device)[None, :]
    # Multiples of 1/16 in [-14/16, 14/16] are exact in BF16 and FP8 E4M3.  Vary every hidden
    # element and every generation so stale slots, truncated vector copies, and late RDMA writes
    # cannot pass a scalar-only payload check.
    value = ((hidden + token * 5 + rank * 3 + round_id * 7) % 29 - 14).float()
    return (value * (1.0 / 16.0)).to(torch.bfloat16).to(dtype)


def _stress_routes(round_id, rank, max_tokens, args, device):
    token = torch.arange(max_tokens, dtype=torch.int32, device=device)[:, None]
    route = torch.arange(args.topk, dtype=torch.int32, device=device)[None, :]
    world_size = args.nodes * args.gpu_per_node
    experts_per_rank = args.num_experts // world_size
    pattern = round_id % 7
    if pattern == 0:
        indices = (
            rank * 19 + token * 13 + route * 31 + round_id * 7
        ) % args.num_experts
    elif pattern == 1:
        # Exercise the tail counters as well as the common low expert IDs. This catches
        # block-0 reset implementations that accidentally use a full-grid stride.
        indices = (
            rank * experts_per_rank
            + (
                max(experts_per_rank - args.topk, 0)
                + route.expand(max_tokens, -1)
            )
            % experts_per_rank
        )
    elif pattern == 2:
        remote_rank = (rank + args.gpu_per_node) % world_size
        indices = remote_rank * experts_per_rank + route.expand(max_tokens, -1)
    elif pattern == 3:
        indices = route.expand(max_tokens, -1).clone()
    elif pattern == 4:
        dest_rank = (rank + round_id) % world_size
        indices = dest_rank * experts_per_rank + route.expand(max_tokens, -1)
    elif pattern == 5:
        indices = (rank * 23 + token * 11 + route * 29 + round_id) % args.num_experts
        indices = indices.clone()
        indices[(token + route + round_id) % 3 == 0] = -1
    else:
        local_node = rank // args.gpu_per_node
        num_nodes = world_size // args.gpu_per_node
        remote_node = (local_node + 1) % num_nodes
        local_rank_in_node = (rank + route) % args.gpu_per_node
        remote_rank_in_node = (rank + route * 3 + 1) % args.gpu_per_node
        dest_rank = torch.where(
            route < (args.topk // 2),
            local_node * args.gpu_per_node + local_rank_in_node,
            remote_node * args.gpu_per_node + remote_rank_in_node,
        )
        indices = (
            dest_rank.expand(max_tokens, -1) * experts_per_rank
            + (token + route * 5) % experts_per_rank
        )
    # Powers of two make the arithmetic exactly representable while still varying the route
    # weights. This lets the Torch model check the transport/reduction order bit-for-bit rather
    # than hiding errors behind a tolerance.
    exponent = ((token + route + round_id + rank) % 5 + 1).float()
    weights = torch.pow(torch.tensor(2.0, device=device), -exponent)
    return indices.contiguous(), weights.contiguous()


def _stress_scales(round_id, rank, max_tokens, scale_dim, device):
    if scale_dim <= 0:
        return None
    token = torch.arange(max_tokens, dtype=torch.int32, device=device)[:, None]
    scale = torch.arange(scale_dim, dtype=torch.int32, device=device)[None, :]
    exponent = ((token + scale + round_id + rank) % 4).float()
    return torch.pow(torch.tensor(2.0, device=device), -exponent).contiguous()


def _bitwise_equal(actual, expected):
    if actual.dtype == torch.bfloat16:
        return torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if actual.element_size() == 1:
        return torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    if actual.dtype == torch.float32:
        return torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    return torch.equal(actual, expected)


def _raise_bitwise_mismatch(label, rank, round_id, actual, expected):
    if _bitwise_equal(actual, expected):
        return
    mismatch = actual.view(torch.uint8) != expected.view(torch.uint8)
    first_byte = int(torch.nonzero(mismatch.flatten(), as_tuple=False)[0])
    value_mismatch = actual != expected
    detail = ""
    if bool(value_mismatch.any()):
        first_value = torch.nonzero(value_mismatch, as_tuple=False)[0]
        index = tuple(int(value) for value in first_value)
        detail = (
            f", first_value_index={index}, actual={float(actual[index])}, "
            f"expected={float(expected[index])}"
        )
        if "scales" in label and actual.ndim == 2:
            row = index[0]
            detail += (
                f", actual_row={actual[row].cpu().tolist()}, "
                f"expected_row={expected[row].cpu().tolist()}"
            )
    raise AssertionError(
        f"{label} bitwise mismatch: rank={rank}, round={round_id}, "
        f"first_byte={first_byte}, shape={tuple(actual.shape)}, dtype={actual.dtype}{detail}"
    )


def _stress_expected_rows(source_ranks, token_ids, round_id, args, dtype, device):
    source_ranks = torch.as_tensor(
        source_ranks, device=device, dtype=torch.int32
    ).flatten()[:, None]
    token_ids = torch.as_tensor(token_ids, device=device, dtype=torch.int32).flatten()[
        :, None
    ]
    hidden = torch.arange(args.hidden_dim, dtype=torch.int32, device=device)[None, :]
    value = (
        (hidden + token_ids * 5 + source_ranks * 3 + round_id * 7) % 29 - 14
    ).float()
    return (value * (1.0 / 16.0)).to(torch.bfloat16).to(dtype)


def _verify_stress_dispatch(
    rank,
    round_id,
    dispatch_result,
    all_counts,
    all_indices,
    all_weights,
    all_scales,
    args,
    layout,
):
    world_size = args.nodes * args.gpu_per_node
    experts_per_rank = args.num_experts // world_size
    source_stride = world_size * args.max_tokens
    device = dispatch_result[0].device
    dtype = dispatch_result[0].dtype
    all_indices_cpu = [value.cpu() for value in all_indices]
    if layout == "token_major":
        (
            token_x,
            token_count,
            token_weights,
            token_indices,
            token_scales,
            token_src,
        ) = dispatch_result
        expected = []
        for source_rank in range(world_size):
            route_pes = (
                all_indices_cpu[source_rank][: all_counts[source_rank]]
                // experts_per_rank
            )
            for token_id in range(all_counts[source_rank]):
                if bool((route_pes[token_id] == rank).any()):
                    expected.append((source_rank, token_id))
        actual_count = int(token_count[0].item())
        if actual_count != len(expected):
            raise AssertionError(
                f"token-major count mismatch: rank={rank}, round={round_id}, "
                f"actual={actual_count}, expected={len(expected)}"
            )
        flat = token_src[:actual_count].cpu().tolist()
        sources = [divmod(value, source_stride) for value in flat]
        if len(set(sources)) != actual_count or set(sources) != set(expected):
            raise AssertionError(
                f"token-major source map mismatch: rank={rank}, round={round_id}"
            )
        if actual_count:
            expected_x = _stress_expected_rows(
                [source_rank for source_rank, _ in sources],
                [token_id for _, token_id in sources],
                round_id,
                args,
                dtype,
                device,
            )
            _raise_bitwise_mismatch(
                "Dispatch token-major X",
                rank,
                round_id,
                token_x[:actual_count],
                expected_x,
            )
            expected_weights = torch.stack([all_weights[s][t] for s, t in sources])
            expected_indices = torch.stack([all_indices[s][t] for s, t in sources])
            _raise_bitwise_mismatch(
                "Dispatch token-major weights",
                rank,
                round_id,
                token_weights[:actual_count],
                expected_weights,
            )
            _raise_bitwise_mismatch(
                "Dispatch token-major indices",
                rank,
                round_id,
                token_indices[:actual_count],
                expected_indices,
            )
            if token_scales is not None:
                expected_scales = torch.stack([all_scales[s][t] for s, t in sources])
                _raise_bitwise_mismatch(
                    "Dispatch token-major scales",
                    rank,
                    round_id,
                    token_scales[:actual_count],
                    expected_scales,
                )
        return

    packed_x, packed_count, packed_weights, packed_scales, packed_src = dispatch_result
    routed_experts = torch.cat(
        [
            all_indices[source_rank][: all_counts[source_rank]].flatten()
            for source_rank in range(world_size)
        ]
    )
    routed_experts = routed_experts[routed_experts >= 0]
    expected_counts = (
        torch.bincount(routed_experts.to(torch.int64), minlength=args.num_experts)
        .cpu()
        .tolist()
    )
    for local_expert in range(experts_per_rank):
        expert = rank * experts_per_rank + local_expert
        expected_count = expected_counts[expert]
        actual_count = int(packed_count[local_expert].item())
        if actual_count != expected_count:
            raise AssertionError(
                f"expert count mismatch: rank={rank}, round={round_id}, expert={expert}, "
                f"actual={actual_count}, expected={expected_count}"
            )
        if not actual_count:
            continue
        flat = packed_src[local_expert, :actual_count].cpu().tolist()
        sources = [divmod(value, source_stride) for value in flat]
        if len(set(sources)) != actual_count:
            raise AssertionError(
                f"duplicate expert source: rank={rank}, round={round_id}, expert={expert}"
            )
        route_slots = []
        for source_rank, token_id in sources:
            if source_rank >= world_size or token_id >= all_counts[source_rank]:
                raise AssertionError(
                    f"invalid source: rank={rank}, round={round_id}, source={(source_rank, token_id)}"
                )
            matches = torch.nonzero(
                all_indices_cpu[source_rank][token_id] == expert, as_tuple=False
            ).flatten()
            if matches.numel() != 1:
                raise AssertionError(
                    f"route map mismatch: rank={rank}, round={round_id}, expert={expert}, "
                    f"source={(source_rank, token_id)}, matches={matches.numel()}"
                )
            route_slots.append(int(matches[0]))
        expected_x = _stress_expected_rows(
            [source_rank for source_rank, _ in sources],
            [token_id for _, token_id in sources],
            round_id,
            args,
            dtype,
            device,
        )
        _raise_bitwise_mismatch(
            "Dispatch expert-major X",
            rank,
            round_id,
            packed_x[local_expert, :actual_count],
            expected_x,
        )
        expected_weights = torch.stack(
            [all_weights[s][t, route] for (s, t), route in zip(sources, route_slots)]
        )
        _raise_bitwise_mismatch(
            "Dispatch expert-major weights",
            rank,
            round_id,
            packed_weights[local_expert, :actual_count],
            expected_weights,
        )
        if packed_scales is not None:
            expected_scales = torch.stack([all_scales[s][t] for s, t in sources])
            _raise_bitwise_mismatch(
                f"Dispatch expert-major scales expert={expert} sources={sources}",
                rank,
                round_id,
                packed_scales[local_expert, :actual_count],
                expected_scales,
            )


def _expert_factor(indices):
    valid = indices >= 0
    exponent = torch.where(valid, indices, torch.zeros_like(indices)) % 4
    factor = torch.pow(torch.tensor(2.0, device=indices.device), -exponent.float())
    return torch.where(valid, factor, torch.zeros_like(factor))


def _fill_stress_combine_input(rank, dispatch_result, combine_input, args, layout):
    world_size = args.nodes * args.gpu_per_node
    experts_per_rank = args.num_experts // world_size
    combine_input.zero_()
    if layout == "token_major":
        token_x, token_count, token_weights, token_indices, _, _ = dispatch_result
        count = int(token_count[0].item())
        if not count:
            return
        local_begin = rank * experts_per_rank
        local_end = local_begin + experts_per_rank
        acc = torch.zeros(
            (count, args.hidden_dim), dtype=torch.float32, device=combine_input.device
        )
        for route in range(args.topk):
            expert = token_indices[:count, route]
            mask = (expert >= local_begin) & (expert < local_end)
            expert_out = (token_x[:count].float() * _expert_factor(expert)[:, None]).to(
                torch.bfloat16
            )
            acc.add_(
                expert_out.float()
                * (token_weights[:count, route] * mask.float())[:, None]
            )
        combine_input[:count].copy_(acc.to(torch.bfloat16))
        return
    packed_x, packed_count, _, _, _ = dispatch_result
    counts = packed_count.cpu().tolist()
    for local_expert, count in enumerate(counts):
        if count:
            expert = rank * experts_per_rank + local_expert
            factor = 2.0 ** (-(expert % 4))
            combine_input[local_expert, :count].copy_(
                (packed_x[local_expert, :count].float() * factor).to(torch.bfloat16)
            )


def _torch_v2ll_reference(
    source_x, weights, indices, rank, args, layout, factor_mode="stress"
):
    world_size = args.nodes * args.gpu_per_node
    experts_per_rank = args.num_experts // world_size
    my_node = rank // args.gpu_per_node
    expert_out = []
    for route in range(args.topk):
        if factor_mode == "main":
            expert = indices[:, route]
            factor = torch.where(
                expert >= 0,
                1.0 + expert.float() * 0.001,
                torch.zeros_like(expert, dtype=torch.float32),
            )
        else:
            factor = _expert_factor(indices[:, route])
        expert_out.append((source_x.float() * factor[:, None]).to(torch.bfloat16))
    node_partials = []
    for node in range(args.nodes):
        node_acc = torch.zeros_like(source_x, dtype=torch.float32)
        if layout == "expert_major":
            for route in range(args.topk):
                route_node = (
                    indices[:, route] // experts_per_rank
                ) // args.gpu_per_node
                valid = indices[:, route] >= 0
                scale = weights[:, route] * (valid & (route_node == node)).float()
                node_acc.add_(expert_out[route].float() * scale[:, None])
        else:
            # Token-major Group GEMM preprocessing reduces all routes for one destination PE,
            # casts that row to BF16, and V2LL then adds the first occurrence of each PE once.
            for first_route in range(args.topk):
                pe = indices[:, first_route] // experts_per_rank
                valid_first = indices[:, first_route] >= 0
                earlier_same_pe = torch.zeros_like(valid_first)
                for earlier in range(first_route):
                    earlier_same_pe |= (indices[:, earlier] >= 0) & (
                        indices[:, earlier] // experts_per_rank == pe
                    )
                first_for_pe = valid_first & ~earlier_same_pe
                pe_row = torch.zeros_like(source_x, dtype=torch.float32)
                for route in range(args.topk):
                    same_pe = (indices[:, route] >= 0) & (
                        indices[:, route] // experts_per_rank == pe
                    )
                    pe_row.add_(
                        expert_out[route].float()
                        * (weights[:, route] * same_pe.float())[:, None]
                    )
                pe_row = pe_row.to(torch.bfloat16)
                pe_node = pe // args.gpu_per_node
                node_acc.add_(
                    pe_row.float() * (first_for_pe & (pe_node == node)).float()[:, None]
                )
        node_partials.append(node_acc.to(torch.bfloat16))
    result = node_partials[my_node]
    for step in range(1, args.nodes):
        remote_node = (my_node + step) % args.nodes
        result = (result.float() + node_partials[remote_node].float()).to(
            torch.bfloat16
        )
    return result


def _run_variable_stress(op, rank, dispatch_dtype, args, layout):
    world_size = args.nodes * args.gpu_per_node
    device = torch.device("cuda", rank % args.gpu_per_node)
    combine_input = op.get_v2_registered_combine_input_buffer(
        torch.bfloat16, args.hidden_dim
    )
    covered = set()
    for round_id in range(args.variable_stress_rounds):
        count = _stress_token_count(round_id, rank, args.max_tokens)
        covered.add(count)
        all_counts = [
            _stress_token_count(round_id, source_rank, args.max_tokens)
            for source_rank in range(world_size)
        ]
        all_indices = []
        all_weights = []
        all_scales = (
            [] if dispatch_dtype != torch.bfloat16 and args.scale_dim > 0 else None
        )
        for source_rank in range(world_size):
            source_indices, source_weights = _stress_routes(
                round_id, source_rank, args.max_tokens, args, device
            )
            all_indices.append(source_indices)
            all_weights.append(source_weights)
            if all_scales is not None:
                all_scales.append(
                    _stress_scales(
                        round_id,
                        source_rank,
                        args.max_tokens,
                        args.scale_dim,
                        device,
                    )
                )
        source_x = _stress_source_x(
            round_id,
            rank,
            args.max_tokens,
            args.hidden_dim,
            dispatch_dtype,
            device,
        )[:count]
        source_indices = all_indices[rank][:count]
        source_weights = all_weights[rank][:count]
        source_scales = all_scales[rank][:count] if all_scales is not None else None
        launch_kwargs = (
            {}
            if args.use_api_defaults
            else {"block_num": args.blocks, "warp_per_block": args.warps}
        )
        dispatch_result = op.dispatch_v2_standard_moe(
            source_x,
            source_weights,
            source_scales,
            source_indices,
            **launch_kwargs,
        )
        torch.cuda.synchronize()
        _verify_stress_dispatch(
            rank,
            round_id,
            dispatch_result,
            all_counts,
            all_indices,
            all_weights,
            all_scales,
            args,
            layout,
        )
        _fill_stress_combine_input(rank, dispatch_result, combine_input, args, layout)
        combine_kwargs = {"rdma_block_num": 1}
        if not args.use_api_defaults:
            combine_kwargs.update(
                block_num=args.combine_blocks,
                warp_per_block=args.combine_warps,
            )
        output, _ = op.combine_standard_moe(
            combine_input,
            None,
            source_indices,
            **combine_kwargs,
        )
        torch.cuda.synchronize()
        expected = _torch_v2ll_reference(
            source_x.to(torch.bfloat16),
            source_weights,
            source_indices,
            rank,
            args,
            layout,
        )
        _raise_bitwise_mismatch(
            "Combine output", rank, round_id, output[:count], expected
        )
        if rank == 0 and (round_id + 1) % 32 == 0:
            print(
                "MORI_V2_STRESS_PROGRESS "
                + json.dumps(
                    {"round": round_id + 1, "total": args.variable_stress_rounds}
                ),
                flush=True,
            )
    covered_tensor = torch.zeros(args.max_tokens + 1, dtype=torch.int32)
    for count in covered:
        covered_tensor[count] = 1
    gathered = _all_gather(covered_tensor, world_size)
    if rank == 0:
        missing = [
            [index for index, value in enumerate(rank_coverage.tolist()) if not value]
            for rank_coverage in gathered
        ]
        print(
            "MORI_V2_STRESS_RESULT "
            + json.dumps(
                {
                    "bitwise": True,
                    "code_sha256": _runtime_code_sha256(),
                    "dispatch_dtype": args.dispatch_dtype,
                    "layout": layout,
                    "max_tokens": args.max_tokens,
                    "missing_counts_per_rank": missing,
                    "rounds": args.variable_stress_rounds,
                    "routing_patterns": 7,
                    "world_size": world_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _worker(local_rank, args):
    node_rank = int(os.environ["RANK"])
    world_size = args.nodes * args.gpu_per_node
    rank = node_rank * args.gpu_per_node + local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group("cpu:gloo", rank=rank, world_size=world_size)
    torch._C._distributed_c10d._register_process_group(
        "default", torch.distributed.group.WORLD
    )
    mori.shmem.shmem_torch_process_group_init("default")

    dispatch_dtype = _dispatch_dtype(args.dispatch_dtype)
    config = mori.ops.EpDispatchCombineConfig(
        data_type=dispatch_dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden_dim,
        scale_dim=args.scale_dim if dispatch_dtype != torch.bfloat16 else 0,
        scale_type_size=4,
        max_token_type_size=2,
        max_num_inp_token_per_rank=args.max_tokens,
        num_experts_per_rank=args.num_experts // world_size,
        num_experts_per_token=args.topk,
        warp_num_per_block=max(args.warps, args.combine_warps),
        block_num=args.blocks,
        use_external_inp_buf=True,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV2LL,
        gpu_per_node=args.gpu_per_node,
        rdma_block_num=1,
        num_qp_per_pe=args.rdma_qps,
        quant_type="none",
        v2_layout=args.layout,
        v2_copy_block_num=(0 if args.use_api_defaults else args.copy_blocks),
    )
    op = mori.ops.EpDispatchCombineOp(config)
    torch.cuda.set_device(local_rank)
    _run_variable_stress(op, rank, dispatch_dtype, args, args.layout)
    if args.validate_lifecycle:
        stream = torch.cuda.current_stream().cuda_stream
        mori.cpp.begin_v2_dispatch(op._handle, stream)
        mori.cpp.abort_v2_dispatch(op._handle)
        try:
            mori.cpp.begin_v2_dispatch(op._handle, stream)
        except RuntimeError as exc:
            if "poisoned-after-launch-failure" not in str(exc):
                raise
        else:
            raise AssertionError("V2 lifecycle reused a poisoned handle")
        if rank == 0:
            print("MORI_V2_ABORT_FAIL_CLOSED true", flush=True)
    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpu-per-node", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--rounds", dest="variable_stress_rounds", type=int, default=258
    )
    parser.add_argument("--dispatch-dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--layout", choices=("expert_major", "token_major"), default="expert_major"
    )
    parser.add_argument("--scale-dim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--copy-blocks", type=int, default=32)
    parser.add_argument("--combine-blocks", type=int, default=56)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--combine-warps", type=int, default=8)
    parser.add_argument("--validate-lifecycle", action="store_true")
    parser.add_argument("--rdma-qps", type=int, default=2)
    parser.add_argument("--use-api-defaults", action="store_true")
    args = parser.parse_args()
    if args.max_tokens > 128:
        raise ValueError("max_tokens must be <= 128")
    if args.variable_stress_rounds < args.max_tokens + 1:
        raise ValueError("rounds must cover at least one full 0..max_tokens cycle")
    world_size = args.nodes * args.gpu_per_node
    if args.num_experts % world_size:
        raise ValueError("num_experts must be divisible by world size")
    if not 1 <= args.rdma_qps <= 8:
        raise ValueError("rdma_qps must be in [1, 8]")
    if args.rdma_qps > min(args.warps, args.combine_warps):
        raise ValueError("each QP requires a sender/waiter warp")
    if args.copy_blocks < args.rdma_qps or args.blocks - args.copy_blocks < args.nodes:
        raise ValueError("invalid copy/scatter CTA split")
    torch.multiprocessing.spawn(
        _worker, args=(args,), nprocs=args.gpu_per_node, join=True
    )


if __name__ == "__main__":
    main()
