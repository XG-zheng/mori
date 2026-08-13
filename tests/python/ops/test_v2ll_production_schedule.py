# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

from types import SimpleNamespace

import pytest
import torch

from mori.ops import dispatch_combine


@pytest.mark.parametrize("model", ["mi355x", "mi350x"])
def test_v2ll_bf16_ep16_decode_schedule(monkeypatch, model):
    monkeypatch.setattr(
        "mori.ops.utils.detect_model", lambda: model
    )
    config = SimpleNamespace(
        kernel_type=dispatch_combine.EpDispatchCombineKernelType.InterNodeV2LL,
        world_size=16,
        gpu_per_node=8,
        hidden_dim=7168,
        num_experts_per_token=8,
    )

    assert dispatch_combine._v2ll_default_launch(
        config, 32, torch.bfloat16, is_dispatch=True
    ) == (128, 4, 32)
    assert dispatch_combine._v2ll_default_launch(
        config, 32, torch.bfloat16, is_dispatch=False
    ) == (56, 4, 0)
    assert dispatch_combine._v2ll_default_launch(
        config, 33, torch.bfloat16, is_dispatch=True
    ) == (160, 4, 32)
    assert dispatch_combine._v2ll_default_launch(
        config, 33, torch.bfloat16, is_dispatch=False
    ) == (112, 4, 0)
