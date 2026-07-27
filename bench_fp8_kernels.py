# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark FP8 schemes through the linear-kernel selector.

Weight quantization and backend-specific preprocessing happen before timing. The
timed path includes activation quantization and GEMM through ``apply_weights``.
"""

import argparse
import copy
import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import per_block_cast_to_fp8


@dataclass(frozen=True)
class FP8Scheme:
    weight_quant_key: QuantKey
    activation_quant_key: QuantKey


PROVIDER_CONFIGS = {
    "fp8-per-tensor": FP8Scheme(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8DynamicTokenSym,
    ),
    "fp8-static-per-tensor": FP8Scheme(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8StaticTensorSym,
    ),
    "fp8-per-channel": FP8Scheme(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
    ),
    "fp8-per-block": FP8Scheme(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
    ),
}
DEFAULT_PROVIDERS = ["torch-bf16", *PROVIDER_CONFIGS]
DEFAULT_BATCH_SIZES = [
    1,
    16,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
]
_SEEN_BACKENDS: set[tuple[str, int, int]] = set()


class FP8Linear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        example_input: torch.Tensor,
        scheme: FP8Scheme,
    ) -> None:
        super().__init__()
        n, k = weight.shape
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.input_scale_ub = None
        self.weight_block_size = None

        weight_group_shape = scheme.weight_quant_key.scale.group_shape
        is_block_wise = scheme.activation_quant_key.scale.group_shape.is_per_group()
        if is_block_wise:
            block_size = [weight_group_shape.row, weight_group_shape.col]
            qweight, weight_scale = per_block_cast_to_fp8(
                weight, block_size=block_size, use_ue8m0=False
            )
            self.weight = torch.nn.Parameter(qweight, requires_grad=False)
            self.weight_scale = None
            self.weight_scale_inv = torch.nn.Parameter(
                weight_scale, requires_grad=False
            )
            self.weight_block_size = block_size
        else:
            per_channel = weight_group_shape.is_per_channel()
            qweight, weight_scale = ops.scaled_fp8_quant(
                weight, use_per_token_if_dynamic=per_channel
            )
            self.weight = torch.nn.Parameter(qweight.t(), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
            self.weight_scale_inv = None

        if scheme.activation_quant_key.scale.static:
            _, input_scale = ops.scaled_fp8_quant(example_input)
            self.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)
        else:
            self.input_scale = None

        self.kernel = init_fp8_linear_kernel(
            activation_quant_key=scheme.activation_quant_key,
            weight_quant_key=scheme.weight_quant_key,
            weight_shape=(n, k),
            input_dtype=example_input.dtype,
            out_dtype=example_input.dtype,
            module_name=type(self).__name__,
        )
        self.kernel.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self, x)


def describe_kernel(kernel) -> str:
    description = type(kernel).__name__
    branches = [
        type(getattr(kernel, name)).__name__
        for name in ("base", "fallback")
        if hasattr(kernel, name)
    ]
    if branches:
        description += f" ({' / '.join(branches)})"
    return description


def build_benchmark(batch_sizes: list[int], providers: list[str]):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=batch_sizes,
            x_log=False,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="TFLOP/s (larger is better)",
            plot_name="BF16 vs dispatched FP8 linear kernels",
            args={},
        )
    )
    def benchmark(batch_size, provider, N, K):
        m = batch_size
        device = "cuda"
        dtype = torch.bfloat16
        x = torch.randn((m, K), device=device, dtype=dtype)
        weight = torch.randn((N, K), device=device, dtype=dtype)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch-bf16":
            run = lambda: torch.nn.functional.linear(x, weight)
        else:
            layer = FP8Linear(weight, x, PROVIDER_CONFIGS[provider])
            backend_key = (provider, N, K)
            if backend_key not in _SEEN_BACKENDS:
                print(f"  {provider}: {describe_kernel(layer.kernel)}")
                _SEEN_BACKENDS.add(backend_key)
            run = lambda: layer(x)

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(run, quantiles=quantiles)

        def to_tflops(t_ms):
            return (2 * m * N * K) * 1e-12 / (t_ms * 1e-3)

        return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)

    return benchmark


def prepare_shapes(args: argparse.Namespace) -> list[tuple[int, int, str]]:
    if args.shape:
        return [(k, n, "custom") for n, k in args.shape]

    shapes = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for kn, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            kn[tp_dim] //= tp_size
            shapes.append((*kn, model))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FP8 linear schemes through vLLM kernel dispatch."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        choices=list(WEIGHT_SHAPES),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--shape",
        action="append",
        nargs=2,
        type=int,
        metavar=("N", "K"),
        help="Benchmark a direct N K shape; may be repeated and overrides --models.",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=DEFAULT_PROVIDERS,
        default=DEFAULT_PROVIDERS,
    )
    parser.add_argument(
        "--linear-backend",
        default="auto",
        help="Set the vLLM linear backend selector (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not current_platform.is_cuda():
        raise RuntimeError("This benchmark requires a CUDA device with FP8 support.")

    benchmark = build_benchmark(args.batch_sizes, args.providers)
    vllm_config = VllmConfig(
        kernel_config=KernelConfig(linear_backend=args.linear_backend)
    )
    vllm_config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=None)
    )

    with set_current_vllm_config(vllm_config):
        for K, N, model in prepare_shapes(args):
            print(f"{model}, N={N} K={K}, BF16 vs FP8 linear TFLOP/s:")
            benchmark.run(print_data=True, N=N, K=K)

    print("Benchmark finished!")


if __name__ == "__main__":
    main()