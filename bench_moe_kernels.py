# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark MoE fused expert kernels: bf16 vs fp8 vs nvfp4.

Compares throughput of fused MoE expert dispatch + GEMM across quantization
schemes. Default shapes match Gemma 4 26B-A4B (128 experts, top-8, hidden=2816,
intermediate=2112).
"""

import argparse

import torch

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton
from vllm.v1.worker.workspace import init_workspace_manager

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

DEFAULT_BATCH_SIZES = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_PROVIDERS = ["torch-bf16", "triton-fp8", "triton-fp8-block"]

_RESULTS: dict[int, dict[str, float]] = {}


def make_fp8_weights(w_bf16, num_experts):
    w_fp8 = torch.empty_like(w_bf16, dtype=torch.float8_e4m3fn)
    w_scale = torch.empty((num_experts, 1, 1), device=w_bf16.device,
                          dtype=torch.float32)
    for e in range(num_experts):
        w_fp8[e], w_scale[e] = ops.scaled_fp8_quant(w_bf16[e])
    return w_fp8, w_scale


def make_fp8_block_weights(w_bf16, num_experts, block_shape):
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
    E, N, K = w_bf16.shape
    w_fp8 = torch.empty_like(w_bf16, dtype=torch.float8_e4m3fn)
    block_n, block_k = block_shape
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    w_scale = torch.empty((E, n_tiles, k_tiles), device=w_bf16.device,
                          dtype=torch.float32)
    for e in range(num_experts):
        w_fp8[e], w_scale[e] = per_block_cast_to_fp8(
            w_bf16[e], block_size=block_shape, use_ue8m0=False)
    return w_fp8, w_scale


def make_nvfp4_weights(w_bf16, num_experts):
    E, N, K = w_bf16.shape
    w_fp4 = torch.empty((E, N, K // 2), device=w_bf16.device,
                        dtype=torch.uint8)
    w_blockscale = torch.empty((E, N, K // 16), device=w_bf16.device,
                               dtype=torch.float8_e4m3fn)
    w_gs = torch.empty((E,), device=w_bf16.device, dtype=torch.float32)
    for e in range(num_experts):
        amax = w_bf16[e].abs().max().to(torch.float32)
        w_gs[e] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
        w_fp4[e], w_blockscale[e] = ops.scaled_fp4_quant(w_bf16[e], w_gs[e])
    return w_fp4, w_blockscale, w_gs


def build_benchmark(batch_sizes, providers):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=batch_sizes,
            x_log=False,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel="TFLOP/s",
            plot_name="MoE fused experts: BF16 vs FP8 vs NVFP4",
            args={},
        )
    )
    def benchmark(batch_size, provider, num_experts, topk, hidden_size,
                  intermediate_size):
        m = batch_size
        E = num_experts
        K = hidden_size
        N = intermediate_size
        device = "cuda"
        dtype = torch.bfloat16

        x = torch.randn((m, K), device=device, dtype=dtype)
        score = torch.randn((m, E), device=device, dtype=torch.float32)
        topk_weights, topk_ids, _ = fused_topk(x, score, topk,
                                               renormalize=True)

        # FLOPs: topk experts per token, each does two GEMMs:
        #   w1: (m_expert, K) x (K, 2*N) and w2: (m_expert, N) x (N, K)
        # Total per token: topk * (2*K*2*N + 2*N*K) = topk * 2*K*N * 3
        # But actually: w1 is gate+up fused (2*N output), w2 is down (K output)
        # w1 FLOP = 2 * m * K * 2N, w2 FLOP = 2 * m * N * K
        # per token with topk experts: topk * (2*K*2N + 2*N*K) = topk * 6*K*N
        total_flops = m * topk * 6 * K * N

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch-bf16":
            w1 = torch.randn((E, 2 * N, K), device=device, dtype=dtype)
            w2 = torch.randn((E, K, N), device=device, dtype=dtype)
            run = lambda: fused_experts(x, w1, w2, topk_weights, topk_ids)
        elif provider == "triton-fp8":
            w1_bf16 = torch.randn((E, 2 * N, K), device=device, dtype=dtype)
            w2_bf16 = torch.randn((E, K, N), device=device, dtype=dtype)
            w1_fp8, w1_scale = make_fp8_weights(w1_bf16, E)
            w2_fp8, w2_scale = make_fp8_weights(w2_bf16, E)
            w1_fp8 = w1_fp8.transpose(1, 2)
            w2_fp8 = w2_fp8.transpose(1, 2)
            _, a_scale = ops.scaled_fp8_quant(x)
            qc = fp8_w8a8_moe_quant_config(
                w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a_scale)
            run = lambda: fused_experts(x, w1_fp8, w2_fp8, topk_weights,
                                        topk_ids, quant_config=qc)
        elif provider == "triton-fp8-block":
            block_shape = [128, 128]
            w1_bf16 = torch.randn((E, 2 * N, K), device=device, dtype=dtype)
            w2_bf16 = torch.randn((E, K, N), device=device, dtype=dtype)
            w1_fp8, w1_scale = make_fp8_block_weights(w1_bf16, E, block_shape)
            w2_fp8, w2_scale = make_fp8_block_weights(w2_bf16, E, block_shape)
            _, a_scale = ops.scaled_fp8_quant(x)
            qc = FusedMoEQuantConfig.make(
                quant_dtype=torch.float8_e4m3fn,
                w1_scale=w1_scale, w2_scale=w2_scale,
                a1_scale=a_scale, a2_scale=a_scale,
                block_shape=block_shape)
            run = lambda: fused_experts(x, w1_fp8, w2_fp8, topk_weights,
                                        topk_ids, quant_config=qc)
        elif provider == "cutlass-nvfp4":
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from tests.kernels.moe.utils import make_dummy_moe_config
            from vllm.model_executor.layers.fused_moe.all2all_utils import (
                maybe_make_prepare_finalize,
            )
            from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
                CutlassExpertsFp4,
            )
            w1_bf16 = torch.randn((E, 2 * N, K), device=device, dtype=dtype)
            w2_bf16 = torch.randn((E, K, N), device=device, dtype=dtype)
            w1_fp4, w1_bs, w1_gs = make_nvfp4_weights(w1_bf16, E)
            w2_fp4, w2_bs, w2_gs = make_nvfp4_weights(w2_bf16, E)
            a1_gs = torch.ones((E,), device=device, dtype=torch.float32)
            a2_gs = torch.ones((E,), device=device, dtype=torch.float32)
            qc = nvfp4_moe_quant_config(
                a1_gscale=a1_gs, a2_gscale=a2_gs,
                w1_scale=w1_bs, w2_scale=w2_bs,
                g1_alphas=w1_gs, g2_alphas=w2_gs)
            moe_config = make_dummy_moe_config(
                num_experts=E, hidden_dim=K,
                intermediate_size_per_partition=N, in_dtype=dtype)
            kernel = mk.FusedMoEKernel(
                maybe_make_prepare_finalize(
                    moe=moe_config, quant_config=qc,
                    allow_new_interface=True, use_monolithic=False),
                CutlassExpertsFp4(moe_config=moe_config, quant_config=qc))
            run = lambda: kernel(hidden_states=x, w1=w1_fp4, w2=w2_fp4,
                                 topk_weights=topk_weights, topk_ids=topk_ids)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)

        def to_tflops(t_ms):
            return total_flops * 1e-12 / (t_ms * 1e-3)

        median_tflops = to_tflops(ms)
        _RESULTS.setdefault(batch_size, {})
        _RESULTS[batch_size][provider] = (
            _RESULTS[batch_size].get(provider, 0.0) + median_tflops
        )

        return median_tflops, to_tflops(max_ms), to_tflops(min_ms)

    return benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MoE fused expert kernels.")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2816)
    parser.add_argument("--intermediate-size", type=int, default=2112)
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--providers", nargs="+",
                        choices=DEFAULT_PROVIDERS + ["cutlass-nvfp4"],
                        default=DEFAULT_PROVIDERS)
    args = parser.parse_args()

    torch.set_default_device("cuda")

    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1))
    with set_current_vllm_config(vllm_config):
        init_workspace_manager(1, False, 0)

        print(f"MoE config: E={args.num_experts}, topk={args.topk}, "
              f"K={args.hidden_size}, N={args.intermediate_size}")
        print(f"Providers: {args.providers}")
        print()

        benchmark = build_benchmark(args.batch_sizes, args.providers)
        benchmark.run(
            print_data=True,
            num_experts=args.num_experts,
            topk=args.topk,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
        )

    if _RESULTS:
        providers = args.providers
        bf16_key = "torch-bf16"
        batch_sizes = sorted(_RESULTS.keys())

        print("\n=== Summed TFLOP/s ===")
        header = f"{'batch_size':>12}" + "".join(f"{p:>20}" for p in providers)
        print(header)
        for bs in batch_sizes:
            row = f"{bs:>12}"
            for p in providers:
                row += f"{_RESULTS[bs].get(p, 0.0):>20.2f}"
            print(row)

        if bf16_key in providers:
            print(f"\n=== Speedup vs {bf16_key} ===")
            header = f"{'batch_size':>12}" + "".join(
                f"{p:>20}" for p in providers)
            print(header)
            for bs in batch_sizes:
                bf16_val = _RESULTS[bs].get(bf16_key, 1.0)
                row = f"{bs:>12}"
                for p in providers:
                    ratio = (_RESULTS[bs].get(p, 0.0) / bf16_val
                             if bf16_val else 0.0)
                    row += f"{ratio:>19.2f}x"
                print(row)

    print("\nBenchmark finished!")


if __name__ == "__main__":
    main()
