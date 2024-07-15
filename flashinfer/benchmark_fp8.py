# uv pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.9/flashinfer-0.0.9+cu121torch2.3-cp310-cp310-linux_x86_64.whl
import torch
import flashinfer
import time
import csv
import itertools


def benchmark_flashinfer(q_dtype, kv_dtype, sequence_lengths):
    num_iters = 10000
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128

    results = []

    for kv_len in sequence_lengths:
        q = torch.randn((num_qo_heads, head_dim), device="cuda:0").to(q_dtype)
        k = torch.randn((kv_len, num_kv_heads, head_dim), device="cuda:0").to(kv_dtype)
        v = torch.randn((kv_len, num_kv_heads, head_dim), device="cuda:0").to(kv_dtype)

        # Warm-up
        for _ in range(10):
            o = flashinfer.single_decode_with_kv_cache(q, k, v)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iters):
            o = flashinfer.single_decode_with_kv_cache(q, k, v)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        latency = (end_time - start_time) / num_iters * 1e6  # Convert to microseconds
        total_bytes = (
            q.nelement() * q.element_size()
            + k.nelement() * k.element_size()
            + v.nelement() * v.element_size()
            + o.nelement() * o.element_size()
        )
        bw_util = total_bytes / latency  # GB/s

        results.append(
            {
                "sequence_length": kv_len,
                "q_dtype": str(q_dtype),
                "kv_dtype": str(kv_dtype),
                "bw_util": bw_util,
                "latency": latency,
            }
        )

    return results


sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

all_results = []

# Run benchmarks for all combinations of fp16 and fp8 for q and kv, ensuring k and v have the same type
for q_dtype, kv_dtype in [
    [torch.float16, torch.float16],
    [torch.float16, torch.float8_e4m3fn],
    [torch.float8_e4m3fn, torch.float8_e4m3fn],
]:
    print(f"Running benchmark for q:{q_dtype}, k/v:{kv_dtype}")
    results = benchmark_flashinfer(q_dtype, kv_dtype, sequence_lengths)
    all_results.extend(results)

# Write results to CSV
with open("flashinfer_benchmark_results.csv", "w", newline="") as csvfile:
    fieldnames = ["sequence_length", "q_dtype", "kv_dtype", "bw_util", "latency"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

print("Benchmark results have been saved to 'flashinfer_benchmark_results.csv'")
