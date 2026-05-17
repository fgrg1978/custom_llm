"""
Benchmark simple de GPU: valida CUDA y mide TFLOPS reales en matmul.

Uso: python cloud/gpu_bench.py

Mide fp32, tf32 y bf16 con multiplicaciones de matrices grandes.
FLOPs de un matmul NxN = 2 * N^3. TFLOPS = FLOPs / tiempo / 1e12.
"""

import time
import torch


def bench(dtype, n=8192, iters=50, tf32=False):
    """Mide TFLOPS para matmul NxN en el dtype dado."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    # Warmup
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12
    return tflops


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA no disponible")
        return

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    mem_gb = props.total_memory / 1e9

    print("=" * 55)
    print(f"  GPU: {name}")
    print(f"  Compute capability: {cap[0]}.{cap[1]}")
    print(f"  Memoria: {mem_gb:.0f} GB")
    print(f"  Multiprocessors (SM): {props.multi_processor_count}")
    print("=" * 55)
    print()
    print("Benchmark matmul 8192x8192 (TFLOPS reales alcanzados):")
    print()

    results = {
        "fp32":          bench(torch.float32, tf32=False),
        "tf32":          bench(torch.float32, tf32=True),
        "bf16":          bench(torch.bfloat16, tf32=False),
        "fp16":          bench(torch.float16, tf32=False),
    }

    # Referencia teorica A100 SXM4
    theoretical = {"fp32": 19.5, "tf32": 156, "bf16": 312, "fp16": 312}

    print(f"  {'dtype':<8} {'TFLOPS':>10} {'teorico A100':>14} {'eficiencia':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*12}")
    for dt, tf in results.items():
        theo = theoretical.get(dt, 0)
        eff = f"{tf/theo*100:.0f}%" if theo else "-"
        print(f"  {dt:<8} {tf:>10.1f} {theo:>14.1f} {eff:>12}")

    print()
    bw_gb = props.total_memory / 1e9
    # Test rapido de ancho de banda de memoria
    x = torch.randn(int(2e8), device="cuda", dtype=torch.float32)  # 800 MB
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        y = x * 2.0
    torch.cuda.synchronize()
    dt_bw = time.perf_counter() - t0
    # cada iter: lee x + escribe y = 2 * 800 MB
    bw = 2 * x.numel() * 4 * 50 / dt_bw / 1e9
    print(f"  Ancho de banda memoria: {bw:.0f} GB/s (teorico A100 SXM4: ~2039 GB/s)")
    print()
    print("Benchmark completo. GPU operativa.")


if __name__ == "__main__":
    main()
