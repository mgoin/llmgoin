from transformers import AutoImageProcessor
from vllm.assets.image import ImageAsset
from vllm.assets.base import get_vllm_public_assets
import torch
import torchvision
import time
import numpy

def run_benchmark(processor, image, num_runs=100):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = processor(image, return_tensors="pt")
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        'mean': numpy.mean(times),
        'median': numpy.median(times),
        'stdev': numpy.std(times),
        'min': min(times),
        'max': max(times)
    }

def print_stats(name, stats):
    print(f"\n{'-' * 50}")
    print(f"{name} Statistics (milliseconds):")
    print(f"{'Mean:':>15} {stats['mean']:.3f}")
    print(f"{'Median:':>15} {stats['median']:.3f}")
    print(f"{'Std Dev:':>15} {stats['stdev']:.3f}")
    print(f"{'Min:':>15} {stats['min']:.3f}")
    print(f"{'Max:':>15} {stats['max']:.3f}")

def main():
    print("\nInitializing processors and loading image...")
    
    # Load image
    pil_image = ImageAsset("stop_sign").pil_image.convert("RGB")
    image_path = get_vllm_public_assets(filename="stop_sign.jpg", s3_prefix="vision_model_images")
    torch_image = torchvision.io.read_image(image_path)
    print(torch_image.shape)
    
    # Initialize processors
    slow_processor = AutoImageProcessor.from_pretrained("mistral-community/pixtral-12b", use_fast=False)
    fast_processor = AutoImageProcessor.from_pretrained("mistral-community/pixtral-12b", use_fast=True)
    compiled_processor = torch.compile(fast_processor, mode="reduce-overhead")
    
    # Warmup runs
    print("Performing warmup runs...")
    for processor in [slow_processor, fast_processor, compiled_processor]:
        _ = processor(pil_image, return_tensors="pt")
        _ = processor(torch_image, return_tensors="pt")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    
    slow_stats = run_benchmark(slow_processor, pil_image)
    print_stats("Slow Processor (PIL Image)", slow_stats)
    
    fast_stats = run_benchmark(fast_processor, pil_image)
    print_stats("Fast Processor (PIL Image)", fast_stats)
    
    compiled_stats = run_benchmark(compiled_processor, pil_image)
    print_stats("Compiled Processor (PIL Image)", compiled_stats)
    
    slow_stats = run_benchmark(slow_processor, pil_image)
    print_stats("Slow Processor (Torch Image)", slow_stats)
    
    fast_stats = run_benchmark(fast_processor, torch_image)
    print_stats("Fast Processor (Torch Image)", fast_stats)
    
    compiled_stats = run_benchmark(compiled_processor, torch_image)
    print_stats("Compiled Processor (Torch Image)", compiled_stats)
    
    # # Print comparison
    # print("\nPerformance Comparison:")
    # print(f"Fast vs Slow speedup: {slow_stats['mean'] / fast_stats['mean']:.2f}x")
    # print(f"Compiled vs Slow speedup: {slow_stats['mean'] / compiled_stats['mean']:.2f}x")
    # print(f"Compiled vs Fast speedup: {fast_stats['mean'] / compiled_stats['mean']:.2f}x")

if __name__ == "__main__":
    main()
