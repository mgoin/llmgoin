from vllm import LLM, SamplingParams
import time

num_iters = 5

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = LLM(model_id)
params = SamplingParams(max_tokens=10)

for i in range(num_iters):
    t0 = time.perf_counter()

    output = model.generate("Hello, my name is Misha and I ", params)

    t = time.perf_counter() - t0

    num_prompt_tokens = len(output[0].prompt_token_ids)
    num_tokens_generated = len(output[0].outputs[0].token_ids)
    tokens_sec = num_tokens_generated / t
    model_size = model.llm_engine.model_executor.driver_worker.model_runner.model_memory_usage
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Time for inference {i+1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
    print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
