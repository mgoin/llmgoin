
Install vllm and lm-eval:
```
pip install vllm==0.5.0 lm-eval
```

Running an eval:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_openllm.sh "neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8" "tensor_parallel_size=4,add_bos_token=True,gpu_memory_utilization=0.7"
```

Reading the scores afterwards:
```
python summarize_openllm_scores.py

No files found for task: hellaswag
No files found for task: mmlu
Scores from results/arc_challenge/neuralmagic__Mixtral-8x7B-Instruct-v0.1-FP8/results_2024-06-12T19-25-45.666273.json:
  acc,none: 0.6791808873720137
  acc_stderr,none: 0.013640943091946528
  acc_norm,none: 0.7107508532423208
  acc_norm_stderr,none: 0.013250012579393445
  alias: arc_challenge
  num_fewshot: 25

Scores from results/winogrande/neuralmagic__Mixtral-8x7B-Instruct-v0.1-FP8/results_2024-06-12T19-00-52.904743.json:
  acc,none: 0.823993685872139
  acc_stderr,none: 0.010703090882320705
  alias: winogrande
  num_fewshot: 5

Scores from results/truthfulqa_mc2/neuralmagic__Mixtral-8x7B-Instruct-v0.1-FP8/results_2024-06-12T19-28-41.493490.json:
  acc,none: 0.6420281627911943
  acc_stderr,none: 0.015188328851840087
  alias: truthfulqa_mc2
  num_fewshot: 0

Scores from results/gsm8k/neuralmagic__Mixtral-8x7B-Instruct-v0.1-FP8/results_2024-06-12T19-10-33.519155.json:
  exact_match,strict-match: 0.6376042456406369
  exact_match_stderr,strict-match: 0.013240654263574755
  exact_match,flexible-extract: 0.640636846095527
  exact_match_stderr,flexible-extract: 0.013216456309851523
  alias: gsm8k
  num_fewshot: 5
```
