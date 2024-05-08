from huggingface_hub import HfApi, Repository
from transformers import AutoModelForCausalLM

# List of model repository IDs
model_repos = [
    "neuralmagic/Llama-2-7b-pruned50-retrained",
    "neuralmagic/Llama-2-7b-pruned70-retrained",
    "neuralmagic/Llama-2-7b-ultrachat",
    "neuralmagic/Llama-2-7b-pruned50-retrained-ultrachat",
    "neuralmagic/Llama-2-7b-pruned70-retrained-ultrachat",
    "neuralmagic/Llama-2-7b-instruct",
    "neuralmagic/Llama-2-7b-pruned50-retrained-instruct",
    "neuralmagic/Llama-2-7b-pruned70-retrained-instruct",
    "neuralmagic/Llama-2-7b-evolcodealpaca",
    "neuralmagic/Llama-2-7b-pruned50-retrained-evolcodealpaca",
    "neuralmagic/Llama-2-7b-pruned70-retrained-evolcodealpaca",
]

# Hugging Face API token
hf_token = "XXXX"

# Create an instance of the Hugging Face API
api = HfApi(token=hf_token)

# Iterate over each model repository
for repo_id in model_repos:
    # Clone the repository
    repo = Repository(local_dir=repo_id.split("/")[-1], clone_from=repo_id, use_auth_token=hf_token)

    # Create a new branch for the FP32 version
    repo.git_checkout(revision="fp32", create_branch_ok=True)

    # Push the FP32 version to the fp32 branch
    repo.git_push(upstream="origin", set_upstream=True)

    # Checkout the main branch
    repo.git_checkout(revision="main")

    # Remove files with "safetensors" or "pytorch" in their names from the main branch
    files_to_remove = [
        file for file in repo.list_files() 
        if "safetensors" in file or "pytorch" in file
    ]
    repo.git_rm(recursive=True, pathspec=files_to_remove)

    # Load the tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype="auto", low_cpu_mem_usage=True)

    # Convert the model to BFloat16 and shard it
    model.to(dtype="bfloat16")
    model.save_pretrained(repo_id.split("/")[-1], safe_serialization=True, max_shard_size="5GB")

    # Stage and commit the changes
    repo.git_add(auto_lfs_track=True)
    repo.git_commit(f"Convert model to BFloat16 and shard using SafeTensors")

    # Push the changes to the main branch
    repo.git_push(upstream="origin", set_upstream=True)

print("Conversion and upload completed for all models.")
