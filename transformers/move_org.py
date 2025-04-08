from huggingface_hub import HfApi, move_repo
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# models = api.list_models(author="neuralmagic")
models = ["neuralmagic/granite-3.1-8b-base-quantized.w4a16"]

for model in iter(models):
    # Replace with the destination org
    move_repo(from_id=model.id, to_id=f"RedHatAI/{model.id.split("/")[-1]}")
