from huggingface_hub import HfApi
from sparsezoo import Model
import argparse

def upload_model_from_sparsezoo_to_huggingface(stub, model_id, token):
    # Download the PyTorch checkpoint from SparseZoo
    model = Model(stub)
    model.training.download()
    model_path = model.training.path

    # Upload the checkpoint to Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=model_id,
        repo_type="model",
        token=token,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model from SparseZoo to Hugging Face.")
    parser.add_argument("--stub", required=True, help="The SparseZoo stub for the model.")
    parser.add_argument("--model_id", required=True, help="The Hugging Face model ID.")
    parser.add_argument("--token", required=True, help="The Hugging Face API token.")
    
    args = parser.parse_args()
    
    upload_model_from_sparsezoo_to_huggingface(args.stub, args.model_id, args.token)
