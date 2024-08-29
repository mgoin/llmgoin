import argparse
import os
import json
import torch
import safetensors.torch

def per_tensor_quantize(tensor):
    """Quantize a tensor to FP8 using per-tensor static scaling factor."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        min_val, max_val = torch.tensor(-16.0, dtype=tensor.dtype), torch.tensor(16.0, dtype=tensor.dtype)
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale

def process_safetensors_file(file_path):
    """Process a single safetensors file in-place, quantizing weights to FP8."""
    print(f"Processing {file_path}")
    tensors = safetensors.torch.load_file(file_path)
    
    modified_tensors = {}
    for name, tensor in tensors.items():
        if name.endswith('_proj.weight'):
            print("Quantizing", name)
            qweight, scale = per_tensor_quantize(tensor)
            modified_tensors[name] = qweight
            modified_tensors[f"{name}_scale"] = scale
        else:
            modified_tensors[name] = tensor

    safetensors.torch.save_file(modified_tensors, file_path)
    print(f"Updated {file_path} with quantized tensors")

def update_index_file(index_file_path):
    """Update the index file for the quantized model."""
    print(f"Updating index file: {index_file_path}")
    with open(index_file_path, 'r') as f:
        index = json.load(f)
    
    new_weight_map = {}
    for tensor_name, file_name in index['weight_map'].items():
        new_weight_map[tensor_name] = file_name
        if tensor_name.endswith('_proj.weight'):
            new_weight_map[f"{tensor_name}_scale"] = file_name
    
    index['weight_map'] = new_weight_map
    
    # Recalculate total_size
    total_size = sum(os.path.getsize(os.path.join(os.path.dirname(index_file_path), file)) 
                     for file in set(index['weight_map'].values()))
    index['metadata']['total_size'] = total_size
    
    with open(index_file_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Updated index file {index_file_path}")

def process_directory(directory):
    """Process all safetensors files in the given directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.safetensors'):
            process_safetensors_file(file_path)
        elif filename == 'model.safetensors.index.json':
            index_file_path = file_path

    update_index_file(index_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert safetensors model to FP8 in-place.')
    parser.add_argument('directory', type=str, help='The directory containing the safetensors files and index file.')
    
    args = parser.parse_args()
    process_directory(args.directory)
