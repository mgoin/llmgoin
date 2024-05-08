import torch
from transformers import AutoModelForCausalLM

def calculate_sparsity(model):
    sparsity_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Calculate sparsity: the fraction of elements that are exactly zero
            total_elements = module.weight.nelement()
            zero_elements = (module.weight == 0).sum().item()
            sparsity = zero_elements / total_elements
            print(name, total_elements)
            sparsity_dict[name] = sparsity
    return sparsity_dict


model_name = 'nm-testing/OpenHermes-2.5-Mistral-7B-pruned50'
model = AutoModelForCausalLM.from_pretrained(model_name)
sparsity_dict = calculate_sparsity(model)

# Print the sparsity of each Linear module
for name, sparsity in sparsity_dict.items():
    print(f"{name}: {sparsity:.2%}")
