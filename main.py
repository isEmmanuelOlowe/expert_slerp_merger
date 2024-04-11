import argparse
import os
import json
import re
import torch
import requests
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from tqdm import tqdm

def tensor_load(file_name, map_location=None):
    tensors = {}
    with safe_open(file_name, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

def get_weight_byte_size(weight):
    if isinstance(weight, torch.Tensor):
        weight_byte_size = weight.nelement() * weight.element_size()
    else:
        weight_byte_size = sum(p.nelement() * p.element_size() for p in weight.parameters())
    return weight_byte_size

def merge_tensor(tensorA, tensorB):
    t = 0.5
    dot = torch.sum(tensorA * tensorB, dim=1)
    norm_v0 = torch.norm(tensorA, dim=1)
    norm_v1 = torch.norm(tensorB, dim=1)
    cos_omega = dot / (norm_v0 * norm_v1)

    eps = 1e-6
    cos_omega = torch.clamp(cos_omega, -1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)
    v_t = (torch.sin((1 - t) * omega) / torch.sin(omega)).unsqueeze(1) * tensorA \
          + (torch.sin(t * omega) / torch.sin(omega)).unsqueeze(1) * tensorB
    return v_t

def download_file(url, dest_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)

def setup_model_directory(model_id, temp_dir):
    model_name = model_id.split("/")[-1]
    target_dir = os.path.join(temp_dir, model_name)
    os.makedirs(target_dir, exist_ok=True)
    
    base_url = f"https://huggingface.co/{model_id}/resolve/main/"
    files = [
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json"
    ]
    
    for file_name in files:
        download_file(f"{base_url}{file_name}", os.path.join(target_dir, file_name))
    
    # Downloading all parts of the model
    for i in tqdm(range(1, 22)):
        file_count_str = str(i).zfill(5)
        file_path = f"{target_dir}/model-{file_count_str}-of-00021.safetensors"
        if not os.path.exists(file_path):
            download_file(f"{base_url}model-{file_count_str}-of-00021.safetensors?download=true", file_path)
            print(f"Downloaded {file_path}")
        else:
            print(f"{file_path} already exists")


def merge_model(model_name_or_path, tmp_dir, save_dir, num_experts_per_tok=2, num_local_experts=2):
    
    os.makedirs(save_dir, exist_ok=True)

    def compute_total_size(weight_map):
        total_size = 0
        for key in weight_map.keys():
            weight = tensor_load(f"{save_dir}/{weight_map[key]}", map_location="cpu")
            total_size += get_weight_byte_size(weight[key])
        return total_size

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(save_dir)
    config_path = f"{save_dir}/config.json"
    # currently only support for Jamba model
    current_num_experts_per_tok = 2
    current_num_local_experts = 16
    config = None
    with open(config_path, "r") as f:
        config = json.load(f)
        config["num_experts_per_tok"] = num_experts_per_tok
        config["num_experts"] = num_local_experts

    divisor = current_num_local_experts // num_local_experts
    print("Divisor:", divisor)
    # save config
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)


    # weight
    weight_map = {}
    first_weights = ["lm_head.weight", "model.embed_tokens.weight", "model.final_layernorm.weight"]

    # load weight map
    bin_index_path = f"{tmp_dir}/model.safetensors.index.json"
    with open(bin_index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    


    # load weight map
    layers = {}
    for key in weight_map.keys():
        if key in first_weights:
            continue
        
        # print("key", key)
        # keyが"model.layers.[0-9]+."にmatchする場合はlayers_listに追加する
        layer_str = re.match(r"model\.layers\.[0-9]+\.", key)[0]
        if layer_str:
            layer_no = re.findall(r"\d+",layer_str)
            layer_no = layer_no[0]
            if layer_no not in layers.keys():
                layers[layer_no] = []

            layers[layer_no].append({ "key":key, "file_name":weight_map[key] })

    # new weight_map index
    new_weight_map = {
    "metadata": {
        "total_size": 0
    },
    "weight_map": {
    }
    }

    # load tensors
    tensor_weights = {}
    tensors = {}
    current_file_name = ""

    file_count = 0
    file_count_str = str(file_count).zfill(5)

    for key in first_weights:
        file_name = weight_map[key]
        if current_file_name != file_name:

            # load safetensor
            tensors = tensor_load(f"{tmp_dir}/{file_name}", map_location="cpu")
            current_file_name = file_name

        tensor_weights[key] = tensors[key]
        new_weight_map["weight_map"][key] = f"model-{file_count_str}.safetensors"

    # save tensor
    save_file(tensor_weights, f"{save_dir}/model-{file_count_str}.safetensors", metadata={"format":"pt"})
    file_count += 1

    layer_keys = sorted([ int(k) for k in layers.keys()])
    print("num_layer_keys", len(layer_keys))
    for layer_no in layer_keys:
        print("starting layer:",layer_no)
        file_count_str = str(file_count).zfill(5)
        tensor_weights = {}

        stock_expert_weights = {}

        current_file_name = ""
        for info in layers[str(layer_no)]:
            file_name = info["file_name"]
            if current_file_name != file_name:
                print("Loading Tensors ", file_name)
                tensors = tensor_load(f"{tmp_dir}/{file_name}", map_location="cpu")
                current_file_name = file_name

            layer_key = info["key"]
            layer_weights = tensors[layer_key]

            if '.moe.experts' in layer_key:
                

                lk = re.findall(r"[.]experts[.][0-9]+.", layer_key)[0]
                exp_index = int( re.findall(r"\d+",lk)[0] )

                new_layer_key = re.sub(r"\.experts\.\d+\.", f".experts.{exp_index // divisor}.", layer_key)

                if new_layer_key not in stock_expert_weights.keys():
                    tensor_weights[new_layer_key] = layer_weights
                    new_weight_map["weight_map"][new_layer_key] = f"model-{file_count_str}.safetensors"
                else:
                    # merge experts
                    tensor_weights[new_layer_key] = merge_tensor(tensor_weights[layer_key] , layer_weights)

                print("merging expert", exp_index, "to", new_layer_key, tensor_weights[new_layer_key].shape)

                if exp_index % divisor == divisor - 1:
                    print("new experts", new_layer_key, tensor_weights[new_layer_key].shape, "from", layer_key)


            elif '.moe.router' in layer_key:
                # print("reshape", layer_weights.shape, "-> view(2, 4, 4096) -> (2, 4096)", layer_key)
                print("reshape", layer_weights.shape, f"-> view({num_local_experts}, {divisor}, 4096) -> ({num_local_experts}, 4096)", layer_key)

                # calc gate merge
                weights_reshaped = layer_weights.view(num_local_experts, divisor, 4096)

                for i in range(divisor):
                    if i == 0:
                        tensor_weights[layer_key] = weights_reshaped[:, i, :]
                        new_weight_map["weight_map"][layer_key] = f"model-{file_count_str}.safetensors"
                    else:
                        tensor_weights[layer_key] = merge_tensor(tensor_weights[layer_key], weights_reshaped[:, i, :])


            else:
                tensor_weights[layer_key] = layer_weights

                new_weight_map["weight_map"][layer_key] = f"model-{file_count_str}.safetensors"

        # save tensor
        save_file(tensor_weights, f"{save_dir}/model-{file_count_str}.safetensors", metadata={"format":"pt"})
        print("Save Tensors ", f"{save_dir}/model-{file_count_str}.safetensors")
        file_count += 1

    # save new_weight_map
    new_weight_map["metadata"]["total_size"] = compute_total_size(new_weight_map["weight_map"])
    with open(f"{save_dir}/model.safetensors.index.json", "w") as f:
        json.dump(new_weight_map, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Merge and save model with specified configurations.")
    parser.add_argument('--model_id', type=str, required=True, help='Model identifier from Hugging Face Hub')
    parser.add_argument('--temp_dir', type=str, required=True, help='Temporary directory for processing model files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the final merged model')
    parser.add_argument('--num_experts_per_tok', type=int, required=True, help='Number of experts per token')
    parser.add_argument('--num_local_experts', type=int, required=True, help='Number of local experts')

    args = parser.parse_args()
    setup_model_directory(args.model_id, args.temp_dir)
    merge_model(args.model_id, args.temp_dir, args.save_dir, args.num_experts_per_tok, args.num_local_experts)
    print("Merge completed. Model saved at:", args.save_dir)

if __name__ == "__main__":
    main()