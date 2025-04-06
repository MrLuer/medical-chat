
import torch
from peft import PeftModel
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_merge(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
   

    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default='/autodl-fs/data/model', help='dataset.yaml path')
    parser.add_argument('--adapter_model_path', type=str, default="/root/huanhuan-chat/data/output/", help='model path(s)')
    parser.add_argument('--merged_model_save_path', type=str, default="merged_model", help='model path(s)')
    opt = parser.parse_args()
    if not os.path.exists(opt.merged_model_save_path):
        os.makedirs(opt.merged_model_save_path)

    apply_merge(opt.base_model_path,opt.merged_model_save_path,opt.adapter_model_path)


