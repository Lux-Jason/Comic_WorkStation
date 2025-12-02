import os
# 优化：使用 HF 镜像加速下载，解决连接不稳定的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
from diffusers import DiffusionPipeline
from colorama import init, Fore, Style
from huggingface_hub import snapshot_download

# Initialize colorama
init(autoreset=True)

def download_model_with_retry(model_id, max_retries=10):
    """
    Downloads the model with robust retry logic and resume capability.
    """
    print(f"{Fore.CYAN}Starting robust download for model: {model_id}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}This is a large model (approx 6GB+). Using HF Mirror for stability.{Style.RESET_ALL}")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n{Fore.BLUE}Attempt {attempt}/{max_retries}...{Style.RESET_ALL}")
            
            # Use snapshot_download which handles resuming and retries better than from_pretrained
            local_dir = snapshot_download(
                repo_id=model_id,
                # resume_download=True, # Deprecated, enabled by default
                local_files_only=False,
                # Exclude flax/tf weights to save space/bandwidth if you only use torch
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
                max_workers=4, # Reduce concurrency to improve stability
            )
            
            print(f"{Fore.GREEN}Download complete! Files stored at: {local_dir}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Download interrupted: {e}{Style.RESET_ALL}")
            if attempt < max_retries:
                wait_time = 5 * attempt # Exponential backoff
                print(f"{Fore.YELLOW}Retrying in {wait_time} seconds...{Style.RESET_ALL}")
                time.sleep(wait_time)
            else:
                print(f"{Fore.RED}Max retries reached. Download failed.{Style.RESET_ALL}")
                return False

def load_pipeline(model_id):
    """
    Loads the pipeline after successful download.
    """
    try:
        print(f"\n{Fore.MAGENTA}Verifying and loading pipeline...{Style.RESET_ALL}")
        # User requested configuration: bfloat16 and device_map="cuda"
        pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            dtype=torch.bfloat16, 
            device_map="cuda"
        )
        print(f"{Fore.GREEN}Successfully loaded model pipeline!{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}Error loading pipeline: {e}{Style.RESET_ALL}")
        return False

if __name__ == "__main__":
    model_id = "ogkalu/Comic-Diffusion"
    
    if download_model_with_retry(model_id):
        load_pipeline(model_id)
    else:
        print(f"{Fore.RED}Failed to prepare model.{Style.RESET_ALL}")

