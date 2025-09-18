import torch 
import random
import yaml

def dev_sel():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            random_gpu_index = random.randint(a=0,b=num_gpus-1)
        else:
            random_gpu_index = 0
        device = torch.device(f"cuda:{random_gpu_index}")
        print(f"CUDA is available. Selected GPU: {random_gpu_index}")
        print(f"Device string: {device}")   
        
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    return device

def load_yaml(path: str,)-> dict:

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config
 

    
