import torch

def cuda_available():
    return torch.cuda.is_available()

if __name__ == "__main__":
    if cuda_available():
        print("CUDA is available on this system.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for device in range(torch.cuda.device_count()):
            print(f"\nDevice {device}: {torch.cuda.get_device_name(device)}")
            print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB")
            print(f"Allocated memory: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB")
            print(f"Cached memory: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB\n")
    else:
        print("CUDA is not available on this system.\n")
