import torch

def get_gpu_info():
    available = torch.cuda.is_available()
    num = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(num)]
    current = torch.cuda.current_device() if available else None
    return {
        "available": available,
        "num": num,
        "names": names,
        "current": current
    }

if __name__ == "__main__":
    info = get_gpu_info()
    print(f"GPU Available: {info['available']}")
    print(f"Number of GPUs: {info['num']}")
    for i, name in enumerate(info['names']):
        print(f"GPU {i}: {name}")
    print(f"Current Device: {info['current']}")