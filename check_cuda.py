import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Mem:", torch.cuda.get_device_properties(0).total_mem // 1024**2, "MB")
else:
    print("CPU only")
