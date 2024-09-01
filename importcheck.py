import torch
print(torch.__version__)
print(torch.cuda.is_available())  # This should return False if you're on a CPU-only system
