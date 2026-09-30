import torch
print("CUDA disponível?", torch.cuda.is_available())
print("cuDNN ativado?", torch.backends.cudnn.enabled)