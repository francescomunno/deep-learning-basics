import sys

import torch
import torchvision

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

x = torch.rand(1000, 1000, device=device)
y = x @ x
print("Calcolo riuscito:", y.shape)
