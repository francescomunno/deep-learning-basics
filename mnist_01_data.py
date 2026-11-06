import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

to_tensor = transforms.ToTensor()

train_full = datasets.MNIST("data", train=True, download=True, transform=to_tensor)
test_set = datasets.MNIST("data", train=False, download=True, transform=to_tensor)

generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(train_full, [50000, 10000], generator=generator)

print("Train:", len(train_set))
print("Validation:", len(val_set))
print("Test:", len(test_set))

image, label = train_set[0]
print("Forma immagine:", image.shape)
print("Valori:", image.min().item(), image.max().item())
print("Etichetta:", label)

fig, axes = plt.subplots(3, 5, figsize=(8, 5))
for i, ax in enumerate(axes.flat):
    image, label = train_set[i]
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(label)
    ax.axis("off")
plt.tight_layout()
plt.show()
