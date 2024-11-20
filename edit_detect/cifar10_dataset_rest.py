import torch
from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    batch_size: int = 1
    num: int = 128
    trainset = CIFAR10(root='datasets', train=True, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    for i, (image, label) in enumerate(trainset):
        if i > num:
            break
        image.save(f"/workspace/research/edit_detect/real_images/cifar10/cifar10_{i}.jpg")