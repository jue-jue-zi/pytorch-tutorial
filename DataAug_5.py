import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

train_data_path = './train'

transforms = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ColorJitter(brightness=(0.2, 1), contrast=(0, 1),
                           saturation=(0, 1), hue=0.1),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path, transform=transforms
)

if __name__ == "__main__":
    train_data = iter(train_data)
    img, _ = next(train_data)
    plt.figure()
    plt.imshow(img.permute(1,2,0))
    plt.show()

