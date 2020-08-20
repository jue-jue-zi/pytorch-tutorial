#%%
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from  torchvision import transforms
import random
from FCN_1 import train_data_loader, val_data_loader, test_data_loader, train, test


#%% write a scalar
writer = SummaryWriter()
writer.add_scalar('example', 3)

#%%
value = 10
writer.add_scalar('test_loop', value, 0)
for i in range(1, 10000):
    value += random.random() - 0.5
    writer.add_scalar('test_loop', value, i)

#%% load test dataset
train_data_path = './train'
transforms = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path, transform=transforms
)
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size)

#%% write a model graph
model = torchvision.models.alexnet(num_classes=2)
images, labels = next(iter(train_data_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)

