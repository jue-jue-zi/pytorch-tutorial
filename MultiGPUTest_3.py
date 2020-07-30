import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim as optim

# assgin GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 3'  # 2 is easy to overheating

#%% creat a great GPU assuming Network (just for test in 1080Ti)
class GreatNetwork(nn.Module):

    def __init__(self):
        super(GreatNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


#%% creat fake pictures for test in 1080Ti
# (with GreatNetwork, one picture can occupy
#  all the memory of one 1080Ti GPU (12GB) )
class FakeDataset(Dataset):

    def __init__(self):
        super(FakeDataset, self).__init__()
        self.count = 90000

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        image = torch.randn(3, 512, 512)
        return image


#%% test one Batch images to multi GPUs for training
# (one image to one GPU)
def main():
    print('-------------------RUN FOR TEST----------------------')
    if not (torch.cuda.is_available() or torch.cuda.device_count() > 1):
        return
    # assign GPUs for training devices
    default_device = torch.device('cuda')
    default_type = torch.float32
    # creat GreatNetwork
    model = GreatNetwork()
    model = nn.DataParallel(model)
    model.to(default_device).type(default_type)
    # set loss function and optimizer
    loss_function = nn.MSELoss()
    # loss_function.to(default_device).type(default_type)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # batch_size = 3 -> use three GPUs for training
    batch_size = 3
    ds = DataLoader(FakeDataset(), batch_size=batch_size, shuffle=True, drop_last=True)
    # start training
    position = 0
    for epoch in range(20):
        for image in ds:
            position += 1
            timestamp = time.time()
            optimizer.zero_grad()
            # input, target to 'device'
            image = image.to(default_device).type(default_type)
            # input -> model -> output
            image_hat = model(image)
            loss = loss_function(image, image_hat)
            loss.backward()
            optimizer.step()
            print('TRAIN[%05d] Time: %10.4fs' % (position, time.time() - timestamp))


if __name__ == '__main__':
    main()