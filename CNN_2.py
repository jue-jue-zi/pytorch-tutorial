import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FCN_1 import train_data_loader, val_data_loader, test_data_loader, train, test


#%% construct CNN
class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2)
        )
        # self._init_weight()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # x.size() = 64 * (256*6*6) = 64 * 9216
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x

    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)

if __name__=="__main__":
    cnnnet = CNNNet()
    if os.path.exists('./cnnnet.pkl'):
        print('Loading exist CNN parameters...')
        cnnnet_state_dict = torch.load('./cnnnet.pkl', map_location=torch.device('cpu'))
        cnnnet.load_state_dict(cnnnet_state_dict)
    else:
        print('No exist CNN parameters!')
    # GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Multi GPU
    if torch.cuda.device_count() > 1:
        torch.nn.DataParallel(cnnnet)
    # to CPU or GPU
    cnnnet.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnnnet.parameters(), lr=0.1)
    train(cnnnet, optimizer, loss_fn, train_data_loader, val_data_loader, 10, device)
    # torch.save(cnnnet.state_dict(), './cnnnet.pkl')
    test(cnnnet, test_data_loader, device)

