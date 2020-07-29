import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from FCN_1 import train_data_loader, val_data_loader, test_data_loader, train, test


#%% transfer learning with ResNet-50
# load exist model
transfer_model = models.resnet50(pretrained=True)
print(transfer_model)
# freeze all the layers except batchnorm
for name, param in transfer_model.named_parameters():
    if('bn' not in name):
        param.requires_grad = False
# replace the final classification block (fc -> myNet)
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features, 500),
                                  nn.ReLU(),
                                  nn.Dropout(), nn.Linear(500, 2))
print(transfer_model)
#%%
if __name__=="__main__":
    if os.path.exists('./transfer_model.pkl'):
        print('Loading exist transfer model parameters...')
        transfer_model_state_dict = torch.load('./transfer_model.pkl', map_location=torch.device('cpu'))
        transfer_model.load_state_dict(transfer_model_state_dict)
    else:
        print('No exist transfer model parameters!')
    # GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Multi GPU
    if torch.cuda.device_count() > 1:
        torch.nn.DataParallel(transfer_model)
    # to CPU or GPU
    transfer_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.0003)
    train(transfer_model, optimizer, loss_fn, train_data_loader, val_data_loader, 10, device)
    # torch.save(transfer_model.state_dict(), './transfer_model.pkl')
    transfer_model.eval()
    test(transfer_model, test_data_loader, device)
