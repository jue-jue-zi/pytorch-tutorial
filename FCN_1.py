# torch version: 1.5.1+cpu
# torchvision version: 0.6.1+cpu
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms


#%% load train dataset
train_data_path = './train'

transforms = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path, transform=transforms
)

#%% load test and validation dataset
val_data_path = './val/'
val_data = torchvision.datasets.ImageFolder(
    root=val_data_path, transform=transforms
)

test_data_path = './test/'
test_data = torchvision.datasets.ImageFolder(
    root=test_data_path, transform=transforms
)

#%% create dataloader
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size)


#%% creat a network
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2= nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


#%% train
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()  # train flag

        for batch in train_data_loader:
            optimizer.zero_grad()
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()  # compute the gradients
            optimizer.step()  # adjust the weights
            training_loss += loss.data.item()
        training_loss /= len(train_loader)

        model.eval()  # validation flag
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            input , target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)

        print('Epoch: {}, Training Loss: {:.2f}, '
              'Validation Loss: {:.2f}, '
              'Accuracy = {:.2f}'.format(epoch, training_loss, valid_loss,
                                         num_correct / num_examples))


#%% test
def test(model, test_loader, device="cpu"):
    num_correct = 0
    num_examples = 0
    for batch in test_loader:
        input, target = batch
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        correct = torch.eq(output.argmax(dim=1), target).view(-1)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    print('Test Accuracy = {:.2f}'.format(num_correct / num_examples))


if __name__ == "__main__":
    # 01. creat Net
    simplenet = SimpleNet()
    # 02. load exist model parameters
    if os.path.exists('./simplenet.pkl'):
        print('Loading exist simplenet parameters...')
        simplenet_state_dict = torch.load('./simplenet.pkl')
        simplenet.load_state_dict(simplenet_state_dict)
    else:
        print('No exist simplenet parameters!')
    # 03. load Net and data to GPU if existed
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    simplenet.to(device)
    # 04. create loss function & optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simplenet.parameters(), lr=0.0003)
    # 05. train
    train(simplenet, optimizer, loss_fn, train_data_loader, val_data_loader, device=device)
    # 06. save model
    torch.save(simplenet.state_dict(), './simplenet.pkl')
    # 07. test
    test(simplenet, test_data_loader, device)




