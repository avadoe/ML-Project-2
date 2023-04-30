import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters

batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 10


datatransform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

trainset = datasets.MNIST(
    "../MNIST_Train", train=True, download=True, transform=datatransform
)

testset = datasets.MNIST(
    "../MNIST_Test", train=False, download=True, transform=datatransform
)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


def one_hot_encode(label: int) -> torch.Tensor:
    x = torch.zeros([label.shape[0], num_classes], dtype=torch.float32)
    for i in range(label.shape[0]):
        x[i, label[i]] = 1.0
    return x


class Net(nn.Module):
    def __init__(self, num_layers, hidden_size, activation,) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16,kernel_size=5,stride=3)
        self.actv = activation()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=16*8*8,out_features=hidden_size),
            nn.Linear(in_features=hidden_size,out_features=hidden_size),
            nn.Linear(in_features=hidden_size,out_features=num_classes),
        ) if (num_layers == 3) else nn.Sequential(
            nn.Linear(in_features=16*8*8,out_features=hidden_size),
            nn.Linear(in_features=hidden_size,out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input(x.shape)
        x = self.conv(x)
        # input(x.shape)
        x = self.actv(x)
        # input(x.shape)
        x = self.flatten(x)
        # input(x.shape)
        x = self.fc(x)
        # input(x.shape)
        return x

# Combinations

model_list = [
    {'num_layers': 2, 'hidden_size': 100, 'activation': nn.Tanh},
    {'num_layers': 2, 'hidden_size': 100, 'activation': nn.Sigmoid},
    {'num_layers': 2, 'hidden_size': 100, 'activation': nn.ReLU},
    {'num_layers': 2, 'hidden_size': 150, 'activation': nn.Tanh},
    {'num_layers': 2, 'hidden_size': 150, 'activation': nn.Sigmoid},
    {'num_layers': 2, 'hidden_size': 150, 'activation': nn.ReLU},
    {'num_layers': 3, 'hidden_size': 100, 'activation': nn.Tanh},
    {'num_layers': 3, 'hidden_size': 100, 'activation': nn.Sigmoid},
    {'num_layers': 3, 'hidden_size': 100, 'activation': nn.ReLU},
    {'num_layers': 3, 'hidden_size': 150, 'activation': nn.Tanh},
    {'num_layers': 3, 'hidden_size': 150, 'activation': nn.Sigmoid},
    {'num_layers': 3, 'hidden_size': 150, 'activation': nn.ReLU},
]

optimizer_list = [optim.SGD, optim.Adam]


for model_dict in model_list:

    model = Net(**model_dict)

    num_layers = model_dict['num_layers']
    hidden_size = model_dict['hidden_size']
    activation_func = {nn.Tanh: "tanh", nn.ReLU: "relu", nn.Sigmoid: "sigmoid"}[model_dict['activation']]

    for optimizer_func in optimizer_list:

        optimizer_fn = {optim.Adam: "Adam", optim.SGD: "SGD"}[optimizer_func]

        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer_func(params=model.parameters(), lr=learning_rate)

        # Training

        model.train()

        training_loss = 0.0
        training_acc = 0.0

        print(f"+{'-'*36:36s}+")
        print(f"|{f'TRAINING: ({num_layers},{hidden_size},{activation_func},{optimizer_fn})':^36s}|")
        print(f"+{'-'*10:^10s}+{'-'*12:^12s}+{'-'*12:^12s}+")
        print(f"|{'Epoch':^10s}|{'Loss':^12s}|{'Accuracy':^12s}|")
        print(f"+{'-'*10:^10s}+{'-'*12:^12s}+{'-'*12:^12s}+")

        for epoch in range(num_epochs):
            for i, d in enumerate(trainloader):
                tensor, label = d  # Shapes - (4,1,28,28) # (4)
                output = model(tensor)
                target = one_hot_encode(label)
                acc = torch.sum(torch.argmax(output,-1)==torch.argmax(target,1))
                training_acc += acc.item()
                optimizer.zero_grad()
                loss = criterion(output,target)
                training_loss += loss.item()
                loss.backward()
                optimizer.step()
                del loss
            training_loss /= float(len(trainloader.dataset))
            training_acc /= float(len(trainloader.dataset))
            print(f"|{f'{epoch+1:02d}/{num_epochs:02d}':^10s}|{f'{training_loss:.6f}':^12s}|{f'{training_acc:.6f}':^12s}|")

        print(f"+{'-'*10:^10s}+{'-'*12:^12s}+{'-'*12:^12s}+\n")

        # Testing

        model.eval()

        val_loss = 0.0
        val_acc = 0.0

        print(f"+{'-'*25:25s}+")
        print(f"|{'TESTING':^25s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+")
        print(f"|{'Loss':^12s}|{'Accuracy':^12s}|")
        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+")


        with torch.inference_mode():
            for i, d in enumerate(trainloader):
                tensor, label = d  # Shapes - (4,1,28,28) # (4)
                output = model(tensor)
                target = one_hot_encode(label)
                acc = torch.sum(torch.argmax(output,-1)==torch.argmax(target,1))
                val_acc += acc.item()
                loss = criterion(output,target)
                val_loss += loss.item()
                del loss
            val_loss /= float(len(trainloader.dataset))
            val_acc /= float(len(trainloader.dataset))
            print(f"|{f'{val_loss:.6f}':^12s}|{f'{val_acc:.6f}':^12s}|")

        print(f"+{'-'*12:^12s}+{'-'*12:^12s}+\n\n")
