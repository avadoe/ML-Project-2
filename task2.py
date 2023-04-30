import warnings
warnings.filterwarnings('ignore')

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, random_split, DataLoader


def main():

    # Hyperparameters

    batch_size = 32
    num_epochs = 10
    num_splits = 10
    learning_rate = 0.001
    num_classes = 10
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datatransform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    trainmnist = datasets.MNIST(
        "../MNIST_Train", train=True, download=True, transform=datatransform
    )

    testmnist = datasets.MNIST(
        "../MNIST_Test", train=False, download=True, transform=datatransform
    )

    mnistset = ConcatDataset([trainmnist,testmnist])

    def one_hot_encode(label: int) -> torch.Tensor:
        x = torch.zeros([label.shape[0], num_classes], dtype=torch.float32)
        for i in range(label.shape[0]):
            x[i, label[i]] = 1.
        return x

    class Net(nn.Module):
        def __init__(self, num_layers: int, hidden_size: int, activation: nn.Module) -> None:
            super().__init__()
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.fc = nn.Sequential(
                nn.Linear(in_features=28*28,out_features=hidden_size),
                activation(),
                nn.Linear(in_features=hidden_size,out_features=hidden_size),
                activation(),
                nn.Linear(in_features=hidden_size,out_features=num_classes),
                activation(),
            ) if (num_layers == 3) else nn.Sequential(
                nn.Linear(in_features=28*28,out_features=hidden_size),
                activation(),
                nn.Linear(in_features=hidden_size,out_features=num_classes),
                activation(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.flatten(x)
            x = self.fc(x)
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

        model.to(model_device)

        num_layers = model_dict['num_layers']
        hidden_size = model_dict['hidden_size']
        activation_func = {nn.Tanh: "tanh", nn.ReLU: "relu", nn.Sigmoid: "sigmoid"}[model_dict['activation']]

        for optimizer_func in optimizer_list:

            optimizer_fn = {optim.Adam: "Adam", optim.SGD: "SGD"}[optimizer_func]

            criterion = nn.CrossEntropyLoss()
            optimizer = optimizer_func(params=model.parameters(), lr=learning_rate)

            test_losses = torch.zeros([num_splits])
            test_accuracies = torch.zeros([num_splits])

            print(f"+{'-'*36:36s}+")
            print(f"|{f'RUNNING MODEL: ({num_layers},{hidden_size},{activation_func},{optimizer_fn})':^36s}|")
            print(f"+{'-'*36:36s}+")

            for split_id in range(num_splits):

                trainset, testset = random_split(mnistset,[0.67,0.33])

                trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
                testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

                # Training

                model.train()

                training_loss = 0.0
                training_acc = 0.0

                print(f"+{'-'*36:36s}+")
                print(f"|{f'SPLIT: {split_id+1}/{num_splits}':^36s}|")
                print(f"+{'-'*36:36s}+")

                for epoch in range(num_epochs):
                    for i, d in enumerate(trainloader):
                        tensor, label = d  # Shapes - (batch_size,1,28,28) # (batch_size)
                        output = model(tensor.to(model_device)).to(training_device)
                        target = one_hot_encode(label).to(training_device)
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

                # Testing

                model.eval()

                val_loss = 0.0
                val_acc = 0.0

                conf_matrix = torch.zeros([num_classes,num_classes],dtype=torch.int32)

                with torch.inference_mode():
                    for i, d in enumerate(trainloader):
                        tensor, label = d  # Shapes - (batch_size,1,28,28) # (batch_size)
                        output = model(tensor.to(model_device)).to(training_device)
                        target = one_hot_encode(label).to(training_device)
                        predictions = torch.argmax(output,-1)
                        targets = torch.argmax(target,1)
                        for i in range(predictions.shape[0]):
                            conf_matrix[predictions[i],targets[i]] += 1
                        acc = torch.sum(torch.argmax(output,-1)==torch.argmax(target,1))
                        val_acc += acc.item()
                        loss = criterion(output,target)
                        val_loss += loss.item()
                        del loss
                    val_loss /= float(len(trainloader.dataset))
                    val_acc /= float(len(trainloader.dataset))
                    test_losses[split_id,] = val_loss
                    test_accuracies[split_id,] = val_acc
                    print(f" {' '*5}+",f"{'-'*5}+"*num_classes,sep='')
                    print(f" {' '*5}|",*[f"{_:^5d}|" for _ in range(num_classes)],sep='')
                    print(f"+{'-'*5}+",f"{'-'*5}+"*num_classes,sep='')
                    for i in range(num_classes):
                        print(f"|{i:^5d}|",*[f"{conf_matrix[i,j]:>5d}|" for j in range(num_classes)],sep='')
                    print(f"+{'-'*5}+",f"{'-'*5}+"*num_classes,sep='')

            print(f"+{'-'*10:^10s}+{'-'*12:^12s}+{'-'*12:^12s}+")
            print(f"|{'Split':^10s}|{'Loss':^12s}|{'Accuracy':^12s}|")
            print(f"+{'-'*10:^10s}+{'-'*12:^12s}+{'-'*12:^12s}+")
            for split_id in range(num_splits):
                print(f"|{f'{split_id+1}/{num_splits}':^10s}|{f'{test_losses[split_id]:.6f}':^12s}|{f'{test_accuracies[split_id]:.6f}':^12s}|")
            print(f"+{'-'*10:^10s}+{'-'*12:^12s}+{'-'*12:^12s}+\n")

            print(f" {' '*12}+{'-'*12:^12s}+{'-'*12:^12s}+")
            print(f" {' '*12}|{'Mean':^12s}|{'Variance':^12s}|")
            print(f"+{'-'*12}+{'-'*12:^12s}+{'-'*12:^12s}+")
            print(f"|{'Loss':^12s}|{f'{torch.mean(test_losses):.6f}':^12s}|{f'{torch.var(test_losses):.6f}':^12s}|")
            print(f"|{'Accuracy':^12s}|{f'{torch.mean(test_accuracies):.6f}':^12s}|{f'{torch.var(test_accuracies):.6f}':^12s}|")
            print(f"+{'-'*12}+{'-'*12:^12s}+{'-'*12:^12s}+\n\n")


if __name__ == '__main__':
    with open('output.txt','w') as f:
        sys.stdout = f
        main()
