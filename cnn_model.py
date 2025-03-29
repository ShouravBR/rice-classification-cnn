import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision import transforms, utils, datasets, models
from torch.autograd import Variable
# from keras.optimizers import Adam
import glob
import os
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib

from sklearn.metrics import f1_score
from tqdm import tqdm

torch.manual_seed(0)
    
path = pathlib.Path("Rice_Image_Dataset")

arborio = list(path.glob('Arborio/*'))
basmati = list(path.glob('Basmati/*'))
ipsala = list(path.glob('Ipsala/*'))
jasmine = list(path.glob('Jasmine/*'))
karacadag = list(path.glob('Karacadag/*'))

total_list=arborio + basmati + ipsala + jasmine + karacadag


data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((100,100)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        ),
    ]
)
model_dataset = datasets.ImageFolder(path, transform=data_transform) 


class CustomizedConvNet(nn.Module):
    def __init__(self,number_of_classes, relu_activation=True, use_dropout=False):
        super().__init__() #Inheritance

        self.use_dropout = use_dropout

        act = nn.ReLU if relu_activation else nn.Sigmoid
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,padding=1,kernel_size=3)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=act()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12,out_channels=20,padding=1,kernel_size=3)
        self.bn2=nn.BatchNorm2d(num_features=20)
        self.relu2=act()
        
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=32,padding=1,kernel_size=3)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=act()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        

        self.fc1=nn.Linear(32*25*25, number_of_classes)

        self.dropout = nn.Dropout(0.2)

    
    def forward(self, Input):
        
        output=self.conv1(Input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool1(output)

        output = self.dropout(output) if self.use_dropout else output
        
        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu2(output)

        output = self.dropout(output) if self.use_dropout else output

        
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=self.pool3(output)

        output = self.dropout(output) if self.use_dropout else output
    
        
        output = torch.flatten(output, 1)
        output = output.view(-1,32*25*25)
        output=self.fc1(output)

        output = self.dropout(output) if self.use_dropout else output
        
        return output


train_count = int(0.7 * len(total_list)) 
valid_count = int(0.2 * len(total_list))
test_count = len(total_list) - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(model_dataset, (train_count, valid_count, test_count))


def train_and_test(batch_size=256, relu_activation=True, use_dropout=False):
    torch.manual_seed(0)
    
    # BATCH_SIZE=256
    BATCH_SIZE=batch_size

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  
    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    test_dataset_loader  = torch.utils.data.DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False)

    dataloaders = {'train': train_dataset_loader, 'val': valid_dataset_loader, 'test': test_dataset_loader}


    
    model = CustomizedConvNet(5, relu_activation=relu_activation, use_dropout=use_dropout)
    device='cuda'
    model = model.to(device)
    model.train()

    def accuracy(pred, label):
        _, out = torch.max(pred, dim=1)
        return torch.tensor(torch.sum(out == label).item()/len(pred))

    def validation_step(valid_dl, model, loss_fn):
        for image, label in valid_dl:
            out = model(image)
            loss = loss_fn(out, label)
            acc = accuracy(out, label)
            return {"val_loss": loss.item(), "val_acc": acc.item()}
        
    def fit_to_model(train_dl, valid_dl, epochs, optimizer, loss_fn, model):
        history = []
        pbar = tqdm(range(epochs))
        val = validation_step(valid_dl, model, loss_fn)
        for epoch in pbar:
            for i, (image, label) in enumerate(train_dl):
                out = model(image)
                loss = loss_fn(out, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description((f"Epoch [{epoch}/{epochs}]; batch: {i} => loss: {loss.item():.3f}, "
                                      f"val_loss: {val['val_loss']:.3f}, val_acc: {val['val_acc']:.3f}"))
                
            val = validation_step(valid_dl, model, loss_fn)
            # print(f"Epoch [{epoch}/{epochs}] => loss: {loss}, val_loss: {val['val_loss']}, val_acc: {val['val_acc']}")
            history.append({"loss": loss.item(), 
                            "val_loss": val['val_loss'], 
                            "val_acc": val['val_acc']
                        })
        return history

    def to_device(data, device):
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
        
        def __iter__(self):
            for x in self.dl:
                yield to_device(x, self.device)

                

    train_dataset_loader= DeviceDataLoader(train_dataset_loader, device)
    valid_dataset_loader = DeviceDataLoader(valid_dataset_loader, device)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 5
    history = fit_to_model(train_dataset_loader, valid_dataset_loader, epochs, optimizer, Loss, model)


    y_preds = []
    y_trues = []

    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for img,label in test_dataset_loader:
            output_model = model(img)
            logits = output_model.argmax(dim=1)
            y_preds.append(logits)
            y_trues.append(label)

    y_preds = torch.concat(y_preds)
    y_trues = torch.concat(y_trues)

    score = f1_score(y_trues, y_preds, average='weighted')
    del model

    return score, history