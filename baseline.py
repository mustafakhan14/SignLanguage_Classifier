import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import cv
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torchvision
from torchvision.transforms import transforms
from torchvision import datasets
import string
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

warnings.filterwarnings("ignore")


# Primary constants
batch_size = 256
learning_rate = 1e-5
input_size = 28*28
num_classes = 26

"""
Read in images from path and store in data-frame with each image
concatenated to its label. Returns a shuffled data-frame.
"""
train_dataset = pd.read_csv('Data/sign_mnist_train/sign_mnist_train.csv')
test_dataset = pd.read_csv('Data/sign_mnist_test/sign_mnist_test.csv')
num_rows = train_dataset.shape[0]
# To map each label number to its corresponding letter
letters = dict(enumerate(string.ascii_uppercase))
# print(train_dataset.head())

def dataframe_to_array(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outputs as numpy arrays
    inputs_array = dataframe1.iloc[:, 1:].to_numpy()
    targets_array = dataframe1['label'].to_numpy()
    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_array(train_dataset)
testinputs_array, testtargets_array = dataframe_to_array(train_dataset)

pic1 = np.reshape(inputs_array[0], (28, 28))
plt.imshow(pic1, cmap = "gray")
print("Letter: ", letters[targets_array[0].item()])

inputs = torch.from_numpy(inputs_array).float()
targets = torch.from_numpy(targets_array).long()
testinputs = torch.from_numpy(testinputs_array).float()
testtargets = torch.from_numpy(testtargets_array).long()

# Training validation & test dataset
dataset = TensorDataset(inputs, targets)
testdataset = TensorDataset(testinputs, testtargets)

# Let's use 15% of our training dataset to validate our model
val_percent = 0.15
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size*2)
test_loader = DataLoader(testdataset, batch_size*2)

img, label = train_ds[0]
plt.imshow(img.reshape((28,28)), cmap = 'gray')
print("Letter: ", letters[label.item()])


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


model = MnistModel()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

print(evaluate(model, val_loader))

history = fit(50, 1e-4, model, train_loader, val_loader)

# Visualizing how our model performed after each epoch
accuracies = [r['val_acc'] for r in history]
plt.plot(accuracies, 'x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()

# Evaluate on test train_dataset
result = evaluate(model, test_loader)
print(result)

#lower learning rate, accuracy too low
model_2 = MnistModel()
# Evaluating the model prior to training
evaluate(model_2, val_loader)
history_2 = fit(50, 1e-5, model_2, train_loader, val_loader)
# Visualizing how our model performed after each epoch
accuracies_2 = [r['val_acc'] for r in history_2]
plt.plot(accuracies_2, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()
history_3 = fit(50, 1e-5, model_2, train_loader, val_loader)


# Visualizing total performance of the model across 100 epoch
accuracies_3 = [r['val_acc'] for r in history_3]
accuracies_model2 = accuracies_2 + accuracies_3
plt.plot(accuracies_model2, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()
# Evaluate on test train_dataset
result_2 = evaluate(model_2, test_loader)
print(result_2)

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    return torch.max(yb, dim=1)[1][0].item()

img, label = test_dataset[929]
plt.imshow(img.reshape(28,28), cmap='gray')
print('Label:', letters[label.item()], ', Predicted:', letters[predict_image(img, model_2)])
img, label = test_dataset[47]
plt.imshow(img.reshape(28,28), cmap='gray')
print('Label:', letters[label.item()], ', Predicted:', letters[predict_image(img, model_2)])