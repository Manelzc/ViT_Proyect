from torch.utils.data import random_split
import torch

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor,Resize,Compos, RandomHorizontalFlip, RandomResizedCrop, ColorJitter
from torchvision import datasets
from collections import Counter

from ViT import ViT

# Preparación de la Data
#------------------------------------------------------------------------------------------
from YogaDatasetClass import YogaDataset

ds_link = '/content/drive/MyDrive/ViT_proyect/Images'
archivo_train_txt = '/content/drive/MyDrive/ViT_proyect/yoga_train.txt'
num_clases = 20
seed = 42
torch.manual_seed(seed)

mean = [0.669, 0.649, 0.623]
std = [0.218, 0.218, 0.238]

archivo_train_txt = '/content/drive/MyDrive/ViT_proyect/yoga_train.txt'

# Transformaciones 
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
])

# Transformaciones con data augmentation
transformaciones_aumentada = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
])

# Crear el conjunto de datos para entrenamiento y validación
train_set = YogaDataset(archivo_train_txt, ds_link, num_clases, transform=transformaciones_aumentada)

# Definir la proporción de datos para el conjunto de validación
train_size = int(0.7 * len(train_set))  #70% para entrenamiento y 30% para validación
val_size = len(train_set) - train_size

torch.random.manual_seed(seed)
train_set, val_set = random_split(train_set, [train_size, val_size])

# Dataloader para el conjunto de entrenamiento
train_loader = DataLoader(train_set, batch_size=64, shuffle=True) # Cambio de batch size
val_loader  = DataLoader(val_set, batch_size=64, shuffle=True)

# Conjunto y loaders de la data de prueba
archivo_test_txt = '/content/drive/MyDrive/ViT_proyect/yoga_test.txt'

test_set = YogaDataset(archivo_test_txt, ds_link, num_clases, transform=transformaciones)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
#------------------------------------------------------------------------------------------
#Entrenamiento

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
N_PATCHES = 14


assert IMAGE_WIDTH%N_PATCHES==0

PATCH_WIDTH = IMAGE_WIDTH//N_PATCHES
PATCH_HEIGHT = IMAGE_HEIGHT//N_PATCHES


N_EPOCHS = 50
LR = 0.000001
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

print(f"ROI divided into regions of {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{IMAGE_CHANNELS}\nUsing {N_PATCHES}x{N_PATCHES} patches of {PATCH_WIDTH}x{PATCH_HEIGHT}x{IMAGE_CHANNELS}")
model = ViT(
        in_channels= IMAGE_CHANNELS,
        img_size=IMAGE_WIDTH,
        patch_size=IMAGE_WIDTH//N_PATCHES,
        emb_size=768,
        num_heads=12,
        depth=12,
        n_classes=20, # Numero de clases del modelo que se quiere cargar
        dropout=0.1,
        forward_expansion=4
    ).to(device)


# Cargar los pesos preentrenados (comentar si hay que entrenar desde cero)
model.load_state_dict(torch.load("/content/drive/MyDrive/ViT_proyect/models/best_model_6_20_MLP.pth"))

# Congelar todas las capas excepto la capa lineal final
for param in model.parameters():
    param.requires_grad = False

# Habilitar el gradiente para los parámetros de la capa lineal final
for param in model[2][2].parameters():
    param.requires_grad = True

# Modificar la capa Linear en la ClassificationHead con el nuevo número de clases y pasarla al dispositivo
model[2][2] = nn.Linear(768, num_clases).to(device)


pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.2f}MB'.format(size_all_mb))


BEST_MODEL_PATH = "/content/drive/MyDrive/ViT_proyect/models/best_model_6_20_82_MLP.pth"



# Training loop

optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

train_losses = []
val_losses = []
best_val_loss = float('inf')
ep = 0


for epoch in trange(N_EPOCHS, desc="Training"):
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss)

    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

# Validation loop
    with torch.no_grad():
        correct, total = 0, 0
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validating"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            val_loss += loss.detach().cpu().item() / len(val_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"val loss: {val_loss:.2f}")
        print(f"val accuracy: {correct / total * 100:.2f}%")
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            print("Best model saved!")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_val_loss = val_loss
        ep = ep + 1

        # Early stopping
        patience = 5
        if len(val_losses) > patience and all(val_losses[i-1] <= val_losses[i] for i in range(-patience, 0)):
            print(f"Stopping early at epoch {epoch + 1}")
            break

# %%
fig, ax = plt.subplots()

ax.plot(np.arange(ep), train_losses)
ax.plot(np.arange(ep), val_losses)
