from YogaDatasetClass import YogaDataset

ds_link = '/content/drive/MyDrive/ViT_proyect/Images'
archivo_train_txt = '/content/drive/MyDrive/ViT_proyect/yoga_train.txt'
num_clases = 82
clases = ["Clase {}".format(i) for i in range(num_clases)]

with open(archivo_train_txt, 'r') as archivo:
    lineas = archivo.readlines()

# Transformaciones 
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Crear el conjunto de datos y el cargador de datos para entrenamiento
conjunto_entrenamiento = YogaDataset(archivo_train_txt, ds_link, num_clases, transform=transformaciones)



# Normalización de la data
from torch.utils.data import random_split
import torch
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

# Crear el conjunto de datos para entrenamiento y validación
train_set = YogaDataset(archivo_train_txt, ds_link, num_clases, transform=transformaciones)

# Definir la proporción de datos para el conjunto de validación
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size

torch.random.manual_seed(seed)
train_set, val_set = random_split(train_set, [train_size, val_size])

# Dataloader para el conjunto de entrenamiento
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader  = DataLoader(val_set, batch_size=32, shuffle=True)

# Conjunto y loaders de la data de prueba
archivo_test_txt = '/content/drive/MyDrive/ViT_proyect/yoga_test.txt'

test_set = YogaDataset(archivo_test_txt, ds_link, num_clases, transform=transformaciones)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
