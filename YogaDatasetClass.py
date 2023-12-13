import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class YogaDataset(Dataset):
    def __init__(self, archivo_txt, ruta_imagenes, num_clases, transform=None):
        self.ruta_imagenes = ruta_imagenes
        self.transform = transform
        self.num_clases = num_clases

        # Leer el archivo de texto y cargar las rutas de las imágenes y etiquetas
        with open(archivo_txt, 'r') as archivo:
            lineas = archivo.readlines()

        # Filtrar las líneas para excluir las muestras sin imagen
        self.lineas = [linea.strip().split(',') for linea in lineas]

    def __len__(self):
        return len(self.lineas)

    def __getitem__(self, idx):
        linea = self.lineas[idx]

        # Obtener la dirección de la imagen
        direccion_imagen = linea[0]

        # Cargar la imagen si existe y manejar excepciones
        ruta_completa_imagen = os.path.join(self.ruta_imagenes, direccion_imagen)

        imagen = Image.open(ruta_completa_imagen)
        if imagen.mode == 'P':
            imagen = imagen.convert('RGBA')
            imagen = imagen.convert('RGB')
        else:
            imagen = imagen.convert('RGB')

        # Aplicar transformaciones si se proporcionan
        if self.transform:
            imagen = self.transform(imagen)

        # Ajustar la etiqueta según la cantidad de clases
        if self.num_clases == 6:
            etiqueta = int(linea[1])

        elif self.num_clases == 20:
            etiqueta = int(linea[2])

        elif self.num_clases == 82:
            etiqueta = int(linea[3])

        return imagen, etiqueta
