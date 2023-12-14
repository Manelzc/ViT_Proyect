import os
from PIL import Image

def eliminar_imagenes_invalidas(directorio_raiz):
    # Recorrer recursivamente el directorio
    for directorio_actual, carpetas, archivos in tqdm(os.walk(directorio_raiz), desc="Procesando directorio"):
        for archivo in tqdm(archivos, desc=f"Procesando carpeta {directorio_actual}"):
            # Ruta completa del archivo
            ruta_archivo = os.path.join(directorio_actual, archivo)

            try:
                # Intentar abrir la imagen
                imagen = Image.open(ruta_archivo).convert('RGB')

                # Cerrar la imagen después de abrirla para liberar recursos
                imagen.close()
            except Exception as e:
                # Si hay un error al abrir la imagen, eliminar el archivo
                print(f"Error al abrir la imagen {ruta_archivo}: {e}")
                os.remove(ruta_archivo)
                print(f"La imagen {ruta_archivo} ha sido eliminada.")

ds_link = '/content/drive/MyDrive/ViT_proyect/Images'
eliminar_imagenes_invalidas(ds_link)

import os
import tqdm

def limpiar_archivo_txt(archivo_txt, ruta_imagenes):
    # Lista para almacenar las líneas válidas
    lineas_validas = []

    # Leer el archivo de texto y cargar las rutas de las imágenes
    with open(archivo_txt, 'r') as archivo:
        lineas = archivo.readlines()

    # Recorrer cada línea del archivo
    for linea in lineas:
        # Obtener la dirección de la imagen
        direccion_imagen = linea.strip().split(',')[0]

        # Comprobar si la imagen existe en el dataset
        ruta_completa_imagen = os.path.join(ruta_imagenes, direccion_imagen)
        if os.path.exists(ruta_completa_imagen):
            # Si la imagen existe, agregar la línea a la lista de líneas válidas
            lineas_validas.append(linea)

    # Escribir las líneas válidas de vuelta al archivo
    with open(archivo_txt, 'w') as archivo:
        archivo.writelines(lineas_validas)

archivo_txt = '/content/drive/MyDrive/ViT_proyect/yoga_train.txt'
ruta_imagenes = '/content/drive/MyDrive/ViT_proyect/Images'
limpiar_archivo_txt(archivo_txt, ruta_imagenes)
contar_lineas('/content/drive/MyDrive/ViT_proyect/yoga_train.txt')

archivo_txt = '/content/drive/MyDrive/ViT_proyect/yoga_test.txt'
ruta_imagenes = '/content/drive/MyDrive/ViT_proyect/Images'
limpiar_archivo_txt(archivo_txt, ruta_imagenes)
contar_lineas('/content/drive/MyDrive/ViT_proyect/yoga_test.txt')
