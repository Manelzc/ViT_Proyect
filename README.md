# ViT_Proyect
En este repositorio se encuentran los módulos necesarios para la creación, entrenamiento y evaluación de un modelo ViT a través del dataset Yoga-82

-Library_info contiene todas las versiones de las librerias instaladas. Es importante mencionar que este proyecto se generó en Google Colab, por lo que se utilizan las versiones de las librerías de aquella plataforma.

-DataClean es un código para limpiar la data, tanto en imágenes como en archivos de texto.

-En ViT.py se implementa el modelo utilizado.

-YogaDatasetClass contiene la clase para generar el dataset utilizado, a través de los archivos de texto e imágenes ya filtradas.

-YogaDataLoaders genera los dataloaders para la implementación 1 y 2 mencionadas en el informe. De todas formas, dentro dentro de cada código de entrenamiento se definen estos.

-La carpeta train contiene los códigos utilizados para entrenar las 4 implementaciones (los números indican a qué implementación corresponde), tanto directamente como a través de transferlearning.

-La implementación 4 es la que finalmente funcionó mejor (es decir, train_4.py y TransferLearning_4.py).
