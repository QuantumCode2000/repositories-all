1. Preprocesamiento de los Datos
El preprocesamiento de datos es esencial para asegurar que las imágenes estén en un formato adecuado para el entrenamiento del modelo. En este proyecto, el preprocesamiento incluye la lectura de imágenes desde las carpetas, su redimensionamiento, y la asignación de etiquetas.

python
Copy code
import os
import numpy as np
import cv2

# se guardan las rutas de las carpetas de entrenamiento y validacion
entrenamiento = '../data/dataset/entrenamiento'
validacion = '../data/dataset/validacion'

listaEntrenamiento = os.listdir(entrenamiento) # Obtiene la lista de nombres de archivos en la carpeta de entrenamiento
listaValidacion = os.listdir(validacion) # Obtiene la lista de nombres de archivos en la carpeta de validación

ancho, alto = 224, 224

etiquetas = [] # Lista para almacenar las etiquetas de las imágenes
fotos = [] # Lista para almacenar las imágenes
datos_entrenamiento = [] # Lista para almacenar los datos de entrenamiento
con = 0

etiquetas2 = [] # Lista para almacenar las etiquetas de las imágenes de validación
fotos2 = [] # Lista para almacenar las imágenes de validación
datos_validacion = [] # Lista para almacenar los datos de validación
con2 = 0

# Lectura de las imágenes de las carpetas de entrenamiento
for nameDir in listaEntrenamiento:
    nombre = entrenamiento + "/" + nameDir
    for nameFile in os.listdir(nombre):
        etiquetas.append(con) # Agrega la etiqueta (con) a la lista de etiquetas
        img = cv2.imread(nombre + "/" + nameFile, 0) # Lee la imagen en escala de grises
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_CUBIC) # Cambia el tamaño de la imagen
        img = img.reshape(ancho, alto, 1) # Cambia la forma de la imagen
        datos_entrenamiento.append([img, con]) # Agrega la imagen y su etiqueta a los datos de entrenamiento
        fotos.append(img) # Agrega la imagen a la lista de fotos
    con += 1

# Lectura de las imágenes de las carpetas de validación
for nameDir2 in listaValidacion:
    nombre2 = validacion + "/" + nameDir2
    for nameFile2 in os.listdir(nombre2):
        etiquetas2.append(con2) # Agrega la etiqueta (con2) a la lista de etiquetas de validación
        img2 = cv2.imread(nombre2 + "/" + nameFile2, 0) # Lee la imagen en escala de grises
        img2 = cv2.resize(img2, (ancho, alto), interpolation=cv2.INTER_CUBIC) # Cambia el tamaño de la imagen
        img2 = img2.reshape(ancho, alto, 1) # Cambia la forma de la imagen
        datos_validacion.append([img2, con2]) # Agrega la imagen y su etiqueta a los datos de validación
        fotos2.append(img2) # Agrega la imagen a la lista de fotos de validación
    con2 += 1
2. Conversión a Escala de Grises
La conversión de las imágenes a escala de grises se realiza para simplificar el procesamiento y reducir la complejidad del modelo, ya que en algunos casos la información de color no es necesaria para la tarea de clasificación.

python
Copy code
# Lectura de las imágenes en escala de grises ya realizada en el preprocesamiento:
# cv2.imread(nombre + "/" + nameFile, 0)
3. Normalización
La normalización de las imágenes implica escalar los valores de los píxeles al rango [0, 1]. Esto mejora la eficiencia del modelo durante el entrenamiento.

python
Copy code
# Normalización de las imágenes
fotos = np.array(fotos).astype('float') / 255
print(fotos.shape)
fotos2 = np.array(fotos2).astype('float') / 255
print(fotos2.shape)

etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)
4. Data Augmentation (Aumentación de Datos)
La aumentación de datos se utiliza para aumentar la diversidad del conjunto de datos de entrenamiento mediante la creación de variaciones de las imágenes existentes. Esto ayuda a mejorar la generalización del modelo y prevenir el sobreajuste.

python
Copy code
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creación de un generador de imágenes para el aumento de datos
imgTrainGen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
)
imgTrainGen.fit(fotos)

# Visualización de imágenes generadas
import matplotlib.pyplot as plt

for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size=1, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[0], cmap='gray')
    plt.show()
    break

imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size=32)
5. Métricas: Pérdida (Loss) y Precisión (Accuracy)
El modelo se compila utilizando el optimizador 'adam' y la función de pérdida 'binary_crossentropy'. Durante el entrenamiento, se monitoriza la precisión (accuracy) y la pérdida (loss) para evaluar el rendimiento del modelo.

python
Copy code
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Definición del modelo de red neuronal convolucional
ModeloCNN2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ancho, alto, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
ModeloCNN2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
ModeloCNN2.fit(imgTrain, batch_size=32, epochs=10, validation_data=(fotos2, etiquetas2), callbacks=[TensorBoard(log_dir='./logs/ModeloCNN2')])
