import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Rutas
ruta_base = './'
ruta_train = os.path.join(ruta_base, 'train')
ruta_test = os.path.join(ruta_base, 'test')

# Tamaño de imagen
IMG_SIZE = (64, 64)

# Visualización de ejemplos
def mostrar_ejemplos():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    clases = os.listdir(ruta_train)
    for i, clase in enumerate(clases):
        clase_path = os.path.join(ruta_train, clase)
        img_name = os.listdir(clase_path)[0]
        img_path = os.path.join(clase_path, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        axes[i].imshow(img)
        axes[i].set_title(f'Clase: {clase}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("distribucion_clases.png")
    plt.show()

mostrar_ejemplos()

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    ruta_train,
    target_size=IMG_SIZE,
    batch_size=64,
    class_mode='sparse',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    ruta_train,
    target_size=IMG_SIZE,
    batch_size=64,
    class_mode='sparse',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    ruta_test,
    target_size=IMG_SIZE,
    batch_size=64,
    class_mode='sparse',
    shuffle=False
)

# Modelo base con MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
#history = model.fit(train_gen, validation_data=val_gen, epochs=3)

# Evaluación
val_loss, val_acc = model.evaluate(val_gen)
print(f'Precisión en validación: {val_acc * 100:.2f}%')

# --- NUEVO MODELO KNN SIN AUGMENTACIÓN ---

# Generador sin augmentación
datagen_no_aug = ImageDataGenerator(rescale=1./255)

train_gen_no_aug = datagen_no_aug.flow_from_directory(
    os.path.join(ruta_base, 'train'),
    target_size=(64, 64),
    batch_size=10000,
    class_mode='sparse',
    shuffle=True
)

test_gen_no_aug = datagen_no_aug.flow_from_directory(
    os.path.join(ruta_base, 'test'),
    target_size=(64, 64),
    batch_size=10000,
    class_mode='sparse',
    shuffle=False
)

# Carga los datos del generador
x_train, y_train = next(train_gen_no_aug)
x_test, y_test = next(test_gen_no_aug)

# Aplanar imágenes
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Normalizar
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_scaled, y_train)

# Evaluación
acc_knn = knn.score(x_test_scaled, y_test)
print(f'KNN sin augmentación: {acc_knn * 100:.2f}%')

