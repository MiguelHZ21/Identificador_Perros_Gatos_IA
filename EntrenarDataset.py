import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# CARGAR DATASET
# -----------------------------
ruta_dataset = r"D:\Universidad\9Noveno Semestre\Inteligencia Artificial\Archivos Unidad 1\Reconocimiento de animales\datos_etiquetados_21_02_26_aumentado.pickle"

with open(ruta_dataset, "rb") as handle:
    X, y = pickle.load(handle)


# -----------------------------
# NORMALIZACIÓN
# -----------------------------
X = X.astype('float32') / 255.0

# -----------------------------
# ONE-HOT ENCODING
# -----------------------------
y = to_categorical(y, 2)

# -----------------------------
# DIVISIÓN TRAIN / TEST
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# APLANAR IMÁGENES (500x500x3 → 750000 features)
# -----------------------------
#X_train_flat = X_train.reshape((X_train.shape[0], -1))
#X_test_flat = X_test.reshape((X_test.shape[0], -1))


# -----------------------------
# MODELO MLP AJUSTADO (más pequeño)
# -----------------------------
model_mlp = Sequential([
    #Flatten(input_shape=(100*100*3,)),
    #Dense(1024, activation='relu'),
    #Dense(1024, activation='relu'),
    #Dense(512, activation='relu'),
    #Dense(2, activation='softmax')
    Conv2D(512, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')  # Binario: perro vs gato
])

model_mlp.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

# Entrenar el modelo
history_mlp = model_mlp.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

# -----------------------------
# EVALUACIÓN FINAL
# -----------------------------
loss, acc = model_mlp.evaluate(X_test, y_test)
print(f"Accuracy en test: {acc:.4f}")
