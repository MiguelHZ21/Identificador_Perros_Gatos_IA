import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Evita que TensorFlow tome toda la VRAM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada correctamente")
    except RuntimeError as e:
        print(e)
else:
    print("No se detectó GPU, usando CPU")