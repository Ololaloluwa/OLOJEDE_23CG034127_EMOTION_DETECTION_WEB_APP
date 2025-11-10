# model.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf

def load_trained_model(path="model.h5"):
    """
    Loads a Keras model from path.
    Returns (model, label_map)
    label_map: list of string labels in order of model outputs. If your model does not embed labels,
    you can customize this list to match training. Default emotion labels provided.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please place your model.h5 in project root.")
    model = tf.keras.models.load_model(path, compile=False)
    # Default label map (change to match how your model was trained)
    default_map = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    # Try to detect if model has attribute label_map (user-saved). If not, use default.
    label_map = default_map
    # If your model outputs length different from default_map, create generic labels
    out_dim = int(model.output_shape[-1])
    if out_dim != len(label_map):
        label_map = [f"label_{i}" for i in range(out_dim)]
    return model, label_map

def preprocess_image_for_model(pil_img, model):
    """
    Converts a PIL image to the model input, using the model.input_shape if available.
    Returns a batch (1, H, W, C) numpy array.
    """
    # Determine expected input shape: (None, H, W, C) or (None, C, H, W)
    input_shape = model.input_shape  # e.g. (None, 48, 48, 1) or (None, 224, 224, 3)
    if len(input_shape) == 4:
        _, h, w, c = input_shape
    elif len(input_shape) == 3:
        # (H, W, C)
        h, w, c = input_shape
    else:
        # fallback
        h, w, c = (224, 224, 3)

    # If model expects grayscale (c == 1), convert
    if c == 1:
        img = pil_img.convert("L").resize((w, h))
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape((h, w, 1))
    else:
        img = pil_img.resize((w, h)).convert("RGB")
        arr = np.array(img).astype("float32") / 255.0

    # If model expects channels-first, convert:
    # Most Keras models expect channels-last; we will assume channels-last for simplicity.
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr

# Optional tiny training stub (not required)
def train_stub():
    """
    A placeholder showing how training code could be structured.
    Replace dataset paths and augmentation code as required.
    """
    # Example: simple flow_from_directory approach
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = train_datagen.flow_from_directory(
        "data/train",
        target_size=(48,48),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )
    val_gen = train_datagen.flow_from_directory(
        "data/train",
        target_size=(48,48),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )
    model = models.Sequential([
        layers.Input(shape=(48,48,3)),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(train_gen.num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, epochs=10, validation_data=val_gen)
    model.save("model.h5")
