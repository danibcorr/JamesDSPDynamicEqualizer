# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import layers
from .architecture import build_model_pretrained, build_model_scratch
import pickle
import numpy as np
import config as cg

# %% Functions

def load_model(model_path: str, input_shape: tuple = cg.INPUT_SHAPE, num_classes: int = len(cg.DICT_LABELS)) -> tf.keras.Model:

    """
    Load a pre-trained model from a file and return it.

    Args:
        model_path (str): Path to the model weights file.
        input_shape (tuple): Input shape of the model.
        num_classes (int): Number of classes in the classification problem.

    Returns:
        tf.keras.Model: Loaded model.
    """

    #  Build and load the model
    model = build_model_scratch((128, 129, 1), num_classes)
    model.load_weights(model_path)

    return model

def make_prediction(model: tf.keras.Model, data: np.ndarray) -> str:

    """
    Make a prediction using a loaded model.

    Args:
        model (tf.keras.Model): Loaded model.
        data: Input data to make a prediction on.

    Returns:
        str: Predicted label.
    """

    # Make a prediction using the model
    prediction = model.predict(data)
    product_probs = np.prod(prediction, axis = 0)

    # Normalize probabilities
    product_probs /= np.sum(product_probs)

    # Obtain the key of the dictionary to obtain the label
    key_genre = np.argmax(product_probs)
    
    # Get the corresponding label from the dictionary
    return cg.DICT_LABELS_PREDICT[key_genre]