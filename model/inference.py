# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from .architecture import build_model_pretrained
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

    # Build the model
    model = build_model_pretrained(input_shape, num_classes)
    
    # Load the weights of the model
    model.load_weights(model_path)
    
    # Return the model
    return model

def make_prediction(model: tf.keras.Model, data) -> str:

    """
    Make a prediction using a loaded model.

    Args:
        model (tf.keras.Model): Loaded model.
        data: Input data to make a prediction on.

    Returns:
        str: Predicted label.
    """

    # Make a prediction using the model
    prediction = np.argmax(model.predict(data))
    
    # Get the corresponding label from the dictionary
    return cg.DICT_LABELS[prediction]