# %% Libraries

import tensorflow as tf
from .architecture import build_model_pretrained
import pickle
import numpy as np

# %% Globals

# String to number class
DICT_LABELS = {
    0: "classical", 
    1: "flamenco", 
    2: "hiphop", 
    3: "jazz", 
    4: "pop",
    5: "r&b",
    6: "reggaeton"
}

# %% Functions

def load_label_dict(file_path: str) -> dict:
    
    """
    Load the label dictionary from a file using pickle.

    Args:
        file_path (str): File path to load the dictionary.

    Returns:
        dict: Loaded label dictionary.
    """

    with open(file_path, 'rb') as f:
        
        return pickle.load(f)

    
def load_model(model_path: str, input_shape: tuple = (224, 224, 3), num_classes: int = len(DICT_LABELS)) -> tf.keras.Model:

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
    return DICT_LABELS[prediction]