# %% Libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ConvNeXtSmall

# %% Functions

def data_augmentation_layer(inputs: tf.Tensor, seed: int = 42) -> tf.Tensor:

    """
    Apply data augmentation to the input data.

    Args:
        inputs (tf.Tensor): The input data.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        tf.Tensor: The augmented input data.
    """
    
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed = seed),
            layers.SpatialDropout2D(rate = 0.2, seed = seed),
        ],
        name = "data_augmentation"
    )

    return data_augmentation(inputs)

def build_model_pretrained(input_shape: tuple, num_classes: int, fc_layers: list = [256, 128], dropout: float = 0.2) -> keras.Model:
    
    """
    Build a classification model using a pre-trained ConvNeXtSmall model.

    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of classes in the classification problem.
        fc_layers (list): A list of integers specifying the number of neurons in each fully connected layer. Defaults to [256, 128].
        dropout (float): The dropout rate for the fully connected layers. Defaults to 0.2.

    Returns:
        keras.Model: The built classification model.
    """

    # Load the pre-trained ConvNeXtSmall model
    base_model = ConvNeXtSmall(weights = "imagenet", include_top = False, input_shape = input_shape)
    
    # Freeze the layers of the pre-trained model due to limited memory
    for layer in base_model.layers:

        # Set the trainable attribute of each layer to False
        layer.trainable = False

    # Create an input layer to introduce the data
    inputs = layers.Input(shape=input_shape)

    # Use data augmentation
    x = data_augmentation_layer(inputs)
    
    # Use the features obtained from the pre-trained model
    x = base_model(x)
    
    # Reduce the dimensionality using global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Use a few fully connected layers (MLPs)
    for fc in fc_layers:

        # Add a dense layer with the specified number of neurons and Gelu activation
        x = layers.Dense(fc, activation='gelu')(x)

        # Add a dropout layer with the specified dropout rate
        x = layers.Dropout(dropout)(x)
    
    # Get the output of the classification using a softmax activation function
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Create the model
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ClassifierModel")