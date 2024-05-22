# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from keras_cv.layers import CutMix, ChannelShuffle, RandomCutout, RandAugment
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import dask.array as da
from dask.delayed import delayed

# %% Let's use mixed precision to reduce memory consumption.

mixed_precision.set_global_policy('mixed_float16')

# %% Constants

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100

# %% Functions

def seed_init(seed: int) -> None:

    """
    Initialize random seeds for NumPy, TensorFlow, and Python's random module.

    This function ensures reproducibility of results by setting the same seed value
    for all random number generators used in the program.

    Args:
        seed (int): The seed value to use for random initialization.

    Returns:
        None
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

import os
import numpy as np
from PIL import Image
from dask.delayed import delayed
import dask.array as da

def load_image(file_path, input_shape):

    """
    Load an image from a file path and resize it to the input shape.

    Args:
        file_path (str): The path to the image file.
        input_shape (tuple): The target size for the loaded image.

    Returns:
        numpy array: The loaded image as a numpy array, or None if there's an error.
    """

    try:

        # Open the image file
        img = Image.open(file_path)
        
        # Convert the image to RGB mode
        img = img.convert('RGB')
        
        # Resize the image to the input shape
        img = img.resize(input_shape)
        
        # Convert the image to a numpy array
        img_array = np.array(img)
        
        return img_array

    except OSError:

        # Print an error message if there's an issue loading the image
        print(f"Error loading image: {file_path}")
        return None

def load_data_from_folder(path, input_shape=(224, 224)):

    """
    Load image data from a folder structure where each subfolder represents a class.

    Args:
        path (str): The path to the root folder containing the class subfolders.
        input_shape (tuple): The target size for the loaded images. Defaults to (224, 224).

    Returns:
        tuple: A tuple containing the loaded data and labels as numpy arrays.
    """

    data = []
    labels = []

    for i, class_name in enumerate(os.listdir(path)):

        # Construct the full path to the class folder
        class_folder = os.path.join(path, class_name)
        
        # Get a list of file paths for the images in the class folder
        file_paths = [os.path.join(class_folder, file_name) for file_name in os.listdir(class_folder) if file_name.endswith('.png')]
        
        # Load the images in parallel using dask
        img_arrays = [delayed(load_image)(file_path, input_shape) for file_path in file_paths]
        img_arrays = da.compute(*img_arrays)
        
        # Filter out any images that failed to load
        img_arrays = [img_array for img_array in img_arrays if img_array is not None]
        
        # Add the loaded images and labels to the data and labels lists
        data.extend(img_arrays)
        labels.extend([i] * len(img_arrays))

    # Convert the data and labels lists to numpy arrays and return them
    return np.array(data, dtype=np.uint8), np.array(labels)

def create_data_generators(train_data: np.ndarray, train_labels: np.ndarray, val_data: np.ndarray, val_labels: np.ndarray, batch_size: int = BATCH_SIZE) -> tuple:
    
    """
    Create data generators for training and validation data.

    Args:
        train_data (np.ndarray): Training data.
        train_labels (np.ndarray): Training labels.
        val_data (np.ndarray): Validation data.
        val_labels (np.ndarray): Validation labels.
        batch_size (int): Batch size.

    Returns:
        train_generator: Data generator for training data.
        val_generator: Data generator for validation data.
    """

    # Create an instance of the ImageDataGenerator for training data
    train_datagen = ImageDataGenerator(horizontal_flip = True)
    
    # Create an instance of the ImageDataGenerator for validation data
    val_datagen = ImageDataGenerator()

    # Create a data generator for the training data using the flow method
    # This will generate batches of the training data and labels
    train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
    
    # Create a data generator for the validation data using the flow method
    # This will generate batches of the validation data and labels
    val_generator = val_datagen.flow(val_data, val_labels, batch_size=batch_size)

    # Return the training and validation data generators
    return train_generator, val_generator

def data_augmentation(data: np.ndarray, labels: np.ndarray, seed: int) -> None:

    data = {"images": data, "labels": tf.cast(labels, dtype = np.float32)}

    cutmix_layer = CutMix(alpha = 1.0, seed = seed)
    channel_shuffle_layer = ChannelShuffle(groups = 3, seed = seed)

    data = cutmix_layer(data)
    data = channel_shuffle_layer(data)

    return data["images"], data["labels"]

def build_model(input_shape: tuple, num_classes: int, fc_layers: list = [256, 128], dropout: float = 0.2) -> keras.Model:
    
    """
    Build a classification model using a pre-trained VGG16 model.

    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of classes in the classification problem.
        fc_layers (list): A list of integers specifying the number of neurons in each fully connected layer. Defaults to [256, 128].
        dropout (float): The dropout rate for the fully connected layers. Defaults to 0.2.

    Returns:
        keras.Model: The built classification model.
    """

    # Load the pre-trained VGG16 model
    base_model = tf.keras.applications.VGG16(weights = "imagenet", include_top = False, input_shape = (input_shape[0], input_shape[1], 3))
    
    # Freeze the layers of the pre-trained model due to limited memory
    for layer in base_model.layers:

        # Set the trainable attribute of each layer to False
        layer.trainable = False

    # Create an input layer to introduce the data
    inputs = layers.Input(shape=(input_shape[0], input_shape[1], 3))
    
    # Preprocess the input data to apply the adjustments of VGG16
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    
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
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ClassifierModel")
    
    # Compile the model with the AdamW optimizer and categorical cross-entropy loss
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate = 1e-3, weight_decay = 2e-4), 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    
    # Print the summary of the model
    print(model.summary())
    
    # Return the model
    return model

def class_weights_calculation(labels: np.ndarray, y_train: np.ndarray) -> dict:

    """
    Calculate class weights to address class imbalance in the dataset.

    This function calculates the class weights using the 'balanced' method, which
    assigns more weight to classes with lower frequencies in the dataset.

    Args:
        labels (np.ndarray): The labels of the dataset.
        y_train (np.ndarray): The one-hot encoded labels of the training set.

    Returns:
        dict: The class weights.
    """

    # Calculate the class weights using the 'balanced' method
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(labels), 
                                         y=np.argmax(y_train, axis=-1))
    
    # Convert the class weights to a dictionary
    class_weights = dict(enumerate(class_weights))
    
    # Return the class weights
    return class_weights

def train_model(model: keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, class_weights: dict, epochs: int = EPOCHS) -> tf.keras.callbacks.History:
    
    """
    Train the model on the training dataset.

    Args:
        model (keras.Model): The model to be trained.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        class_weights (dict): The class weights to address class imbalance.
        epochs (int): The number of epochs to train the model. Defaults to EPOCHS.

    Returns:
        tf.keras.callbacks.History: The training history.
    """

    # Train the model on the training dataset with the specified class weights
    return model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, class_weight=class_weights)

def evaluate_model(model: keras.Model, val_dataset: tf.data.Dataset) -> float:

    """
    Evaluate the model on the validation dataset.

    Args:
        model (keras.Model): The model to be evaluated.
        val_dataset (tf.data.Dataset): The validation dataset.

    Returns:
        float: The accuracy of the model on the validation dataset.
    """

    # Evaluate the model on the validation dataset
    loss, accuracy = model.evaluate(val_dataset)
    
    # Return the accuracy
    return accuracy

def main(path: str) -> None:

    """
    Main function to execute the image classification pipeline.

    Args:
        path (str): The path to the folder containing the image data.

    Returns:
        None
    """

    # Initialize the seeds for reproducible results
    seed_init(seed = SEED)

    # Load data from the specified folder
    data, labels = load_data_from_folder(path)
    num_classes = len(np.unique(labels))

    # Print information about the data
    print(f"Num data: {data.shape}")
    print(f"Num classes: {num_classes}")

    """
    # Apply data augmentation
    data, labels = data_augmentation(data, labels, SEED)

    # Print information about the data
    print(f"Num data: {data.shape}")
    print(f"Num classes: {num_classes}")
    """

    # Check if the number of images is equal to the number of labels
    assert len(data) == len(labels), "Number of images is not equal to the number of labels"

    # One-hot encode the labels
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes = num_classes)

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels_onehot, test_size=0.2, random_state=42, stratify=labels)
    
    # Calculate class weights to mitigate the class imbalance problem
    class_weights = class_weights_calculation(labels = labels, y_train = train_labels)

    # Create data generators for the training and validation data
    train_dataset, val_dataset = create_data_generators(train_data, train_labels, val_data, val_labels)

    # Build and train the classification model
    input_shape = (train_data.shape[1], train_data.shape[2], 3)
    model = build_model(input_shape, num_classes)
    history = train_model(model, train_dataset, val_dataset, class_weights, epochs = EPOCHS)

    # Save model
    model.save_weights('./model/trained_model/music_classifier')

    # Evaluate the model on the validation data
    accuracy = evaluate_model(model, val_dataset)
    print(f'Validation accuracy: {accuracy:.2f}')

if __name__ == '__main__':

    path = './dataset'
    main(path)