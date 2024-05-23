# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import dask.array as da
from dask.delayed import delayed
from model.architecture import build_model_pretrained
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from model.schedulers.warmupcosine import WarmUpCosine
from model.architecture import build_model_pretrained

# %% Let's use mixed precision to reduce memory consumption.

mixed_precision.set_global_policy('mixed_float16')

# %% Constants

SEED = 42
BATCH_SIZE = 32
EPOCHS = 1
INPUT_SHAPE = (224, 224, 3)

# String to number class
DICT_LABELS = {
    "classical": 0, 
    "flamenco": 1, 
    "hiphop": 2, 
    "jazz": 3, 
    "pop": 4,
    "r&b": 5,
    "reggaeton": 6
}

# %% Functions

def create_label_dict(labels: np.ndarray) -> dict:
    
    """
    Create a dictionary mapping label indices to label names.

    Args:
        labels (np.ndarray): List of unique labels.

    Returns:
        dict: Label dictionary.
    """
    
    return {i: label for i, label in enumerate(labels)}

def save_label_dict(label_dict: dict, file_path: str) -> None:

    """
    Save the label dictionary to a file using pickle.

    Args:
        file_path (str): File path to save the dictionary.
    """

    with open(file_path, 'wb') as f:

        pickle.dump(label_dict, f)

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

def load_image(file_path: str, input_shape: tuple) -> np.ndarray or None:

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

def load_data_from_folder(path: str, input_shape: tuple = INPUT_SHAPE) -> tuple:

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
        img_arrays = [delayed(load_image)(file_path, (input_shape[0], input_shape[1])) for file_path in file_paths]
        img_arrays = da.compute(*img_arrays)
        
        # Filter out any images that failed to load
        img_arrays = [img_array for img_array in img_arrays if img_array is not None]
        
        # Add the loaded images and labels to the data and labels lists
        data.extend(img_arrays)
        labels.extend([DICT_LABELS[class_name]] * len(img_arrays))

    # Convert the data and labels lists to numpy arrays and return them
    return np.array(data, dtype = np.uint8), np.array(labels)

def create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> None:

    """
    Create a confusion matrix from true labels and predicted labels.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        labels (list): List of unique labels.

    Returns:
        np.ndarray: Confusion matrix.
    """

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

    # Display the confusion matrix
    plt.figure(figsize = (10, 7))
    sns.heatmap(conf_mat, annot = True, fmt = 'd', cmap = 'Reds')
    plt.xlabel('Predict', fontsize = 15, weight = 'bold')
    plt.ylabel('True', fontsize = 15, weight = 'bold')
    plt.show()

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
    train_datagen = ImageDataGenerator()
    
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

def main(dataset_path: str, model_save_path: str, labels_dict_path: str) -> None:

    """
    Main function to execute the image classification pipeline.

    Args:
        dataset_path (str): The path to the folder containing the image data.

    Returns:
        None
    """

    # Initialize the seeds for reproducible results
    seed_init(seed = SEED)

    # Load data from the specified folder
    data, labels = load_data_from_folder(dataset_path)
    classes = np.unique(labels)
    num_classes = len(classes)

    # Print information about the data
    print(f"Num data: {data.shape}")
    print(f"Num classes: {num_classes}")
    print(f"Min val: {np.min(data)}")
    print(f"Max val: {np.max(data)}")

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

    # Total steps for the scheduler
    total_steps = int((len(train_data) / BATCH_SIZE) * EPOCHS)
    warmup_steps = int(total_steps * 0.15)
    scheduled_lrs = WarmUpCosine(lr_start = 1e-5, lr_max = 1e-3, warmup_steps = warmup_steps, total_steps = total_steps)
                 
    # Build the classification model
    model = build_model_pretrained(INPUT_SHAPE, num_classes)

    # Compile the model with the AdamW optimizer and categorical cross-entropy loss
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate = scheduled_lrs, weight_decay = 2e-4), 
                  loss=tf.keras.losses.CategoricalCrossentropy(), 
                  metrics=['accuracy'])
    print(model.summary())

    # Train the model
    history = train_model(model, train_dataset, val_dataset, class_weights, epochs = EPOCHS)

    # Save model
    model.save_weights(model_save_path)

    # Evaluate the model using the confusion matrix
    y_pred = model.predict(val_dataset)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(val_labels, axis=1)
    labels = np.unique(y_true_class)
    conf_mat = create_confusion_matrix(y_true_class, y_pred_class, labels)

# %% Main

if __name__ == '__main__':

    dataset_path = 'dataset'
    model_save_path = 'model/trained_model/music_classifier'
    labels_dict_path = 'model/labels_dict.pkl'

    main(dataset_path, model_save_path, labels_dict_path)