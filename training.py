# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import mlflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import dask.array as da
from dask.delayed import delayed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from model.schedulers.warmupcosine import WarmUpCosine
from model.architecture import build_model_pretrained, build_model_scratch
from model.layers.CutMix import cut_mix
import config as cg

# %% Let's use mixed precision to reduce memory consumption.

mixed_precision.set_global_policy('mixed_float16')

# %% Functions

def mlflow_connection() -> str:

    """
    Establishes a connection with MLflow for experiment tracking and logging.

    This function sets the tracking server URI to http://127.0.0.1:8080 for logging.
    It creates a new MLflow experiment named "Music Classifier" and enables system metrics logging.

    Note:
        - Make sure MLflow is installed and running on the specified URI before calling this function.
        - Ensure that the tracking server URI is correctly configured for your MLflow setup.

    Returns:
        str: The name of the run.
    """

    # Set the tracking server URI for logging
    mlflow.set_tracking_uri(uri = "http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Music Classifier")

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()

    # Autolog for tensorflow
    mlflow.tensorflow.autolog(log_datasets = False)

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

def load_image(file_path: str) -> np.ndarray or None:

    """
    Load an image from a file path and resize it to the input shape.

    Args:
        file_path (str): The path to the image file.

    Returns:
        numpy array: The loaded image as a numpy array, or None if there's an error.
    """

    try:

        # Open the image file and convert the image to grayscale mode
        # Convert the image to a numpy array
        return np.array(Image.open(file_path).convert('L'), dtype = np.uint8)

    except OSError:

        # Print an error message if there's an issue loading the image
        print(f"Error loading image: {file_path}")

        return None

def load_data_from_folder(path: str, input_shape: tuple) -> tuple:

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
        img_arrays = [delayed(load_image)(file_path) for file_path in file_paths]
        img_arrays = da.compute(*img_arrays)
        
        # Filter out any images that failed to load
        img_arrays = [img_array for img_array in img_arrays if img_array is not None]
        
        # Add the loaded images and labels to the data and labels lists
        data.extend(img_arrays)
        labels.extend([cg.DICT_LABELS[class_name]] * len(img_arrays))

    # Convert the data and labels lists to numpy arrays and return them
    return np.expand_dims(np.array(data, dtype = np.uint8), -1), np.array(labels, dtype = np.float32)

def create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, run_name: str) -> None:

    """
    Create a confusion matrix from true labels and predicted labels.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)

    # Display the confusion matrix
    plt.figure(figsize = (10, 7))
    sns.heatmap(conf_mat, annot = True, fmt = 'd', cmap = 'Reds')
    plt.xlabel('Predict', fontsize = 15, weight = 'bold')
    plt.ylabel('True', fontsize = 15, weight = 'bold')
    plt.savefig(f'model/trained_model/confusion_matrix_{run_name}.png')

def train_model(model: keras.Model, train_dataset: tuple, val_dataset: tuple, epochs: int) -> tf.keras.callbacks.History:
    
    """
    Train the model on the training dataset.

    Args:
        model (keras.Model): The model to be trained.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        epochs (int): The number of epochs to train the model. Defaults to EPOCHS.

    Returns:
        tf.keras.callbacks.History: The training history.
    """

    with mlflow.start_run() as run:

        history = model.fit(x = train_dataset[0], y = train_dataset[1], epochs = epochs, batch_size = cg.BATCH_SIZE, validation_data = val_dataset)

    return history, run

def evaluate_model(model: keras.Model, val_data: np.ndarray, val_labels: np.ndarray, run_name: str) -> None:

    """
    Evaluate the model on the validation dataset.

    Args:
        model (keras.Model): The model to be evaluated.
        val_dataset (tf.data.Dataset): The validation dataset.

    Returns:
        float: The accuracy of the model on the validation dataset.
    """

    # Evaluate the model using the confusion matrix
    y_pred = model.predict(val_data)
    y_pred_class = np.argmax(y_pred, axis = 1)
    y_true_class = np.argmax(val_labels, axis = 1)
    conf_mat = create_confusion_matrix(y_true_class, y_pred_class, run_name)

def main(dataset_path: str, model_save_path: str, labels_dict_path: str) -> None:

    """
    Main function to execute the image classification pipeline.

    Args:
        dataset_path (str): The path to the folder containing the image data.

    Returns:
        None
    """

    # MLflow connection
    mlflow_connection()

    # Initialize the seeds for reproducible results
    seed_init(seed = cg.SEED)

    # Load data from the specified folder
    data, labels = load_data_from_folder(dataset_path, cg.INPUT_SHAPE)
    classes = np.unique(labels)
    num_classes = len(classes)

    # Print information about the data
    print(f"Data shape: {data.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Num classes: {num_classes}")
    print(f"Min val: {np.min(data)}")
    print(f"Max val: {np.max(data)}")

    # Check if the number of images is equal to the number of labels
    assert len(data) == len(labels), "Number of images is not equal to the number of labels"

    # One-hot encode the labels
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes = num_classes)

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels_onehot, test_size = cg.test_size_split, random_state = cg.SEED, stratify = labels)
    print("Split data, training and validation, done.")
    del data, labels, labels_onehot

    # Split training data in two datasets to use CutMix
    train_data_1, train_data_2, train_labels_1, train_labels_2 = train_test_split(train_data, train_labels, test_size = 0.5, random_state = cg.SEED)
    print("Split data, 2 training, done.")

    # Apply CutMix
    train_data_cutmix_data, train_data_cutmix_labels = cut_mix((train_data_1, train_labels_1), (train_data_2, train_labels_2), alpha = 1, beta = 1)
    del train_data_1, train_data_2, train_labels_1, train_labels_2

    # Combine the datasets
    combined_train_data = np.concatenate((train_data, train_data_cutmix_data), axis = 0)
    combined_train_labels = np.concatenate((train_labels, train_data_cutmix_labels), axis = 0)
    del train_data, train_labels, train_data_cutmix_data, train_data_cutmix_labels
    
    # Print information about the data
    print(f"Data shape: {combined_train_data.shape}")
    print(f"Label shape: {combined_train_labels.shape}")
    print(f"Num classes: {num_classes}")

    # Total steps for the scheduler
    total_steps = int((len(combined_train_data) / cg.BATCH_SIZE) * cg.EPOCHS)
    warmup_steps = int(total_steps * cg.warmup_p)
    scheduled_lrs = WarmUpCosine(lr_start = cg.lr_start, lr_max = cg.lr_max, warmup_steps = warmup_steps, total_steps = total_steps)
                 
    # Build the classification model
    #model = build_model_pretrained(INPUT_SHAPE, num_classes)
    model = build_model_scratch(cg.INPUT_SHAPE, num_classes)

    # Compile the model with the AdamW optimizer and categorical cross-entropy loss
    model.compile(optimizer = tf.keras.optimizers.AdamW(learning_rate = scheduled_lrs, weight_decay = cg.weight_decay), 
                  loss = tf.keras.losses.CategoricalCrossentropy(), 
                  metrics = ['accuracy'])
    print(model.summary())

    # Train the model
    history, run = train_model(model, (combined_train_data, combined_train_labels), (val_data, val_labels), epochs = cg.EPOCHS)

    # Save model
    model.save_weights(model_save_path)

    # Get the ID of the current run
    run_info = mlflow.get_run(run.info.run_id)
    run_name = run_info.data.tags['mlflow.runName']

    # Show confusion matrix for model evaluation
    evaluate_model(model, val_data, val_labels, run_name)

# %% Main

if __name__ == '__main__':

    dataset_path = 'dataset'
    model_save_path = 'model/trained_model/music_classifier'
    labels_dict_path = 'model/labels_dict.pkl'

    main(dataset_path, model_save_path, labels_dict_path)