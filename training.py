import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model.utils.Supervised.MCNN import mcnn_classifier
from tensorflow.keras import layers, mixed_precision
import cv2
import pickle

# Constants
BATCH_SIZE = 32
EPOCHS = 10

def load_data_from_folder(path: str, input_shape: tuple = (128, 128)) -> tuple:

    """Load data from folders where each subfolder represents a class."""

    data = []
    labels = []

    for i, class_name in enumerate(os.listdir(path)):

        class_folder = os.path.join(path, class_name)

        for file_name in os.listdir(class_folder):

            if file_name.endswith('.png'):

                file_path = os.path.join(class_folder, file_name)
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=input_shape, color_mode='grayscale')
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                data.append(img_array)
                labels.append(i)
    
    return np.array(data, dtype=np.uint8), np.array(labels)

def create_data_generators(train_data: np.ndarray, train_labels: np.ndarray, val_data: np.ndarray, val_labels: np.ndarray, batch_size: int = BATCH_SIZE) -> tuple:

    """Create data generators for training and validation data."""

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
    val_generator = val_datagen.flow(val_data, val_labels, batch_size=batch_size)

    return train_generator, val_generator

def build_model(input_shape: tuple, num_classes: int) -> keras.Model:

    """Build a CNN model for genre classification."""

    model = mcnn_classifier(input_shape, num_classes)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4), 
                  loss=tf.keras.losses.CategoricalFocalCrossentropy(), 
                  metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def train_model(model: keras.Model, train_generator: ImageDataGenerator, val_generator: ImageDataGenerator, epochs: int = EPOCHS) -> tf.keras.Model:

    """Train the CNN model."""
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
    return history

def evaluate_model(model: keras.Model, val_generator: ImageDataGenerator) -> float:

    """Evaluate the model on validation data."""
    loss, accuracy = model.evaluate(val_generator)
    return accuracy

if __name__ == '__main__':

    # Load data from folder
    dataset_folder = './dataset/'
    data, labels = load_data_from_folder(dataset_folder)
    num_classes = len(np.unique(labels))

    print(f"Num data: {data.shape}")
    print(f"max data: {np.max(data)}, min data: {np.min(data)}")
    print(f"Num classes {num_classes}")
    print(f"{labels}")

    # Check if the number of images is equal to the number of labels
    assert len(data) == len(labels), "Number of images is not equal to the number of labels"

    # One-hot encode labels
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels_onehot, test_size=0.15, random_state=42, stratify=labels_onehot)

    # Create data generators
    train_generator, val_generator = create_data_generators(train_data, train_labels, val_data, val_labels)

    # Build and train the model
    input_shape = (train_data.shape[1], train_data.shape[2], 1)
    model = build_model(input_shape, num_classes)
    history = train_model(model, train_generator, val_generator, epochs=EPOCHS)

    # Evaluate the model
    accuracy = evaluate_model(model, val_generator)
    print(f'Validation accuracy: {accuracy:.2f}')