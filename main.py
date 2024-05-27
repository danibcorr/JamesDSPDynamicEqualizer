# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
import tensorflow as tf
from data.data_acquisition import process_audio_in_real_time
from model.inference import load_model, make_prediction
from data.command_equalization import change_equalizer_settings
import numpy as np
from data import config_audio as cg 
from PIL import Image
import io

# %% Functions

def main(model: tf.keras.Model, num_samples: int = 3) -> None:

    """
    Main function to process audio in real time, make predictions, and adjust equalizer settings.

    Args:
        model (tf.keras.Model): The trained model for genre classification.
    """

    try:

        while True:

            # Obtain 3 spectrograms
            spectrograms = []
            
            for _ in range(num_samples):

                spectrogram = process_audio_in_real_time()
                spectrogram = np.array(Image.open(io.BytesIO(spectrogram)).convert('L'), dtype = np.uint8)
                spectrogram = np.expand_dims(spectrogram, [0, -1])
                spectrograms.append(spectrogram)

            spectrograms = np.array(spectrograms).reshape(num_samples, 128, 129, 1)

            # Make predictions for each spectrogram
            genre = make_prediction(model, spectrograms)
            print(f"Genre: {genre}")

            # Apply the EQ profiles to the JamesDSP program based on the genre
            change_equalizer_settings(genre)

            # Pause for 30 seconds before the next analysis
            print("Waiting for 30 seconds before the next audio analysis...")
            time.sleep(30)

    except KeyboardInterrupt:

        print("User interruption. Closing the program...")

# %% Main

if __name__ == '__main__':

    # Load the model at the beginning of the code
    model = load_model(model_path = "model/trained_model/music_classifier")
    print("Model loaded.")

    # In an infinite loop, the program will constantly listen to the audio
    main(model)