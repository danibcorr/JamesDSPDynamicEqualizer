# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from data.data_acquisition import process_audio_in_real_time
from model.inference import load_model, make_prediction, load_label_dict
from data.command_equalization import change_equalizer_settings
import numpy as np

# %% Functions

def main(model: tf.keras.Model) -> None:

    """
    Main function to process audio in real time, make predictions, and adjust equalizer settings.

    Parameters:
        model (tf.keras.Model): The trained model for genre classification.
    """

    try:

        while True:

            # Obtain the current audio for processing to a spectrogram
            spectrogram = process_audio_in_real_time()
            print("Obtained spectrogram.")

            # Add batch dimension and channel dimension
            spectrogram = spectrogram.reshape((1, 224, 224, 1))

            # Repeat the channel dimension 3 times  
            spectrogram = np.repeat(spectrogram, 3, axis=3)  

            # Perform the prediction of the model to obtain the genre of the song
            genre = make_prediction(model, spectrogram)
            print(f"Genre: {genre}")
            
            # Apply the EQ profiles to the JamesDSP program based on the genre
            change_equalizer_settings(genre)

    except KeyboardInterrupt:

        print("User interruption. Closing the program...")

# %% Main

if __name__ == '__main__':

    # Load the model at the beginning of the code
    model = load_model(model_path = "model/trained_model/music_classifier")
    print("Model loaded.")

    # In an infinite loop, the program will constantly listen to the audio
    main(model)