# %% Libraries

import os
import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from PIL import Image
import config_audio as cg

# %% Functions

def music_2_melspectrogram(input_dir: str, duration: int = cg.RECORD_SEC, sr: int = cg.SAMPLE_RATE) -> None:

    """
    This function converts audio files in a directory to Mel spectrograms and saves them as numpy arrays.

    Args:
        input_dir (str): The directory containing the audio files to process.
        duration (int, optional): The duration of each spectrogram in seconds. Defaults to 10.
        sr (int, optional): The target sampling rate. Defaults to 22050.

    Returns:
        None
    """

    # Loop over each genre in the input directory
    for genre in os.listdir(input_dir):

        print(f"Genre: {genre}")
        genre_path = os.path.join(input_dir, genre)

        # Skip if the genre is not a directory
        if not os.path.isdir(genre_path):

            continue

        # Loop over each file in the genre directory
        for filename in os.listdir(genre_path):

            # Process only audio files
            if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):

                # Load the audio file using librosa
                audio, orig_sr = librosa.load(os.path.join(genre_path, filename))

                # Resample the audio to the target sampling rate
                audio = librosa.resample(y = audio, orig_sr = orig_sr, target_sr = sr)

                # Calculate the number of frames needed for a spectrogram of the specified duration
                num_frames = int(duration * sr)

                # Initialize a variable to store the previous chunk
                prev_chunk = np.array([])

                # Loop over the audio in chunks of size num_frames
                for i in range(0, len(audio), num_frames):

                    chunk = audio[i:i+num_frames]

                    # If the chunk is smaller than num_frames, concatenate it with the previous chunk
                    if len(chunk) < num_frames:

                        if len(prev_chunk) + len(chunk) >= num_frames:

                            chunk = np.concatenate((prev_chunk[-(num_frames-len(chunk)):], chunk))
                        else:

                            continue

                    # Compute the Mel spectrogram of the chunk
                    melspectrogram = librosa.feature.melspectrogram(y = chunk, sr = sr, n_mels = cg.N_MELS)
                    melspectrogram = librosa.power_to_db(melspectrogram, ref = np.max)
                    melspectrogram = ((melspectrogram - melspectrogram.min()) / (melspectrogram.max() - melspectrogram.min())) * 255.0

                    # Convert the Mel spectrogram to an image
                    img = Image.fromarray(melspectrogram)

                    # Use plt.imsave with a buffer to reduce disk I/O
                    buf = BytesIO()
                    plt.imsave(buf, melspectrogram, cmap = 'viridis', format = 'png')

                    # Save the image to a file
                    with open(os.path.join(input_dir, genre, f'{filename}_{i//num_frames}.png'), 'wb') as f:

                        f.write(buf.getvalue())

                    # Update the previous chunk
                    prev_chunk = chunk

# %% Main

if __name__ == '__main__':
    
    # Define the directory containing the dataset
    input_dir = '/home/dani/Pictures/music_dataset'

    # Define the duration of each spectrogram
    duration = 10

    # Call the function to convert the audio files to Mel spectrograms
    music_2_melspectrogram(input_dir, duration)