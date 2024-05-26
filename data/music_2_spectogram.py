# %% Libraries

import os
import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from PIL import Image
import config_audio as cg
from concurrent.futures import ProcessPoolExecutor

# %% Functions

def circular_shift(img: np.ndarray) -> np.ndarray:

    """
    Perform a circular shift on a given image.

    Args:
        img (numpy array): The input image.

    Returns:
        numpy array: The circularly shifted image.
    """

    shift = np.random.randint(img.shape[0])

    if img.ndim > 1:
    
        return np.roll(img, shift, axis = 1)
    
    else:
    
        return np.roll(img, shift)

def process_file(filename: str, genre_path: str, genre: str, sr: int, num_frames: int, num_extra_samples: int) -> None:

    """
    Process an audio file by loading, resampling, and converting it into multiple Mel spectrogram images.

    Args:
        filename (str): The name of the audio file.
        genre_path (str): The path to the directory containing the audio file.
        genre (str): The genre of the audio file.
        sr (int): The target sampling rate.
        num_frames (int): The number of frames to process at a time.
        num_extra_samples (int): The number of additional samples to generate for each chunk.

    Returns:
        None
    """

    # Load the audio file using librosa
    audio, orig_sr = librosa.load(os.path.join(genre_path, filename))

    # Resample the audio to the target sampling rate
    audio = librosa.resample(y = audio, orig_sr = orig_sr, target_sr = sr)

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

        og_chunk = chunk

        for extra_sample in range(num_extra_samples):

            if extra_sample != 0:

                chunk = circular_shift(chunk)

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
            with open(os.path.join(input_dir, genre, f'{filename}_{i//num_frames}_{extra_sample}.png'), 'wb') as f:

                f.write(buf.getvalue())

        # Update the previous chunk
        prev_chunk = og_chunk

def music_2_melspectrogram(input_dir: str, duration: int = cg.RECORD_SEC, sr: int = cg.SAMPLE_RATE, num_extra_samples: int = 6) -> None:

    """
    This function converts audio files in a directory to Mel spectrograms and saves them as numpy arrays.

    Args:
        input_dir (str): The directory containing the audio files to process.
        duration (int, optional): The duration of each spectrogram in seconds. Defaults to 10.
        sr (int, optional): The target sampling rate. Defaults to 22050.

    Returns:
        None
    """

    # Calculate the number of frames needed for a spectrogram of the specified duration
    num_frames = int(duration * sr)

    # Loop over each genre in the input directory
    for genre in os.listdir(input_dir):

        print(f"Genre: {genre}")
        genre_path = os.path.join(input_dir, genre)

        # Skip if the genre is not a directory
        if not os.path.isdir(genre_path):

            continue

        # Create a list to store the futures
        futures = []

        # Create a process pool executor
        with ProcessPoolExecutor() as executor:

            # Loop over each file in the genre directory
            for filename in os.listdir(genre_path):

                # Process only audio files
                if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):

                    # Submit the task to the executor
                    future = executor.submit(process_file, filename, genre_path, genre, sr, num_frames, num_extra_samples)
                    futures.append(future)

            # Wait for all tasks to complete
            for future in futures:
                future.result()

# %% Main

if __name__ == '__main__':
    
    # Define the directory containing the dataset
    input_dir = '/home/dani/Pictures/music_dataset'

    # Call the function to convert the audio files to Mel spectrograms
    music_2_melspectrogram(input_dir)