# %% Libraries

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

# %% Functions

def music_2_spectogram(input_dir: str, output_dir: str, duration: int = 30) -> None:

    """
    Convert audio files in a directory to spectrograms and save them as PNG images.

    Args:
        input_dir (str): The directory containing the audio files to process
        output_dir (str): The directory where the spectrogram images will be saved
        duration (int, optional): The duration of each spectrogram in seconds (default: 30)

    Returns:
        None
    """

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):

        # Check if the file is an audio file (e.g. MP3, WAV, etc.)
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):

            # Load the audio file using librosa
            audio, sr = librosa.load(os.path.join(input_dir, filename))

            # Calculate the number of frames needed for a 30-second spectrogram
            num_frames = int(duration * sr)

            # Loop through the audio file in 30-second chunks
            for i in range(0, len(audio), num_frames):

                # Extract a 30-second chunk of audio
                chunk = audio[i:i+num_frames]

                # Compute the short-time Fourier transform (STFT) of the chunk
                stft = librosa.stft(chunk)

                # Convert the STFT to a spectrogram
                spectrogram = librosa.amplitude_to_db(stft, ref=np.max)

                # Create a figure and axis object
                fig, ax = plt.subplots()

                # Plot the spectrogram
                ax.imshow(spectrogram, cmap='inferno', origin='lower')

                # Remove axis labels and title
                ax.set_axis_off()

                # Save the spectrogram to a PNG file
                plt.savefig(os.path.join(output_dir, f'{filename}_{i//num_frames}.png'), dpi=300, bbox_inches='tight', pad_inches=0)

                # Close the figure to free up memory
                plt.close(fig)

# %% Main

if __name__ == '__main__':

    # Set the directory path and the output directory path
    input_dir = '/home/dani/Pictures/music_dataset/hiphop'
    output_dir = './dataset/hiphop'

    # Set the duration of each spectrogram (30 seconds)
    duration = 30

    music_2_spectogram(music_folder, input_dir, output_dir, duration)