# %% Libraries

import os
import librosa
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from PIL import Image

# %% Globals

np.random.seed(42)

# %% Functions

def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:

    """
    Add random noise to an audio signal.

    Args:
        audio (numpy array): The original audio signal
        noise_factor (float, optional): The factor by which to scale the noise (default: 0.005)

    Returns:
        numpy array: The audio signal with added noise
    """

    noise = np.random.normal(0, 1, len(audio))
    return np.clip(audio + noise_factor * noise, -1.0, 1.0)

def time_shift(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:

    """
    Shift an audio signal in time.

    Args:
        audio (numpy array): The original audio signal
        shift_max (float, optional): The maximum fraction of the audio length to shift (default: 0.2)

    Returns:
        numpy array: The time-shifted audio signal
    """

    shift = np.random.randint(int(shift_max * len(audio)))
    return np.roll(audio, shift)

def frequency_masking(mel_spectrogram: np.ndarray, mask_percentage: float = 0.1) -> np.ndarray:

    """
    Apply frequency masking to a Mel spectrogram.

    Args:
        mel_spectrogram (numpy array): The Mel spectrogram
        mask_percentage (float, optional): The percentage of the spectrogram to mask (default: 0.1)

    Returns:
        numpy array: The frequency-masked Mel spectrogram
    """

    n_mels, t = mel_spectrogram.shape
    mask_value = mel_spectrogram.mean()
    num_mels_to_mask = int(mask_percentage * n_mels)
    f = np.random.randint(0, n_mels - num_mels_to_mask)
    mel_spectrogram[f:f + num_mels_to_mask, :] = mask_value

    return mel_spectrogram

def time_masking(mel_spectrogram: np.ndarray, mask_percentage: float = 0.1) -> np.ndarray:

    """
    Apply time masking to a Mel spectrogram.

    Args:
        mel_spectrogram (numpy array): The Mel spectrogram
        mask_percentage (float, optional): The percentage of the spectrogram to mask (default: 0.1)

    Returns:
        numpy array: The time-masked Mel spectrogram
    """

    n_mels, t = mel_spectrogram.shape
    mask_value = mel_spectrogram.mean()
    num_times_to_mask = int(mask_percentage * t)
    t_ = np.random.randint(0, t - num_times_to_mask)
    mel_spectrogram[:, t_:t_ + num_times_to_mask] = mask_value

    return mel_spectrogram

def apply_random_augmentations(chunk: np.ndarray, idx: int) -> np.ndarray:

    """
    Apply random augmentations to the audio chunk.

    Args:
        chunk (numpy array): The audio chunk

    Returns:
        numpy array: The augmented audio chunk
    """

    if idx != 0:
            
        augmentation_type = np.random.randint(6)
        
        if augmentation_type >= 0:

            chunk = add_noise(chunk)

        if augmentation_type > 1:

            chunk = time_shift(chunk)

        if augmentation_type > 2:

            melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=22050, n_mels=128)

            if augmentation_type > 3:

                melspectrogram = frequency_masking(melspectrogram)

            if augmentation_type > 4:

                melspectrogram = time_masking(melspectrogram)

            chunk = librosa.griffinlim(melspectrogram)

    return chunk

def music_2_melspectrogram(input_dir: str, duration: int = 30, sr: int = 22050, target_size: tuple = (224, 224), num_samples: int = 6) -> None:
    
    """
    Convert audio files in a directory to Mel spectrograms and save them as numpy arrays.

    Args:
        input_dir (str): The directory containing the audio files to process
        duration (int, optional): The duration of each spectrogram in seconds (default: 30)
        sr (int, optional): The target sampling rate (default: 22050)
        target_size (tuple, optional): The target size of the image (default: (224, 224))

    Returns:
        None
    """

    for genre in os.listdir(input_dir):

        print(f"Genre: {genre}")
        genre_path = os.path.join(input_dir, genre)

        if not os.path.isdir(genre_path):

            continue

        for filename in os.listdir(genre_path):

            if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):

                # Load the audio file using librosa
                audio, orig_sr = librosa.load(os.path.join(genre_path, filename))

                # Resample the audio to the target sampling rate
                audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=sr)

                # Calculate the number of frames needed for a 30-second spectrogram
                num_frames = int(duration * sr)

                # Initialize a variable to store the previous chunk
                prev_chunk = np.array([])

                for i in range(0, len(audio), num_frames):

                    chunk = audio[i:i+num_frames]

                    if len(chunk) < num_frames:

                        if len(prev_chunk) + len(chunk) >= num_frames:

                            chunk = np.concatenate((prev_chunk[-(num_frames-len(chunk)):], chunk))
                        
                        else:

                            continue
                    
                    og_chunk = chunk

                    for idx in range(num_samples):
                            
                        chunk = apply_random_augmentations(og_chunk, idx)
                        melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
                        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
                        melspectrogram = ((melspectrogram - melspectrogram.min()) / (melspectrogram.max() - melspectrogram.min())) * 255

                        # Resize the Mel spectrogram to the target size
                        img = Image.fromarray(melspectrogram)
                        img = img.resize(target_size)
                        melspectrogram = np.array(img)

                        # Use plt.imsave with a buffer to reduce disk I/O
                        buf = BytesIO()
                        plt.imsave(buf, melspectrogram, cmap='viridis', format='png')
                        
                        with open(os.path.join(input_dir, genre, f'{filename}_{i//num_frames}_{idx}.png'), 'wb') as f:
                            
                            f.write(buf.getvalue())

                    prev_chunk = og_chunk

# %% Main

if __name__ == '__main__':
    
    input_dir = '/home/dani/Pictures/music_dataset/'
    duration = 30
    music_2_melspectrogram(input_dir, duration)