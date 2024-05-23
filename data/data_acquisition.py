# %% Libraries

import soundcard as sc
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# %% Functions

def generate_spectrogram_image(S_log: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    
    """
    Generate and return the spectrogram image of the audio signal.

    Args:
        S_log (np.ndarray): Log-scaled spectrogram.
        target_size (tuple): The target size of the spectrogram image. Default is (224, 224).

    Returns:
        np.ndarray: Image array of the spectrogram.
    """

    # Normalize the spectrogram
    S_normalized = (S_log - S_log.min()) / (S_log.max() - S_log.min()) * 255
    
    # Convert to image and resize
    img = Image.fromarray(S_normalized)
    img = img.resize(target_size)
    S_resized = np.array(img)
    
    return S_resized


def process_audio_in_real_time(SAMPLE_RATE: int = 22050, RECORD_SEC: int = 30, target_size: tuple = (224, 224)) -> np.ndarray:
    
    """
    Record and process audio in real time to return its spectrogram.

    Args:
        SAMPLE_RATE (int): Sampling rate of the audio signal.
        RECORD_SEC (int): Number of seconds to record audio.
        target_size (tuple): The target size of the spectrogram image. Default is (224, 224).

    Returns:
        np.ndarray: The spectrogram image array of the recorded audio.
    """

    with sc.get_microphone(id = str(sc.default_speaker().name), include_loopback = True).recorder(samplerate = SAMPLE_RATE) as mic:
        
        print(f"Recording audio...")

        # Initialize an empty array to store audio data
        chunk = np.empty((0,))

        # Record audio in chunks
        for _ in range(int(SAMPLE_RATE * RECORD_SEC / 1024)):

            data = mic.record(numframes = 1024)
            chunk = np.append(chunk, data[:, 0])
        
        # Compute the spectrogram using Librosa
        melspectrogram = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=128)
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        
        # Generate the spectrogram image
        return generate_spectrogram_image(melspectrogram, target_size=target_size)