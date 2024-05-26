# %% Libraries

import soundcard as sc
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from . import config_audio as cg

# %% Functions

def process_audio_in_real_time(SAMPLE_RATE: int = cg.SAMPLE_RATE, RECORD_SEC: int = cg.RECORD_SEC_INFERENCE) -> np.ndarray:
    
    """
    Record and process audio in real time to return its spectrogram.

    Args:
        SAMPLE_RATE (int): Sampling rate of the audio signal.
        RECORD_SEC (int): Number of seconds to record audio.

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
        melspectrogram = librosa.feature.melspectrogram(y = chunk, sr = SAMPLE_RATE, n_mels = cg.N_MELS)
        melspectrogram = librosa.power_to_db(melspectrogram, ref = np.max)
        melspectrogram = ((melspectrogram - melspectrogram.min()) / (melspectrogram.max() - melspectrogram.min())) * 255.0

        # Convert the spectrogram to a PIL Image
        img = Image.fromarray(melspectrogram.astype(np.uint8))

        # Create a BytesIO buffer
        buffer = io.BytesIO()

        # Save the image to the buffer in PNG format
        img.save(buffer, format = "PNG")

        # Return the buffer contents as a byte array
        return buffer.getvalue()