# %% Libraries

import soundcard as sc
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

# %% Functions

def generate_spectrogram_image(S_log: np.ndarray, sr: int, display: bool = False) -> None:

    """
    Generate and return the spectrogram image of the audio signal.

    Parameters:
        S_log (np.ndarray): Log-scaled spectrogram.
        sr (int): Sampling rate of the audio signal.
        display (bool): Whether to display the spectrogram image. Default is False.

    Returns:
        np.ndarray: Image array of the spectrogram.
    """

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(S_log, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    if display:

        plt.show()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Read the buffer into an image array
    img = plt.imread(buf, format='png')
    buf.close()

    return img

def process_audio_in_real_time(SAMPLE_RATE: int, RECORD_SEC: int, display_spectrogram: bool = False) -> None:

    """
    Record and process audio in real time to display its spectrogram.
    """

    with sc.get_microphone(id = str(sc.default_speaker().name), include_loopback = True).recorder(samplerate = SAMPLE_RATE) as mic:
        
        # Initialize an empty array to store audio data
        all_data = np.empty((0, 1))

        # Record audio in chunks and process each chunk
        for _ in range(int(SAMPLE_RATE * RECORD_SEC / 1024)):

            data = mic.record(numframes=1024)
            all_data = np.append(all_data, data[:, 0])
            
            # Compute the spectrogram using Librosa
            S = librosa.feature.melspectrogram(y=all_data, sr=SAMPLE_RATE, n_mels=128)
            
            # Convert the spectrogram to a log-scale representation
            S_log = librosa.power_to_db(S, ref=np.max)
            
            # Display the spectrogram
            return generate_spectrogram_image(S_log, SAMPLE_RATE, display_spectrogram)