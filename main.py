# %% Libraries

from .data.data_acquisition import process_audio_in_real_time
from .model.inference import prediction 

# %% Globals

# [Hz]. sampling rate.
SAMPLE_RATE = 48000       
# [sec]. duration recording audio.       
RECORD_SEC = 20      

# %% Functions

def main(SAMPLE_RATE: int, RECORD_SEC: int, display_spectrogram: bool = False) -> None:

    # We obtain the current audio for processing to a spectrogram.
    spectrogram = process_audio_in_real_time(SAMPLE_RATE, RECORD_SEC, display_spectrogram)

    # Obtained the spectrogram, we perform the prediction of the model to obtain
    # the genre of the song
    genre = prediction(spectrogram)

    # Once we get the genre of the song, we perform the commands needed to
    # apply the EQ profiles to the JamesDSP program
    change_equalizer_settings(genre)

# %% Main

if __name__ == '__main__':

    main(SAMPLE_RATE, RECORD_SEC)