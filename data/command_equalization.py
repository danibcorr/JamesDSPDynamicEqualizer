# %% Libraries

import subprocess

# %% Functions

def change_equalizer_settings(genre: str) -> None:

    """
    Changes the equalizer settings to the specified genre.

    Args:
        genre (str): The name of the genre to apply (e.g. "Rock", "Pop", etc.)

    Returns:
        None
    """

    # Activate equalization
    command = f"flatpak run me.timschneeberger.jdsp4linux --set tone_enable=True"
    subprocess.run(command, shell=True)

    # Activate the equalization profile
    command = f"flatpak run me.timschneeberger.jdsp4linux --load-preset '{genre}'"
    subprocess.run(command, shell=True)