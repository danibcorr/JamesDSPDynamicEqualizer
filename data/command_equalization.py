# %% Libraries

import subprocess

# %% Functions

def change_equalizer_settings(genre: str) -> None:

    # Activamos la ecualizacion
    command = f"flatpak run me.timschneeberger.jdsp4linux --set tone_enable=True"
    subprocess.run(command, shell=True)

    # Activamos el perfil de ecualizacion
    command = f"flatpak run me.timschneeberger.jdsp4linux --load-preset '{genre}'"
    subprocess.run(command, shell=True)
    