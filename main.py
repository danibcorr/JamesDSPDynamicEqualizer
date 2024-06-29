# %% Libraries

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import threading
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import customtkinter as ctk
from data.data_acquisition import process_audio_in_real_time
from model.inference import load_model, make_prediction
from data.command_equalization import change_equalizer_settings
from tkinter import PhotoImage

# %% GUI Class definitions

class MusicGenreClassifierGUI:

    def __init__(self, master):

        """
        Initialize the GUI

        master: The main window
        """

        self.master = master
        master.title("Music Genre Classifier")
        master.geometry("500x450")

        self.model = load_model(model_path="model/trained_model/music_classifier")
        
        self.setup_ui()
        
        self.is_running = False
        self.current_genre = "None"

    def setup_ui(self) -> None:

        """
        Set up the UI components
        """

        # Main content
        self.content_frame = ctk.CTkFrame(self.master)
        self.content_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Genre display
        self.genre_frame = ctk.CTkFrame(self.content_frame)
        self.genre_frame.pack(pady=20, padx=20, fill="x")
        self.genre_label = ctk.CTkLabel(self.genre_frame, text="Detected Genre", 
                                        font=ctk.CTkFont(size=18, weight="bold"))
        self.genre_label.pack(pady=(10, 0))
        self.genre_value = ctk.CTkLabel(self.genre_frame, text="None", 
                                        font=ctk.CTkFont(size=24))
        self.genre_value.pack(pady=(0, 10))

        # Control buttons
        self.button_frame = ctk.CTkFrame(self.content_frame)
        self.button_frame.pack(pady=20)
        self.start_button = ctk.CTkButton(self.button_frame, text="Start Classification", 
                                          command=self.start_classification, width=150)
        self.start_button.pack(side="left", padx=10)
        self.stop_button = ctk.CTkButton(self.button_frame, text="Stop Classification", 
                                         command=self.stop_classification, width=150, 
                                         state="disabled")
        self.stop_button.pack(side="left", padx=10)

        # Status and progress
        self.status_label = ctk.CTkLabel(self.content_frame, text="Status: Idle", 
                                         font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=10)
        self.progress_bar = ctk.CTkProgressBar(self.content_frame, width=400, height=20)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        # Theme selection
        self.appearance_mode_label = ctk.CTkLabel(self.content_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.pack(pady=(20, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.content_frame, 
                                                             values=["Light", "Dark", "System"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.pack(pady=10)

        # Set default appearance mode
        self.appearance_mode_optionemenu.set("System")

    def start_classification(self) -> None:

        """
        Start the classification process
        """

        self.is_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="Status: Running")

        threading.Thread(target=self.classification_loop, daemon=True).start()

    def stop_classification(self) -> None:

        """
        Stop the classification process
        """

        self.is_running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="Status: Stopped")
        self.progress_bar.set(0)

    def classification_loop(self) -> None:

        """
        Main classification loop
        """

        while self.is_running:

            try:

                spectrograms = []

                # Update status to indicate recording
                self.master.after(0, self.update_status, "Recording audio...")
                
                for _ in range(3):

                    spectrogram = process_audio_in_real_time()
                    spectrogram = np.array(Image.open(io.BytesIO(spectrogram)).convert('L'), dtype=np.uint8)
                    spectrogram = np.expand_dims(spectrogram, [0, -1])
                    spectrograms.append(spectrogram)

                # Update status to indicate processing
                self.master.after(0, self.update_status, "Processing audio...")
                
                spectrograms = np.array(spectrograms).reshape(3, 128, 129, 1)
                
                genre = make_prediction(self.model, spectrograms)
                self.master.after(0, self.update_genre, genre)
                
                change_equalizer_settings(genre)
                
                # Update status to indicate waiting
                self.master.after(0, self.update_status, "Waiting for next analysis...")
                
                for i in range(30, 0, -1):

                    if not self.is_running:

                        break

                    self.master.after(0, self.update_status, f"Next analysis in {i} seconds")
                    self.master.after(0, self.update_progress, (30 - i) / 30)
                    time.sleep(1)

            except Exception as e:

                print(f"An error occurred: {e}")
                self.master.after(0, self.stop_classification)

                break

    def update_genre(self, genre) -> None:

        """
        Update the genre label
        """

        self.genre_value.configure(text=genre)

        if genre!= self.current_genre:

            self.current_genre = genre

    def update_status(self, status) -> None:

        """
        Update the status label
        """

        self.status_label.configure(text=f"Status: {status}")

    def update_progress(self, value) -> None:
        
        """
        Update the progress bar
        """

        self.progress_bar.set(value)

    def change_appearance_mode_event(self, new_appearance_mode: str) -> None:

        """
        Change the appearance mode
        """

        ctk.set_appearance_mode(new_appearance_mode)

# %% Functions

def main():

    ctk.set_default_color_theme("blue")
    root = ctk.CTk()

    # Set the icon
    if os.name == 'nt':  

        # For Windows
        root.iconbitmap('images/icon.ico')

    elif os.name == 'posix':  

        # For macOS and Linux
        img = PhotoImage(file='images/icon.png')
        root.tk.call('wm', 'iconphoto', root._w, img)

    app = MusicGenreClassifierGUI(root)
    root.mainloop()

# %% Main

if __name__ == '__main__':

    main()