# 🎵 Music Genre Classifier and Equalizer

## 📋 Description

This project analyzes the audio being played on your computer, converts it into a MEL spectrogram, and uses a deep learning model to classify the music genre. Based on the detected genre, it sends a command to [JamesDSP](https://github.com/Audio4Linux/JDSP4Linux) to adjust the music equalization according to the genre-specific profile.

## 🧩 Features

- **Audio Capture**: Monitors and captures the audio being played on your computer.
- **MEL Spectrogram Conversion**: Converts the captured audio into a MEL spectrogram.
- **Music Genre Classification**: Utilizes a deep learning model to classify the genre of the audio.
- **Automatic Equalization Adjustment**: Sends commands to JamesDSP to adjust the equalization based on the detected music genre.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🕹️ Usage

1. Start the main script to begin monitoring audio and adjusting equalization:
   ```bash
   python main.py
   ```

2. The system will capture the audio, generate the MEL spectrogram, classify the music genre, and adjust the equalization in real-time.

## 🎼 Project Structure

- `data/`
  - `balance_dataset.py`: Balances the dataset for training.
  - `command_equalization.py`: Sends commands to JamesDSP for equalization adjustments.
  - `data_acquisition.py`: Captures audio being played on the computer.
  - `music_2_spectrogram.py`: Converts audio to MEL spectrogram.
- `model/`
  - `schedulers/`: Contains learning rate schedulers.
  - `architecture.py`: Defines the architecture of the deep learning model.
  - `inference.py`: Handles inference for genre classification.
- `main.py`: Main script to run the complete audio analysis and equalization adjustment flow.
- `training.py`: Script for training the deep learning model.

## 🫂 Contributing

Contributions are welcome. If you would like to contribute, please follow these steps:

1. Fork the project.
2. Create a new branch (`git checkout -b feature/new_feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new_feature`).
5. Open a Pull Request.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Thank you for using our automatic music genre classification and equalization system!