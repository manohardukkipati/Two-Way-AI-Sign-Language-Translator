# Two-Way AI Sign Language Translator

## Overview
This project is a real-time sign language translation system designed to bridge the communication gap between signers and non-signers. It utilizes **MediaPipe** for high-precision hand landmark detection and **OpenCV** for gesture recognition.

## Key Features
* **Real-Time Detection:** Uses MediaPipe for low-latency tracking suitable for live conversation.
* **Bi-Directional Communication:** Translates signs to text/speech and text to sign videos.
* **Extensible:** The system allows for easy retraining with new sign data using the provided scripts.

## File Structure
* **`sign_language_app.py`**: The main application script. Run this to launch the real-time translation system.
* **`train_two_hand_model.py`**: The script used to train the Random Forest classifier on hand landmark data.
* **`collect_two_hand_data.py`**: Utility script for capturing new gesture data from the webcam.
* **`ASLCoordinateDictionary.py`**: Helper module containing coordinate mappings for ASL signs.

## Installation & Usage

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Collect Data (Optional)
To add your own signs, run the data collection script:
```bash
python collect_two_hand_data.py
```

### 3. Train the Model
Once data is collected, train the classifier:
```bash
python train_two_hand_model.py
```

### 4. Run the Application
Launch the main translation interface:
```bash
python sign_language_app.py
```

## Note on Model Files
> **Disclaimer:** The pre-trained model file (`two_hand_model.p`) and video assets are excluded from this repository due to GitHub's file size limits (25MB+). To run the app, you must first run the training script (Step 3) to generate your own local model file.

## Tech Stack
* **Language:** Python
* **Libraries:** MediaPipe, OpenCV, Scikit-Learn, NumPy, Pyttsx3
