# Speech-to-Text Transcription Web App

## Overview
This Flask-based web application implements a speech-to-text transcription system with real-time correction highlighting. It features:
- A light green–themed UI.
- Display of an original sentence (loaded from text files in the **originaltxt/** folder).
- Audio recording via the browser using the MediaRecorder API.
- Transcription of the recorded audio using a Whisper-based ASR model.
- Real-time, character-by-character error highlighting (mismatches appear with a yellow background).
- A “Next” button to load subsequent sentences.


## How to run

source ~/miniconda3/bin/activate
cd NursultanWorkspace/project08.04/
conda activate project_typing
python app.py
