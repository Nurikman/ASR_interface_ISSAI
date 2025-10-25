# Speech-to-Text Transcription Web App

This web application was developed at ISSAI for conducting experiments for the research paper "User Independent Multilingual Automatic Speech Recognition Interface for Typing: Usability Study and Performance Evaluation"

## Overview
This Flask-based web application implements a speech-to-text transcription system with real-time correction highlighting. It features:
- A light green–themed UI.
- Display of an original sentence (loaded from text files in the **texts/** folder).
- Audio recording via the browser using the MediaRecorder API.
- Transcription of the recorded audio using a Whisper-based ASR model which was fine-tuned at ISSAI.
- Real-time, character-by-character error highlighting (mismatches appear with a yellow background).
- A “Next” button to load subsequent sentences.
- Also it logs following information for data collection and further analysis:
P: Presented text (string). 
P_words: Number of words in P. 
S: Returned text by ASR (string).
S_words: Number of words in S. 
T: Transcribed text after user editing (string). 
T_words: Number of words in T. 
Time_talking (seconds): Time elapsed from pressing the Record button until pressing Stop. 
Time_asr (seconds): Time from pressing Stop until the ASR result is received. 
Time_edit (seconds): Time from the first keyboard press during editing until the “Next”  button is pressed.
Time_server (seconds): Computed as Time_talking + Time_asr. 
Time_total (seconds): Computed as Time_talking + Time_asr + Time_edit. 
WPM_asr: Calculated as S_words divided by (Time_asr/60).
WPM_server: Calculated as S_words divided by (Time_server/60). 
WPM_user: Calculated as T_words divided by (Time_total/60). 
CPM_server: Calculated as Number_of_characters divided by (Time_server/60). 
CPM_user: Calculated as Number_of_characters divided by (Time_total/60). 
CER_asr (%): Use the provided library (see below) to compute the character error rate between S and P, then multiply by 100.
CER_user (%): Compute the CER between T and P, then multiply by 100. 
WER_asr (%): Use the provided library to compute the word error rate between S and P, then multiply by 100.
WER_user (%): Compute the WER between T and P, then multiply by 100. Backspaces: Number of backspaces recorded during text editing in the current trial. 
