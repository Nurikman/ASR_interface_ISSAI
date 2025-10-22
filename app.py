import os
import time
import csv
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
from huggingface_hub import login
import evaluate
import numpy as np
import soundfile as sf

# Initialize evaluation metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

hf_token = "your_huggingface_token"
login(hf_token)

# Set folders – recordings, transcribed texts, and original texts.
RECORDINGS_FOLDER = 'recordings'
RECORDINGS_FOLDER_OYLAN = 'oylan_recordings'
TRANSCRIBED_FOLDER = 'transcribedtxt'
ORIGINAL_FOLDER = 'big_texts'
for folder in [RECORDINGS_FOLDER, RECORDINGS_FOLDER_OYLAN, TRANSCRIBED_FOLDER, ORIGINAL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Existing code for loading sentences (kept for backward compatibility)
original_files = sorted(os.listdir(ORIGINAL_FOLDER), key=lambda x: int(os.path.splitext(x)[0]))
sentences = []
for filename in original_files:
    filepath = os.path.join(ORIGINAL_FOLDER, filename)
    with open(filepath, 'r') as f:
        sentences.append(f.read().strip())

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "issai/whisper-turbo"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_name = "issai/whisper-turbo"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    checkpoint_path,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)
model = torch.compile(model)

# -------------------------------
# NEW REGISTRATION & EXPERIMENT ROUTES
# -------------------------------

@app.route('/')
def registration():
    return render_template('registration.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    surname = request.form.get('surname')
    # Capture new Gender field
    gender = request.form.get('gender')
    age = request.form.get('age')
    profession = request.form.get('profession')
    education = request.form.get('education')
    participant_id = request.form.get('id')
    group = request.form.get('group')
    # Capture new Audio Data Consent checkbox (will have a value if checked)
    audio_consent = request.form.get('audio_consent')
   
    participants_file = os.path.join('data', 'participants.csv')
    os.makedirs('data', exist_ok=True)
    file_exists = os.path.isfile(participants_file)
    with open(participants_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Name", "Surname", "Gender", "Age", "Profession", "Education", "ID", "Group", "Audio Data Consent"])
        writer.writerow([name, surname, gender, age, profession, education, participant_id, group, audio_consent])
   
    # Save registration data in session and determine language order.
    session['registration'] = {
        'name': name,
        'surname': surname,
        'gender': gender,
        'age': age,
        'profession': profession,
        'education': education,
        'id': participant_id,
        'group': group,
        'audio_consent': audio_consent
    }
    if group == 'A':
        order = ['en', 'kz', 'ru']
    elif group == 'B':
        order = ['kz', 'ru', 'en']
    elif group == 'C':
        order = ['ru', 'en', 'kz']
    else:
        order = ['en', 'kz', 'ru']
    session['order'] = order
    session['current_language_index'] = 0
    session['phase'] = 'demo'
   
    # --- New Logging Setup ---
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    log_filename = f"{participant_id}_{group}_{int(time.time())}.csv"
    log_filepath = os.path.join(data_folder, log_filename)
    session['log_filepath'] = log_filepath
   
    session['trial_logs'] = {'demo': [], 'session1': [], 'session2': []}
    session['language_averages'] = {}
   
    return redirect(url_for('experiment'))



@app.route('/experiment')
def experiment():
    if 'registration' not in session:
        return redirect(url_for('registration'))
    order = session.get('order')
    lang = order[session.get('current_language_index', 0)]
    phase = session.get('phase', 'demo')
    # Map phase to folder name: for demo use e.g. "en-demo"; for sessions use "en-session1" or "en-session2"
    folder = f"{lang}-" + ("demo" if phase == 'demo' else phase)
    return render_template('experiment_page.html', textFolder=folder, language=lang, phase=phase)

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/advance', methods=['POST'])
def advance():
    phase = session.get('phase', 'demo')
    if phase == 'demo':
        session['phase'] = 'session1'
        return redirect(url_for('experiment'))
    elif phase == 'session1':
        session['phase'] = 'session2'
        return redirect(url_for('experiment'))
    elif phase == 'session2':
        group = session['registration']['group']
        order = session['order']
        current_lang = order[session['current_language_index']]
        # For English, if group B; for Kazakh, if group C; for Russian, if group A, finish experiment.
        if (group == 'B' and current_lang == 'en') or \
           (group == 'C' and current_lang == 'kz') or \
           (group == 'A' and current_lang == 'ru'):
            # NEW: Compute and log the language average for the final language before finishing.
            lang_avgs = session.get('language_averages', {})
            numeric_keys = ["P_words", "S_words", "T_words", "Time_talking", "Time_asr",
                            "Time_edit", "Time_server", "Time_total", "WPM_asr", "WPM_server",
                            "WPM_user", "CER_asr", "CER_user", "WER_asr", "WER_user", "Backspaces"]
            overall_avg = {}
            for key in numeric_keys:
                values = []
                for phase in ["demo", "session1", "session2"]:
                    if phase in lang_avgs:
                        values.append(lang_avgs[phase].get(key, 0))
                overall_avg[key] = sum(values)/len(values) if values else 0

            log_filepath = session.get('log_filepath')
            language = session.get('order')[session.get('current_language_index', 0)]
            write_language_average_csv(language, lang_avgs, overall_avg, log_filepath)
            return redirect(url_for('thank_you'))
        else:
            return redirect(url_for('intermediate'))
    return redirect(url_for('experiment'))


@app.route('/intermediate')
def intermediate():
    return render_template('intermediate_page.html')

@app.route('/next_language', methods=['POST'])
def next_language():
    # Before moving to next language, compute overall averages for the current language.
    lang_avgs = session.get('language_averages', {})
    # For the keys, we use the same numeric keys as before.
    numeric_keys = ["P_words", "S_words", "T_words", "Time_talking", "Time_asr",
                    "Time_edit", "Time_server", "Time_total", "WPM_asr", "WPM_server",
                    "WPM_user", "CER_asr", "CER_user", "WER_asr", "WER_user", "Backspaces"]
    
    overall_avg = {}
    for key in numeric_keys:
        # Average the averages from demo, session1, and session2
        values = []
        for phase in ["demo", "session1", "session2"]:
            if phase in lang_avgs:
                values.append(lang_avgs[phase].get(key, 0))
        overall_avg[key] = sum(values)/len(values) if values else 0

    # Append the language average block to the CSV file.
    log_filepath = session.get('log_filepath')
    language = session.get('order')[session.get('current_language_index', 0)]
    write_language_average_csv(language, lang_avgs, overall_avg, log_filepath)
    
    # Reset trial logs and language averages for the new language.
    session['trial_logs'] = {'demo': [], 'session1': [], 'session2': []}
    session['language_averages'] = {}
    session['current_language_index'] = session.get('current_language_index', 0) + 1
    session['phase'] = 'demo'
    return redirect(url_for('experiment'))


@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

# -------------------------------
# NEW TEXT FETCHING ROUTE
# -------------------------------

@app.route('/get_text', methods=['GET'])
def get_text():
    folder = request.args.get('folder')
    index = int(request.args.get('index', 0))
    base_path = os.path.join('texts', folder)
    if not os.path.exists(base_path):
        return jsonify({"error": "Folder not found"}), 404
    files = sorted([f for f in os.listdir(base_path) if f.endswith('.txt')])
    total = len(files)
    if index < 0 or index >= total:
        return jsonify({"error": "No more texts.", "total": total}), 404
    filepath = os.path.join(base_path, files[index])
    session['file_path'] = "{}_{}".format(folder, files[index].split('.')[0])
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return jsonify({"text": text, "index": index, "total": total})

# -------------------------------
# ORIGINAL ROUTES (kept unchanged)
# -------------------------------

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    index = int(request.args.get('index', 0))
    if index < 0 or index >= len(sentences):
        return jsonify({"error": "No more sentences."}), 404
    return jsonify({"sentence": sentences[index], "index": index})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400
    audio_file = request.files['audio_data']
    filename = secure_filename(audio_file.filename)
    save_path = os.path.join(RECORDINGS_FOLDER, filename)
    audio_file.save(save_path)

    start_time = time.time()
    audio_array, sampling_rate = librosa.load(save_path, sr=16000)
    if session['registration']['audio_consent'] == "on":
    	save_path_for_oylan = os.path.join(RECORDINGS_FOLDER_OYLAN, "{}_{}_{}.wav".format(session['registration'] ['id'], session['order'][session.get('current_language_index', 0)], session['file_path']))
    	sf.write(save_path_for_oylan, audio_array, 16000)
    
    chunk_duration = 25
    samples_per_chunk = int(sampling_rate * chunk_duration)
    total_samples = len(audio_array)
    num_chunks = int(np.ceil(total_samples / samples_per_chunk))
    transcriptions = []
    for i in range(num_chunks):
        start_idx = i * samples_per_chunk
        end_idx = min(start_idx + samples_per_chunk, total_samples)
        chunk = audio_array[start_idx:end_idx]
        input_features = processor.feature_extractor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        input_features = input_features.to(device, non_blocking=True).to(torch_dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                task="transcribe"
            )
        chunk_transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        transcriptions.append(chunk_transcription)
    transcription = " ".join(transcriptions)
    transcription = re.sub(r'[\"\'“”‘’]', '', transcription) # Remove all types of quotation marks: " ' “ ” ‘ ’
    inference_duration = time.time() - start_time
    trans_filename = os.path.splitext(filename)[0] + '.txt'
    trans_save_path = os.path.join(TRANSCRIBED_FOLDER, trans_filename)
    with open(trans_save_path, 'w') as f:
        f.write(transcription)
    return jsonify({
        "transcription": transcription,
        "inference_time": inference_duration
    })

# -------------------------------
# CSV LOGGING (unchanged aside from placement)
# -------------------------------


def count_words(text: str) -> int:
    words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    return len(words)

@app.route('/log_trial', methods=['POST'])
def log_trial():
    data = request.get_json()
    trial_num = data.get("trial")
    # Get texts; if T is missing, use S.
    P = data.get("P", "") or ""
    S = data.get("S", "") or ""
    T = data.get("T", "") or S

    # Use count_words for word counts
    P_words = count_words(P)
    S_words = count_words(S)
    T_words = count_words(T)
    
    # Parse times; if missing, default to 0.
    try:
        user_talking_time = float(data.get("user_talking_time", 0))
    except:
        user_talking_time = 0
    try:
        server_process_time = float(data.get("server_process_time", 0))
    except:
        server_process_time = 0
    try:
        correction_time = float(data.get("correction_time", 0))
    except:
        correction_time = 0
    backspaces = int(data.get("backspaces", 0))
    
    Time_talking = user_talking_time
    Time_asr = server_process_time
    Time_edit = correction_time
    Time_server = Time_talking + Time_asr
    Time_total = Time_talking + Time_asr + Time_edit
    
    # Compute WPM values (if denominator is 0, default to 0)
    WPM_asr = S_words / (Time_asr/60) if Time_asr > 0 else 0
    WPM_server = S_words / (Time_server/60) if Time_server > 0 else 0
    WPM_user = T_words / (Time_total/60) if Time_total > 0 else 0
    
    # Compute error rates using evaluate library, multiplied by 100
    CER_asr = cer_metric.compute(references=[P], predictions=[S]) * 100
    CER_user = cer_metric.compute(references=[P], predictions=[T]) * 100
    WER_asr = wer_metric.compute(references=[P], predictions=[S]) * 100
    WER_user = wer_metric.compute(references=[P], predictions=[T]) * 100
    
    trial_entry = {
        "trial_num": trial_num,
        "P": P,
        "P_words": P_words,
        "S": S,
        "S_words": S_words,
        "T": T,
        "T_words": T_words,
        "Time_talking": Time_talking,
        "Time_asr": Time_asr,
        "Time_edit": Time_edit,
        "Time_server": Time_server,
        "Time_total": Time_total,
        "WPM_asr": WPM_asr,
        "WPM_server": WPM_server,
        "WPM_user": WPM_user,
        "CER_asr": CER_asr,
        "CER_user": CER_user,
        "WER_asr": WER_asr,
        "WER_user": WER_user,
        "Backspaces": backspaces
    }
    
    current_phase = session.get('phase', 'demo')
    trial_logs = session.get('trial_logs', {})
    if current_phase not in trial_logs:
        trial_logs[current_phase] = []
    trial_logs[current_phase].append(trial_entry)
    session['trial_logs'] = trial_logs  # update session

    return jsonify({"status": "logged"})



def compute_average(trials, numeric_keys):
    """Compute the average for each numeric field in the trials list.
       For non-numeric fields (like text), return an empty string."""
    avg = {}
    count = len(trials)
    for key in numeric_keys:
        total = sum(trial.get(key, 0) for trial in trials)
        avg[key] = total / count if count > 0 else 0
    return avg

def write_session_csv(session_name, language, trials, average, log_filepath):
    """
    Append a block for the given session (e.g. "demo", "Session 1", or "Session 2")
    to the CSV file. The header will include dynamic trial columns and an "Average" column.
    """
    # Mapping for rows: key, description, is_numeric
    parameters = [
        ("P", "Presented text (string)", False),
        ("P_words", "Number of words in P", True),
        ("S", "Returned text by ASR (string)", False),
        ("S_words", "Number of words in S", True),
        ("T", "Transcribed text after user editing (string)", False),
        ("T_words", "Number of words in T", True),
        ("Time_talking", "Time from start recording to stop recording", True),
        ("Time_asr", "Time from stop recording till user receives the results from ASR", True),
        ("Time_edit", "Time from any keyboard till next", True),
        ("Time_server", "Time_talking + Time_asr", True),
        ("Time_total", "Time_talking + Time_asr + Time_edit", True),
        ("WPM_asr", "S_words/Time_asr", True),
        ("WPM_server", "S_words/Time_server", True),
        ("WPM_user", "T_words/Time_total", True),
        ("CER_asr", "Levenshtein distance of S and P / P_length", True),
        ("CER_user", "Levenshtein distance of T and P / P_length", True),
        ("WER_asr", "Levenshtein distance of S and P / P_words", True),
        ("WER_user", "Levenshtein distance of T and P / P_words", True),
        ("Backspaces", "Number of backspaces during text editing by user", True)
    ]
    
    # Determine the header row for trials: "Trial 1", "Trial 2", ... based on the number of trials.
    num_trials = len(trials)
    trial_headers = [f"Trial {i+1}" for i in range(num_trials)]
    
    # Prepare session title (capitalize as needed)
    title_map = {
        "demo": f"{language} demo session",
        "session1": f"{language} Session 1",
        "session2": f"{language} Session 2"
    }
    session_title = title_map.get(session_name, f"{language} {session_name}")
    
    with open(log_filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the session title
        writer.writerow([session_title])
        # Write the header row: Parameter, Description, then trial headers, then "Average"
        writer.writerow(["Parameter", "Description"] + trial_headers + ["Average"])
        
        # For each parameter, write its row.
        for key, desc, is_numeric in parameters:
            row = [key, desc]
            # Append each trial's value (convert non-numeric values to string)
            for trial in trials:
                value = trial.get(key, 0)
                row.append(str(value) if not is_numeric else f"{value:.2f}")
            # For numeric fields, add average; for text fields, leave blank.
            if is_numeric:
                row.append(f"{average.get(key, 0):.2f}")
            else:
                row.append("")
            writer.writerow(row)
        # Write an empty row for separation
        writer.writerow([])
        
def write_language_average_csv(language, lang_averages, overall_avg, log_filepath):
    """
    Append a block for the "Average of the sessions" for the given language.
    lang_averages is expected to be a dict with keys "demo", "session1", "session2".
    """
    parameters = [
        ("P_words", "Number of words in P", True),
        ("S_words", "Number of words in S", True),
        ("T_words", "Number of words in T", True),
        ("Time_talking", "Time from start recording to stop recording", True),
        ("Time_asr", "Time from stop recording till user receives the results from ASR", True),
        ("Time_edit", "Time from any keyboard till next", True),
        ("Time_server", "Time_talking + Time_asr", True),
        ("Time_total", "Time_talking + Time_asr + Time_edit", True),
        ("WPM_asr", "S_words/Time_asr", True),
        ("WPM_server", "S_words/Time_server", True),
        ("WPM_user", "T_words/Time_total", True),
        ("CER_asr", "Levenshtein distance of S and P / P_length", True),
        ("CER_user", "Levenshtein distance of T and P / P_length", True),
        ("WER_asr", "Levenshtein distance of S and P / P_words", True),
        ("WER_user", "Levenshtein distance of T and P / P_words", True),
        ("Backspaces", "Number of backspaces during text editing by user", True)
    ]
    
    with open(log_filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{language} Average of the sessions"])
        writer.writerow(["Parameter", "Description", "Demo Session", "Session 1", "Session 2", "Average"])
        for key, desc, is_numeric in parameters:
            demo_val = lang_averages.get("demo", {}).get(key, 0)
            sess1_val = lang_averages.get("session1", {}).get(key, 0)
            sess2_val = lang_averages.get("session2", {}).get(key, 0)
            overall = overall_avg.get(key, 0)
            writer.writerow([
                key,
                desc,
                f"{demo_val:.2f}",
                f"{sess1_val:.2f}",
                f"{sess2_val:.2f}",
                f"{overall:.2f}"
            ])
        writer.writerow([])


@app.route('/end_experiment', methods=['POST'])
def end_experiment():
    current_phase = session.get('phase', 'demo')
    trial_logs = session.get('trial_logs', {}).get(current_phase, [])
    if not trial_logs:
        return jsonify({"error": "No trial data."}), 400
    
    # Define numeric keys for averaging (only those which are numeric)
    numeric_keys = ["P_words", "S_words", "T_words", "Time_talking", "Time_asr",
                    "Time_edit", "Time_server", "Time_total", "WPM_asr", "WPM_server",
                    "WPM_user", "CER_asr", "CER_user", "WER_asr", "WER_user", "Backspaces"]
    
    avg = compute_average(trial_logs, numeric_keys)
    
    # Write the current session block to the CSV file
    log_filepath = session.get('log_filepath')
    language = session.get('order')[session.get('current_language_index', 0)]
    write_session_csv(current_phase, language, trial_logs, avg, log_filepath)
    
    # Store the average for this session for later language averaging.
    lang_avgs = session.get('language_averages', {})
    lang_avgs[current_phase] = avg
    session['language_averages'] = lang_avgs
    session['trial_logs'][current_phase] = []

    return jsonify({
        "Time_talking": avg.get("Time_talking", 0),
        "Time_asr": avg.get("Time_asr", 0),
        "Time_edit": avg.get("Time_edit", 0),
        "Time_server": avg.get("Time_server", 0),
        "Time_total": avg.get("Time_total", 0),
        "T_words": avg.get("T_words", 0),
        "S_words": avg.get("S_words", 0),
        "P_words": avg.get("P_words", 0),
        "WPM_user": avg.get("WPM_user", 0),
        "WPM_server": avg.get("WPM_server", 0),
        "WPM_asr": avg.get("WPM_asr", 0),  # if needed adjust if a separate value is desired
        "CER_asr": avg.get("CER_asr", 0),
        "WER_asr": avg.get("WER_asr", 0),
        "CER_user": avg.get("CER_user", 0),
        "WER_user": avg.get("WER_user", 0),
        "Backspaces": avg.get("Backspaces", 0)
    })



if __name__ == '__main__':
    app.run(debug=True, port=5001)
