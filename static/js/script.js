// Global variables for timing, state, and text tracking
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let recordStartTime = null;
let stopTime = null;
let serverReceiveTime = null;
let correctionStarted = false;
let correctionStartTime = null;
let backspacesCount = 0;
let asrTranscription = "";
let currentIndex = 0;
let totalTexts = 0;
let originalText = "";

// On page load, fetch the first text.
// (If running in the new experiment pages, a global "textFolder" variable is set via template.)
document.addEventListener('DOMContentLoaded', function() {
  if (typeof textFolder !== 'undefined') {
    fetchText(currentIndex);
  } else {
    fetchSentence(currentIndex);
  }
});

// Fetch text from the backend using /get_text route.
function fetchText(index) {
  fetch(`/get_text?folder=${textFolder}&index=${index}`)
    .then(response => {
      if (!response.ok) {
        // If no more texts, redirect to results page.
        window.location.href = "/results";
        throw new Error("No more texts.");
      }
      return response.json();
    })
    .then(data => {
      originalText = data.text;
      totalTexts = data.total;
      document.getElementById('heading').innerText = `${phase.charAt(0).toUpperCase() + phase.slice(1)} ${language} ${data.index + 1}/${data.total} texts`;
      document.getElementById('original-text').innerText = originalText;
      document.getElementById('transcription').innerText = "";
    })
    .catch(err => console.error(err));
}

// (For backward compatibility, in case /get_sentence is used.)
function fetchSentence(index) {
  fetch(`/get_sentence?index=${index}`)
    .then(response => {
      if (!response.ok) {
        endExperiment();
        throw new Error("No more sentences.");
      }
      return response.json();
    })
    .then(data => {
      originalText = data.sentence;
      document.getElementById('original-text').innerText = originalText;
      document.getElementById('transcription').innerText = "";
    })
    .catch(err => console.error(err));
}

const recordBtn = document.getElementById('record-btn');
const nextBtn = document.getElementById('next-btn');
const statusDiv = document.getElementById('status');
const transcriptionDiv = document.getElementById('transcription');

recordBtn.addEventListener('click', function() {
  if (!isRecording) {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        recordStartTime = Date.now();
        audioChunks = [];
        mediaRecorder.addEventListener("dataavailable", event => {
          audioChunks.push(event.data);
        });
        isRecording = true;
        recordBtn.textContent = "Stop";
        statusDiv.innerText = "Recording...";
      })
      .catch(err => {
        console.error("Error accessing microphone: ", err);
        alert("Error accessing microphone.");
      });
  } else {
    mediaRecorder.stop();
    stopTime = Date.now();
    isRecording = false;
    recordBtn.textContent = "Record";
    statusDiv.innerText = "Processing transcription...";
    mediaRecorder.addEventListener("stop", () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      const formData = new FormData();
      const filename = `recording_${Date.now()}.wav`;
      formData.append("audio_data", audioBlob, filename);
      fetch('/transcribe', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          serverReceiveTime = Date.now();
          statusDiv.innerText = `Transcription complete in ${data.inference_time.toFixed(2)} seconds.`;
          asrTranscription = data.transcription;
          transcriptionDiv.innerText = data.transcription;
          highlightDifferences();
        })
        .catch(err => {
          console.error("Transcription error: ", err);
          statusDiv.innerText = "Error during transcription.";
        });
    }, { once: true });
  }
});

transcriptionDiv.addEventListener('keydown', function(e) {
  if (!correctionStarted) {
    correctionStarted = true;
    correctionStartTime = Date.now();
  }
  if (e.key === "Backspace") {
    backspacesCount++;
  }
});

transcriptionDiv.addEventListener('input', highlightDifferences);
function highlightDifferences() {
  let userText = transcriptionDiv.innerText.replace(/\u00A0/g, ' ');
  let highlightedHTML = "";
  let maxLength = Math.max(userText.length, originalText.length);
  const cursorPos = getCursorPosition(transcriptionDiv);
  for (let i = 0; i < maxLength; i++) {
    let origChar = originalText[i] || "";
    let userChar = userText[i] || "";
    highlightedHTML += (userChar === origChar) ? userChar : (userChar ? `<span class="highlight">${userChar}</span>` : "");
  }
  transcriptionDiv.innerHTML = highlightedHTML;
  if (cursorPos) {
    setCursorPosition(transcriptionDiv, cursorPos.start, cursorPos.end);
  }
}

function getCursorPosition(div) {
  const sel = window.getSelection();
  if (sel.rangeCount === 0) return null;
  const range = sel.getRangeAt(0);
  const preRange = document.createRange();
  preRange.selectNodeContents(div);
  preRange.setEnd(range.startContainer, range.startOffset);
  const start = preRange.toString().length;
  const end = start + range.toString().length;
  return { start, end };
}

function setCursorPosition(div, startOffset, endOffset) {
  const sel = window.getSelection();
  const range = document.createRange();
  let nodeStack = [div], node, foundStart = false, currentOffset = 0, startNode, startOffsetInNode, endNode, endOffsetInNode;
  while ((node = nodeStack.pop())) {
    if (node.nodeType === Node.TEXT_NODE) {
      let nodeLength = node.length;
      let nextOffset = currentOffset + nodeLength;
      if (!foundStart && startOffset <= nextOffset) {
        startNode = node;
        startOffsetInNode = Math.max(0, startOffset - currentOffset);
        foundStart = true;
      }
      if (foundStart && endOffset <= nextOffset) {
        endNode = node;
        endOffsetInNode = Math.max(0, endOffset - currentOffset);
        break;
      }
      currentOffset = nextOffset;
    } else {
      for (let i = node.childNodes.length - 1; i >= 0; i--) {
        nodeStack.push(node.childNodes[i]);
      }
    }
  }
  if (startNode) {
    range.setStart(startNode, startOffsetInNode);
    range.setEnd(endNode || startNode, endOffsetInNode || startOffsetInNode);
    sel.removeAllRanges();
    sel.addRange(range);
  }
}

nextBtn.addEventListener('click', function() {
  let correctionEndTime = Date.now();
  let correctionTime = correctionStarted ? (correctionEndTime - correctionStartTime) / 1000 : 0;
  let userTalkingTime = recordStartTime && stopTime ? (stopTime - recordStartTime) / 1000 : 0;
  let serverProcessTime = stopTime && serverReceiveTime ? (serverReceiveTime - stopTime) / 1000 : 0;
  let presented = originalText;
  let asrResult = asrTranscription;
  let edited = transcriptionDiv.innerText;
  function countWords(text) {
    return text.trim().split(/\s+/).filter(Boolean).length;
  }
  let L_T = countWords(edited);
  let L_S = countWords(asrResult);
  let payload = {
    trial: currentIndex + 1,
    P: presented,
    S: asrResult,
    T: edited,
    user_talking_time: userTalkingTime,
    server_process_time: serverProcessTime,
    correction_time: correctionTime,
    backspaces: backspacesCount
  };
  fetch('/log_trial', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(response => response.json())
  .then(data => {
    console.log("Trial logged:", data);
    currentIndex++;
    fetchText(currentIndex);
    correctionStarted = false;
    correctionStartTime = null;
    backspacesCount = 0;
    asrTranscription = "";
  })
  .catch(err => console.error("Logging error:", err));
});

function endExperiment() {
  fetch('/end_experiment', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      statusDiv.innerHTML = `
        <strong>Averages:</strong><br>
        User talking time: ${data.user_talking_time.toFixed(2)} s<br>
        Server process time: ${data.server_process_time.toFixed(2)} s<br>
        Correction time: ${data.correction_time.toFixed(2)} s<br>
        Total time: ${data.t_i.toFixed(2)} s<br>
        L_T: ${data.L_T_i}<br>
        L_S: ${data.L_S_i}<br>
        WPM: ${data.WPM.toFixed(2)}<br>
        WPM_asr: ${data.WPM_asr.toFixed(2)}<br>
        CER_ASR: ${data.CER_ASR_i.toFixed(2)}%<br>
        WER_ASR: ${data.WER_ASR_i.toFixed(2)}%<br>
        CER_Edit: ${data.CER_Edit_i.toFixed(2)}%<br>
        WER_Edit: ${data.WER_Edit_i.toFixed(2)}%<br>
        Backspaces: ${data.Backspaces_i}
      `;
    })
    .catch(err => console.error("End experiment logging error:", err));
}
