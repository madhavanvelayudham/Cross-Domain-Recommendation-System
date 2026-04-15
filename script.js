window.onerror = function(msg, src, line) {
  console.error("GLOBAL ERROR:", msg, "at", line);
};

const input = document.getElementById("user-input");
const textButton = document.getElementById("send-btn");
const recordButton = document.getElementById("record-btn");
const uploadButton = document.getElementById("uploadBtn");
const audioFileInput = document.getElementById("audioFileInput");
const fileNameElement = document.getElementById("fileName");
const playAudioButton = document.getElementById("play-audio-btn");
const titleElement = document.getElementById("category-title");
const resultsContainer = document.getElementById("results");
const voiceAudio = document.getElementById("voice-audio");

const TEXT_URL = "http://127.0.0.1:8000/recommend";
const AUDIO_FILE_URL = "http://127.0.0.1:8000/recommend-audio-file";
const VOICE_URL = "http://127.0.0.1:8000/recommend-voice";
const API_BASE = "http://127.0.0.1:8000";

let mediaRecorder = null;
let audioChunks = [];
let activeStream = null;
let latestAudioUrl = null;
let isRecording = false;

function setStatus(msg) {
  const el = document.getElementById("status");
  if (!el) return;

  const text = msg || "";
  el.innerText = text;
  const loadingTerms = ["Processing", "Uploading", "Transcribing", "Loading"];
  el.classList.toggle("loading", loadingTerms.some((term) => text.startsWith(term)));
}

function showError(msg) {
  setStatus(msg);
  console.error(msg);
}

function clearResults() {
  if (resultsContainer) {
    resultsContainer.innerHTML = "";
  }
}

function formatRating(item) {
  const value = Number(item.rating || item.display_rating);
  const normalized = Number.isFinite(value) ? value.toFixed(2) : "0.00";
  return `\u2B50 ${normalized} stars`;
}

function playAudio(url) {
  latestAudioUrl = url || null;
  const fullUrl = API_BASE + url;

  if (voiceAudio) {
    voiceAudio.src = fullUrl;
  }

  const audio = voiceAudio || new Audio(fullUrl);
  audio.play().catch((err) => console.error("Audio failed:", err));
}

function updateAudioControls(url) {
  latestAudioUrl = url || null;
  if (!playAudioButton) return;
  playAudioButton.classList.toggle("hidden-control", !url);
}

function displayRecommendations(recs) {
  const container = document.getElementById("results");

  if (!container) {
    console.error("Missing #results container");
    return;
  }

  container.innerHTML = "";

  if (!Array.isArray(recs) || recs.length === 0) {
    container.innerHTML = "<p class=\"empty-state\">No recommendations found.</p>";
    return;
  }

  recs.forEach((item) => {
    const card = document.createElement("div");
    card.className = "card";

    card.innerHTML = `
      <h3>${item.name}</h3>
      <p>ASIN: ${item.asin}</p>
      <p class="rating">${formatRating(item)}</p>
      <button type="button" class="play-btn">\uD83D\uDD0A</button>
    `;

    card.querySelector(".play-btn")?.addEventListener("click", () => {
      if (!latestAudioUrl) {
        showError("No audio summary available.");
        return;
      }
      playAudio(latestAudioUrl);
    });

    container.appendChild(card);
  });
}

async function fetchData(url, options) {
  try {
    const res = await fetch(url, options);

    if (!res.ok) {
      throw new Error("HTTP error " + res.status);
    }

    return await res.json();
  } catch (err) {
    console.error("FETCH ERROR:", err);
    showError("Something went wrong. Check backend.");
    return null;
  }
}

async function submitText() {
  const query = input?.value?.trim();

  if (!query) {
    showError("Please enter movie reviews");
    return;
  }

  clearResults();
  setStatus("Processing audio...");
  if (titleElement) {
    titleElement.textContent = "Loading recommendations (this may take a moment)...";
  }

  const data = await fetchData(TEXT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ reviews: query })
  });

  if (!data) return;

  if (titleElement) {
    titleElement.textContent = "Music recommendations based on your movie taste:";
  }

  displayRecommendations(data.recommendations);
  updateAudioControls(data.audio_url || null);

  if (data.audio_url) {
    playAudio(data.audio_url);
  }

  setStatus("Done");
  input.value = "";
}

async function submitVoiceBlob(audioBlob) {
  clearResults();
  setStatus("Transcribing...");
  if (titleElement) {
    titleElement.textContent = "Processing your voice input...";
  }

  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  const data = await fetchData(VOICE_URL, {
    method: "POST",
    body: formData
  });

  if (!data) return;

  if (titleElement) {
    titleElement.textContent = "Music recommendations based on your movie taste:";
  }

  displayRecommendations(data.recommendations);
  updateAudioControls(data.audio_url || null);

  if (data.audio_url) {
    playAudio(data.audio_url);
  }

  setStatus("Done");
}

function stopStream() {
  if (activeStream) {
    activeStream.getTracks().forEach((track) => track.stop());
    activeStream = null;
  }
}

function resetRecorderUi() {
  isRecording = false;
  if (recordButton) {
    recordButton.textContent = "Record";
    recordButton.classList.remove("is-recording");
  }
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
    showError("Voice recording is not supported in this browser.");
    return;
  }

  try {
    activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(activeStream);

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      stopStream();
      resetRecorderUi();
      setStatus("Processing audio...");
      await submitVoiceBlob(audioBlob);
    });

    mediaRecorder.start();
    isRecording = true;
    if (recordButton) {
      recordButton.textContent = "Stop Recording";
      recordButton.classList.add("is-recording");
    }
    setStatus("Recording...");
  } catch (err) {
    console.error("Mic error:", err);
    stopStream();
    resetRecorderUi();
    showError("Microphone access failed");
  }
}

textButton?.addEventListener("click", (e) => {
  e.preventDefault();
  submitText();
});

input?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    submitText();
  }
});

recordButton?.addEventListener("click", async (e) => {
  e.preventDefault();

  if (isRecording) {
    mediaRecorder?.stop();
    return;
  }

  await startRecording();
});

uploadButton?.addEventListener("click", async (e) => {
  e.preventDefault();

  if (!audioFileInput || !audioFileInput.files.length) {
    setStatus("Please select a file");
    return;
  }

  const formData = new FormData();
  formData.append("file", audioFileInput.files[0]);

  try {
    clearResults();
    setStatus("Processing audio...");
    if (titleElement) {
      titleElement.textContent = "Processing uploaded audio...";
    }

    const data = await fetchData(AUDIO_FILE_URL, {
      method: "POST",
      body: formData
    });

    if (!data) return;

    if (titleElement) {
      titleElement.textContent = "Music recommendations based on your movie taste:";
    }

    displayRecommendations(data.recommendations);
    updateAudioControls(data.audio_url || null);

    if (data.audio_url) {
      playAudio(data.audio_url);
    }

    setStatus("Done");
  } catch (err) {
    console.error("Upload error:", err);
    setStatus("Error occurred");
  }
});

playAudioButton?.addEventListener("click", (e) => {
  e.preventDefault();

  if (!latestAudioUrl) {
    showError("No audio summary available.");
    return;
  }

  playAudio(latestAudioUrl);
});

audioFileInput?.addEventListener("change", (e) => {
  const name = e.target.files[0]?.name || "No file selected";
  if (fileNameElement) {
    fileNameElement.innerText = name;
  }
});

window.addEventListener("load", () => {
  setTimeout(() => {
    const intro = document.getElementById("intro");
    const mainContent = document.getElementById("main-content");
    if (intro) {
      intro.style.display = "none";
    }
    if (mainContent) {
      mainContent.classList.remove("hidden");
      mainContent.style.opacity = "1";
    }
  }, 5500);
});

document.querySelector("form")?.addEventListener("submit", (e) => {
  e.preventDefault();
});
