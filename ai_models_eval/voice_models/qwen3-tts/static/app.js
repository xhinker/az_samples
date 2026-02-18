const form = document.getElementById("stream-form");
const modeEl = document.getElementById("mode");
const cloneFields = document.getElementById("clone-fields");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const logEl = document.getElementById("status-log");

let audioContext = null;
let gainNode = null;
let nextPlaybackTime = 0;
let activeSources = new Set();
let controller = null;

function log(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  logEl.textContent = `${line}\n${logEl.textContent}`.slice(0, 10000);
}

function updateModeUI() {
  const showClone = modeEl.value === "voice_clone";
  cloneFields.style.display = showClone ? "grid" : "none";
}

function ensureAudioContext(sampleRate) {
  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate });
    gainNode = audioContext.createGain();
    gainNode.connect(audioContext.destination);
    nextPlaybackTime = audioContext.currentTime;
  }
  if (audioContext.state === "suspended") {
    return audioContext.resume();
  }
  return Promise.resolve();
}

function enqueuePcmChunk(pcmBytes, sampleRate) {
  if (!pcmBytes || pcmBytes.byteLength === 0) {
    return;
  }
  const int16 = new Int16Array(
    pcmBytes.buffer,
    pcmBytes.byteOffset,
    Math.floor(pcmBytes.byteLength / 2)
  );
  if (int16.length === 0) {
    return;
  }
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) {
    float32[i] = int16[i] / 32768.0;
  }
  const buffer = audioContext.createBuffer(1, float32.length, sampleRate);
  buffer.getChannelData(0).set(float32);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(gainNode);
  const startAt = Math.max(nextPlaybackTime, audioContext.currentTime + 0.02);
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;
  activeSources.add(source);
  source.onended = () => activeSources.delete(source);
}

function stopPlayback({ silent = false } = {}) {
  if (controller) {
    controller.abort();
    controller = null;
  }
  activeSources.forEach((source) => {
    try {
      source.stop();
    } catch (err) {
      console.debug("stop source error", err);
    }
  });
  activeSources.clear();
  if (audioContext) {
    nextPlaybackTime = audioContext.currentTime;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  if (!silent) {
    log("Stopped stream.");
  }
}

async function streamAudio(formData) {
  controller = new AbortController();
  const response = await fetch("/api/stream", {
    method: "POST",
    body: formData,
    signal: controller.signal,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.error) {
        detail = errorJson.error;
      }
    } catch (e) {
      console.debug("error parsing error payload", e);
    }
    throw new Error(detail);
  }

  const sampleRate = Number(response.headers.get("x-sample-rate")) || 24000;
  await ensureAudioContext(sampleRate);
  log(`Streaming started. sample_rate=${sampleRate}`);

  const reader = response.body.getReader();
  let carry = new Uint8Array(0);

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (!value || value.byteLength === 0) {
      continue;
    }

    const merged = new Uint8Array(carry.byteLength + value.byteLength);
    merged.set(carry, 0);
    merged.set(value, carry.byteLength);

    const evenLength = merged.byteLength - (merged.byteLength % 2);
    if (evenLength === 0) {
      carry = merged;
      continue;
    }

    const pcm = merged.slice(0, evenLength);
    carry = merged.slice(evenLength);
    enqueuePcmChunk(pcm, sampleRate);
  }

  if (carry.byteLength > 0) {
    log("Dropped trailing odd byte at stream end.");
  }
}

modeEl.addEventListener("change", updateModeUI);
stopBtn.addEventListener("click", stopPlayback);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPlayback({ silent: true });

  const formData = new FormData(form);
  const mode = formData.get("mode");
  const text = String(formData.get("text") || "").trim();
  if (!text) {
    log("Text is required.");
    return;
  }
  if (mode === "voice_clone") {
    const audioFile = formData.get("reference_audio");
    const referenceText = String(formData.get("reference_text") || "").trim();
    if (!(audioFile instanceof File) || audioFile.size === 0) {
      log("Voice clone mode requires a reference audio file.");
      return;
    }
    if (!referenceText) {
      log("Voice clone mode requires reference transcript.");
      return;
    }
  }

  startBtn.disabled = true;
  stopBtn.disabled = false;
  try {
    await streamAudio(formData);
    log("Stream completed.");
  } catch (err) {
    if (err.name === "AbortError") {
      log("Stream aborted.");
    } else {
      log(`Stream failed: ${err.message}`);
    }
  } finally {
    controller = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
});

updateModeUI();
log("Ready.");
