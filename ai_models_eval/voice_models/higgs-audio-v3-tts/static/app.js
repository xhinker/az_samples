const form = document.getElementById('stream-form');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const logEl = document.getElementById('status-log');
const tempSlider = document.getElementById('temperature');
const tempVal = document.getElementById('temp-val');

let audioContext = null;
let gainNode = null;
let nextPlaybackTime = 0;
let activeSources = new Set();
let controller = null;
let pcmStash = new Uint8Array(0);

const STARTUP_BUFFER_SEC = 1.0;
const MIN_SOURCE_SAMPLES = 4096;

function log(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}\n`;
  logEl.textContent += line;
  logEl.scrollTop = logEl.scrollHeight;
}

tempSlider.addEventListener('input', () => {
  tempVal.textContent = tempSlider.value;
});

function ensureAudioContext(sampleRate) {
  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate });
    gainNode = audioContext.createGain();
    gainNode.connect(audioContext.destination);
    nextPlaybackTime = audioContext.currentTime + STARTUP_BUFFER_SEC;
  }
  if (audioContext.state === 'suspended') {
    return audioContext.resume();
  }
  return Promise.resolve();
}

function enqueuePcmChunk(pcmBytes, sampleRate) {
  if (!pcmBytes || pcmBytes.byteLength === 0) return;
  const int16 = new Int16Array(
    pcmBytes.buffer, pcmBytes.byteOffset,
    Math.floor(pcmBytes.byteLength / 2)
  );
  if (int16.length === 0) return;
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

function pushPcmBytes(pcmBytes, sampleRate, flushAll = false) {
  if (pcmBytes && pcmBytes.byteLength > 0) {
    const merged = new Uint8Array(pcmStash.byteLength + pcmBytes.byteLength);
    merged.set(pcmStash, 0);
    merged.set(new Uint8Array(pcmBytes), pcmStash.byteLength);
    pcmStash = merged;
  }
  const minFlushBytes = MIN_SOURCE_SAMPLES * 2;
  while (pcmStash.byteLength >= minFlushBytes || (flushAll && pcmStash.byteLength >= 2)) {
    const flushBytes = flushAll ? pcmStash.byteLength - (pcmStash.byteLength % 2) : minFlushBytes;
    if (flushBytes <= 0) break;
    const chunk = pcmStash.slice(0, flushBytes);
    pcmStash = pcmStash.slice(flushBytes);
    enqueuePcmChunk(chunk, sampleRate);
  }
}

async function stopPlayback({ silent = false } = {}) {
  if (controller) {
    controller.abort();
    controller = null;
  }
  activeSources.forEach(source => {
    try { source.stop(); } catch (err) {}
  });
  activeSources.clear();
  pcmStash = new Uint8Array(0);
  if (audioContext) {
    nextPlaybackTime = audioContext.currentTime;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  if (!silent) log('Stopped.');
}

async function streamAudioChunked(body) {
  controller = new AbortController();
  const response = await fetch('/v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.error) detail = errorJson.error.message || JSON.stringify(errorJson.error);
    } catch (e) {}
    throw new Error(detail);
  }

  const sampleRate = Number(response.headers.get('x-sample-rate')) || 24000;
  await ensureAudioContext(sampleRate);
  nextPlaybackTime = audioContext.currentTime + STARTUP_BUFFER_SEC;
  pcmStash = new Uint8Array(0);
  log('Streaming started...');

  const reader = response.body.getReader();
  let carry = new Uint8Array(0);
  let totalBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value || value.byteLength === 0) continue;

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
    totalBytes += pcm.byteLength;
    pushPcmBytes(pcm, sampleRate, false);
  }
  pushPcmBytes(carry, sampleRate, true);
  log(`Done. Total: ${(totalBytes / 1024).toFixed(1)} KB`);
}

stopBtn.addEventListener('click', async () => {
  await stopPlayback();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  await stopPlayback({ silent: true });

  const text = document.getElementById('text').value.trim();
  if (!text) { log('Text is required.'); return; }

  const format = document.getElementById('response_format').value;
  const streamMode = document.getElementById('stream_mode').value;
  const voice = document.getElementById('voice').value;
  const temperature = parseFloat(document.getElementById('temperature').value);

  const body = {
    input: text,
    model: 'higgs-audio-v3-tts',
    voice: voice,
    response_format: format,
    temperature: temperature,
  };

  // Voice cloning
  const refAudio = document.getElementById('reference_audio');
  const refText = document.getElementById('reference_text').value.trim();
  if (refAudio.files && refAudio.files.length > 0) {
    const file = refAudio.files[0];
    const buf = await file.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
    body.reference_audio = b64;
    if (refText) body.reference_text = refText;
  }

  startBtn.disabled = true;
  stopBtn.disabled = false;

  try {
    if (format === 'mp3') {
      // MP3 is non-streaming — download the file
      const resp = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error?.message || `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'speech.mp3';
      a.click();
      URL.revokeObjectURL(url);
      log('MP3 downloaded.');
    } else {
      // WAV/PCM streaming
      await streamAudioChunked(body);
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      log('Stream aborted.');
    } else {
      log(`Error: ${err.message}`);
    }
  } finally {
    controller = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
});

log('Ready. Higgs Audio v3 TTS test page.');
