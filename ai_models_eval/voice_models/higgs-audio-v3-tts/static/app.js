const form = document.getElementById('stream-form');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const logEl = document.getElementById('status-log');
const tempSlider = document.getElementById('temperature');
const tempVal = document.getElementById('temp-val');

let audioContext = null;
let gainNode = null;
// Monotonic scheduling clock — only advances by buffer durations, never drifts.
let scheduleClock = 0;
let activeSources = new Set();
let controller = null;
let pcmStash = new Uint8Array(0);

const STARTUP_BUFFER_SEC = 0.3;   // Shorter initial buffer for lower latency
const MIN_SOURCE_SAMPLES = 2048;  // Smaller flush threshold = more frequent scheduling

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
    gainNode.gain.value = 1.0;
    gainNode.connect(audioContext.destination);
    // Start the monotonic clock from a point slightly in the future so the first
    // buffer has time to arrive and fill before playback begins.
    scheduleClock = audioContext.currentTime + STARTUP_BUFFER_SEC;
  }
  if (audioContext.state === 'suspended') {
    return audioContext.resume().then(() => {
      // After resume, adjust scheduleClock to current time + small offset
      // to avoid scheduling in the past.
      scheduleClock = Math.max(scheduleClock, audioContext.currentTime + 0.01);
    });
  }
  return Promise.resolve();
}

/**
 * Schedule a PCM chunk onto the Web Audio timeline.
 * Uses a monotonic scheduleClock that only advances by each buffer's duration —
 * never looks at currentTime — so late-arriving chunks catch up seamlessly
 * without creating gaps.
 */
function enqueuePcmChunk(pcmBytes, sampleRate) {
  if (!pcmBytes || pcmBytes.byteLength === 0) return;
  const int16 = new Int16Array(
    pcmBytes.buffer, pcmBytes.byteOffset,
    Math.floor(pcmBytes.byteLength / 2)
  );
  if (int16.length === 0) return;

  // Convert int16 -> float32 in-place for efficiency
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }

  const buffer = audioContext.createBuffer(1, float32.length, sampleRate);
  buffer.getChannelData(0).set(float32);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(gainNode);

  // Schedule at the monotonic clock position — no drift, no gaps.
  const startAt = scheduleClock;
  source.start(startAt);
  scheduleClock = startAt + buffer.duration;

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
    // Reset scheduleClock so next playback starts fresh without scheduling in the past
    scheduleClock = audioContext.currentTime + STARTUP_BUFFER_SEC;
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
  // Flush remaining bytes
  pushPcmBytes(carry, sampleRate, true);
  
  // Wait for all scheduled audio to finish playing before logging "Done"
  const remainingSec = scheduleClock - audioContext.currentTime;
  if (remainingSec > 0.05) {
    log(`Streaming done (${(totalBytes / 1024).toFixed(1)} KB), ${remainingSec.toFixed(1)}s of audio still playing...`);
  } else {
    log(`Done. Total: ${(totalBytes / 1024).toFixed(1)} KB`);
  }
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
      // WAV/PCM streaming with gap-free playback
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
