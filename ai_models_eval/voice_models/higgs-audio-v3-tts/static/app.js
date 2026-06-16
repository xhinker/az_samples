/**
 * Higgs Audio v3 TTS — Web Test Page
 *
 * Features:
 * - Batch TTS (full audio download + playback)
 * - Streaming TTS (SSE with real-time Web Audio API playback)
 * - Voice cloning with reference audio upload
 * - Waveform visualization
 * - Gapless streaming playback
 */

// --- State ---
let audioContext = null;
let audioBuffer = null;
let sourceNode = null;
let isPlaying = false;
let isPaused = false;
let pauseStartTime = 0;
let playbackStartTime = 0;
let abortController = null;
let currentWriteStream = null;  // reference to active streaming playback
let streamingChunks = [];
let streamingStartTime = 0;

// --- DOM refs ---
const $ = id => document.getElementById(id);
const textInput = $('text-input');
const voiceSelect = $('voice-select');
const tempSlider = $('temp-slider');
const tempValue = $('temp-value');
const formatSelect = $('format-select');
const modeSelect = $('mode-select');
const refAudioFile = $('ref-audio-file');
const refFileName = $('ref-file-name');
const refTextInput = $('ref-text-input');
const btnGenerate = $('btn-generate');
const btnStop = $('btn-stop');
const btnDownload = $('btn-download');
const btnPlay = $('btn-play');
const btnPause = $('btn-pause');
const btnReplay = $('btn-replay');
const audioDuration = $('audio-duration');
const audioElement = $('audio-element');
const waveformCanvas = $('waveform-canvas');
const statusLog = $('status-log');
const statusDot = $('status-indicator');
const statusText = $('status-text');

// --- Init ---
tempSlider.addEventListener('input', () => {
    tempValue.textContent = parseFloat(tempSlider.value).toFixed(2);
});

refAudioFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        refFileName.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    }
});

checkHealth();
setInterval(checkHealth, 10000);

// --- Health check ---
async function checkHealth() {
    try {
        const resp = await fetch('/v1/health');
        const data = await resp.json();
        if (data.model_loaded) {
            statusDot.className = 'status-dot connected';
            statusText.textContent = `Connected — ${data.device} | ${data.sample_rate}Hz`;
        } else {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Model not loaded';
        }
    } catch (e) {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Disconnected';
    }
}

// --- Logging ---
function log(msg, level = '') {
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    entry.innerHTML = `<span class="time">[${time}]</span> ${msg}`;
    statusLog.appendChild(entry);
    statusLog.scrollTop = statusLog.scrollHeight;
}

// --- Audio Context ---
function getAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    }
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    return audioContext;
}

// --- Waveform drawing ---
function drawWaveform(data, color = '#58a6ff') {
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const mid = height / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    if (!data || data.length === 0) return;

    const step = Math.ceil(data.length / width);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i < width; i++) {
        let min = 1.0, max = -1.0;
        for (let j = 0; j < step; j++) {
            const idx = i * step + j;
            if (idx < data.length) {
                min = Math.min(min, data[idx]);
                max = Math.max(max, data[idx]);
            }
        }
        ctx.moveTo(i, mid - max * mid * 0.9);
        ctx.lineTo(i, mid - min * mid * 0.9);
    }
    ctx.stroke();
}

function drawProgress(data, progress, color = '#3fb950') {
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    ctx.fillStyle = 'rgba(63, 185, 80, 0.15)';
    ctx.fillRect(0, 0, width * progress, height);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(width * progress, 0);
    ctx.lineTo(width * progress, height);
    ctx.stroke();
}

// --- Encode reference audio to base64 ---
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result;
            const base64 = result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// --- Build request payload ---
function buildPayload() {
    const payload = {
        input: textInput.value.trim(),
        voice: voiceSelect.value,
        response_format: formatSelect.value,
        temperature: parseFloat(tempSlider.value),
    };
    return payload;
}

// --- Generation ---
async function startGeneration() {
    const text = textInput.value.trim();
    if (!text) {
        log('Error: No text input', 'error');
        return;
    }

    // Reset state
    stopPlayback();
    streamingChunks = [];
    btnGenerate.disabled = true;
    btnStop.disabled = false;
    btnDownload.disabled = true;
    statusDot.className = 'status-dot generating';
    statusText.textContent = 'Generating...';
    statusLog.innerHTML = '';

    abortController = new AbortController();
    const payload = buildPayload();

    // Add reference audio if uploaded
    if (refAudioFile.files[0]) {
        try {
            payload.reference_audio = await fileToBase64(refAudioFile.files[0]);
            if (refTextInput.value.trim()) {
                payload.reference_text = refTextInput.value.trim();
            }
            log(`Voice cloning: ${refAudioFile.files[0].name}`);
        } catch (e) {
            log(`Error reading reference audio: ${e.message}`, 'error');
            return;
        }
    }

    const mode = modeSelect.value;
    log(`Mode: ${mode} | Voice: ${payload.voice} | Temp: ${payload.temperature} | Format: ${payload.response_format}`);
    log(`Text: ${text.substring(0, 100)}${text.length > 100 ? '...' : ''}`);

    try {
        if (mode === 'stream') {
            await startStreaming(payload, abortController);
        } else {
            await startBatch(payload, abortController);
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            log('Generation stopped by user', 'warning');
        } else {
            log(`Error: ${e.message}`, 'error');
        }
    } finally {
        btnGenerate.disabled = false;
        btnStop.disabled = true;
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Ready';
    }
}

// --- Batch mode ---
async function startBatch(payload, abortCtrl) {
    log('Sending batch request...');
    const startTime = Date.now();

    const resp = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortCtrl.signal,
    });

    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || err.error || resp.statusText);
    }

    const blob = await resp.blob();
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    log(`Received ${blob.size} bytes in ${elapsed}s`, 'success');

    // Decode for playback
    const arrayBuffer = await blob.arrayBuffer();
    const ctx = getAudioContext();

    try {
        audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    } catch {
        // Might be raw PCM — wrap in WAV
        const wavBlob = pcmToWav(arrayBuffer, 24000);
        audioBuffer = await ctx.decodeAudioData(await wavBlob.arrayBuffer());
    }

    const duration = audioBuffer.duration;
    audioDuration.textContent = `${duration.toFixed(1)}s`;
    log(`Decoded: ${duration.toFixed(1)}s audio, ${audioBuffer.sampleRate}Hz`, 'success');

    // Draw waveform
    const rawData = audioBuffer.getChannelData(0);
    drawWaveform(rawData);

    // Set up audio element for download
    audioElement.src = URL.createObjectURL(blob);
    btnDownload.disabled = false;
    btnPlay.disabled = false;
    btnReplay.disabled = false;
}

// --- Streaming mode ---
async function startStreaming(payload, abortCtrl) {
    log('Starting streaming request...');
    streamingStartTime = Date.now();

    const resp = await fetch('/v1/audio/speech-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortCtrl.signal,
    });

    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || err.error || resp.statusText);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let totalBytes = 0;
    let chunkCount = 0;
    let pcmData = [];
    let sampleRate = 24000;

    // Start real-time playback
    const ctx = getAudioContext();
    const ws = createPcmStream(ctx);
    currentWriteStream = ws;  // track for stop button

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            // Check if user pressed Stop
            if (abortCtrl.signal.aborted) {
                await reader.cancel();
                throw new DOMException('Aborted', 'AbortError');
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.type === 'metadata') {
                        sampleRate = data.sample_rate || 24000;
                        log(`Stream metadata: ${sampleRate}Hz, ${data.format}`);
                    } else if (data.type === 'audio') {
                        const binaryStr = atob(data.data);
                        const bytes = new Uint8Array(binaryStr.length);
                        for (let i = 0; i < binaryStr.length; i++) {
                            bytes[i] = binaryStr.charCodeAt(i);
                        }
                        pcmData.push(bytes);
                        totalBytes += bytes.length;
                        chunkCount++;

                        // Write to audio stream for real-time playback
                        ws.write(bytes);

                        // Update progress and waveform every 5 chunks
                        if (chunkCount % 5 === 0) {
                            const elapsed = ((Date.now() - streamingStartTime) / 1000).toFixed(1);
                            const duration = (totalBytes / 2 / sampleRate).toFixed(1);
                            statusText.textContent = `Streaming... ${duration}s audio (${elapsed}s elapsed)`;
                            // Draw progressive waveform
                            const allData = new Int16Array(totalBytes / 2);
                            let off = 0;
                            for (const c of pcmData) {
                                allData.set(new Int16Array(c.buffer, c.byteOffset, c.byteLength / 2), off);
                                off += c.byteLength / 2;
                            }
                            const floatData = new Float32Array(allData.length);
                            for (let i = 0; i < allData.length; i++) floatData[i] = allData[i] / 32768;
                            drawWaveform(floatData, '#3fb950');
                        }
                        if (chunkCount % 20 === 0) {
                            const elapsed = ((Date.now() - streamingStartTime) / 1000).toFixed(1);
                            const duration = (totalBytes / 2 / sampleRate).toFixed(1);
                            log(`  ${chunkCount} chunks: ${duration}s audio (${elapsed}s)`, 'info');
                        }
                    } else if (data.type === 'done') {
                        const elapsed = ((Date.now() - streamingStartTime) / 1000).toFixed(1);
                        log(`Stream complete: ${data.total_chunks} chunks, ${data.total_bytes} bytes, ${data.duration_seconds}s (${elapsed}s total)`, 'success');
                        // Keep play buttons disabled until streaming processor fully drains
                        // They will be enabled after assembly + by _onStreamPlaybackDone callback
                    }
                }
            }
        }
    } finally {
        ws.close();
        currentWriteStream = null;
    }

    // Assemble full audio for playback/download
    // Stop streaming playback first to prevent overlap
    stopPlayback();

    const totalLength = pcmData.reduce((sum, c) => sum + c.length, 0);
    if (totalLength === 0) {
        log('Warning: No audio data received from stream', 'warning');
        btnGenerate.disabled = false;
        btnStop.disabled = true;
        return;
    }
    const fullPcm = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of pcmData) {
        fullPcm.set(chunk, offset);
        offset += chunk.length;
    }

    // Convert to AudioBuffer
    const int16Array = new Int16Array(fullPcm.buffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768;
    }

    audioBuffer = ctx.createBuffer(1, float32Array.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32Array);

    const duration = audioBuffer.duration;
    audioDuration.textContent = `${duration.toFixed(1)}s`;

    // Draw waveform
    drawWaveform(float32Array);

    // Set up download
    const wavBlob = pcmToWav(fullPcm.buffer, sampleRate);
    audioElement.src = URL.createObjectURL(wavBlob);
    btnDownload.disabled = false;

    // Enable play buttons only after streaming processor has fully drained
    // Set up a callback for when the stream processor disconnects
    window._onStreamPlaybackDone = () => {
        btnPlay.disabled = false;
        btnReplay.disabled = false;
        log('Playback controls ready', 'info');
    };
    // If streaming wasn't playing (buffer never filled), enable immediately
    setTimeout(() => {
        btnPlay.disabled = false;
        btnReplay.disabled = false;
    }, 2000); // Fallback timeout
}

// --- PCM streaming playback — ring buffer + ScriptProcessorNode, no overlaps ---
let _streamProcessor = null;
let _streamRingBuffer = new Float32Array(24000 * 30); // 30s max ring buffer at 24kHz
let _streamRingWrite = 0;
let _streamRingRead = 0;
let _streamRingSize = 0;
const _STREAM_RING_MAX = 24000 * 30;

function _ringPush(samples) {
    const len = samples.length;
    const half = _STREAM_RING_MAX - _streamRingWrite;
    if (len <= half) {
        _streamRingBuffer.set(samples, _streamRingWrite);
    } else {
        _streamRingBuffer.set(samples.subarray(0, half), _streamRingWrite);
        _streamRingBuffer.set(samples.subarray(half), 0);
    }
    _streamRingWrite = (_streamRingWrite + len) % _STREAM_RING_MAX;
    _streamRingSize += len;
}

function _ringPop(count) {
    let available = Math.min(count, _streamRingSize);
    if (available <= 0) return new Float32Array(count);
    const out = new Float32Array(count);
    for (let i = 0; i < available; i++) {
        out[i] = _streamRingBuffer[_streamRingRead];
        _streamRingRead = (_streamRingRead + 1) % _STREAM_RING_MAX;
        _streamRingSize--;
    }
    // Fill remainder with silence
    for (let i = available; i < count; i++) out[i] = 0;
    return out;
}

function createPcmStream(ctx) {
    _streamRingWrite = 0;
    _streamRingRead = 0;
    _streamRingSize = 0;

    if (_streamProcessor) {
        try { _streamProcessor.disconnect(); } catch(e) {}
        _streamProcessor = null;
    }

    let started = false;
    let closed = false;
    const bufferSize = 4096;
    const MIN_BUFFER_SAMPLES = 24000; // Don't start until 1s of audio buffered

    function ensureStarted() {
        if (started) return;
        // Only start when we have enough buffered audio to avoid underruns
        if (_streamRingSize < MIN_BUFFER_SAMPLES) return;
        started = true;
        _streamProcessor = ctx.createScriptProcessor(bufferSize, 1, 1);
        _streamProcessor.onaudioprocess = function(e) {
            const output = e.outputBuffer.getChannelData(0);
            const samples = _ringPop(bufferSize);
            output.set(samples);
        };
        _streamProcessor.connect(ctx.destination);
        log('Playback started (buffered ' + (_streamRingSize / ctx.sampleRate).toFixed(1) + 's)', 'info');
    }

    return {
        write(bytes) {
            if (!bytes || bytes.length === 0 || closed) return;
            const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
            _ringPush(float32);
            ensureStarted();
        },
        close() {
            closed = true;
            if (!started) return;
            // Calculate drain time based on remaining samples
            const drainTime = Math.ceil((_streamRingSize / ctx.sampleRate) * 1000) + 200;
            setTimeout(() => {
                try { _streamProcessor.disconnect(); } catch(e) {}
                _streamProcessor = null;
                _streamRingSize = 0;
                started = false;
                // Signal that streaming playback is fully done
                if (window._onStreamPlaybackDone) window._onStreamPlaybackDone();
            }, drainTime);
        },
        isPlaying() { return started && !closed; }
    };
}

// --- PCM to WAV conversion ---
function pcmToWav(pcmBuffer, sampleRate) {
    const pcm = new Int16Array(pcmBuffer);
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcm.length * (bitsPerSample / 8);
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // PCM data
    for (let i = 0; i < pcm.length; i++) {
        view.setInt16(44 + i * 2, pcm[i], true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// --- Playback controls ---
function playAudio() {
    if (!audioBuffer) return;
    stopPlayback();

    const ctx = getAudioContext();
    sourceNode = ctx.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(ctx.destination);
    sourceNode.start();
    isPlaying = true;
    playbackStartTime = ctx.currentTime;

    btnPlay.disabled = true;
    btnPause.disabled = false;

    sourceNode.onended = () => {
        if (isPlaying && !isPaused) {
            isPlaying = false;
            btnPlay.disabled = false;
            btnPause.disabled = true;
        }
    };
}

function pauseAudio() {
    if (!isPlaying || isPaused) return;
    const ctx = getAudioContext();
    ctx.suspend();
    pauseStartTime = ctx.currentTime;
    isPaused = true;
    btnPlay.disabled = false;
    btnPause.disabled = true;
    btnPlay.textContent = '▶ Resume';
}

function replayAudio() {
    if (!audioBuffer) return;
    stopPlayback();
    playAudio();
}

function stopPlayback() {
    if (sourceNode) {
        try { sourceNode.stop(); } catch (e) {}
        sourceNode.disconnect();
        sourceNode = null;
    }
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
    isPlaying = false;
    isPaused = false;
    btnPlay.disabled = !audioBuffer;
    btnPause.disabled = true;
    btnPlay.textContent = '▶ Play';
}

async function stopGeneration() {
    // Stop streaming playback immediately (ScriptProcessorNode)
    if (currentWriteStream) {
        currentWriteStream.close();
        currentWriteStream = null;
    }
    // Abort the fetch/reader
    if (abortController) {
        abortController.abort();
    }
    // Stop batch playback (AudioBufferSourceNode)
    stopPlayback();
}

function downloadAudio() {
    if (!audioElement.src) return;
    const a = document.createElement('a');
    a.href = audioElement.src;
    a.download = `higgs_tts_${Date.now()}.wav`;
    a.click();
    log('Download started', 'info');
}
