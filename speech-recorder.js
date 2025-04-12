// speech-recorder.js - Audio recording and speech-to-text functionality

class SpeechRecorder {
  constructor(options = {}) {
    // Default configuration
    this.config = {
      silenceThreshold: options.silenceThreshold || 0.01,
      silenceTimeout: options.silenceTimeout || 5000, // 5 seconds
      serverEndpoint:
        options.serverEndpoint || "http://localhost:8000/upload-audio",
      onStartRecording: options.onStartRecording || function () {},
      onStopRecording: options.onStopRecording || function () {},
      onProcessing: options.onProcessing || function () {},
      onResult: options.onResult || function () {},
      onError: options.onError || function () {},
    };

    // Internal state
    this.isRecording = false;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.silenceStart = null;
    this.audioContext = null;
    this.scriptProcessor = null;
    this.analyser = null;
    this.microphone = null;
  }

  async startRecording() {
    if (this.isRecording) {
      console.log("Already recording");
      return;
    }

    try {
      console.log("Starting audio recording...");
      this.config.onStartRecording();

      // Request microphone access with specific constraints for better audio quality
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      });

      // Set up audio context for silence detection
      this.audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();
      this.microphone = this.audioContext.createMediaStreamSource(stream);
      this.scriptProcessor = this.audioContext.createScriptProcessor(
        2048,
        1,
        1
      );

      this.analyser.smoothingTimeConstant = 0.8;
      this.analyser.fftSize = 1024;

      this.microphone.connect(this.analyser);
      this.analyser.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioContext.destination);

      // Setup silence detection
      this.silenceStart = null;
      this.scriptProcessor.onaudioprocess = this._handleAudioProcess.bind(this);

      // Find supported MIME type
      const mimeTypes = [
        "audio/webm",
        "audio/webm;codecs=opus",
        "audio/ogg;codecs=opus",
        "audio/mp4",
      ];

      let selectedMimeType = "";
      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          selectedMimeType = type;
          console.log(`Using MIME type: ${selectedMimeType}`);
          break;
        }
      }

      const mediaRecorderOptions = selectedMimeType
        ? { mimeType: selectedMimeType }
        : {};
      this.mediaRecorder = new MediaRecorder(stream, mediaRecorderOptions);

      // Event handler for data available
      this.audioChunks = [];
      this.mediaRecorder.addEventListener("dataavailable", (e) => {
        console.log(`Received audio chunk of size: ${e.data.size} bytes`);
        this.audioChunks.push(e.data);
      });

      // Event handler when recording stops
      this.mediaRecorder.addEventListener(
        "stop",
        this._handleRecordingStopped.bind(this)
      );

      // Start recording with timeslice to get data periodically
      this.mediaRecorder.start(1000); // Get data every second
      this.isRecording = true;
    } catch (error) {
      console.error("Error accessing microphone:", error);
      this.config.onError(
        `Microphone error: ${error.message}. Please check your microphone permissions.`
      );
    }
  }

  stopRecording() {
    if (!this.isRecording) {
      console.log("Not currently recording");
      return;
    }

    console.log("Manually stopping recording");
    this.config.onStopRecording();

    // Stop all tracks first to ensure clean shutdown
    if (this.mediaRecorder && this.mediaRecorder.stream) {
      this.mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    }

    // Then stop the media recorder if it's still recording
    if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
      this.mediaRecorder.stop();
    }

    // Clean up audio processing resources
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
    }

    if (this.analyser) {
      this.analyser.disconnect();
    }

    if (this.microphone) {
      this.microphone.disconnect();
    }

    // Update state
    this.isRecording = false;
  }

  _handleAudioProcess(event) {
    if (!this.isRecording) return;

    const array = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(array);

    // Calculate volume level (simple average)
    let sum = 0;
    for (let i = 0; i < array.length; i++) {
      sum += array[i];
    }
    const average = sum / array.length / 255; // Normalize to 0-1

    // For debugging
    // console.log("Audio level:", average.toFixed(4));

    // Check for silence
    if (average < this.config.silenceThreshold) {
      if (!this.silenceStart) {
        this.silenceStart = Date.now();
        console.log("Silence detected, starting timer...");
      } else if (Date.now() - this.silenceStart > this.config.silenceTimeout) {
        console.log("Silence timeout reached, stopping recording");
        // Stop recording after silence threshold
        if (this.isRecording) {
          this.stopRecording();
        }
      }
    } else {
      // Reset silence timer when sound is detected
      this.silenceStart = null;
    }
  }

  async _handleRecordingStopped() {
    console.log(
      `Recording stopped, processing ${this.audioChunks.length} chunks`
    );

    // Convert audio chunks to Blob
    const audioBlob = new Blob(this.audioChunks, {
      type: this.mediaRecorder.mimeType || "audio/webm",
    });
    console.log(
      `Created audio blob of type ${audioBlob.type}, size: ${audioBlob.size} bytes`
    );

    if (audioBlob.size <= 100) {
      console.warn("Audio blob is too small, likely no audio was recorded");
      this.config.onError("No audio detected. Please try again.");
      return;
    }

    try {
      this.config.onProcessing();

      // For debugging: save blob to a variable accessible in the console
      window.lastAudioBlob = audioBlob;

      // Convert WebM to WAV
      console.log("Converting WebM to WAV...");
      let fileToSend;
      let fileName;

      try {
        const wavBlob = await this.convertWebmToWav(audioBlob);
        console.log("Successfully converted to WAV");
        fileToSend = wavBlob;
        fileName = `recording_${Date.now()}.wav`;
      } catch (conversionError) {
        console.error("WebM to WAV conversion failed:", conversionError);
        console.log("Falling back to original WebM file");
        fileToSend = audioBlob;
        fileName = `recording_${Date.now()}.webm`;
      }

      // Create FormData to send the file
      const formData = new FormData();
      formData.append("file", fileToSend, fileName);

      console.log(`Sending file ${fileName} to server...`);

      // Send to server endpoint
      const response = await fetch(this.config.serverEndpoint, {
        method: "POST",
        body: formData,
      });

      console.log(`Server response status: ${response.status}`);

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}: ${response.statusText}`
        );
      }

      // Parse the response
      const data = await response.json();
      console.log(`Transcription result:`, data);

      // If we got a transcription
      if (data.text && data.text.trim()) {
        this.config.onResult(data.text);
      } else {
        this.config.onError(
          "I couldn't detect any speech. Please try again or type your message."
        );
      }
    } catch (error) {
      console.error("Transcription error:", error);
      this.config.onError(
        "Sorry, I had trouble understanding that. Please try again or type your message."
      );
    }
  }

  // WebM to WAV conversion
  async convertWebmToWav(webmBlob) {
    return new Promise((resolve, reject) => {
      console.log("Starting WebM to WAV conversion...");

      // Create audio context
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();

      // Create file reader to read the blob
      const fileReader = new FileReader();

      fileReader.onload = async function () {
        try {
          console.log("FileReader loaded the WebM blob");

          // Decode the audio file
          const arrayBuffer = fileReader.result;
          console.log(`Array buffer size: ${arrayBuffer.byteLength} bytes`);

          // Decode the audio data
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          console.log(
            `Decoded audio buffer: ${audioBuffer.duration}s, ${audioBuffer.numberOfChannels} channels, ${audioBuffer.sampleRate}Hz`
          );

          // Convert to WAV
          const wavBlob = await this._audioBufferToWav(audioBuffer);
          console.log(`Converted to WAV, size: ${wavBlob.size} bytes`);

          resolve(wavBlob);
        } catch (error) {
          console.error("Error converting audio:", error);
          reject(error);
        }
      }.bind(this);

      fileReader.onerror = function () {
        console.error("FileReader error");
        reject(new Error("FileReader error"));
      };

      // Read the blob as array buffer
      fileReader.readAsArrayBuffer(webmBlob);
    });
  }

  // Convert AudioBuffer to WAV Blob
  _audioBufferToWav(buffer) {
    return new Promise((resolve) => {
      const numberOfChannels = buffer.numberOfChannels;
      const sampleRate = buffer.sampleRate;
      const format = 1; // PCM
      const bitDepth = 16;

      let result;
      if (numberOfChannels === 2) {
        result = this._interleave(
          buffer.getChannelData(0),
          buffer.getChannelData(1)
        );
      } else {
        result = buffer.getChannelData(0);
      }

      // Convert to 16-bit PCM
      const dataLength = result.length * (bitDepth / 8);
      const view = new DataView(new ArrayBuffer(44 + dataLength));

      // RIFF identifier
      this._writeString(view, 0, "RIFF");
      // File length
      view.setUint32(4, 36 + dataLength, true);
      // RIFF type
      this._writeString(view, 8, "WAVE");
      // Format chunk identifier
      this._writeString(view, 12, "fmt ");
      // Format chunk length
      view.setUint32(16, 16, true);
      // Sample format (PCM)
      view.setUint16(20, format, true);
      // Channel count
      view.setUint16(22, numberOfChannels, true);
      // Sample rate
      view.setUint32(24, sampleRate, true);
      // Byte rate (sample rate * block align)
      view.setUint32(28, sampleRate * numberOfChannels * (bitDepth / 8), true);
      // Block align (channel count * bytes per sample)
      view.setUint16(32, numberOfChannels * (bitDepth / 8), true);
      // Bits per sample
      view.setUint16(34, bitDepth, true);
      // Data chunk identifier
      this._writeString(view, 36, "data");
      // Data chunk length
      view.setUint32(40, dataLength, true);

      // Write the PCM samples
      this._floatTo16BitPCM(view, 44, result);

      // Create a Blob
      const wavBlob = new Blob([view], { type: "audio/wav" });

      resolve(wavBlob);
    });
  }

  // Helper function to write a string to a DataView
  _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Helper function to interleave audio channels
  _interleave(inputL, inputR) {
    const length = inputL.length + inputR.length;
    const result = new Float32Array(length);

    let index = 0;
    let inputIndex = 0;

    while (index < length) {
      result[index++] = inputL[inputIndex];
      result[index++] = inputR[inputIndex];
      inputIndex++;
    }
    return result;
  }

  // Helper function to convert float to 16-bit PCM
  _floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }
}

// Export the class
if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
  module.exports = SpeechRecorder;
} else {
  window.SpeechRecorder = SpeechRecorder;
}
