"""
Audio Handler Module - Component A: The Audio Buffer (Thread 1)

Handles microphone input with Voice Activity Detection (VAD) using Silero.
Uses a circular buffer and state machine for speech detection.

State Machine:
- SILENCE -> SPEECH_DETECTED: Start accumulating audio
- SPEECH -> SILENCE (pause > 300ms): Flag "Utterance Complete" and send buffer
"""

import threading
import queue
import time
import numpy as np
from enum import Enum, auto
from collections import deque
from typing import Optional, Callable
import logging

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import torch
except ImportError:
    torch = None

from config import AudioConfig, VADConfig

logger = logging.getLogger(__name__)


class SpeechState(Enum):
    """Voice activity state machine states."""
    SILENCE = auto()
    SPEECH_DETECTED = auto()
    SPEECH_ENDING = auto()


class AudioHandler:
    """
    Handles real-time audio capture with Voice Activity Detection.

    Uses a circular buffer to continuously record audio and Silero VAD
    to detect speech segments. When speech ends (silence > threshold),
    the accumulated audio is sent to the transcription queue.
    """

    def __init__(
        self,
        audio_config: AudioConfig,
        vad_config: VADConfig,
        transcription_queue: queue.Queue,
        on_state_change: Optional[Callable[[SpeechState], None]] = None
    ):
        self.audio_config = audio_config
        self.vad_config = vad_config
        self.transcription_queue = transcription_queue
        self.on_state_change = on_state_change

        # Calculate chunk size in samples
        self.chunk_samples = int(
            audio_config.sample_rate * audio_config.chunk_duration_ms / 1000
        )

        # Maximum buffer size in samples
        self.max_buffer_samples = int(
            audio_config.sample_rate * audio_config.max_buffer_duration_s
        )

        # State machine
        self.state = SpeechState.SILENCE
        self._state_lock = threading.Lock()

        # Audio buffer (circular buffer using deque)
        self.audio_buffer: deque = deque(maxlen=self.max_buffer_samples)
        self._buffer_lock = threading.Lock()

        # Silence tracking
        self.silence_start_time: Optional[float] = None
        self.speech_start_time: Optional[float] = None

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None

        # Initialize VAD
        self.vad_model = None
        self._init_vad()

    def _init_vad(self):
        """Initialize Silero VAD model."""
        if torch is None:
            logger.warning("PyTorch not available, VAD disabled")
            return

        try:
            # Load Silero VAD
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.vad_model.eval()
            logger.info("Silero VAD initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            self.vad_model = None

    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Run VAD on audio chunk to detect speech.

        Args:
            audio_chunk: Audio samples as float32 array

        Returns:
            True if speech detected, False otherwise
        """
        if self.vad_model is None:
            # Fallback: simple energy-based detection
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            return energy > 0.01

        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()

            # Run VAD
            speech_prob = self.vad_model(
                audio_tensor,
                self.audio_config.sample_rate
            ).item()

            return speech_prob > self.vad_config.threshold
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    def _update_state(self, is_speech: bool):
        """
        Update state machine based on speech detection.

        Args:
            is_speech: Whether speech was detected in current chunk
        """
        current_time = time.time()

        with self._state_lock:
            old_state = self.state

            if self.state == SpeechState.SILENCE:
                if is_speech:
                    self.state = SpeechState.SPEECH_DETECTED
                    self.speech_start_time = current_time
                    self.silence_start_time = None
                    logger.debug("State: SILENCE -> SPEECH_DETECTED")

            elif self.state == SpeechState.SPEECH_DETECTED:
                if not is_speech:
                    self.state = SpeechState.SPEECH_ENDING
                    self.silence_start_time = current_time
                    logger.debug("State: SPEECH_DETECTED -> SPEECH_ENDING")

            elif self.state == SpeechState.SPEECH_ENDING:
                if is_speech:
                    # Speech resumed
                    self.state = SpeechState.SPEECH_DETECTED
                    self.silence_start_time = None
                    logger.debug("State: SPEECH_ENDING -> SPEECH_DETECTED")
                else:
                    # Check if silence threshold reached
                    silence_duration_ms = (current_time - self.silence_start_time) * 1000
                    if silence_duration_ms >= self.vad_config.min_silence_duration_ms:
                        # Utterance complete - send to transcription
                        self._flush_buffer()
                        self.state = SpeechState.SILENCE
                        self.speech_start_time = None
                        self.silence_start_time = None
                        logger.debug("State: SPEECH_ENDING -> SILENCE (utterance complete)")

            # Notify state change
            if self.state != old_state and self.on_state_change:
                self.on_state_change(self.state)

    def _flush_buffer(self):
        """Send accumulated audio buffer to transcription queue."""
        with self._buffer_lock:
            if len(self.audio_buffer) == 0:
                return

            # Convert buffer to numpy array
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            self.audio_buffer.clear()

        # Check minimum speech duration
        duration_ms = len(audio_data) / self.audio_config.sample_rate * 1000
        if duration_ms < self.vad_config.min_speech_duration_ms:
            logger.debug(f"Audio too short ({duration_ms:.0f}ms), discarding")
            return

        logger.info(f"Sending {duration_ms:.0f}ms audio to transcription")

        try:
            self.transcription_queue.put_nowait(audio_data)
        except queue.Full:
            logger.warning("Transcription queue full, dropping audio")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback for audio stream.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Stream status
        """
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Convert to float32 and flatten
        audio_chunk = indata[:, 0].astype(np.float32)

        # Run VAD
        is_speech = self._detect_speech(audio_chunk)

        # Update state machine
        self._update_state(is_speech)

        # Accumulate audio if in speech state
        with self._state_lock:
            if self.state in (SpeechState.SPEECH_DETECTED, SpeechState.SPEECH_ENDING):
                with self._buffer_lock:
                    self.audio_buffer.extend(audio_chunk)

    def start(self):
        """Start audio capture."""
        if self._running:
            logger.warning("Audio handler already running")
            return

        if sd is None:
            raise RuntimeError("sounddevice not available")

        self._running = True

        # Open audio stream
        self._stream = sd.InputStream(
            samplerate=self.audio_config.sample_rate,
            channels=self.audio_config.channels,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=self._audio_callback
        )
        self._stream.start()
        logger.info("Audio capture started")

    def stop(self):
        """Stop audio capture."""
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Flush any remaining audio
        self._flush_buffer()

        logger.info("Audio capture stopped")

    def get_state(self) -> SpeechState:
        """Get current speech state."""
        with self._state_lock:
            return self.state

    def is_running(self) -> bool:
        """Check if audio capture is running."""
        return self._running
