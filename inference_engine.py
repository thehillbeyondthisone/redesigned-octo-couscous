"""
Inference Engine Module - Components B & C: Transcriber and Oracle

Component B (Thread 2): Transcriber
- Watches transcription queue
- Uses faster-whisper to convert audio to English text

Component C (Thread 3): The Oracle (LLM)
- Predicts Spanish completion using Llama-3-8B
- Uses raw completion prompt (not chat template)
"""

import threading
import queue
import time
import logging
from typing import Optional, Callable, Tuple
import numpy as np

from config import WhisperConfig, LLMConfig

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
faster_whisper = None
llama_cpp = None


def _import_faster_whisper():
    global faster_whisper
    if faster_whisper is None:
        try:
            import faster_whisper as fw
            faster_whisper = fw
        except ImportError:
            logger.error("faster-whisper not installed")
    return faster_whisper


def _import_llama_cpp():
    global llama_cpp
    if llama_cpp is None:
        try:
            from llama_cpp import Llama
            llama_cpp = Llama
        except ImportError:
            logger.error("llama-cpp-python not installed")
    return llama_cpp


class Transcriber:
    """
    Component B: Audio to text transcription using faster-whisper.

    Watches the transcription queue and converts audio to English text.
    Uses beam_size=1 for maximum speed.
    """

    def __init__(
        self,
        config: WhisperConfig,
        transcription_queue: queue.Queue,
        llm_queue: queue.Queue,
        on_transcription: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.transcription_queue = transcription_queue
        self.llm_queue = llm_queue
        self.on_transcription = on_transcription

        self.model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_model()

    def _init_model(self):
        """Initialize faster-whisper model."""
        fw = _import_faster_whisper()
        if fw is None:
            logger.error("Cannot initialize Whisper model")
            return

        try:
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            self.model = fw.WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as float32 array (16kHz)

        Returns:
            Transcribed text
        """
        if self.model is None:
            return ""

        try:
            start_time = time.time()

            # Transcribe with beam_size=1 for speed
            segments, info = self.model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                language=self.config.language,
                vad_filter=self.config.vad_filter
            )

            # Combine all segments
            text = " ".join(segment.text.strip() for segment in segments)

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Transcription ({elapsed:.0f}ms): {text}")

            return text.strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _worker(self):
        """Worker thread for transcription."""
        logger.info("Transcriber worker started")

        while self._running:
            try:
                # Wait for audio with timeout
                audio = self.transcription_queue.get(timeout=0.1)

                # Transcribe
                text = self._transcribe(audio)

                if text:
                    # Notify callback
                    if self.on_transcription:
                        self.on_transcription(text)

                    # Send to LLM queue
                    try:
                        self.llm_queue.put_nowait(text)
                    except queue.Full:
                        logger.warning("LLM queue full")

                self.transcription_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcriber worker error: {e}")

        logger.info("Transcriber worker stopped")

    def start(self):
        """Start transcription worker."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop transcription worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None


class Oracle:
    """
    Component C: The LLM Oracle for Spanish completion.

    Uses Llama-3-8B with a raw completion prompt to predict
    the Spanish completion of partial English sentences.
    """

    def __init__(
        self,
        config: LLMConfig,
        llm_queue: queue.Queue,
        on_completion: Optional[Callable[[str, str], None]] = None
    ):
        self.config = config
        self.llm_queue = llm_queue
        self.on_completion = on_completion

        self.model = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_model()

    def _init_model(self):
        """Initialize llama-cpp-python model."""
        Llama = _import_llama_cpp()
        if Llama is None:
            logger.error("Cannot initialize LLM model")
            return

        if not self.config.model_path:
            logger.warning("No LLM model path specified")
            return

        try:
            logger.info(f"Loading LLM model: {self.config.model_path}")
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                verbose=False
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None

    def _complete(self, user_text: str) -> str:
        """
        Generate Spanish completion for user text.

        Args:
            user_text: Partial English text from transcription

        Returns:
            Spanish completion
        """
        if self.model is None:
            # Fallback: return placeholder
            return "[LLM not loaded]"

        try:
            start_time = time.time()

            # Build prompt using shadow prompt template
            prompt = self.config.prompt_template.format(user_input=user_text)

            # Generate completion (raw completion, not chat)
            output = self.model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=["\n", "Input:", "Output:"],  # Stop at newline or next example
                echo=False
            )

            # Extract generated text
            completion = output["choices"][0]["text"].strip()

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"LLM completion ({elapsed:.0f}ms): {completion}")

            return completion
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            return ""

    def _worker(self):
        """Worker thread for LLM completion."""
        logger.info("Oracle worker started")

        while self._running:
            try:
                # Wait for text with timeout
                user_text = self.llm_queue.get(timeout=0.1)

                # Generate completion
                completion = self._complete(user_text)

                # Notify callback with both original and completion
                if self.on_completion and completion:
                    self.on_completion(user_text, completion)

                self.llm_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Oracle worker error: {e}")

        logger.info("Oracle worker stopped")

    def start(self):
        """Start LLM worker."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop LLM worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None


class InferenceEngine:
    """
    Combined inference engine managing both Transcriber and Oracle.

    Provides a unified interface for the main application.
    """

    def __init__(
        self,
        whisper_config: WhisperConfig,
        llm_config: LLMConfig,
        transcription_queue: queue.Queue,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_completion: Optional[Callable[[str, str], None]] = None
    ):
        # Create LLM queue (internal)
        self.llm_queue: queue.Queue = queue.Queue(maxsize=10)

        # Initialize components
        self.transcriber = Transcriber(
            config=whisper_config,
            transcription_queue=transcription_queue,
            llm_queue=self.llm_queue,
            on_transcription=on_transcription
        )

        self.oracle = Oracle(
            config=llm_config,
            llm_queue=self.llm_queue,
            on_completion=on_completion
        )

    def start(self):
        """Start both inference workers."""
        self.transcriber.start()
        self.oracle.start()

    def stop(self):
        """Stop both inference workers."""
        self.transcriber.stop()
        self.oracle.stop()

    def is_ready(self) -> Tuple[bool, bool]:
        """
        Check if models are loaded.

        Returns:
            Tuple of (whisper_ready, llm_ready)
        """
        return (
            self.transcriber.model is not None,
            self.oracle.model is not None
        )
