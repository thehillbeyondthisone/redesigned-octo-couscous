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
import os
import sys
from typing import Optional, Callable, Tuple
from pathlib import Path
import numpy as np

from config import WhisperConfig, LLMConfig

logger = logging.getLogger(__name__)

# Configure DLL search path for cuDNN on Windows
# This allows cuDNN DLLs placed next to main.py to be found
if sys.platform == "win32":
    script_dir = Path(__file__).parent.absolute()
    cudnn_dlls = list(script_dir.glob("cudnn*.dll"))
    if cudnn_dlls:
        # Add script directory to DLL search path
        os.add_dll_directory(str(script_dir))
        print(f"[DLL] Added {script_dir} to DLL search path ({len(cudnn_dlls)} cuDNN DLLs found)")
    # Also check for zlibwapi.dll
    if (script_dir / "zlibwapi.dll").exists():
        print(f"[DLL] Found zlibwapi.dll")

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
        """Initialize faster-whisper model with automatic CUDA fallback."""
        import time as _time
        import os
        fw = _import_faster_whisper()
        if fw is None:
            print("[Whisper] ERROR: faster-whisper not installed")
            logger.error("Cannot initialize Whisper model")
            return

        # Try configurations in order of preference
        configs_to_try = [
            (self.config.device, self.config.compute_type),
            ("cuda", "int8"),  # Try int8 if float16 fails
            ("cpu", "int8"),   # Fallback to CPU
        ]
        
        model_name = self.config.model_size
        print(f"[Whisper] Loading {model_name}...", end=" ", flush=True)
        
        for device, compute_type in configs_to_try:
            try:
                start = _time.time()
                self.model = fw.WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type
                )
                elapsed = _time.time() - start
                print(f"OK ({device}/{compute_type}, {elapsed:.1f}s)")
                logger.info(f"Whisper model loaded: {model_name} on {device}/{compute_type}")
                
                # Update config to reflect what actually worked
                self.config.device = device
                self.config.compute_type = compute_type
                return
                
            except Exception as e:
                error_msg = str(e).lower()
                if "cudnn" in error_msg or "cuda" in error_msg:
                    if device == "cuda":
                        print(f"\n[Whisper] CUDA failed ({e}), trying fallback...", end=" ", flush=True)
                        continue
                # If we're already on CPU or it's a different error, give up
                if device == "cpu":
                    print(f"FAILED: {e}")
                    logger.error(f"Failed to load Whisper model: {e}")
                    self.model = None
                    return
                continue
        
        print("FAILED: All configurations failed")
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
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            text = " ".join(text_parts)

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Transcription ({elapsed:.0f}ms): {text}")

            return text.strip()
        except Exception as e:
            error_msg = str(e).lower()
            # Check for CUDA/cuDNN errors at runtime
            if "cudnn" in error_msg or "cuda" in error_msg or "invalid handle" in error_msg:
                print(f"[Whisper] CUDA error, switching to CPU...", flush=True)
                logger.warning(f"CUDA error during transcription: {e}, falling back to CPU")
                self.config.device = "cpu"
                self.config.compute_type = "int8"
                self._init_model()
                if self.model is not None:
                    return self._transcribe(audio)
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
        import time as _time
        import os
        Llama = _import_llama_cpp()
        if Llama is None:
            print("[LLM] ERROR: llama-cpp-python not installed")
            logger.error("Cannot initialize LLM model")
            return

        if not self.config.model_path:
            print("[LLM] No model path specified - completion disabled")
            logger.warning("No LLM model path specified")
            return

        # Extract model info from filename
        model_file = os.path.basename(self.config.model_path)
        # Try to extract quant from filename (e.g., Q4_K_M)
        quant = "unknown"
        for q in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]:
            if q in model_file.upper():
                quant = q
                break
        
        print(f"[LLM] Loading {model_file} ({quant})...", end=" ", flush=True)
        
        try:
            start = _time.time()
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                verbose=False
            )
            elapsed = _time.time() - start
            
            gpu_info = f"GPU:{self.config.n_gpu_layers}" if self.config.n_gpu_layers != 0 else "CPU"
            print(f"OK ({gpu_info}, {elapsed:.1f}s)")
            logger.info(f"LLM model loaded: {model_file}")
        except Exception as e:
            print(f"FAILED: {e}")
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None

    def _complete(self, user_text: str) -> str:
        """
        Generate completion for user text based on output_mode.

        Args:
            user_text: Partial English text from transcription

        Returns:
            Completion based on mode (English/Spanish/Both)
        """
        if self.model is None:
            return "[LLM not loaded]"

        try:
            start_time = time.time()

            # Select prompt based on output mode
            mode = self.config.output_mode
            if mode == 0:  # English continuation
                prompt = self.config.prompt_english.format(user_input=user_text)
                stop_tokens = ["\n", "Sentence:", "Continuation:"]
            elif mode == 2:  # Both languages
                prompt = self.config.prompt_both.format(user_input=user_text)
                stop_tokens = ["\n", "English:", "Response:"]
            else:  # Default: Spanish translation (mode 1)
                prompt = self.config.prompt_spanish.format(user_input=user_text)
                stop_tokens = ["\n", "English:", "Spanish:"]

            # Generate completion (raw completion, not chat)
            output = self.model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=stop_tokens,
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
