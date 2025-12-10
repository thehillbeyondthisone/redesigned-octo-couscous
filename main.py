#!/usr/bin/env python3
"""
Real-Time Spanish Autocomplete "Ghost Text" Aid

Main application entry point implementing multi-threaded producer-consumer architecture:
- Thread 1 (AudioHandler): Microphone capture with VAD
- Thread 2 (Transcriber): faster-whisper English transcription
- Thread 3 (Oracle): Llama-3-8B Spanish completion
- Main Thread: PyQt6 overlay UI

Target: NVIDIA RTX 4070 (12GB VRAM)
Target Latency: <500ms
"""

import sys
import queue
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional

from config import AppConfig, get_default_config, get_low_vram_config
from audio_handler import AudioHandler, SpeechState
from inference_engine import InferenceEngine
from overlay import create_overlay, PYQT_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpanishGhostTextApp:
    """
    Main application class for Real-Time Spanish Autocomplete.

    Coordinates all components:
    - AudioHandler (Thread 1)
    - Transcriber (Thread 2)
    - Oracle (Thread 3)
    - Overlay UI (Main Thread)
    """

    def __init__(self, config: AppConfig, use_terminal: bool = False):
        self.config = config
        self.use_terminal = use_terminal

        # Create transcription queue (producer-consumer between audio and transcriber)
        self.transcription_queue: queue.Queue = queue.Queue(maxsize=10)

        # Initialize components
        self.audio_handler: Optional[AudioHandler] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.overlay = None

        # State tracking
        self._current_user_text = ""
        self._current_ghost_text = ""
        self._running = False

    def _on_state_change(self, state: SpeechState):
        """Handle audio state changes."""
        if self.overlay:
            status_map = {
                SpeechState.SILENCE: "Listening...",
                SpeechState.SPEECH_DETECTED: "🎤 Speaking...",
                SpeechState.SPEECH_ENDING: "Processing..."
            }
            self.overlay.set_status(status_map.get(state, "Unknown"))

    def _on_transcription(self, text: str):
        """Handle transcription result."""
        self._current_user_text = text
        if self.overlay:
            self.overlay.set_user_text(text)
        logger.info(f"Transcription: {text}")

    def _on_completion(self, user_text: str, completion: str):
        """Handle LLM completion result."""
        self._current_ghost_text = completion
        if self.overlay:
            # Only update if this matches current user text
            if user_text == self._current_user_text:
                self.overlay.set_ghost_text(completion)
        logger.info(f"Completion: {completion}")

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing Spanish Ghost Text Aid...")

        try:
            # Initialize audio handler
            logger.info("Initializing audio handler...")
            self.audio_handler = AudioHandler(
                audio_config=self.config.audio,
                vad_config=self.config.vad,
                transcription_queue=self.transcription_queue,
                on_state_change=self._on_state_change
            )

            # Initialize inference engine
            logger.info("Initializing inference engine...")
            self.inference_engine = InferenceEngine(
                whisper_config=self.config.whisper,
                llm_config=self.config.llm,
                transcription_queue=self.transcription_queue,
                on_transcription=self._on_transcription,
                on_completion=self._on_completion
            )

            # Check model status
            whisper_ready, llm_ready = self.inference_engine.is_ready()
            if not whisper_ready:
                logger.warning("Whisper model not loaded - transcription disabled")
            if not llm_ready:
                logger.warning("LLM model not loaded - completion disabled")

            # Initialize overlay
            logger.info("Initializing overlay...")
            self.overlay = create_overlay(self.config.ui, use_terminal=self.use_terminal)

            logger.info("Initialization complete")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def start(self):
        """Start all components."""
        if self._running:
            return

        self._running = True

        # Start inference engine (threads 2 & 3)
        if self.inference_engine:
            self.inference_engine.start()

        # Start audio handler (thread 1)
        if self.audio_handler:
            self.audio_handler.start()

        logger.info("Application started")

    def stop(self):
        """Stop all components."""
        self._running = False

        # Stop audio handler
        if self.audio_handler:
            self.audio_handler.stop()

        # Stop inference engine
        if self.inference_engine:
            self.inference_engine.stop()

        logger.info("Application stopped")

    def run_gui(self):
        """Run with PyQt6 GUI."""
        if not PYQT_AVAILABLE:
            logger.error("PyQt6 not available")
            return self.run_terminal()

        from PyQt6.QtWidgets import QApplication

        app = QApplication(sys.argv)

        # Initialize
        if not self.initialize():
            return 1

        # Create GUI overlay
        if self.overlay and hasattr(self.overlay, 'show'):
            self.overlay.show()

        # Start processing
        self.start()

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            logger.info("Shutting down...")
            self.stop()
            app.quit()

        signal.signal(signal.SIGINT, signal_handler)

        # Run Qt event loop
        ret = app.exec()

        # Cleanup
        self.stop()
        return ret

    def run_terminal(self):
        """Run in terminal mode (no GUI)."""
        # Initialize
        if not self.initialize():
            return 1

        # Start processing
        self.start()

        # Handle Ctrl+C
        def signal_handler(sig, frame):
            logger.info("\nShutting down...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)

        print("\n=== Spanish Ghost Text Aid (Terminal Mode) ===")
        print("Speak in English. Press Ctrl+C to exit.\n")

        try:
            # Simple terminal loop
            while self._running:
                import time
                time.sleep(0.1)

                # Update terminal display
                if self.overlay:
                    self.overlay.set_user_text(self._current_user_text)
                    self.overlay.set_ghost_text(self._current_ghost_text)

        except KeyboardInterrupt:
            pass

        # Cleanup
        self.stop()
        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Spanish Autocomplete Ghost Text Aid"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to GGUF LLM model file"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3-turbo",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model size (default: large-v3-turbo)"
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Use low VRAM configuration"
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Run in terminal mode (no GUI)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get configuration
    if args.low_vram:
        config = get_low_vram_config()
    else:
        config = get_default_config()

    # Apply command line overrides
    if args.model:
        config.llm.model_path = args.model

    if args.whisper_model:
        config.whisper.model_size = args.whisper_model

    # Create and run application
    app = SpanishGhostTextApp(config, use_terminal=args.terminal)

    if args.terminal or not PYQT_AVAILABLE:
        return app.run_terminal()
    else:
        return app.run_gui()


if __name__ == "__main__":
    sys.exit(main())
