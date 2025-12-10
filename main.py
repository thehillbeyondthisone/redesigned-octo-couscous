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
from audio_handler import AudioHandler, SpeechState, get_audio_devices, get_default_device_index
from inference_engine import InferenceEngine
from overlay import create_overlay, setup_debug_logging, load_settings, PYQT_AVAILABLE

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
        self._debug_handler = None

        # State tracking
        self._current_user_text = ""
        self._current_ghost_text = ""
        self._running = False
        self._selected_device_index: Optional[int] = None

    def _on_state_change(self, state: SpeechState):
        """Handle audio state changes."""
        if self.overlay:
            status_map = {
                SpeechState.SILENCE: "● Listening...",
                SpeechState.SPEECH_DETECTED: "🎤 Speaking...",
                SpeechState.SPEECH_ENDING: "⏳ Processing..."
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

    def _on_device_change(self, device_index: int):
        """Handle audio device change from UI."""
        self._selected_device_index = device_index
        logger.info(f"Device changed to index: {device_index}")

        # If running, restart with new device
        if self._running and self.audio_handler:
            self.audio_handler.restart_with_device(device_index)

    def _on_settings_change(self, new_settings: dict):
        """Handle settings change from UI."""
        logger.info("Settings changed")

        # Update VAD threshold if changed
        if self.audio_handler and "vad_threshold" in new_settings:
            self.config.vad.threshold = new_settings["vad_threshold"]
            logger.info(f"VAD threshold updated to {new_settings['vad_threshold']}")

        # Update silence duration if changed
        if "silence_duration_ms" in new_settings:
            self.config.vad.min_silence_duration_ms = new_settings["silence_duration_ms"]
            logger.info(f"Silence duration updated to {new_settings['silence_duration_ms']}ms")

    def _on_ui_start(self):
        """Handle start button from UI."""
        logger.info("Start requested from UI")
        self.start()

    def _on_ui_stop(self):
        """Handle stop button from UI."""
        logger.info("Stop requested from UI")
        self.stop()

    def _load_audio_devices(self):
        """Load available audio devices and update UI."""
        devices = get_audio_devices()
        logger.info(f"Found {len(devices)} audio input devices")

        if self.overlay:
            self.overlay.update_devices(devices)

        # Set default device
        if devices:
            for dev in devices:
                if dev.get('is_default'):
                    self._selected_device_index = dev['index']
                    break
            if self._selected_device_index is None:
                self._selected_device_index = devices[0]['index']

    def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing Spanish Ghost Text Aid...")

        try:
            # Initialize overlay first (so we can log to it)
            logger.info("Initializing overlay...")
            self.overlay = create_overlay(self.config.ui, use_terminal=self.use_terminal)

            # Set up debug logging to overlay
            if self.overlay and hasattr(self.overlay, 'append_debug'):
                self._debug_handler = setup_debug_logging(self.overlay)

            # Set UI callbacks
            if self.overlay and hasattr(self.overlay, 'set_callbacks'):
                self.overlay.set_callbacks(
                    on_start=self._on_ui_start,
                    on_stop=self._on_ui_stop,
                    on_device_change=self._on_device_change,
                    on_settings_change=self._on_settings_change
                )

            # Load audio devices
            self._load_audio_devices()

            # Initialize audio handler (but don't start yet)
            logger.info("Initializing audio handler...")
            self.audio_handler = AudioHandler(
                audio_config=self.config.audio,
                vad_config=self.config.vad,
                transcription_queue=self.transcription_queue,
                on_state_change=self._on_state_change,
                device_index=self._selected_device_index
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

            logger.info("Initialization complete - click Start to begin")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        """Start all components."""
        if self._running:
            logger.warning("Already running")
            return

        self._running = True
        logger.info("Starting audio processing...")

        # Get selected device from UI
        if self.overlay and hasattr(self.overlay, 'get_selected_device_index'):
            device_idx = self.overlay.get_selected_device_index()
            if device_idx is not None:
                self._selected_device_index = device_idx

        # Start inference engine (threads 2 & 3)
        if self.inference_engine:
            self.inference_engine.start()

        # Start audio handler (thread 1) with selected device
        if self.audio_handler:
            self.audio_handler.start(self._selected_device_index)

        if self.overlay:
            self.overlay.set_status("● Listening...")
            self.overlay.set_running_state(True)

        logger.info("Application started")

    def stop(self):
        """Stop all components."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping audio processing...")

        # Stop audio handler
        if self.audio_handler:
            self.audio_handler.stop()

        # Stop inference engine
        if self.inference_engine:
            self.inference_engine.stop()

        if self.overlay:
            self.overlay.set_status("⏸ Stopped")
            self.overlay.set_running_state(False)

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

        # Show GUI overlay (don't auto-start)
        if self.overlay and hasattr(self.overlay, 'show'):
            self.overlay.show()

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

        # Remove debug handler
        if self._debug_handler:
            logging.getLogger().removeHandler(self._debug_handler)

        return ret

    def run_terminal(self):
        """Run in terminal mode (no GUI)."""
        # Initialize
        if not self.initialize():
            return 1

        # Auto-start in terminal mode
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
            import time
            while self._running:
                time.sleep(0.1)

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
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load saved settings
    saved_settings = load_settings()

    # Get configuration based on settings or args
    if args.low_vram or saved_settings.get("low_vram_mode", False):
        config = get_low_vram_config()
    else:
        config = get_default_config()

    # Apply saved settings
    if saved_settings.get("llm_model_path"):
        config.llm.model_path = saved_settings["llm_model_path"]
    if saved_settings.get("whisper_model"):
        config.whisper.model_size = saved_settings["whisper_model"]
    if saved_settings.get("vad_threshold"):
        config.vad.threshold = saved_settings["vad_threshold"]
    if saved_settings.get("silence_duration_ms"):
        config.vad.min_silence_duration_ms = saved_settings["silence_duration_ms"]

    # Command line overrides saved settings
    if args.model:
        config.llm.model_path = args.model

    if args.whisper_model != "large-v3-turbo":  # Only override if explicitly set
        config.whisper.model_size = args.whisper_model

    # Create application
    app = SpanishGhostTextApp(config, use_terminal=args.terminal)

    # Set device from command line if provided
    if args.device is not None:
        app._selected_device_index = args.device

    if args.terminal or not PYQT_AVAILABLE:
        return app.run_terminal()
    else:
        return app.run_gui()


if __name__ == "__main__":
    sys.exit(main())
