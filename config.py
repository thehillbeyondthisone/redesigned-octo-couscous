"""
Configuration module for Real-Time Spanish Autocomplete "Ghost Text" Aid.

VRAM Budget for RTX 4070 (12GB):
- Whisper large-v3-turbo: ~1.5GB (float16)
- Llama-3-8B Q4_K_M: ~6.5GB
- OS Overhead: ~2GB
- Total: ~10GB / 12GB (Safe margin)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio recording configuration."""
    sample_rate: int = 16000  # Whisper expects 16kHz
    chunk_duration_ms: int = 512  # Audio chunk size in milliseconds
    channels: int = 1  # Mono audio
    silence_threshold_ms: int = 300  # Pause duration to trigger utterance complete
    max_buffer_duration_s: float = 30.0  # Maximum audio buffer duration


@dataclass
class WhisperConfig:
    """Faster-Whisper transcription configuration."""
    model_size: str = "large-v3-turbo"  # Or "medium" if VRAM is tight
    device: str = "cuda"
    compute_type: str = "float16"  # ~1.5GB VRAM
    beam_size: int = 1  # Critical for speed over accuracy
    language: str = "en"  # Source language
    vad_filter: bool = False  # We handle VAD separately


@dataclass
class LLMConfig:
    """Llama-cpp-python LLM configuration."""
    model_path: str = ""  # Path to GGUF model file
    n_ctx: int = 512  # Context window (small for single sentences)
    n_gpu_layers: int = -1  # Offload all layers to GPU
    n_batch: int = 512  # Batch size for prompt processing
    max_tokens: int = 10  # Only predict immediate next words
    temperature: float = 0.7
    top_p: float = 0.9

    # Prompt template for Spanish autocomplete
    prompt_template: str = '''You are a real-time Spanish tutor. Your job is to complete the user's sentence in Spanish.
Examples:
Input: "I want to go to the..."
Output: "playa hoy."

Input: "Where is the..."
Output: "baño más cercano?"

Input: "{user_input}"
Output:
'''


@dataclass
class VADConfig:
    """Silero Voice Activity Detection configuration."""
    threshold: float = 0.5  # VAD confidence threshold
    min_speech_duration_ms: int = 250  # Minimum speech duration
    min_silence_duration_ms: int = 300  # Silence to mark end of utterance


@dataclass
class UIConfig:
    """PyQt6 Overlay UI configuration."""
    window_width: int = 800
    window_height: int = 150
    font_family: str = "Arial"
    font_size_user: int = 18
    font_size_ghost: int = 16
    color_user: str = "#FFFFFF"  # White
    color_ghost: str = "#888888"  # Grey/Faded
    background_opacity: float = 0.85
    background_color: str = "#1a1a2e"


@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioConfig = None
    whisper: WhisperConfig = None
    llm: LLMConfig = None
    vad: VADConfig = None
    ui: UIConfig = None

    # Target latency
    target_latency_ms: int = 500

    def __post_init__(self):
        self.audio = self.audio or AudioConfig()
        self.whisper = self.whisper or WhisperConfig()
        self.llm = self.llm or LLMConfig()
        self.vad = self.vad or VADConfig()
        self.ui = self.ui or UIConfig()


def get_default_config() -> AppConfig:
    """Get default configuration optimized for RTX 4070."""
    return AppConfig()


def get_low_vram_config() -> AppConfig:
    """Get configuration for lower VRAM GPUs."""
    config = AppConfig()
    config.whisper.model_size = "medium"  # Smaller model
    config.whisper.compute_type = "int8"  # Lower precision
    config.llm.n_ctx = 256  # Smaller context
    return config
