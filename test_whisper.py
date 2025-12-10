"""
Debug script to test Whisper transcription in isolation.
Run this directly to diagnose crashes.
"""
import sys
import os
from pathlib import Path

# Add DLL directory for cuDNN
script_dir = Path(__file__).parent.absolute()
if sys.platform == "win32":
    os.add_dll_directory(str(script_dir))
    print(f"[DLL] Added {script_dir} to DLL search path")

print("\n=== Whisper Debug Test ===\n")

# Test 1: Import check
print("[1] Testing imports...")
try:
    import numpy as np
    print("    numpy: OK")
except Exception as e:
    print(f"    numpy: FAILED - {e}")
    sys.exit(1)

try:
    import faster_whisper
    print(f"    faster_whisper: OK (v{faster_whisper.__version__})")
except Exception as e:
    print(f"    faster_whisper: FAILED - {e}")
    sys.exit(1)

# Test 2: Model loading (CPU)
print("\n[2] Testing model load (CPU mode)...")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("    Model loaded: OK")
except Exception as e:
    print(f"    Model load: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Transcription with dummy audio
print("\n[3] Testing transcription...")
try:
    # Create 1 second of silence (16kHz mono)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(dummy_audio, beam_size=1, language="en")
    # Consume the generator
    text = " ".join(seg.text for seg in segments)
    print(f"    Transcription: OK (result: '{text.strip() or '[silence]'}')")
except Exception as e:
    print(f"    Transcription: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: GPU mode (optional)
print("\n[4] Testing model load (CUDA mode)...")
try:
    model_cuda = WhisperModel("tiny", device="cuda", compute_type="float16")
    print("    CUDA Model loaded: OK")
    
    # Test transcription on CUDA
    segments, info = model_cuda.transcribe(dummy_audio, beam_size=1, language="en")
    text = " ".join(seg.text for seg in segments)
    print(f"    CUDA Transcription: OK")
except Exception as e:
    print(f"    CUDA test: FAILED - {e}")
    print("    (This is expected if CUDA/cuDNN isn't properly configured)")

print("\n=== All basic tests passed! ===\n")
print("If the app still crashes, the issue is likely in:")
print("  - Audio handling (audio_handler.py)")
print("  - Threading/queue issues")
print("  - Memory issues with larger models")
