
import sys
import traceback

try:
    print("Attempting to import faster_whisper...")
    import faster_whisper
    print("Success: faster_whisper imported.")
    print("Version:", faster_whisper.__version__)
except Exception:
    traceback.print_exc()

try:
    print("Attempting to import requests...")
    import requests
    print("Success: requests imported.")
except Exception:
    traceback.print_exc()
