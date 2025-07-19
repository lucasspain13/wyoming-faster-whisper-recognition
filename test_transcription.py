#!/usr/bin/env python3
"""Test transcription with the new NeMo model."""

import os
import tempfile
import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr

def test_nemo_transcription():
    """Test basic transcription functionality."""
    print("Loading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    
    # Use the existing test audio file
    test_audio = "tests/turn_on_the_living_room_lamp.wav"
    
    # Load and convert to mono 16kHz for NeMo
    print(f"Loading and converting {test_audio}...")
    audio, sr = librosa.load(test_audio, sr=16000, mono=True)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        sf.write(temp_path, audio, 16000)
    
    try:
        print(f"Transcribing {temp_path}...")
        transcriptions = model.transcribe([temp_path])
        
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
            if hasattr(result, 'text'):
                text = result.text
            else:
                text = str(result)
            print(f"Transcription: {text}")
            return text
        else:
            print("No transcription result")
            return ""
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

if __name__ == "__main__":
    test_nemo_transcription()
