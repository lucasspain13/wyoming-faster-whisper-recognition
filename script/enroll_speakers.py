#!/usr/bin/env python3
"""Enroll speaker voices using resemblyzer VoiceEncoder."""
import argparse
import logging
import pickle
from pathlib import Path
import warnings
import sys

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_LOGGER = logging.getLogger("enroll_speakers")

def _check_mp3_support():
    """Check if MP3 support is available."""
    try:
        from pydub import AudioSegment
        return AudioSegment
    except ImportError:
        return None

def main():
    AudioSegment = _check_mp3_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory containing reference audio files (speaker_name.wav or speaker_name.mp3)"
    )
    parser.add_argument(
        "--output",
        default="user_embeddings.pkl",
        help="Output path for embeddings file (.pkl)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.debug:
        _LOGGER.setLevel(logging.DEBUG)
        _LOGGER.debug("Debug logging enabled")

    _LOGGER.info("Starting speaker enrollment")
    _LOGGER.info("Reference directory: %s", args.reference_dir)
    _LOGGER.info("Output file: %s", args.output)

    # Initialize voice encoder
    _LOGGER.debug("Initializing VoiceEncoder")
    encoder = VoiceEncoder()
    embeddings = {}

    # Process each reference file
    ref_dir = Path(args.reference_dir)
    audio_files = list(ref_dir.glob("*.wav")) + (list(ref_dir.glob("*.mp3")) if AudioSegment else [])
    _LOGGER.info("Found %d audio files in directory", len(audio_files))
    
    for audio_path in audio_files:
        speaker = audio_path.stem
        _LOGGER.debug("Processing speaker: %s (%s)", speaker, audio_path)
        
        try:
            # Handle MP3 files if supported
            if audio_path.suffix.lower() == ".mp3":
                if not AudioSegment:
                    _LOGGER.error("MP3 support requires pydub and ffmpeg. Install with: pip install pydub")
                    _LOGGER.error("Alternatively, convert MP3 to WAV manually")
                    continue
                    
                _LOGGER.debug("Converting MP3 to WAV")
                sound = AudioSegment.from_mp3(audio_path)
                wav_path = audio_path.with_suffix(".wav")
                sound.export(wav_path, format="wav")
                audio_path = wav_path
                
            # Load and preprocess audio
            _LOGGER.debug("Preprocessing audio")
            wav = preprocess_wav(audio_path)
            
            # Compute embedding
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _LOGGER.debug("Computing embedding")
                embedding = encoder.embed_utterance(wav)
            
            embeddings[speaker] = embedding
            _LOGGER.info("Successfully enrolled speaker: %s", speaker)
            
        except Exception as e:
            _LOGGER.error("Failed to process %s: %s", wav_path, e, exc_info=args.debug)

    # Save embeddings
    _LOGGER.info("Saving embeddings for %d speakers", len(embeddings))
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
        _LOGGER.info("Successfully saved embeddings to %s", output_path)
        
    except Exception as e:
        _LOGGER.error("Failed to save embeddings: %s", e)
        _LOGGER.info("Try using a relative path like './user_embeddings.pkl'")
        sys.exit(1)

if __name__ == "__main__":
    main()