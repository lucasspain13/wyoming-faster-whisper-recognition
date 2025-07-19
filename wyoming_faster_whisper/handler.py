"""Original event handler with minimal speaker ID additions."""
import argparse
import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import nemo.collections.asr as nemo_asr
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
import torch
import librosa
import soundfile as sf
import numpy as np

from .speaker_identifier import load_embeddings, identify_speaker

_LOGGER = logging.getLogger(__name__)

class ParakeetEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: nemo_asr.models.ASRModel,
        model_lock: asyncio.Lock,
        *args,
        initial_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.initial_prompt = initial_prompt
        self._language = self.cli_args.language

        # Optimized VoiceEncoder device selection
        from resemblyzer import VoiceEncoder
        if torch.cuda.is_available():
            encoder_device = "cuda"
        elif torch.backends.mps.is_available():
            encoder_device = "mps"
        else:
            encoder_device = "cpu"
        self.voice_encoder = VoiceEncoder(device=encoder_device)
        self.speaker_embeddings = None
        if getattr(cli_args, "embeddings_file", None):
            self.speaker_embeddings = load_embeddings(Path(cli_args.embeddings_file))

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file = None

    async def handle_event(self, event: Event) -> bool:
        # Respond to Describe event with Info event
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)
            self._wav_file.writeframes(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            self._wav_file.close()
            self._wav_file = None

            # Start both tasks in parallel
            transcription_task = asyncio.create_task(self._transcribe_audio())
            speaker_task = asyncio.create_task(self._identify_speaker_optimized())
            text, speaker = await asyncio.gather(transcription_task, speaker_task)

            payload: str = str({"text": text, "speaker": speaker if speaker else "guest"})
            await self.write_event(Transcript(text=payload).event())
            return False

        return True

    async def _transcribe_audio(self) -> str:
        async with self.model_lock:
            # Convert audio to proper format for NeMo (mono, 16kHz)
            # Load and convert the audio
            audio, sr = librosa.load(self._wav_path, sr=16000, mono=True)
            
            # Create a temporary file with proper format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio, 16000)
            
            try:
                # NeMo transcribe method returns a list of transcription results
                transcriptions = self.model.transcribe([temp_path])
                if transcriptions and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, 'text'):
                        text = result.text
                    else:
                        text = str(result)
                else:
                    text = ""
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        _LOGGER.info(text)
        return text

    async def _identify_speaker_optimized(self) -> Optional[str]:
        if not self.speaker_embeddings:
            return None
        try:
            from resemblyzer import preprocess_wav
            # Load and preprocess audio in memory
            wav = preprocess_wav(self._wav_path)
            # Trim silence using librosa
            trimmed_wav, _ = librosa.effects.trim(wav, top_db=20)
            # If trimming removed everything, fall back to original
            if trimmed_wav.size == 0:
                trimmed_wav = wav
            # Compute embedding (in-memory)
            speaker = identify_speaker(
                trimmed_wav,
                self.speaker_embeddings,
                self.voice_encoder
            )
            _LOGGER.debug("Identified speaker: %s", speaker)
            return speaker
        except Exception as e:
            _LOGGER.error("Speaker identification failed: %s", e)
            return None
