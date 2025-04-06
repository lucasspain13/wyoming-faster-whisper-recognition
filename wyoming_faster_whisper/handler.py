"""Original event handler with minimal speaker ID additions."""
import argparse
import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import faster_whisper
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .speaker_identifier import load_embeddings, identify_speaker

_LOGGER = logging.getLogger(__name__)

class FasterWhisperEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: faster_whisper.WhisperModel,
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

        # Minimal speaker ID addition
        from resemblyzer import VoiceEncoder
        self.voice_encoder = VoiceEncoder()
        self.speaker_embeddings = None
        if getattr(cli_args, "embeddings_file", None):
            self.speaker_embeddings = load_embeddings(Path(cli_args.embeddings_file))

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file = None

    async def handle_event(self, event: Event) -> bool:
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

            async with self.model_lock:
                segments, _info = self.model.transcribe(
                    self._wav_path,
                    beam_size=self.cli_args.beam_size,
                    language=self._language,
                    initial_prompt=self.initial_prompt,
                )

            text = " ".join(segment.text for segment in segments)
            _LOGGER.info(text)

            # Minimal speaker ID addition
            speaker = None
            if self.speaker_embeddings:
                from resemblyzer import preprocess_wav
                try:
                    wav = preprocess_wav(self._wav_path)
                    speaker = identify_speaker(
                        wav,  # Pass preprocessed audio
                        self.speaker_embeddings,
                        self.voice_encoder
                    )
                except Exception as e:
                    _LOGGER.error("Speaker identification failed: %s", e)
                    speaker = None
                _LOGGER.debug("Identified speaker: %s", speaker)

            payload: str = str({"text": text, "speaker": speaker if speaker else "guest"})
            await self.write_event(Transcript(text=payload).event())
            return False

        return True
