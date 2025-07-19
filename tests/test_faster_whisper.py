"""Tests for wyoming-nemo-parakeet"""

import asyncio
import re
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024

# Need to give time for the model to download
_START_TIMEOUT = 120  # Increased for NeMo model loading
_TRANSCRIBE_TIMEOUT = 60


@pytest.mark.asyncio
async def test_nemo_parakeet() -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_faster_whisper",
        "--uri",
        "stdio://",
        "--model",
        "nvidia/parakeet-tdt-0.6b-v2",
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        "en",
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Check info
    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_START_TIMEOUT
        )
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert len(info.asr) == 1, "Expected one asr service"
        asr = info.asr[0]
        assert len(asr.models) > 0, "Expected at least one model"
        assert any(
            m.name == "nvidia/parakeet-tdt-0.6b-v2" for m in asr.models
        ), "Expected nvidia/parakeet-tdt-0.6b-v2 model"
        break

    # We want to use the NeMo model
    await async_write_event(Transcribe(name="nvidia/parakeet-tdt-0.6b-v2").event(), proc.stdin)

    # Test known WAV
    with wave.open(str(_DIR / "turn_on_the_living_room_lamp.wav"), "rb") as example_wav:
        await async_write_event(
            AudioStart(
                rate=example_wav.getframerate(),
                width=example_wav.getsampwidth(),
                channels=example_wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
        )
        assert event is not None

        if not Transcript.is_type(event.type):
            continue

        transcript = Transcript.from_event(event)
        text = transcript.text.lower().strip()
        text = re.sub(r"[^a-z ]", "", text)
        assert text == "turn on the living room lamp"
        break

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()
