#!/usr/bin/env python3
"""Main entry point for Wyoming Faster Whisper server."""
import argparse
import asyncio
import logging
import sys
from functools import partial

import faster_whisper
from wyoming.client import AsyncClient
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import FasterWhisperEventHandler

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, default="tiny", help="Name of faster-whisper model (default: tiny for Apple Silicon)")
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument("--data-dir", required=True, action="append", help="Data directory")
    parser.add_argument("--download-dir", help="Directory to download models")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--language", help="Default language")
    parser.add_argument("--compute-type", default="int8", help="Compute type (default: int8 for Apple Silicon)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--initial-prompt", help="Initial prompt text")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--embeddings-file", help="Speaker embeddings file")

    args = parser.parse_args()

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    # Load model
    try:
        whisper_model = faster_whisper.WhisperModel(
            args.model,
            download_root=args.download_dir,
            device=args.device,
            compute_type=args.compute_type
        )
        _LOGGER.info("Loaded model: %s", args.model)
    except Exception as e:
        _LOGGER.error("Failed to load model: %s", e)
        sys.exit(1)

    # Create Wyoming info
    wyoming_info = Info(
        asr=[AsrProgram(
            name="faster-whisper",
            description="Faster Whisper with speaker ID",
            attribution=Attribution(
                name="Guillaume Klein",
                url="https://github.com/guillaumekln/faster-whisper/",
            ),
            installed=True,
            version=__version__,
            models=[AsrModel(
                name=args.model,
                description=args.model,
                attribution=Attribution(
                    name="Systran",
                    url="https://huggingface.co/Systran",
                ),
                installed=True,
                languages=None if args.language else faster_whisper.tokenizer._LANGUAGE_CODES,
                version=faster_whisper.__version__,
            )]
        )]
    )

    # Server setup
    try:
        server = AsyncServer.from_uri(args.uri)
    except Exception as e:
        _LOGGER.error("Failed to create server: %s", e)
        sys.exit(1)

    # Start server
    _LOGGER.info("Service ready on %s", args.uri)
    await server.run(
        partial(
            FasterWhisperEventHandler,
            wyoming_info,
            args,
            whisper_model,
            asyncio.Lock(),
            initial_prompt=args.initial_prompt,
        )
    )

def run() -> None:
    """Run the server."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run()
