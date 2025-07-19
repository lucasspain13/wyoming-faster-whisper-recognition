#!/usr/bin/env python3
"""Main entry point for Wyoming NeMo Parakeet server."""
import argparse
import asyncio
import logging
import sys
from functools import partial

import nemo.collections.asr as nemo_asr
from wyoming.client import AsyncClient
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import ParakeetEventHandler

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, default="nvidia/parakeet-tdt-0.6b-v2", help="Name of NeMo ASR model (default: nvidia/parakeet-tdt-0.6b-v2)")
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument("--data-dir", required=True, action="append", help="Data directory")
    parser.add_argument("--download-dir", help="Directory to download models")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cuda/cpu)")
    parser.add_argument("--language", help="Default language")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size (not used in NeMo greedy decoding)")
    parser.add_argument("--initial-prompt", help="Initial prompt text (not supported in NeMo)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--embeddings-file", help="Speaker embeddings file")

    args = parser.parse_args()

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    # Auto-detect device if not specified
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
            _LOGGER.info("Auto-detected CUDA device")
        else:
            args.device = "cpu"
            _LOGGER.info("Auto-detected CPU device (CUDA not available)")
    else:
        _LOGGER.info("Using specified device: %s", args.device)

    # Load NeMo model
    try:
        _LOGGER.info("Loading NeMo ASR model: %s", args.model)
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
        
        # Move model to appropriate device
        if args.device == "cuda":
            import torch
            if torch.cuda.is_available():
                asr_model = asr_model.cuda()
                _LOGGER.info("Model moved to CUDA")
            else:
                _LOGGER.warning("CUDA requested but not available, using CPU")
                args.device = "cpu"
        
        _LOGGER.info("Loaded model: %s on device: %s", args.model, args.device)
    except Exception as e:
        _LOGGER.error("Failed to load model: %s", e)
        sys.exit(1)

    # Create Wyoming info
    wyoming_info = Info(
        asr=[AsrProgram(
            name="nemo-parakeet",
            description="NeMo Parakeet TDT with speaker ID",
            attribution=Attribution(
                name="NVIDIA",
                url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
            ),
            installed=True,
            version=__version__,
            models=[AsrModel(
                name=args.model,
                description=args.model,
                attribution=Attribution(
                    name="NVIDIA",
                    url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
                ),
                installed=True,
                languages=["en"] if args.language else ["en"],  # Parakeet is English-only
                version="0.6b-v2",
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
            ParakeetEventHandler,
            wyoming_info,
            args,
            asr_model,
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
