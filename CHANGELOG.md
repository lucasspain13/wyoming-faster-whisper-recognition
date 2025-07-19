# Changelog

## 3.0.0

- **BREAKING CHANGE**: Replaced faster-whisper with NVIDIA NeMo Parakeet TDT 0.6B v2 model
- Updated dependencies to use `nemo_toolkit[asr]` instead of `faster-whisper`
- Improved accuracy and performance with state-of-the-art NeMo model
- Added support for punctuation and capitalization out of the box
- Updated default model to `nvidia/parakeet-tdt-0.6b-v2`
- Enhanced speaker identification to work with preprocessed audio arrays
- Updated Docker configuration for NeMo requirements

## 2.4.0

- Add "auto" for model and beam size (0) to select values based on CPU

## 2.3.0

- Bump faster-whisper package to 1.1.0
- Supports model `turbo` for faster processing

## 2.2.0

- Bump faster-whisper package to 1.0.3

## 2.1.0

- Added `--initial-prompt` (see https://github.com/openai/whisper/discussions/963)

## 2.0.0

- Use faster-whisper PyPI package
- `--model` can now be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

## 1.1.0

- Fix enum use for Python 3.11+
- Add tests and Github actions
- Bump tokenizers to 0.15
- Bump wyoming to 1.5.2

## 1.0.0

- Initial release
