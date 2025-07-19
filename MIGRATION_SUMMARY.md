# Migration from faster-whisper to NeMo Parakeet TDT 0.6B v2

## Summary

Successfully replaced faster-whisper with NVIDIA's NeMo Parakeet TDT 0.6B v2 model in the Wyoming ASR server while maintaining speaker identification capabilities.

## Key Changes Made

### 1. Dependencies Updated (`requirements.txt`)

- **Removed**: `faster-whisper==1.1.0`
- **Added**:
  - `nemo_toolkit[asr]` - Core NeMo ASR functionality
  - `torch` - Required for NeMo
  - `librosa` - Audio processing
  - `soundfile` - Audio file I/O

### 2. Core Module Changes (`wyoming_faster_whisper/__main__.py`)

- **Imports**: Replaced `faster_whisper` with `nemo.collections.asr as nemo_asr`
- **Model Loading**: Changed from `faster_whisper.WhisperModel()` to `nemo_asr.models.ASRModel.from_pretrained()`
- **Default Model**: Updated from `"tiny"` to `"nvidia/parakeet-tdt-0.6b-v2"`
- **Device Handling**: Default device changed from `"cpu"` to `"cuda"` for better performance
- **Wyoming Info**: Updated attribution and model information

### 3. Event Handler (`wyoming_faster_whisper/handler.py`)

- **Class Name**: `FasterWhisperEventHandler` → `ParakeetEventHandler`
- **Model Type**: Parameter type changed to `nemo_asr.models.ASRModel`
- **Transcription Method**: Completely rewritten to:
  - Convert audio to mono 16kHz format required by NeMo
  - Use temporary file approach for format conversion
  - Handle NeMo's transcription API differences
- **Audio Processing**: Added librosa and soundfile imports

### 4. Configuration & Metadata

- **Version**: Bumped from `2.4.0` to `3.0.0` (major version due to breaking changes)
- **Description**: Updated to reflect NeMo Parakeet usage
- **Keywords**: Added "nemo" and "parakeet", removed "whisper"
- **Docker**: Updated default model and added g++ build dependency

### 5. Documentation Updates

- **README.md**: Updated title, descriptions, and examples to reference NeMo Parakeet
- **CHANGELOG.md**: Added entry for v3.0.0 with breaking change notice

### 6. Test Updates

- **Test File**: Updated to expect `nvidia/parakeet-tdt-0.6b-v2` model instead of `tiny-int8`
- **Timeout**: Increased model loading timeout for larger NeMo model

## Technical Improvements

### Model Capabilities

- **Accuracy**: Significantly improved transcription accuracy
- **Punctuation**: Built-in punctuation and capitalization
- **Performance**: Better handling of various audio domains
- **Timestamps**: Native support for word-level timestamps

### Audio Processing

- **Format Handling**: Automatic conversion to required mono 16kHz format
- **Compatibility**: Works with various input audio formats via librosa
- **Quality**: Improved audio preprocessing pipeline

### Speaker Identification

- **Compatibility**: Maintained full speaker identification functionality
- **Performance**: Optimized device selection for voice encoder
- **Flexibility**: Enhanced to handle both file paths and audio arrays

## Migration Notes

### Breaking Changes

1. **Model Parameter**: Default model changed from small whisper models to larger NeMo model
2. **Device Preference**: Now defaults to CUDA for better performance
3. **Audio Format**: Internal handling changed to ensure mono 16kHz compatibility
4. **Dependencies**: Requires NeMo toolkit and associated dependencies

### Compatibility

- **Wyoming Protocol**: Fully compatible with existing Wyoming ASR protocol
- **Speaker ID**: All speaker identification features preserved
- **API**: Command-line interface remains the same

### Performance Considerations

- **Model Size**: NeMo model is significantly larger (~2.5GB)
- **Memory**: Requires more RAM (~2GB minimum)
- **Speed**: Better accuracy but may be slower on CPU-only systems
- **GPU**: Strongly recommended for optimal performance

## Verification

The migration has been tested and verified:

- ✅ NeMo model loads successfully
- ✅ Audio transcription works correctly
- ✅ Format conversion handles various audio inputs
- ✅ Speaker identification functionality preserved
- ✅ Wyoming protocol compatibility maintained

## Usage Examples

### Basic Usage

```bash
python -m wyoming_faster_whisper \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --uri tcp://0.0.0.0:10300 \
    --data-dir ./data \
    --embeddings-file ./user_embeddings.pkl
```

### Docker Deployment

```yaml
version: "3"
services:
  whisper:
    image: yourrepo/wyoming-faster-whisper:latest
    command:
      [
        "--model",
        "nvidia/parakeet-tdt-0.6b-v2",
        "--uri",
        "tcp://0.0.0.0:10300",
        "--data-dir",
        "/data",
        "--embeddings-file",
        "/data/user_embeddings.pkl",
      ]
```

The migration successfully modernizes the ASR capabilities while maintaining backward compatibility with the Wyoming protocol and speaker identification features.
