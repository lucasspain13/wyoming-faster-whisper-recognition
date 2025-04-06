# Audio Preparation Guide

## Supported Formats

### Recommended Format

- **WAV** (16-bit PCM, 16kHz, mono)
  - No external dependencies needed
  - Best performance

### Optional Formats (requires extra dependencies)

- MP3 (converted internally to WAV)
- OGG (converted internally to WAV)
- FLAC (converted internally to WAV)

## Requirements for Optional Formats

1. Install ffmpeg:

```sh
# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

2. Install Python package:

```sh
pip install pydub
```

## File Naming Convention

```
speaker_name.format_extension
```

Example:

```
alice.wav
bob.mp3
carol.flac
```

## Output File Handling

### Recommended (local directory):

```sh
--output ./user_embeddings.pkl
```

### System-wide (requires permissions):

```sh
sudo mkdir -p /data
--output /data/user_embeddings.pkl
```

## Preparation Steps

### For WAV files:

1. Ensure 16kHz mono format
2. Use clear speech with minimal background noise
3. Keep duration between 10-30 seconds

### For other formats:

1. Place files in reference directory
2. The system will automatically:
   - Convert to WAV
   - Maintain original speaker names
   - Store converted temporary files

## Verification Command

Check supported formats:

```sh
python script/enroll_speakers.py --check-formats
```

## Troubleshooting

- "No such file or directory":
  - Use `./` prefix for local paths
  - Or create parent directory first
- "Codec not supported":
  - Re-encode files using: `ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 output.wav`
  - Or use recommended WAV format

## Best Practices

- Test with a few files first
- Verify audio quality before processing
- Keep backups of original files
