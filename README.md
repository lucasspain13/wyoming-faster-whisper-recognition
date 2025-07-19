# Wyoming NeMo Parakeet with Speaker Identification

### Forked From

[wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper)

Whisper transcription and Wyoming communication was provided by the fork. This repository replaces the Whisper transcription engine with NVIDIA's NeMo Parakeet TDT 0.6B v2 model for improved accuracy and performance. It also adds a script and instructions for training an embedding file on 1 or more user voices. These users will be passed to the conversation agent with the transcribed text for personalized threads by user.

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for [NeMo Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) with speaker identification.

See more at [my site](https://lucas-spain.com)

## Features

- Fast, accurate speech-to-text transcription using NVIDIA's NeMo Parakeet TDT 0.6B v2
- Speaker identification using voice embeddings
- Support for punctuation and capitalization
- Timestamp prediction capabilities
- Docker support with compose examples
- Wyoming protocol compatibility

## Preparing Voice Clips

1. Collect 10-30 seconds of clean audio per speaker (minimum 5 seconds)
2. Save as WAV files (16kHz, mono) named after each speaker:
   ```
   /reference_audio/
     ├── alice.wav
     ├── bob.wav
     └── charlie.wav
   ```
3. Audio should be:
   - Clear speech with minimal background noise
   - Consistent microphone/recording conditions
   - Representative of normal speaking voice

## Speaker Identification Setup

1. Enroll speakers from reference audio:

```sh
python script/enroll_speakers.py \
    --reference-dir /path/to/reference_audio \
    --output /data/user_embeddings.pkl
```

2. The system will:
   - Process each WAV file (filename becomes speaker name)
   - Create voice embeddings using resemblyzer
   - Save embeddings to the specified .pkl file

## Labeling Process

During transcription:

1. Audio is compared against enrolled embeddings
2. Cosine similarity scores are calculated (0-1)
3. The speaker with highest score above threshold (default: 0.7) is selected
4. Transcripts include:
   ```json
   {
     "speaker": "alice", // from WAV filename
     "text": "Hello world",
     "similarity": 0.82 // confidence score
   }
   ```

## Docker Deployment

1. Build the image:

```sh
docker build -t yourrepo/wyoming-faster-whisper:latest .
```

2. Push to registry:

```sh
docker push yourrepo/wyoming-faster-whisper:latest
```

3. Example docker-compose.yml:

```yaml
version: "3"
services:
  whisper:
    image: yourrepo/wyoming-faster-whisper:latest
    ports:
      - "10300:10300"
    volumes:
      - ./data:/data
      - ./reference_audio:/reference_audio
    command:
      [
        "--model",
        "nvidia/parakeet-tdt-0.6b-v2",
        "--uri",
        "tcp://0.0.0.0:10300",
        "--device",
        "auto",
        "--data-dir",
        "/data",
        "--embeddings-file",
        "/data/user_embeddings.pkl",
      ]
```

## Running Locally

```sh
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (auto-detects best device)
python -m wyoming_faster_whisper \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --uri tcp://0.0.0.0:10300 \
    --device auto \
    --data-dir ./data \
    --embeddings-file ./user_embeddings.pkl

# For first run (with model downloads):
python -m wyoming_faster_whisper \
    --model nvidia/parakeet-tdt-0.6b-v2 \
    --uri tcp://0.0.0.0:10300 \
    --device auto \
    --data-dir ./data \
    --download-dir ./models \
    --embeddings-file ./user_embeddings.pkl
```

## Notes

- Minimum 3 speakers recommended for reliable identification
- Similarity scores below 0.5 indicate poor matches
- For production, use at least 30s of reference audio per speaker
- Models cache to `~/.cache/huggingface/hub` (set `HF_HUB_CACHE` to override)
