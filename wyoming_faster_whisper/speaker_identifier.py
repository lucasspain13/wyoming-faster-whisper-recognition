"""Speaker identification using pre-computed voice embeddings."""
import logging
import pickle
from typing import Dict, Optional

import numpy as np
from resemblyzer import VoiceEncoder

_LOGGER = logging.getLogger(__name__)

def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """Load speaker embeddings from pickle file."""
    _LOGGER.debug("Loading speaker embeddings from %s", path)
    try:
        with open(path, "rb") as f:
            embeddings = pickle.load(f)
        _LOGGER.info("Loaded embeddings for %d speakers", len(embeddings))
        return embeddings
    except Exception as e:
        _LOGGER.error("Failed to load embeddings: %s", e)
        raise

def identify_speaker(
    audio_path: str,
    embeddings: Dict[str, np.ndarray],
    encoder: VoiceEncoder,
    threshold: float = 0.35,
) -> Optional[str]:
    """
    Identify speaker in audio file by comparing against known embeddings.
    
    Args:
        audio_path: Path to audio file to identify
        embeddings: Dictionary of {speaker_name: embedding}
        encoder: Initialized VoiceEncoder instance
        threshold: Minimum similarity score (0-1) to consider a match
        
    Returns:
        Name of best matching speaker or None if no match meets threshold
    """
    _LOGGER.debug("Identifying speaker for %s", audio_path)
    
    try:
        # Compute embedding for input audio
        _LOGGER.debug("Computing embedding for audio")
        embedding = encoder.embed_utterance(audio_path)
        
        # Find best matching speaker
        best_speaker = None
        best_score = -1
        
        _LOGGER.debug("Comparing against %d enrolled speakers", len(embeddings))
        for speaker, ref_embedding in embeddings.items():
            similarity = np.dot(embedding, ref_embedding)
            if similarity > best_score:
                best_score = similarity
                best_speaker = speaker
        
        _LOGGER.debug(
            "Best match: %s (score: %.2f, threshold: %.2f)",
            best_speaker, best_score, threshold
        )
        
        if best_score >= threshold:
            _LOGGER.info(
                "Identified speaker: %s (score: %.2f)",
                best_speaker, best_score
            )
            return best_speaker
            
        _LOGGER.warning(
            "No speaker matched (best score: %.2f < threshold: %.2f)",
            best_score, threshold
        )
        return None
        
    except Exception as e:
        _LOGGER.error("Speaker identification failed: %s", e)
        raise