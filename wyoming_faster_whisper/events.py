"""Custom Wyoming event types for speaker identification."""
import logging
from typing import Optional
from wyoming.event import Event

_LOGGER = logging.getLogger(__name__)

class SpeakerTranscript:
    """Transcript with identified speaker."""
    TYPE = "speaker-transcript"

    def __init__(self, text: str, speaker: Optional[str] = None) -> None:
        self._event = Event(
            self.TYPE,
            data={
                "text": text,
                "speaker": speaker
            }
        )
        _LOGGER.debug("Created transcript: text='%s', speaker=%s", text, speaker)

    def event(self) -> Event:
        """Return Wyoming Event representation."""
        _LOGGER.debug("Event data: %s", self._event.data)
        return self._event

    @classmethod
    def from_event(cls, event: Event) -> "SpeakerTranscript":
        """Create from Wyoming Event."""
        _LOGGER.debug("Creating from event: %s", event.data)
        return cls(
            text=event.data["text"],
            speaker=event.data.get("speaker")
        )