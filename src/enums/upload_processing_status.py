from enum import Enum

class UploadProcessingStatus(Enum):
    AUDIO_PROCESSING = "AUDIO_PROCESSING"
    QUEUED = "QUEUED"
    RE_ENCODING = "RE_ENCODING"
    TRANSCRIBING = "TRANSCRIBING"
    VISUAL_PROCESSING = "VISUAL_PROCESSING"

    @staticmethod
    def parse_string(status_str):
        for status in UploadProcessingStatus.__members__.values():
            if status.value == status_str:
                return status
        raise ValueError("Invalid status string")