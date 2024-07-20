from enum import Enum


class GoogleScopes(Enum):
    GDRIVE = 'https://www.googleapis.com/auth/drive'
    YOUTUBE = 'https://www.googleapis.com/auth/youtube.readonly'
