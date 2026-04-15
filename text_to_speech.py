from __future__ import annotations

import os

from gtts import gTTS


def speak_text(text: str, output_path: str) -> str:
    """
    Convert text to speech and save it as an MP3 file.

    Args:
        text: Text to synthesize.
        output_path: Destination audio path.

    Returns:
        The saved output path.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("output_path must be a non-empty string.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tts = gTTS(text=text.strip(), lang="en")
    tts.save(output_path)
    return output_path
