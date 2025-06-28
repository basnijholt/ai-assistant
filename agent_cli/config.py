"""Default configuration settings for the Agent CLI package."""

from __future__ import annotations

import os

import pyaudio

# --- ASR (Wyoming) Configuration ---
ASR_SERVER_IP = os.getenv("ASR_SERVER_IP", "192.168.1.143")
ASR_SERVER_PORT = 10300

# --- TTS (Wyoming Piper) Configuration ---
TTS_SERVER_IP = os.getenv("TTS_SERVER_IP", "192.168.1.143")
TTS_SERVER_PORT = 10200

# --- Ollama LLM Configuration ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "devstral:24b"

# --- PyAudio Configuration ---
PYAUDIO_FORMAT = pyaudio.paInt16
PYAUDIO_CHANNELS = 1
PYAUDIO_RATE = 16000
PYAUDIO_CHUNK_SIZE = 1024
