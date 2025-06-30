# Agent CLI

`agent-cli` is a collection of **_local-first_**, AI-powered command-line agents that run entirely on your machine.
It provides a suite of powerful tools for voice and text interaction, designed for privacy, offline capability, and seamless integration with system-wide hotkeys and workflows.

> [!TIP]
> If using [`uv`](https://docs.astral.sh/uv/), you can easily run the tools from this package directly. For example, to see the help message for `autocorrect`:
>
> ```bash
> uvx agent-cli autocorrect --help
> ```

<details><summary><b><u>[ToC]</u></b> 📚</summary>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [`autocorrect`](#autocorrect)
  - [`transcribe`](#transcribe)
  - [`speak`](#speak)
  - [`voice-assistant`](#voice-assistant)
  - [`interactive`](#interactive)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Contributing](#contributing)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

</details>

> [!IMPORTANT]
> **Local and Private by Design**
> All agents in this toolkit are designed to run **100% locally**. Your data, whether it's from your clipboard, microphone, or files, is never sent to any cloud API. This ensures your privacy and allows the tools to work completely offline.

## Features

- **`autocorrect`**: Correct grammar and spelling in your text (e.g., from clipboard) using a local LLM with Ollama.
- **`transcribe`**: Transcribe audio from your microphone to text in your clipboard.
- **`speak`**: Convert text to speech using a local TTS engine.
- **`voice-assistant`**: A voice-powered clipboard assistant that edits text based on your spoken commands.
- **`interactive`**: An interactive, conversational AI agent with tool-calling capabilities.

## Prerequisites

- **Python**: Version 3.11 or higher.
- **Ollama**: For `autocorrect`, `voice-assistant`, and `interactive`, you need [Ollama](https://ollama.ai/) running with a model pulled (e.g., `ollama pull mistral:latest`).
- **Wyoming Piper**: For `speak`, `voice-assistant`, and `interactive`, you need a [Wyoming TTS server](https://github.com/rhasspy/wyoming-piper) running for text-to-speech.
- **Wyoming Faster Whisper**: For `transcribe`, `voice-assistant`, and `interactive`, you need a [Wyoming ASR server](https://github.com/rhasspy/wyoming-faster-whisper) for speech-to-text.
- **Clipboard Tools**: `xsel`, `xclip` (Linux), or `pbcopy`/`pbpaste` (macOS) are used by many agents.
- **PortAudio**: Required for PyAudio to handle microphone and speaker I/O.

## Installation

Install `agent-cli` using pip:

```bash
pip install agent-cli
```

Or for development:

1. **Clone the repository:**

   ```bash
   git clone git@github.com:basnijholt/agent-cli.git
   cd agent-cli
   ```

2. **Install in development mode:**

   ```bash
   uv sync
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

## Usage

This package provides multiple command-line tools, each designed for a specific purpose.

### `autocorrect`

**Purpose:** Quickly fix spelling and grammar in any text you've copied.

**Workflow:** This is a simple, one-shot command.
1.  It reads text from your system clipboard (or from a direct argument).
2.  It sends the text to a local Ollama LLM with a prompt to perform only technical corrections.
3.  The corrected text is copied back to your clipboard, replacing the original.

**How to Use It:** This tool is ideal for integrating with a system-wide hotkey.
- **From Clipboard**: `agent-cli autocorrect`
- **From Argument**: `agent-cli autocorrect "this text has an eror"`

<details>
<summary>See the output of <code>agent-cli autocorrect --help</code></summary>

<!-- CODE:BASH:START -->
<!-- echo '```yaml' -->
<!-- export NO_COLOR=1 -->
<!-- export TERM=dumb -->
<!-- export TERMINAL_WIDTH=80 -->
<!-- agent-cli autocorrect --help -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- OUTPUT:END -->

</details>

### `transcribe`

**Purpose:** A simple tool to turn your speech into text.

**Workflow:** This agent listens to your microphone and converts your speech to text in real-time.
1.  Run the command. It will start listening immediately.
2.  Speak into your microphone.
3.  Press `Ctrl+C` to stop recording.
4.  The transcribed text is copied to your clipboard.
5.  Optionally, use the `--llm` flag to have an Ollama model clean up the raw transcript (fixing punctuation, etc.).

**How to Use It:**
- **Simple Transcription**: `agent-cli transcribe --device-index 1`
- **With LLM Cleanup**: `agent-cli transcribe --device-index 1 --llm`

<details>
<summary>See the output of <code>agent-cli transcribe --help</code></summary>

<!-- CODE:BASH:START -->
<!-- echo '```yaml' -->
<!-- export NO_COLOR=1 -->
<!-- export TERM=dumb -->
<!-- export TERMINAL_WIDTH=80 -->
<!-- agent-cli transcribe --help -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- OUTPUT:END -->

</details>

### `speak`

**Purpose:** Reads any text out loud.

**Workflow:** A straightforward text-to-speech utility.
1.  It takes text from a command-line argument or your clipboard.
2.  It sends the text to a Wyoming TTS server (like Piper).
3.  The generated audio is played through your default speakers.

**How to Use It:**
- **Speak from Argument**: `agent-cli speak "Hello, world!"`
- **Speak from Clipboard**: `agent-cli speak`
- **Save to File**: `agent-cli speak "Hello" --save-file hello.wav`

<details>
<summary>See the output of <code>agent-cli speak --help</code></summary>

<!-- CODE:BASH:START -->
<!-- echo '```yaml' -->
<!-- export NO_COLOR=1 -->
<!-- export TERM=dumb -->
<!-- export TERMINAL_WIDTH=80 -->
<!-- agent-cli speak --help -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- OUTPUT:END -->

</details>

### `voice-assistant`

**Purpose:** A powerful clipboard assistant that you command with your voice.

**Workflow:** This agent is designed for a hotkey-driven workflow to act on text you've already copied.
1.  Copy a block of text to your clipboard (e.g., an email draft).
2.  Press a hotkey to run `agent-cli voice-assistant &` in the background. The agent is now listening.
3.  Speak a command, such as "Make this more formal" or "Summarize the key points."
4.  Press the same hotkey again, which should trigger `agent-cli voice-assistant --stop`.
5.  The agent transcribes your command, sends it along with the original clipboard text to the LLM, and the LLM performs the action.
6.  The result is copied back to your clipboard. If `--tts` is enabled, it will also speak the result.

**How to Use It:** The power of this tool is unlocked with a hotkey manager like Keyboard Maestro (macOS) or AutoHotkey (Windows). See the docstring in `agent_cli/agents/voice_assistant.py` for a detailed Keyboard Maestro setup guide.

<details>
<summary>See the output of <code>agent-cli voice-assistant --help</code></summary>

<!-- CODE:BASH:START -->
<!-- echo '```yaml' -->
<!-- export NO_COLOR=1 -->
<!-- export TERM=dumb -->
<!-- export TERMINAL_WIDTH=80 -->
<!-- agent-cli voice-assistant --help -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- OUTPUT:END -->

</details>

### `interactive`

**Purpose:** A full-featured, conversational AI assistant that can interact with your system.

**Workflow:** This is a persistent, interactive agent that you can have a conversation with.
1.  Run the `interactive` command. It will start listening for your voice.
2.  Speak your command or question (e.g., "What's in my current directory?").
3.  The agent transcribes your speech, sends it to the LLM, and gets a response. The LLM can use tools like `read_file` or `execute_code` to answer your question.
4.  The agent speaks the response back to you and then immediately starts listening for your next command.
5.  The conversation continues in this loop. Conversation history is saved between sessions.

**Interaction Model:**
- **To Interrupt**: Press `Ctrl+C` **once** to stop the agent from either listening or speaking, and it will immediately return to a listening state for a new command. This is useful if it misunderstands you or you want to speak again quickly.
- **To Exit**: Press `Ctrl+C` **twice in a row** to terminate the application.

**How to Use It:**
- **Start the agent**: `agent-cli interactive --device-index 1 --tts`
- **Have a conversation**:
  - *You*: "Read the pyproject.toml file and tell me the project version."
  - *AI*: (Reads file) "The project version is 0.1.0."
  - *You*: "Thanks!"

<details>
<summary>See the output of <code>agent-cli interactive --help</code></summary>

<!-- CODE:BASH:START -->
<!-- echo '```yaml' -->
<!-- export NO_COLOR=1 -->
<!-- export TERM=dumb -->
<!-- export TERMINAL_WIDTH=80 -->
<!-- agent-cli interactive --help -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- OUTPUT:END -->

</details>

## Development

### Running Tests

The project uses `pytest` for testing. To run tests using `uv`:

```bash
uv run pytest
```

### Pre-commit Hooks

This project uses pre-commit hooks (ruff for linting and formatting, mypy for type checking) to maintain code quality. To set them up:

1. Install pre-commit:

   ```bash
   pip install pre-commit
   ```

2. Install the hooks:

   ```bash
   pre-commit install
   ```

   Now, the hooks will run automatically before each commit.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue. If you'd like to contribute code, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
