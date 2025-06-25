import typer

from ai_assistant import config

device_index: int | None = typer.Option(
    None,
    "--device-index",
    help="Index of the PyAudio input device to use.",
)
list_devices: bool = typer.Option(
    False,  # noqa: FBT003
    "--list-devices",
    help="List available audio input devices and exit.",
    is_eager=True,
)
asr_server_ip: str = typer.Option(
    config.ASR_SERVER_IP,
    "--asr-server-ip",
    help="Wyoming ASR server IP address.",
)
asr_server_port: int = typer.Option(
    config.ASR_SERVER_PORT,
    "--asr-server-port",
    help="Wyoming ASR server port.",
)
model: str = typer.Option(
    config.DEFAULT_MODEL,
    "--model",
    "-m",
    help=f"The Ollama model to use. Default is {config.DEFAULT_MODEL}.",
)
ollama_host: str = typer.Option(
    config.OLLAMA_HOST,
    "--ollama-host",
    help=f"The Ollama server host. Default is {config.OLLAMA_HOST}.",
)
daemon: bool = typer.Option(
    False,  # noqa: FBT003
    "--daemon",
    help="Run as a background daemon process.",
)
kill: bool = typer.Option(
    False,  # noqa: FBT003
    "--kill",
    help="Kill any running voice-assistant daemon.",
)
status: bool = typer.Option(
    False,  # noqa: FBT003
    "--status",
    help="Check if voice-assistant daemon is running.",
)
log_level: str = typer.Option(
    "WARNING",
    "--log-level",
    help="Set logging level.",
    case_sensitive=False,
)
log_file: str | None = typer.Option(
    None,
    "--log-file",
    help="Path to a file to write logs to.",
)
quiet: bool = typer.Option(
    False,  # noqa: FBT003
    "-q",
    "--quiet",
    help="Suppress console output from rich.",
)

clipboard: bool = typer.Option(
    True,  # noqa: FBT003
    "--clipboard/--no-clipboard",
    help="Copy transcript to clipboard.",
)
