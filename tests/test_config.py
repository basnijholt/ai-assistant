"""Test the config loading."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click import Command, Option
from typer import Context
from typer.testing import CliRunner

from agent_cli.cli import set_config_defaults
from agent_cli.config_loader import load_config

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config_content = """
[defaults]
model = "wildcard-model"
log_level = "INFO"

[autocorrect]
model = "autocorrect-model"
quiet = true

[transcribe]
model = "transcribe-model"
clipboard = false
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)
    return config_path


def test_config_loader_basic(config_file: Path) -> None:
    """Test the config loader function directly."""
    # Test loading from explicit path
    config = load_config(str(config_file))
    assert config["defaults"]["model"] == "wildcard-model"
    assert config["defaults"]["log_level"] == "INFO"
    assert config["autocorrect"]["model"] == "autocorrect-model"
    assert config["autocorrect"]["quiet"] is True
    assert config["transcribe"]["model"] == "transcribe-model"
    assert config["transcribe"]["clipboard"] is False

    # Test loading from non-existent path
    config = load_config("/non/existent/path.toml")
    assert config == {}

    # Test loading with None (should use default paths)
    config = load_config(None)
    assert isinstance(config, dict)


def test_config_loader_key_replacement(config_file: Path) -> None:
    """Test that dashed keys are replaced with underscores."""
    # Add a config with dashed keys
    config_content = """
[defaults]
log-level = "DEBUG"
ollama-host = "http://example.com"

[test-command]
some-option = "value"
"""
    config_path = config_file.parent / "dashed-config.toml"
    config_path.write_text(config_content)

    config = load_config(str(config_path))
    assert config["defaults"]["log_level"] == "DEBUG"
    assert config["defaults"]["ollama_host"] == "http://example.com"
    assert config["test-command"]["some_option"] == "value"


def test_set_config_defaults(config_file: Path) -> None:
    """Test the set_config_defaults function."""
    # Mock parameters
    mock_model_param = Option(["--model"], default="original-model")
    mock_log_level_param = Option(["--log-level"], default="original-log-level")
    mock_quiet_param = Option(["--quiet"], default=False, is_flag=True)
    mock_clipboard_param = Option(["--clipboard"], default=True, is_flag=True)

    # Mock subcommands
    mock_autocorrect_cmd = Command(
        name="autocorrect",
        params=[mock_model_param, mock_log_level_param, mock_quiet_param],
    )
    mock_transcribe_cmd = Command(
        name="transcribe",
        params=[mock_model_param, mock_log_level_param, mock_clipboard_param],
    )

    # Mock main command
    mock_main_command = MagicMock()
    mock_main_command.commands = {
        "autocorrect": mock_autocorrect_cmd,
        "transcribe": mock_transcribe_cmd,
    }

    ctx = Context(command=mock_main_command)

    # Test with no subcommand (should set default_map)
    ctx.invoked_subcommand = None
    set_config_defaults(ctx, str(config_file))
    assert ctx.default_map == {"model": "wildcard-model", "log_level": "INFO"}

    # Test with autocorrect subcommand
    ctx.invoked_subcommand = "autocorrect"
    set_config_defaults(ctx, str(config_file))

    # Check that the defaults on the parameters themselves have been updated
    assert mock_model_param.default == "autocorrect-model"
    assert mock_log_level_param.default == "INFO"
    assert mock_quiet_param.default is True

    # Test with transcribe subcommand
    # Reset param defaults before testing the next command
    mock_model_param.default = "original-model"
    mock_log_level_param.default = "original-log-level"

    ctx.invoked_subcommand = "transcribe"
    set_config_defaults(ctx, str(config_file))
    assert mock_model_param.default == "transcribe-model"
    assert mock_log_level_param.default == "INFO"
    assert mock_clipboard_param.default is False


@patch("agent_cli.config_loader.CONFIG_PATH")
@patch("agent_cli.config_loader.CONFIG_PATH_2")
def test_default_config_paths(mock_path2: Path, mock_path1: Path, config_file: Path) -> None:
    """Test that default config paths are checked in order."""
    # Neither path exists
    mock_path1.exists.return_value = False  # type: ignore[attr-defined]
    mock_path2.exists.return_value = False  # type: ignore[attr-defined]
    config = load_config(None)
    assert config == {}

    # Only CONFIG_PATH_2 exists
    mock_path1.exists.return_value = False  # type: ignore[attr-defined]
    mock_path2.exists.return_value = True  # type: ignore[attr-defined]
    mock_path2.open.return_value.__enter__.return_value = config_file.open("rb")  # type: ignore[attr-defined]
    config = load_config(None)
    assert config["defaults"]["model"] == "wildcard-model"

    # CONFIG_PATH exists (takes precedence)
    mock_path1.exists.return_value = True  # type: ignore[attr-defined]
    mock_path2.exists.return_value = True  # type: ignore[attr-defined]
    mock_path1.open.return_value.__enter__.return_value = config_file.open("rb")  # type: ignore[attr-defined]
    config = load_config(None)
    assert config["defaults"]["model"] == "wildcard-model"


def test_config_file_error_handling(tmp_path: Path) -> None:
    """Test config loading with invalid TOML."""
    invalid_toml = tmp_path / "invalid.toml"
    invalid_toml.write_text("invalid toml content [[[")

    # The config loader doesn't handle TOML errors currently
    # It lets them propagate. Let's test that they do propagate.

    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(str(invalid_toml))
