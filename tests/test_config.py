"""Test the config loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from agent_cli.cli import app

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
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def dummy_cli(config_file: Path) -> None:  # noqa: ARG001
    # Create a dummy autocorrect command that just prints its parameters
    @app.command("autocorrect", help="dummy help")
    def test_autocorrect(
        model: str = "default-model",
        log_level: str = "WARNING",
        quiet: bool = False,
    ) -> None:
        print(f"model={model}")
        print(f"log_level={log_level}")
        print(f"quiet={quiet}")

    # Create a dummy command that just prints its parameters
    @app.command("wildcard_test", help="dummy help")
    def wildcard_test(
        model: str = "default-model",
        log_level: str = "WARNING",
        quiet: bool = False,
    ) -> None:
        print(f"model={model}")
        print(f"log_level={log_level}")
        print(f"quiet={quiet}")


def test_config_loading(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_cli: None,  # noqa: ARG001
) -> None:
    result = runner.invoke(
        app,
        ["autocorrect", "--config", str(config_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=autocorrect-model" in output
    assert "log_level=INFO" in output
    assert "quiet=True" in output

    result = runner.invoke(
        app,
        ["wildcard_test", "--config", str(config_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=wildcard-model" in output
    assert "log_level=INFO" in output
    assert "quiet=False" in output

    result = runner.invoke(
        app,
        [
            "autocorrect",
            "--config",
            str(config_file),
            "--model",
            "override-model",
            "--quiet=false",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=override-model" in output
    assert "log_level=INFO" in output
    assert "quiet=False" in output

    result = runner.invoke(
        app,
        ["autocorrect", "--config", str(config_file), "--log-level", "DEBUG"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=autocorrect-model" in output
    assert "log_level=DEBUG" in output
    assert "quiet=True" in output

    result = runner.invoke(
        app,
        ["autocorrect"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=default-model" in output
    assert "log_level=WARNING" in output
    assert "quiet=False" in output

    # Test default config file loading
    monkeypatch.setattr("agent_cli.config_loader.CONFIG_PATH", config_file)
    result = runner.invoke(
        app,
        ["autocorrect"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=autocorrect-model" in output
    assert "log_level=INFO" in output
    assert "quiet=True" in output

    # Test --no-config
    monkeypatch.setattr("agent_cli.config_loader.CONFIG_PATH", config_file)
    result = runner.invoke(
        app,
        ["--config", "", "autocorrect"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output = result.stdout
    assert "model=default-model" in output
    assert "log_level=WARNING" in output
    assert "quiet=False" in output

    # Clean up the dummy commands
    app.registered_commands = [
        cmd for cmd in app.registered_commands if cmd.name not in ["autocorrect", "wildcard_test"]
    ]
