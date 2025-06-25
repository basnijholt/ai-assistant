"""Tests for the ai_assistant package."""

from __future__ import annotations

from ai_assistant import cli


def test_get_base_parser() -> None:
    """Test that the base parser has the correct default arguments."""
    parser = cli.get_base_parser()
    args = parser.parse_args([])
    assert args.log_level == "INFO"
    assert not args.quiet
