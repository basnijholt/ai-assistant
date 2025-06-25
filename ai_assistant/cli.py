"""Shared CLI functionality for the AI assistant tools."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Handler


def get_base_parser() -> argparse.ArgumentParser:
    """Gets a base argument parser with common arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level.",
    )
    parser.add_argument("--log-file", help="Path to a file to write logs to.")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output from rich.",
    )
    return parser


def setup_logging(args: argparse.Namespace) -> None:
    """Sets up logging based on parsed arguments."""
    handlers: list[Handler] = []
    if not getattr(args, "quiet", False):
        handlers.append(logging.StreamHandler())
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="w"))

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
