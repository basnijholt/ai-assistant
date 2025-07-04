#!/usr/bin/env python3
"""Development workflow automation script."""

import subprocess
import sys

MIN_ARGS = 2


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"🔄 {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return False


def setup_environment() -> bool:
    """Set up the development environment."""
    commands = [
        (["uv", "pip", "install", "-e", ".[dev]", "--all-extras"], "Installing dependencies"),
        (["uv", "run", "pre-commit", "install"], "Installing pre-commit hooks"),
    ]

    return all(run_command(cmd, desc) for cmd, desc in commands)


def run_tests() -> bool:
    """Run the test suite."""
    return run_command(["uv", "run", "pytest"], "Running tests")


def run_pre_commit() -> bool:
    """Run pre-commit hooks."""
    return run_command(
        ["uv", "run", "pre-commit", "run", "--all-files"],
        "Running pre-commit hooks",
    )


def check_project() -> bool:
    """Run all checks before committing."""
    print("🔍 Running all project checks...")

    if not run_pre_commit():
        return False

    if not run_tests():
        return False

    print("✅ All checks passed! Ready to commit.")
    return True


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < MIN_ARGS:
        print("Usage: python dev.py <command>")
        print("Commands:")
        print("  setup    - Set up development environment")
        print("  test     - Run tests")
        print("  lint     - Run pre-commit hooks")
        print("  check    - Run all checks (lint + test)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "setup":
        success = setup_environment()
    elif command == "test":
        success = run_tests()
    elif command == "lint":
        success = run_pre_commit()
    elif command == "check":
        success = check_project()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
