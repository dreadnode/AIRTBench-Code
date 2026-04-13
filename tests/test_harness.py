import json
from pathlib import Path

import pytest

from airtbench.challenges import Challenge
from airtbench.harness import verify_harness_configuration


def write_notebook(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": "Challenge",
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
        ),
    )


def test_verify_harness_configuration_passes_for_minimal_valid_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_dir = tmp_path / "challenges"
    container_dir = tmp_path / "container"
    challenge_dir.mkdir()
    container_dir.mkdir()
    write_notebook(challenge_dir / "sample.ipynb")
    (container_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    monkeypatch.setattr("airtbench.harness.version", lambda _: "2.0.11")

    result = verify_harness_configuration(
        challenge_dir=challenge_dir,
        container_dir=container_dir,
        challenges=[
            Challenge(
                id="sample",
                name="Sample",
                category="Test",
                difficulty="easy",
                notebook="sample.ipynb",
                is_llm=True,
            ),
        ],
    )

    assert result.passed is True


def test_verify_harness_configuration_reports_missing_notebook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_dir = tmp_path / "challenges"
    container_dir = tmp_path / "container"
    challenge_dir.mkdir()
    container_dir.mkdir()
    (container_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    monkeypatch.setattr("airtbench.harness.version", lambda _: "2.0.11")

    result = verify_harness_configuration(
        challenge_dir=challenge_dir,
        container_dir=container_dir,
        challenges=[
            Challenge(
                id="sample",
                name="Sample",
                category="Test",
                difficulty="easy",
                notebook="missing.ipynb",
            ),
        ],
    )

    checks = {check.name: check for check in result.checks}
    assert result.passed is False
    assert checks["challenge_notebooks_exist"].passed is False
    assert "missing.ipynb" in checks["challenge_notebooks_exist"].detail
