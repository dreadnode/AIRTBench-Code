from pathlib import Path

import pytest

import airtbench.challenges as challenges_module
from airtbench.challenges import load_challenges


def test_load_challenges_reads_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_dir = tmp_path / "challenges"
    manifest_dir.mkdir()
    (manifest_dir / ".challenges.yaml").write_text(
        """
sample:
  category: Prompt Injection
  difficulty: easy
  name: Sample
  notebook: sample.ipynb
  is_llm: true
""",
    )
    monkeypatch.setattr(challenges_module, "challenges_dir", Path(manifest_dir))

    challenges = load_challenges()

    assert len(challenges) == 1
    assert challenges[0].id == "sample"
    assert challenges[0].is_llm is True
