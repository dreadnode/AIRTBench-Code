from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from airtbench.challenges import Challenge, load_challenges
from airtbench.kernel import Notebook


@dataclass(frozen=True)
class HarnessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class HarnessVerification:
    checks: list[HarnessCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def verify_harness_configuration(
    *,
    challenge_dir: Path,
    container_dir: Path,
    challenges: list[Challenge] | None = None,
) -> HarnessVerification:
    loaded_challenges = challenges if challenges is not None else load_challenges()
    checks: list[HarnessCheck] = []

    checks.append(
        HarnessCheck(
            "dreadnode_sdk_v2",
            version("dreadnode").startswith("2."),
            f"dreadnode {version('dreadnode')}",
        ),
    )
    checks.append(
        HarnessCheck(
            "challenge_manifest",
            bool(loaded_challenges),
            f"{len(loaded_challenges)} challenges loaded",
        ),
    )
    checks.append(
        HarnessCheck(
            "llm_challenges",
            any(challenge.is_llm for challenge in loaded_challenges),
            f"{sum(challenge.is_llm for challenge in loaded_challenges)} LLM challenges",
        ),
    )

    missing_notebooks = [
        challenge.notebook
        for challenge in loaded_challenges
        if not (challenge_dir / challenge.notebook).exists()
    ]
    checks.append(
        HarnessCheck(
            "challenge_notebooks_exist",
            not missing_notebooks,
            "all notebooks present" if not missing_notebooks else ", ".join(missing_notebooks),
        ),
    )

    unreadable_notebooks = []
    for challenge in loaded_challenges:
        notebook_path = challenge_dir / challenge.notebook
        if not notebook_path.exists():
            continue
        try:
            Notebook.load(notebook_path)
        except (OSError, ValueError):
            unreadable_notebooks.append(challenge.notebook)

    checks.append(
        HarnessCheck(
            "challenge_notebooks_parse",
            not unreadable_notebooks,
            "all notebooks parse" if not unreadable_notebooks else ", ".join(unreadable_notebooks),
        ),
    )

    dockerfile = container_dir / "Dockerfile"
    checks.append(
        HarnessCheck(
            "container_dockerfile",
            dockerfile.exists(),
            str(dockerfile),
        ),
    )

    return HarnessVerification(checks)
