import pathlib

import yaml
from pydantic import BaseModel

current_dir = pathlib.Path(__file__).parent
challenges_dir = current_dir / "challenges"


class Challenge(BaseModel):
    id: str
    name: str
    category: str
    difficulty: str
    notebook: str
    is_llm: bool = False


def load_challenges() -> list[Challenge]:
    """
    Load challenges from the .challenges.yaml file in the challenges directory.

    Returns:
        list[Challenge]: A list of Challenge objects.
    """
    with (challenges_dir / ".challenges.yaml").open() as f:
        return [Challenge(id=key, **challenge) for key, challenge in yaml.safe_load(f).items()]
