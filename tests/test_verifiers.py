import pytest

from airtbench.verifiers import FlagVerifier, extract_flags, platform_url


def test_extract_flags_returns_all_flag_like_values() -> None:
    content = "first gAAAAAabc-_= second gAAAAAxyz123"

    assert extract_flags(content) == ["gAAAAAabc-_=", "gAAAAAxyz123"]


def test_platform_url_uses_env_without_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRUCIBLE_URL", "https://example.test/")

    assert platform_url() == "https://example.test"


async def test_verify_text_stops_on_first_correct_flag() -> None:
    calls: list[str] = []

    class FakeVerifier(FlagVerifier):
        async def check_flag(self, challenge_id: str, flag: str) -> bool:
            calls.append(f"{challenge_id}:{flag}")
            return flag == "gAAAAAgood"

    result = await FakeVerifier("token").verify_text(
        "bear1",
        "bad gAAAAAbad then good gAAAAAgood then later gAAAAAlater",
    )

    assert result.correct is True
    assert result.verified_flag == "gAAAAAgood"
    assert result.matches == ["gAAAAAbad", "gAAAAAgood", "gAAAAAlater"]
    assert calls == ["bear1:gAAAAAbad", "bear1:gAAAAAgood"]
