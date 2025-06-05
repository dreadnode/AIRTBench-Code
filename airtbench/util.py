import asyncio
import re
import typing as t

T = t.TypeVar("T")


def flatten(list_: t.Sequence[t.Sequence[T]]) -> list[T]:
    return [item for sublist in list_ for item in sublist]


def majority_vote(list_: list[bool]) -> bool:
    return sum(list_) > len(list_) // 2


def all_in(str_: str, *checks: str) -> bool:
    return all(check in str_.lower() for check in checks)


def any_in(str_: str, *checks: str) -> bool:
    return any(check in str_.lower() for check in checks)


def not_in(str_: str, *checks: str) -> bool:
    return not any_in(str_, *checks)


# Async concurrency


async def enforce_concurrency(coros: t.Sequence[t.Awaitable[T]], limit: int) -> list[T]:
    async def run_coroutine_with_semaphore(semaphore: asyncio.Semaphore, coro: t.Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    semaphore = asyncio.Semaphore(limit)
    return await asyncio.gather(*(run_coroutine_with_semaphore(semaphore, coro) for coro in coros))


# Text cleanup

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi_codes(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)
