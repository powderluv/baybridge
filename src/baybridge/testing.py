from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable


@dataclass(frozen=True)
class JitArguments:
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args: Any, **kwargs: Any):
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "kwargs", dict(kwargs))


def benchmark(
    function: Callable[..., Any],
    *,
    kernel_arguments: JitArguments,
    warmup_iterations: int = 1,
    iterations: int = 10,
) -> float:
    if warmup_iterations < 0 or iterations <= 0:
        raise ValueError("warmup_iterations must be >= 0 and iterations must be > 0")
    for _ in range(warmup_iterations):
        function(*kernel_arguments.args, **kernel_arguments.kwargs)
    start = perf_counter()
    for _ in range(iterations):
        function(*kernel_arguments.args, **kernel_arguments.kwargs)
    elapsed = perf_counter() - start
    return (elapsed / iterations) * 1_000_000.0


def autotune_jit(*args: Any, **kwargs: Any):
    del args
    del kwargs

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return decorator


def tune(
    configurations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    function: Callable[..., Any] | None = None,
    kernel_arguments: JitArguments | None = None,
) -> dict[str, Any]:
    del function
    del kernel_arguments
    if not configurations:
        raise ValueError("tune requires at least one configuration")
    return dict(configurations[0])


def get_workspace_count(*args: Any, **kwargs: Any) -> int:
    del args
    del kwargs
    return 1
