from __future__ import annotations

import pytest

import baybridge as bb


def test_nvidia_target_normalizes_integer_sm() -> None:
    target = bb.NvidiaTarget(sm=80)

    assert target.sm == "sm_80"
    assert target.arch == "sm_80"
    assert target.to_dict()["sm"] == "sm_80"


def test_nvidia_target_normalizes_numeric_string_sm() -> None:
    target = bb.NvidiaTarget(sm="80")

    assert target.sm == "sm_80"
    assert target.arch == "sm_80"


def test_nvidia_target_preserves_prefixed_sm_variant() -> None:
    target = bb.NvidiaTarget(sm="SM_90A")

    assert target.sm == "sm_90a"
    assert target.arch == "sm_90a"


def test_nvidia_target_rejects_invalid_sm_text() -> None:
    with pytest.raises(ValueError, match="expects values like 80, '80', or 'sm_80'"):
        bb.NvidiaTarget(sm="sm-80")
