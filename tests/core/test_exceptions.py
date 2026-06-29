"""Tests for the custom exception hierarchy."""

import pytest

from rovingbandit import (
    InvalidArmError,
    InvalidConfigurationError,
    MissingConfigurationError,
    RovingBanditError,
    UninitializedStateError,
)


@pytest.mark.parametrize(
    ("exc", "builtin"),
    [
        (InvalidArmError, ValueError),
        (InvalidConfigurationError, ValueError),
        (MissingConfigurationError, ValueError),
        (UninitializedStateError, RuntimeError),
    ],
)
def test_exception_subclasses_base_and_builtin(exc, builtin):
    """Each concrete error subclasses RovingBanditError and its matching builtin."""
    assert issubclass(exc, RovingBanditError)
    assert issubclass(exc, builtin)


@pytest.mark.parametrize(
    "exc",
    [InvalidArmError, InvalidConfigurationError, MissingConfigurationError],
)
def test_value_error_catches_custom_types(exc):
    """ValueError handlers still catch the value-like custom exceptions."""
    with pytest.raises(ValueError):
        raise exc("boom")


def test_base_is_plain_exception():
    """The base type is a plain Exception (not a ValueError)."""
    assert issubclass(RovingBanditError, Exception)
    assert not issubclass(RovingBanditError, ValueError)


def test_uninitialized_state_is_runtime_error():
    """UninitializedStateError is a RuntimeError, not a ValueError."""
    assert issubclass(UninitializedStateError, RuntimeError)
    assert not issubclass(UninitializedStateError, ValueError)
