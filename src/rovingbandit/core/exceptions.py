"""Custom exception hierarchy for the RovingBandit library.

Each concrete error subclasses the matching builtin (``ValueError``/``RuntimeError``)
so that existing ``except ValueError`` handlers and ``pytest.raises(ValueError)``
checks keep working, while callers who want to can catch the library-specific
``RovingBanditError`` base instead.
"""


class RovingBanditError(Exception):
    """Base class for all rovingbandit errors."""


class InvalidArmError(RovingBanditError, ValueError):
    """Raised when an arm index is outside ``[0, n_arms)``."""


class InvalidConfigurationError(RovingBanditError, ValueError):
    """Raised for invalid construction or parameter values.

    Examples: a mismatched array length, an out-of-range probability, target
    shares that do not sum to one, or a parameter outside its valid interval.
    """


class MissingConfigurationError(RovingBanditError, ValueError):
    """Raised when a required input is absent.

    Examples: neither ``arm_means`` nor ``reward_fn`` provided, a horizon or
    context required but not set, or a policy/optimal reward needed by an
    objective but missing.
    """


class UninitializedStateError(RovingBanditError, RuntimeError):
    """Raised when internal state is accessed before it was initialized.

    These are defensive invariants that should be unreachable through the
    public API.
    """
