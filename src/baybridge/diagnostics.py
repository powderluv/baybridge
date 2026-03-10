class BaybridgeError(Exception):
    """Base exception for baybridge."""


class CompilationError(BaybridgeError):
    """Raised when a kernel cannot be traced into the portable IR."""


class UnsupportedOperationError(BaybridgeError):
    """Raised when the user reaches an unsupported API surface."""


class BackendNotImplementedError(BaybridgeError):
    """Raised when a compiled artifact is launched before AMD lowering exists."""
