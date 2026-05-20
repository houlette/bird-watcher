"""Shared pipeline exceptions."""


class SkipFile(Exception):
    """Raised for a clip we should NOT retry — too large, corrupt, unsupported
    codec, etc. The worker catches this, marks the Visit row processed (with
    the error message recorded), and moves on. Distinguishes itself from
    transient errors (network blips, partial uploads) that *should* retry.
    """
