"""
Bridge NumPy 2-style imports on NumPy 1.x.

Some wheels import ``numpy._core.numeric``. On NumPy 1.26 that lives at
``numpy.core.numeric`` — alias so photo search does not crash.
"""

from __future__ import annotations

import sys
import types


def apply_numpy_core_alias() -> None:
    try:
        import numpy._core.numeric  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    try:
        import numpy.core as core
        import numpy.core.numeric as numeric
    except Exception:
        return

    pkg = types.ModuleType("numpy._core")
    pkg.__dict__.update({k: v for k, v in core.__dict__.items() if not k.startswith("__")})
    pkg.__path__ = list(getattr(core, "__path__", []))  # type: ignore[attr-defined]
    pkg.__file__ = getattr(core, "__file__", None)
    pkg.numeric = numeric

    sys.modules["numpy._core"] = pkg
    sys.modules["numpy._core.numeric"] = numeric

    for name in (
        "multiarray",
        "umath",
        "numerictypes",
        "_multiarray_umath",
        "overrides",
    ):
        try:
            mod = __import__(f"numpy.core.{name}", fromlist=["*"])
            setattr(pkg, name, mod)
            sys.modules[f"numpy._core.{name}"] = mod
        except Exception:
            continue


apply_numpy_core_alias()
