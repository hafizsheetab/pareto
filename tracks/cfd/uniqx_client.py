# uniqx_client.py — Singleton gateway client and module cache for hackaton_cfd.
#
# Both linalg.py and fd_operators.py share the same client and module cache so
# @to_module traces happen once per matrix size and are reused across all time steps.

import uniqx

UNIQX_TARGET = "localhost:50050"

_client = None
_MODULE_CACHE: dict = {}


def get_client():
    global _client
    if _client is None:
        _client = uniqx.connect(UNIQX_TARGET)
    return _client


def get_or_build(key, build_fn):
    """Return a cached Module, building and caching it on first call."""
    if key not in _MODULE_CACHE:
        _MODULE_CACHE[key] = build_fn()
    return _MODULE_CACHE[key]