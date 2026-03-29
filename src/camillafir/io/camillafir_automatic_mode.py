"""Backward-compatible alias for the relocated automatic-mode API.

This module exists only to preserve legacy imports. Do not add new business
logic here; extend `camillafir.auto_mode.api` instead.
"""

import sys

from ..auto_mode import api as _api

sys.modules[__name__] = _api
