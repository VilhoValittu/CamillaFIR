"""Compatibility wrapper for the application-level house-curve service."""

import sys

from ..application import house_curve_service as _service

sys.modules[__name__] = _service
