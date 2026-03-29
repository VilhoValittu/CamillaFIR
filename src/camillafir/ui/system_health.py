"""Compatibility wrapper for the application-level health service."""

import sys

from ..application import health_service as _service

sys.modules[__name__] = _service
