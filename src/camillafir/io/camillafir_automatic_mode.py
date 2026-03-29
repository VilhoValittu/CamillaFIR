import sys

from ..auto_mode import api as _api

sys.modules[__name__] = _api
