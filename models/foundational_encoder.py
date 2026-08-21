"""
Loads the Encoder class from the `foundational` pretraining repo's `models`
package under an aliased module name, avoiding collision with denovo_base's
own top-level `models` package (both are unpackaged namespace packages named
`models`, so a plain `from models.encoder import Encoder` would silently
resolve to denovo_base's own encoder instead).
"""
import sys
import os
import types
import importlib

FOUNDATIONAL_ROOT = "/global/home/lapj/foundation"
_PKG_NAME = "foundational_models"

if _PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [os.path.join(FOUNDATIONAL_ROOT, "models")]
    sys.modules[_PKG_NAME] = _pkg

Encoder = importlib.import_module(f"{_PKG_NAME}.encoder").Encoder
