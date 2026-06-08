"""PyTensor/macOS fix: must run before `import pymc`."""
import os
import subprocess

import pytensor

pytensor.config.cxx = "/usr/bin/clang++"

_sdk = subprocess.run(
    ["xcrun", "--show-sdk-path"],
    check=False,
    capture_output=True,
    text=True,
).stdout.strip()
if _sdk:
    _cxx_inc = f"-I{_sdk}/usr/include/c++/v1"
    _flags = pytensor.config.gcc__cxxflags
    if _cxx_inc not in _flags:
        pytensor.config.gcc__cxxflags = f"{_flags} {_cxx_inc}".strip()

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=/usr/bin/clang++")
