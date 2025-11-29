# %%
import sys
import platform
import os

# Setup UVTOOLS_PATH
UVTOOLS_PATH = None
if platform.system() == "Windows":
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"Software\UVtools", 0, winreg.KEY_READ
        )
        UVTOOLS_PATH = winreg.QueryValueEx(key, "InstallDir")[0]
        if key:
            winreg.CloseKey(key)
    except:
        pass
else:
    UVTOOLS_PATH = os.getenv("UVTOOLS_PATH")

if UVTOOLS_PATH is None or not os.path.exists(UVTOOLS_PATH):
    raise RuntimeError(
        "Unable to find UVtools path. Please install and set UVTOOLS_PATH as an environment variable."
    )

# Bootstrap .NET interop
sys.path.append(UVTOOLS_PATH)

import pythonnet

pythonnet.load("coreclr")

import clr

clr.AddReference("UVtools.Core")

# 🌟 This imports what "from UVtools.Core import *" would
from UVtools.Core import *

# 🌟 Now bring in the rest of the submodules explicitly
from UVtools.Core import EmguCV
from UVtools.Core import Extensions
from UVtools.Core import FileFormats
from UVtools.Core import GCode
from UVtools.Core import Layers
from UVtools.Core import Managers
from UVtools.Core import MeshFormats
from UVtools.Core import Network
from UVtools.Core import Objects
from UVtools.Core import Operations
from UVtools.Core import PixelEditor
from UVtools.Core import Printer
from UVtools.Core import Scripting
from UVtools.Core import Suggestions
from UVtools.Core import SystemOS

__all__ = [
    "About",
    "EmguCV",
    "Extensions",
    "FileFormats",
    "GCode",
    "Layers",
    "Managers",
    "MeshFormats",
    "Network",
    "Objects",
    "Operations",
    "PixelEditor",
    "Printer",
    "Scripting",
    "Suggestions",
    "SystemOS",
]
