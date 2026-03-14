from __future__ import annotations

from pathlib import Path


def _extend_package_path_for_local_build() -> None:
    package_dir = Path(__file__).resolve().parent
    build_root = package_dir.parent / "build"
    for candidate in sorted(build_root.glob("lib.*"), reverse=True):
        built_package_dir = candidate / "rosa_soft"
        if built_package_dir.is_dir():
            built_package_str = str(built_package_dir)
            if built_package_str not in __path__:
                __path__.append(built_package_str)


_extend_package_path_for_local_build()

import torch

from . import _C, ops
from .rosa_sam import RosaContext, RosaWork
from .rosa_scan import rosa_scan_ops
from .rosa_soft import RosaSoftWork, rosa_soft_ops
from .rosa_sufa import rosa_sufa_ops
