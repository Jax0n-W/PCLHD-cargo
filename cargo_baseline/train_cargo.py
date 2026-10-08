"""Public entry point for the fixed CARGO PCLHD-CMhard baseline."""

from __future__ import absolute_import, print_function

import os.path as osp
import runpy
import sys


HERE = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


if __name__ == "__main__":
    engine = osp.join(HERE, "_train_cargo_engine.py")
    runpy.run_path(engine, run_name="__main__")
