"""Makes the repository root and its script folders importable.

Scripts in this folder start with `import _bootstrap`, so they can be run from the
repository root as `python <folder>/<script>.py` and still import `Models`, the data
loaders and the scripts of the other folders by name.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ('visualization', 'evaluation', 'baselines', ''):
    _path = os.path.join(ROOT, _sub) if _sub else ROOT
    if _path not in sys.path:
        sys.path.insert(0, _path)
