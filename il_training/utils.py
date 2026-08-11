"""
Utility imports from isaac-training scripts.
"""

import sys
import importlib.util
from pathlib import Path

isaac_scripts_path = Path(__file__).resolve().parent.parent / "isaac-training" / "training" / "scripts"
if str(isaac_scripts_path) not in sys.path:
    sys.path.insert(0, str(isaac_scripts_path))

# Load isaac-training/training/scripts/utils.py under a unique module name.
# Importing it as the plain name "utils" would collide with THIS module (also named
# "utils" when il_training is on sys.path), causing a circular import.
_spec = importlib.util.spec_from_file_location(
    "_navrl_isaac_scripts_utils", str(isaac_scripts_path / "utils.py")
)
_isaac_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_isaac_utils)

ValueNorm = _isaac_utils.ValueNorm
make_mlp = _isaac_utils.make_mlp
GAE = _isaac_utils.GAE
make_batch = _isaac_utils.make_batch
IndependentBeta = _isaac_utils.IndependentBeta
BetaActor = _isaac_utils.BetaActor
IndependentNormal = _isaac_utils.IndependentNormal
Actor = _isaac_utils.Actor
vec_to_world = _isaac_utils.vec_to_world
vec_to_new_frame = _isaac_utils.vec_to_new_frame
construct_input = _isaac_utils.construct_input
evaluate = _isaac_utils.evaluate

__all__ = [
 'ValueNorm',
 'make_mlp',
 'GAE',
 'make_batch',
 'IndependentBeta',
 'BetaActor',
 'IndependentNormal',
 'Actor',
 'vec_to_world',
 'vec_to_new_frame',
 'construct_input',
 'evaluate',
]

