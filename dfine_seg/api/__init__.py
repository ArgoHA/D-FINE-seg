"""Load a model and know what it is: weight resolution, backend dispatch, checkpoint facts.

Torch-only — nothing here may reach the training stack. The public names are re-exported
from `dfine_seg` itself, so callers write `from dfine_seg import load_model`.
"""
