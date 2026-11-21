"""wingbatch: gz-sim's fixed-wing physics (AdvancedLiftDrag, MulticopterMotorModel, JointPositionController)
read from an SDF model and run for N aircraft at once in NumPy.

Not affiliated with, nor endorsed by, the PX4 project, the Dronecode Foundation or Open Robotics.
"""

from .aircraft import Aircraft, bundled_models
from .physics import MassProperties
from .sdf import Model, World, load_model, load_world
from .systems.wind import UniformWind, WindField
from .trim import steady_state, trim

__version__ = "0.1.0"
__all__ = ["Aircraft", "MassProperties", "Model", "UniformWind", "WindField", "World", "bundled_models",
           "load_model", "load_world", "steady_state", "trim", "__version__"]
