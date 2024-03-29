# Default DSL
from .neural_functions import HeuristicNeuralFunction, ListToListModule, ListToAtomModule, AtomToAtomModule, init_neural_function
from .library_functions import StartFunction, LibraryFunction, MapFunction, MapPrefixesFunction, ITE, SimpleITE, \
                                FoldFunction, FullInputAffineFunction, AddFunction, MultiplyFunction,Multiply01,SoftmaxFunction

# Additional running average functions
from .running_averages import RunningAverageFunction, RunningAverageLast5Function, RunningAverageLast10Function, \
                                 RunningAverageWindow7Function, RunningAverageWindow5Function

# Domain-specific library functions
from .crim13 import Crim13PositionSelection, Crim13DistanceSelection, Crim13DistanceChangeSelection, \
                    Crim13VelocitySelection, Crim13AccelerationSelection, Crim13AngleSelection, Crim13AngleChangeSelection
from .fruitflies import FruitFlyWingSelection, FruitFlyRatioSelection, FruitFlyPositionalSelection, \
                        FruitFlyAngularSelection, FruitFlyLinearSelection
                        
from .basketball import BBallBallSelection, BBallOffenseSelection, BBallDefenseSelection
from .basketball47 import BBallBallSelection, BBallOffenseSelection, BBallDefenseSelection

from .mars import MarsAngleHeadBodySelection, MarsAxisRatioSelection, MarsSpeedSelection, MarsVelocitySelection, \
    MarsAccelerationSelection, MarsResidentTowardIntruderSelection, MarsRelAngleSelection, MarsRelDistSelection, \
    MarsAreaEllipseRatioSelection #MarsFacingAngleSelection #new 1

# from .mars_new import MarsAngleHeadBodySelection, MarsAxisRatioSelection, MarsSpeedSelection, MarsVelocitySelection, \
#     MarsAccelerationSelection, MarsResidentTowardIntruderSelection, MarsRelAngleSelection, MarsRelDistSelection, \
#     MarsAreaEllipseRatioSelection