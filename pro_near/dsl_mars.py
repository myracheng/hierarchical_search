import dsl

DSL_DICT = {('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
                        ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE,
                                            dsl.running_averages.RunningAverageLast10Function, dsl.running_averages.RunningAverageWindow13Function,
                                            dsl.running_averages.RunningAverageWindow5Function,dsl.running_averages.RunningAverageWindow7Function],
('atom', 'atom') : [dsl.SimpleITE, dsl.AddFunction, dsl.MultiplyFunction, dsl.MarsAngleHeadBodySelection, \
                    dsl.MarsAxisRatioSelection, dsl.MarsSpeedSelection, dsl.MarsVelocitySelection, \
                    dsl.MarsAccelerationSelection, dsl.MarsResidentTowardIntruderSelection, dsl.MarsRelAngleSelection,
                    dsl.MarsRelDistSelection, dsl.MarsAreaEllipseRatioSelection]}

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}
