import dsl


DSL_DICT = {
    ('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                        dsl.running_averages.RunningAverageWindow13Function,
                        dsl.running_averages.RunningAverageWindow5Function],
    ('atom', 'atom') : [dsl.AddFunction, dsl.MultiplyFunction, dsl.SimpleITE,
                        dsl.basketball47.BBallBallSelection,dsl.basketball47.BBallOffenseSelection,
                        dsl.basketball47.BBallDefenseSelection,
                        dsl.basketball47.BBallOffenseBallDistSelection,dsl.basketball47.BBallOffenseBhDistSelection,
                        dsl.basketball47.BBallOffenseBasketDistSelection,dsl.basketball47.BBallDefenseBhDistSelection,
                        dsl.basketball47.BBallOffensePaintSelection
                        ]
}


CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}