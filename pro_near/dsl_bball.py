# DSL for 51 features
import dsl


DSL_DICT = {
    ('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                        dsl.running_averages.RunningAverageWindow13Function,
                        dsl.running_averages.RunningAverageWindow5Function,dsl.MultiplyFunction],
    ('atom', 'atom') : [dsl.AddFunction, dsl.MultiplyFunction, dsl.SimpleITE,dsl.SoftmaxFunction,
                    dsl.Multiply01,
    #51
                        dsl.basketball.BBallOffenseBallDistSelection,dsl.basketball.BBallOffenseBhDistSelection,
                        dsl.basketball.BBallDefenseBhDistSelection,
                        dsl.basketball.BBallBhOneHotSelection,
                        dsl.basketball.BBallScreenBhDistSelection,
                        dsl.basketball.BBallScreenPaintSelection,
                        dsl.basketball.BBallBallPaintSelection

                        #47
                        # dsl.basketball.BBallBallSelection,dsl.basketball.BBallOffenseSelection,
                        # dsl.basketball.BBallDefenseSelection,
                        # dsl.basketball.BBallOffenseBallDistSelection,dsl.basketball.BBallOffenseBhDistSelection,
                        # dsl.basketball.BBallOffenseBasketDistSelection,dsl.basketball.BBallDefenseBhDistSelection,
                        # dsl.basketball.BBallOffensePaintSelection
                        ]
}


CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}