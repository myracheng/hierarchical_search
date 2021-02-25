import copy

from .core import ProgramLearningAlgorithm
from program_graph import ProgramGraph
from utils.logging import log_and_print, print_program, print_program_dict

# from utils.logging import log_and_print
from utils.training import execute_and_train_prog


class BALLSCREEN(ProgramLearningAlgorithm):

    def __init__(self):
        log_and_print("Root node is Start(ListToListModule) or Start(ListToAtomModule), both implemented with an RNN.")
        log_and_print("Be sure to set neural_epochs and max_num_units accordingly.\n")

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)
        log_and_print("Training RNN baseline with {} LSTM units ...".format(graph.max_num_units))
        current = copy.deepcopy(graph.root_node)

        child_node = graph.get_bball_prog(current)
        log_and_print("Training child program: {}".format(print_program(child_node.program, ignore_constants=(not verbose))))
                # is_neural = not graph.is_fully_symbolic(child_node.program) #mcheng is not complete
                # child_node.score, l, m = execute_and_train_with_full(base_program_name, hole_node_ind, child_node.program, validset, trainset, train_config, 
                #     graph.output_type, graph.output_size, neural=is_neural, device=device)

        score,prog = execute_and_train_prog(child_node.program, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=False, device=device)
        log_and_print("Score of Program is {:.4f}".format(1-score))
        
        return score, prog