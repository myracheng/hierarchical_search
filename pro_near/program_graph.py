import copy
import dsl


class ProgramNode(object):

    def __init__(self, program, score, parent, depth, cost, order):
        self.score = score
        self.program = program
        self.children = []
        self.parent = parent
        self.depth = depth
        self.cost = cost
        self.order = order


class ProgramGraph(object):

    def __init__(self, dsl_dict, edge_cost_dict, input_type, output_type, input_size, output_size,
        max_num_units, min_num_units, max_num_children, max_depth, penalty, ite_beta=1.0):
        self.dsl_dict = dsl_dict
        self.edge_cost_dict = edge_cost_dict
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.max_num_units = max_num_units
        self.min_num_units = min_num_units
        self.max_num_children = max_num_children
        self.max_depth = max_depth
        self.penalty = penalty
        self.ite_beta = ite_beta

        start = dsl.StartFunction(input_type=input_type, output_type=output_type, 
            input_size=input_size, output_size=output_size, num_units=max_num_units)
        self.root_node = ProgramNode(start, 0, None, 0, 0, 0)

    

    def construct_candidates(self, input_type, output_type, input_size, output_size, num_units):
        candidates = []
        replacement_candidates = self.dsl_dict[(input_type, output_type)]
        for functionclass in replacement_candidates:
            if issubclass(functionclass, dsl.ITE): #todo maybe here? yeah based on the way theyre inialized, e.g. multiply function contains init_neural_function
                candidate = functionclass(input_type, output_type, input_size, output_size, num_units, beta=self.ite_beta)
            else:
                candidate = functionclass(input_size, output_size, num_units)
            candidates.append(candidate)
        return candidates

    def is_fully_symbolic(self, candidate_program):
        queue = [(candidate_program.submodules['program'])]
        while len(queue) != 0:
            current_function = queue.pop()
            if issubclass(type(current_function), dsl.HeuristicNeuralFunction):
                return False
            else:
                for submodule in current_function.submodules:
                    queue.append(current_function.submodules[submodule])
        return True

    def compute_edge_cost(self, expandion_candidate):
        edge_cost = 0
        functionclass = type(expandion_candidate)
        typesig = expandion_candidate.get_typesignature()

        if functionclass in self.edge_cost_dict[typesig]:
            edge_cost = self.edge_cost_dict[typesig][functionclass]
        else:
            # Otherwise, the edge cost scales with the number of HeuristicNeuralFunction
            for submodule, fxnclass in expandion_candidate.submodules.items():
                if isinstance(fxnclass, dsl.HeuristicNeuralFunction):
                    edge_cost += 1

        return edge_cost*self.penalty

    def compute_program_cost(self, candidate_program):
        queue = [candidate_program.submodules['program']]
        total_cost = 0
        depth = 0
        edge_cost = 0
        while len(queue) != 0:
            depth += 1
            current_function = queue.pop()
            current_type = type(current_function)
            current_type_sig = current_function.get_typesignature()
            if current_type in self.edge_cost_dict[current_type_sig]:
                edge_cost = self.edge_cost_dict[current_type_sig][current_type]
            else:
                edge_cost = 0
                # Otherwise, the edge cost scales with the number of neural modules
                for submodule, fxnclass in current_function.submodules.items():
                    edge_cost += 1
            total_cost += edge_cost
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
        return total_cost * self.penalty, depth

    def min_depth2go(self, candidate_program):
        depth2go = 0
        queue = [(candidate_program.submodules['program'])]
        while len(queue) != 0:
            current_function = queue.pop()
            if issubclass(type(current_function), dsl.HeuristicNeuralFunction):
                depth2go += 1
                # in current DSL, ListToList/ListToAtom both need at least 2 depth to be fully symbolic
                if issubclass(type(current_function), dsl.ListToListModule):
                    depth2go += 1
                elif issubclass(type(current_function), dsl.ListToAtomModule):
                    depth2go += 1
            else:
                for submodule in current_function.submodules:
                    queue.append(current_function.submodules[submodule])
        return depth2go

    def num_units_at_depth(self, depth):
        num_units = max(int(self.max_num_units*(0.5**(depth-1))), self.min_num_units)
        return num_units
    
    
    def get_all_children(self, current_node, in_enumeration=False):
        all_children = []
        child_depth = current_node.depth + 1
        child_num_units = self.num_units_at_depth(child_depth)
        queue = [current_node.program]
        while len(queue) != 0:
            current = queue.pop()
            for submod, functionclass in current.submodules.items():
                if issubclass(type(functionclass), dsl.HeuristicNeuralFunction): #TODO start with all neural fns?
                    replacement_candidates = self.construct_candidates(functionclass.input_type,
                                                                   functionclass.output_type,
                                                                   functionclass.input_size,
                                                                   functionclass.output_size,
                                                                   child_num_units)
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    for child_candidate in replacement_candidates:
                        # replace the neural function with a candidate
                        current.submodules[submod] = child_candidate
                        # create the correct child node
                        child_node = copy.deepcopy(current_node)
                        child_node.depth = child_depth
                        # check if child program can be completed within max_depth
                        if child_node.depth + self.min_depth2go(child_node.program) > self.max_depth:
                            continue
                        # if yes, compute costs and add to list of children
                        child_node.cost = current_node.cost + self.compute_edge_cost(child_candidate)
                        all_children.append(child_node)
                        # if len(all_children) >= self.max_num_children and not in_enumeration: #todo what does in_enumeation mean
                        #todo how do we make sure that at the max depth, the programs are all good? or like when do we stop using neural functions
                            # return all_children
                    # once we've copied it, set current back to the original current
                    current.submodules[submod] = orig_fclass
                    if not in_enumeration:
                        return all_children
                else:
                    #add submodules
                    queue.append(functionclass)
        return all_children

    def construct_specific_candidate(self, functionclass, input_type, output_type, input_size, output_size, num_units):
        if issubclass(functionclass, dsl.ITE): #todo maybe here? yeah based on the way theyre inialized, e.g. multiply function contains init_neural_function
            candidate = functionclass(input_type, output_type, input_size, output_size, num_units, beta=self.ite_beta)
        else:
            candidate = functionclass(input_size, output_size, num_units)
        return candidate

    def get_bball_prog2(self, current_node):
        queue_fns = [dsl.running_averages.RunningAverageWindow13Function,dsl.MultiplyFunction,dsl.MultiplyFunction,
                        dsl.basketball.BBallOffenseBallDistSelection,dsl.basketball.BBallOffenseBhDistSelection,
                        dsl.basketball.BBallDefenseBhDistSelection]
                        
        # Window13Avg(Multiply(Multiply(Multiply(Multiply(Multiply(BhOneHot,OffenseBhDist), ScreenBhDist),ScreenPaint),BallPaint),DefenseBhDist)
        all_children = []
        
        i = 0
        # curr = current_node
        temp_fnclass = queue_fns[i]
        for submod, functionclass in current_node.program.submodules.items():
            child_num_units = self.min_num_units
            child_candidate = self.construct_specific_candidate(temp_fnclass, functionclass.input_type,
                                                                    functionclass.output_type,
                                                                    functionclass.input_size,
                                                                    functionclass.output_size,
                                                                    child_num_units)

            current_node.program.submodules['program'] = child_candidate #add window13avg

            print(functionclass)
            print(functionclass.input_size)
            print(functionclass.output_size)
        # curr = child_candidate
        # functionclass = type(current_node.program.submodules['program'])

        i += 1
        temp_fnclass = queue_fns[i]
        prev_submods = []
        for submod, functionclass in current_node.program.submodules['program'].submodules.items():
            
            child_candidate = self.construct_specific_candidate(temp_fnclass, functionclass.input_type,
                                                                    functionclass.output_type,
                                                                    functionclass.input_size,
                                                                    functionclass.output_size,
                                                                    child_num_units)
            current_node.program.submodules['program'].submodules[submod] = child_candidate #add window13avg
            prev_submods.append(submod)
            print(functionclass)
            print(functionclass.input_size)
            print(functionclass.output_size)
        new_submods = {}
        for p in prev_submods:
            new_submods[p] = current_node.program.submodules['program'].submodules[p].submodules.keys()
            for submod, functionclass in current_node.program.submodules['program'].submodules[p].submodules.items():
                i += 1
                temp_fnclass = queue_fns[i]
                child_candidate = self.construct_specific_candidate(temp_fnclass, functionclass.input_type,
                                                                        functionclass.output_type,
                                                                        functionclass.input_size,
                                                                        functionclass.output_size,
                                                                        child_num_units)
                # if child_
                current_node.program.submodules['program'].submodules[p].submodules[submod] = child_candidate
                print(functionclass)
                print(functionclass.input_size)
                print(functionclass.output_size)
        # prev_submods = new_submods 
        # new_submods = []
        for p in prev_submods:
            for s in new_submods[p]:
                for submod, functionclass in current_node.program.submodules['program'].submodules[p].submodules[s].submodules.items():
                    i += 1
                    temp_fnclass = queue_fns[i]
                    child_candidate = self.construct_specific_candidate(temp_fnclass, functionclass.input_type,
                                                                            functionclass.output_type,
                                                                            functionclass.input_size,
                                                                            functionclass.output_size,
                                                                            child_num_units)
                    current_node.program.submodules['program'].submodules[p].submodules[s].submodules[submod] = child_candidate
                    print(functionclass)
                    print(functionclass.input_size)
                    print(functionclass.output_size)
        return [current_node]

    def get_bball_prog(self, current_node):

        #construct functions
        queue_fns = [dsl.MapPrefixesFunction,dsl.running_averages.RunningAverageWindow13Function,dsl.MultiplyFunction,dsl.MultiplyFunction,
                        dsl.basketball.BBallOffenseBallDistSelection,dsl.basketball.BBallOffenseBhDistSelection,
                        dsl.basketball.BBallDefenseBhDistSelection]
        input_types = ["list","list","atom","atom","atom","atom","atom","atom","atom","atom","atom","atom","atom","atom","atom"]
        output_types = ["list"]
        output_types.extend(["atom"] * len(queue_fns))
        output_types.append
        child_num_units = self.min_num_units
        all_children = []
        for i,f in enumerate(queue_fns):
            child_candidate = self.construct_specific_candidate(f, input_types[i],
                                                                    output_types[i],
                                                                    51,
                                                                    2,
                                                                    child_num_units)
            print(child_candidate)
            all_children.append(child_candidate)
            for submod, functionclass in child_candidate.submodules.items():
                print(submod)
        running_ind = 1

        #populate submodules
        for f in all_children:
            for submod, functionclass in f.submodules.items():
                # print(submod)
                f.submodules[submod] = all_children[running_ind]
                running_ind += 1

        current_node.program.submodules['program'] = all_children[0]

        return current_node
