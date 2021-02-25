import pickle
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def save_to_tree(d, G):
    # if G is None:
    #     G = nx.DiGraph()
    for key,val in d.items(): 
        G.add_node(val)
        try:
            if val.submodules is not None:
                kids = val.submodules.values()
                for k in kids:
                    G.add_node(k)
                    G.add_edges(val, k)
                traverse(val.submodules,G) 
        except AttributeError:
            continue

def traverse(d,l,level = 0): 
    for key,val in d.items(): 
        # print(val)
        # for submod, functionclass in current_node.program.submodules['program'].submodules.items():
            
        #     child_candidate = self.construct_specific_candidate(temp_fnclass, functionclass.input_type,
        #                                                             functionclass.output_type,
        #                                                             functionclass.input_size,
        #                                                             functionclass.output_size,
        #                                                             child_num_units)
        #     current_node.program.submodules['program'].submodules[submod] = child_candidate #add window13avg
        #     prev_submods.append(submod)
        #     print(functionclass)
        #     print(functionclass.input_size)
        #     print(functionclass.output_size)
        l.append([val, level]) 
        try:
            if val.submodules is not None:
                traverse(val.submodules,l, level+1) 
        except AttributeError:
            continue

def get_options(dsl_dict, node):
    for key,val in dsl_dict.items(): 
        if type(node) in val:

            print(type(node))

def change_key(d, required_key, new_value):
    for k, v in d.items():
        if isinstance(v, dict):
            change_key(v, required_key, new_value)
        if k == required_key:
            d[k] = new_value
