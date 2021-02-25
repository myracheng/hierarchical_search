# GOOD PROGRAMS
"""
Save programs as trees for use in GNNs
"""
from dsl_mars import DSL_DICT, CUSTOM_EDGE_COSTS
from utils.logging import print_program
import copy
from cpu_unpickle import CPU_Unpickler, traverse
from utils.training import change_key
import networkx as nx
import glob
import re
import dsl
from program_graph import ProgramGraph
import os
import pickle
from datetime import datetime

def save_to_tree(d, G):
    for key,val in d.items(): 
        # G.add_node
        G.add_nodes_from([(val, {"props": val})])
        try:
            if val.submodules is not None:
                kids = val.submodules.values()
                for k in kids:
                    G.add_node(k)
                    G.add_edge(val, k)
                save_to_tree(val.submodules,G) 
        except AttributeError:
            continue
            
# root_dir = 'results/' <- for the astar ones
     
count = 0
root_dir = '../../../../../home/mccheng/near_programs/'


now = datetime.now()
timestamp = str(datetime.timestamp(now)).split('.')[0][4:]
  
fn_dict = set()

count = 0
save_dir = os.path.join('trees_holes',root_dir.split('/')[-2], timestamp)
os.makedirs(save_dir)

def add_nodes_to_feature_dict(g):
    for n, d in list(g.nodes(data=True)):
        
        fn_name = type(d['props'])
        fn_dict.add(fn_name)

    return fn_dict

prog_file = open("%s/graphs.txt" % (save_dir),"w") 
label_file = open("%s/labels.txt" % (save_dir),"w") 
prog_str_file = open("%s/progs.txt" % (save_dir),"w") 

# annotator programs
for filename in glob.iglob(root_dir + '**/*.p', recursive=True): #1.p
    # print(filename)
    full_folder = os.path.dirname(filename)
    log_file = os.path.join(full_folder, 'log.txt')
    try:
        ### GET LABEL
        f = open(log_file, "r")
        lines_list = f.readlines()
        lines = "".join(lines_list)
        score = re.findall("achieved is \d+\.\d+",lines)[-1][12:]
        # c = float(lines_list[-1][-7:-1]) #jank error checking for neurh.csv
        # 


        ### PROCESS PROGRAM
        folder = os.path.dirname(filename).split('/')[-1]
        if count > 20: 
            break
        count += 1
    
        base_program = CPU_Unpickler(open(filename, "rb")).load()
        data = base_program.submodules  
        # print(print_program(base_program, ignore_constants=True))
        # if count > 10:
        #     base_program[0] = 0
        #     exit
        # else: 
        #     continue
        G = nx.Graph()
        save_to_tree(data, G)
        # print(G.nodes[0])
        depths = nx.shortest_path_length(G,list(G.nodes)[0])
        max_depth = sorted(list(depths.values()))[-1]
        hole_count = 0
        # print(depths)
        # found_hole = false
        for node, depth in depths.items():
            if hole_count > 0:
                continue
            if (max_depth - depth) >= 1:
                continue  #skip if too high
                #add tree with hole
            H = G.copy()
            attrs = {node: {"props": "hole"}}
            nx.set_node_attributes(H, attrs)
        
            with open('%s/%s_%d.pkl'%(save_dir, folder, hole_count), 'wb') as output:
                pickle.dump(H, output)

            label_file.write("%s\n" % score)

            prog_file.write("%s_%d\n" % (folder,hole_count))


            hole_count += 1
            # add to labels and files
            

    except (FileNotFoundError, ValueError,IndexError) as e:
        #no log, incomplete log, etc
        print(e)


print("num programs: %d" % count)
count = 0
#NEAR GENERATED (BAD)
root_dir = '../../../../../home/mccheng/near_programs2/'

for filename in glob.iglob(root_dir + '**/*1.p', recursive=True): #1.p
    # print(filename)
    full_folder = os.path.dirname(filename)
    log_file = os.path.join(full_folder, 'log.txt')
    try:
        ### GET LABEL
        f = open(log_file, "r")
        lines_list = f.readlines()
        lines = "".join(lines_list)
        score = re.findall("achieved is \d+\.\d+",lines)[-1][12:]
        # c = float(lines_list[-1][-7:-1]) #jank error checking for neurh.csv
        # 


        ### PROCESS PROGRAM
        folder = os.path.dirname(filename).split('/')[-1]
        # if count > 20: 
            # break
        count += 1
    
        base_program = CPU_Unpickler(open(filename, "rb")).load()
        data = base_program.submodules  
        # print(print_program(base_program, ignore_constants=True))
        # if count > 10:
        #     base_program[0] = 0
        #     exit
        # else: 
        #     continue
        G = nx.Graph()
        save_to_tree(data, G)
        # print(G.nodes[0])
        add_nodes_to_feature_dict(G)
        depths = nx.shortest_path_length(G,list(G.nodes)[0])
        max_depth = sorted(list(depths.values()))[-1]
        hole_count = 0
        # print(depths)
        for node, depth in depths.items():
            if hole_count > 0:
                continue
            if (max_depth - depth) >= 1:
                continue  #skip if too high
                #add tree with hole
            H = G.copy()
            attrs = {node: {"props": "hole"}}
            nx.set_node_attributes(H, attrs)
        
            with open('%s/%s_%d.pkl'%(save_dir, folder, hole_count), 'wb') as output:
                pickle.dump(H, output)

            label_file.write("%s\n" % score)

            prog_file.write("%s_%d\n" % (folder,hole_count))

            prog_str_file.write("%s\n" % (print_program(base_program, ignore_constants=True)))



            hole_count += 1
            # add to labels and files
            

    except (FileNotFoundError, ValueError,IndexError) as e:
        #no log, incomplete log, etc
        print(e)


print("num programs: %d" % count)
count = 0
# MARS (GOOD)
root_dir = 'results/'
# for filename in glob.iglob(root_dir + 'crim**/*1.p', recursive=True): #1.p
#     # print(filename)
#     full_folder = os.path.dirname(filename)
#     log_file = os.path.join(full_folder, 'log.txt')
#     try:
#         ### GET LABEL
#         f = open(log_file, "r")
#         lines_list = f.readlines()
#         lines = "".join(lines_list)
#         score = re.findall("achieved is \d+\.\d+",lines)[-1][12:]
#         # c = float(lines_list[-1][-7:-1]) #jank error checking for neurh.csv
#         # 


#         ### PROCESS PROGRAM
#         folder = os.path.dirname(filename).split('/')[-1]
#         # if count > 10: 
#             # break
#         count += 1
    
#         base_program = CPU_Unpickler(open(filename, "rb")).load()
#         data = base_program.submodules  
#         # print(print_program(base_program, ignore_constants=True))
#         # if count > 10:
#         #     base_program[0] = 0
#         #     exit
#         # else: 
#         #     continue
#         G = nx.Graph()
#         save_to_tree(data, G)
#         # print(G.nodes[0])
#         depths = nx.shortest_path_length(G,list(G.nodes)[0])
#         max_depth = sorted(list(depths.values()))[-1]
#         hole_count = 0
#         # print(depths)
#         for node, depth in depths.items():
#             if (max_depth - depth) >= 2:
#                 continue  #skip if too high
#                 #add tree with hole
#             H = G.copy()
#             attrs = {node: {"props": "hole"}}
#             nx.set_node_attributes(H, attrs)
        
#             with open('%s/%s_%d.pkl'%(save_dir, folder, hole_count), 'wb') as output:
#                 pickle.dump(H, output)

#             label_file.write("%s\n" % score)

#             prog_file.write("%s_%d\n" % (folder,hole_count))


#             hole_count += 1
#             # add to labels and files
            

#     except (FileNotFoundError, ValueError,IndexError) as e:
#         #no log, incomplete log, etc
#         print(e)
# # print("crim: %d")

print("MARS mine")
# mars_count = 0
for filename in glob.iglob(root_dir + 'mars**/full*.p', recursive=True): #1.p
    # print(filename)
    if "neursym" in filename:
        continue
    full_folder = os.path.dirname(filename)
    log_file = os.path.join(full_folder, 'log.txt')
    try:
        ### GET LABEL
        f = open(log_file, "r")
        lines_list = f.readlines()
        lines = "".join(lines_list)
        score = re.findall("achieved is \d+\.\d+",lines)[-1][12:]
        # c = float(lines_list[-1][-7:-1]) #jank error checking for neurh.csv
        # 


        ### PROCESS PROGRAM
        folder = os.path.dirname(filename).split('/')[-1]
        # if count > 10: 
            # break
        count += 1
    
        base_program = CPU_Unpickler(open(filename, "rb")).load()
        data = base_program.submodules  
        # if count < 10:
             
        # print(print_program(base_program, ignore_constants=True))
        G = nx.Graph()
        save_to_tree(data, G)
        # print(G.nodes[0])
        depths = nx.shortest_path_length(G,list(G.nodes)[0])
        max_depth = sorted(list(depths.values()))[-1]
        hole_count = 0
        add_nodes_to_feature_dict(G)
        # print(depths)
        # l = []
        # traverse(data,l)
        # print(l)
        for node, depth in depths.items():
            if hole_count > 0:
                continue
            if (max_depth - depth) >= 1:
                continue  #skip if too high

            #add tree with hole
            near_input_type = node.input_type
            near_output_type = node.output_type
            near_input_size = node.input_size
            near_output_size = node.output_size
            
            program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS, near_input_type, near_output_type, near_input_size, near_output_size,
                4,8, 4, 0, 0) ## max_depth 0
            current = program_graph.root_node
            # #get index
            base_program2 = copy.deepcopy(base_program)
            data = base_program2.submodules
            l = [] #populate AST
            hole_node_ind = -1

            traverse(data,l)
            for i,n in enumerate(l):
                # print(n[0])
                if type(n[0]) == type(node) and n[1] == depth:
                    hole_node_ind = i
            if hole_node_ind == -1:
                # print(node)
                # print(l)
                print("not found")
            change_key(data, [], hole_node_ind, current.program.submodules["program"])

            # program_baby = ListToListModule() #this is a big rnn

            H = G.copy()
            attrs = {node: {"props": current.program.submodules["program"]}}
            nx.set_node_attributes(H, attrs)
            
            # add_to_files(H, save_dir, folder, hole_count, label_file, score, base_program)
        
            with open('%s/%s_%d.pkl'%(save_dir, folder, hole_count), 'wb') as output:
                pickle.dump(H, output)

            label_file.write("%s\n" % score)

            prog_file.write("%s_%d\n" % (folder,hole_count))

            prog_str_file.write("%s\n" % (print_program(base_program2, ignore_constants=True)))

            hole_count += 1
            # add to labels and files
            

    except (FileNotFoundError, ValueError,IndexError) as e:
        #no log, incomplete log, etc
        print(e)

#long and short mix

print("num programs: %d" % count)
count = 0
root_dir = '../../near_programs0/'
# print()
for filename in glob.iglob(root_dir + '**/*.p', recursive=True): #1.p
    # print(root_dir)
    # print(filename)
    full_folder = os.path.dirname(filename)
    log_file = os.path.join(full_folder, 'log.txt')
    try:
        ### GET LABEL
        f = open(log_file, "r")
        lines_list = f.readlines()
        lines = "".join(lines_list)
        score = re.findall("achieved is \d+\.\d+",lines)[-1][12:]
        # c = float(lines_list[-1][-7:-1]) #jank error checking for neurh.csv
        # 


        ### PROCESS PROGRAM
        folder = os.path.dirname(filename).split('/')[-1]
        # if count > 20: 
            # break
        count += 1
    
        base_program = CPU_Unpickler(open(filename, "rb")).load()
        data = base_program.submodules  
        # print(print_program(base_program, ignore_constants=True))
        # if count > 10:
        #     base_program[0] = 0
        #     exit
        # else: 
        #     continue
        G = nx.Graph()
        save_to_tree(data, G)
        # print(G.nodes[0])
        add_nodes_to_feature_dict(G)
        depths = nx.shortest_path_length(G,list(G.nodes)[0])
        max_depth = sorted(list(depths.values()))[-1]
        hole_count = 0
        # print(depths)
        for node, depth in depths.items():
            if hole_count > 0:
                continue
            if (max_depth - depth) >= 1:
                continue  #skip if too high
                #add tree with hole
            H = G.copy()
            attrs = {node: {"props": "hole"}}
            nx.set_node_attributes(H, attrs)
        
            with open('%s/%s_%d.pkl'%(save_dir, folder, hole_count), 'wb') as output:
                pickle.dump(H, output)

            label_file.write("%s\n" % score)

            prog_file.write("%s_%d\n" % (folder,hole_count))
            
            prog_str_file.write("%s\n" % (print_program(base_program, ignore_constants=True)))



            hole_count += 1
            # add to labels and files
            

    except (FileNotFoundError, ValueError,IndexError) as e:
        #no log, incomplete log, etc
        print(e)

label_file.close()
prog_file.close()        
print("Processed %d programs" % count)
print(timestamp)
# print(fn_dict)
fn_ind = 0
res = {}
cands = set([dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE,dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE,
                                            dsl.running_averages.RunningAverageLast10Function, dsl.running_averages.RunningAverageWindow13Function,
                                            dsl.running_averages.RunningAverageWindow5Function,dsl.running_averages.RunningAverageWindow7Function,dsl.SimpleITE, dsl.AddFunction, dsl.MultiplyFunction, dsl.MarsAngleHeadBodySelection, \
                    dsl.MarsAxisRatioSelection, dsl.MarsSpeedSelection, dsl.MarsVelocitySelection, 
                    dsl.MarsAccelerationSelection, dsl.MarsResidentTowardIntruderSelection, dsl.MarsRelAngleSelection,
                    dsl.MarsRelDistSelection, dsl.MarsAreaEllipseRatioSelection])
for elem in fn_dict:
    
    if elem in cands:
        res[elem] = fn_ind
        fn_ind += 1
print(res)

with open('%s/feature_dict.pkl'%save_dir, 'wb') as output:
    pickle.dump(res, output)