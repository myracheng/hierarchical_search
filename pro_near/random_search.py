"""
python3 random_search.py --algorithm astar-near --exp_name crim13 --trial 1 --train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy --test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels_other.npy --valid_labels data/crim13_processed/val_crim13_labels_other.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy --input_type "list" --output_type "list" \
--input_size 19 --output_size 2 --num_labels 1 --lossfxn "crossentropy" --max_depth 3 --frontier_capacity 8 \
--learning_rate 0.001 --neural_epochs 6 --symbolic_epochs 15 --class_weights "0.1,0.9" --base_program_name results/crim13_astar-near_1_1604688497/fullprogram --hole_node_ind 2 --penalty 0 --eval True
"""

"""
Sample command:
cd pronear/pro_near

python3.8 random_search.py --algorithm astar-near --exp_name bball --trial 1 \
--train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz \
--valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz --test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz \
--train_labels "mount" --input_type "list" --output_type "list" --input_size 316 --output_size 2 --num_labels 1 --lossfxn "crossentropy" \
--normalize --max_depth 3 --max_num_units 16 --min_num_units 6 --max_num_children 12 --learning_rate 0.001 --neural_epochs 8 --symbolic_epochs 15 \
--class_weights "0.03,0.97" --base_program_name data/7keypoints/astar_1 --hole_node_ind 3 --penalty 0 --eval True

cd pronear/pro_near;
python3.8 random_search.py --algorithm astar-near --exp_name mars_an --trial 1 \
--train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz \
--valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz --test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz \
--train_labels "sniff" --input_type "list" --output_type "list" --input_size 316 --output_size 2 --num_labels 1 --lossfxn "crossentropy" \
--normalize --max_depth 4 --max_num_units 4 --min_num_units 4 --max_num_children 10 --learning_rate 0.001 --neural_epochs 6 --symbolic_epochs 6 \
--class_weights "0.3,0.7" --base_program_name results/mars_an_astar-near_1_882748/fullprogram --hole_node_ind -1 --batch_size 128

cd pronear/pro_near;
CUDA_VISIBLE_DEVICES=1 
"""
import argparse
import csv
import os
from cpu_unpickle import CPU_Unpickler, traverse
import pickle
import torch
import glob
import torch.nn as nn
from pprint import pprint
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from data import normalize_data, MyDataset
from datetime import datetime
	
import time
from algorithms import ASTAR_NEAR, IDDFS_NEAR, MC_SAMPLING, ENUMERATION, GENETIC, RNN_BASELINE
from program_graph import ProgramGraph
from utils.data import *
from utils.evaluation import label_correctness
from utils.logging import init_logging, print_program_dict, log_and_print,print_program
import dsl
from utils.training import change_key

### random seed
os.environ['PYTHONHASHSEED']='0'
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    # Args for experiment setup
    parser.add_argument('-t', '--trial', type=int, required=True,
                        help="trial ID")
    parser.add_argument('--exp_name', type=str, required=True,
                        help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # Args for data
    parser.add_argument('--train_data', type=str, required=True,
                        help="path to train data")
    parser.add_argument('--test_data', type=str, required=True,
                        help="path to test data")
    parser.add_argument('--valid_data', type=str, required=False, default=None,
                        help="path to val data. if this is not provided, we sample val from train.")
    parser.add_argument('--train_labels', type=str, required=False,
                        help="path to train labels")
    parser.add_argument('--test_labels', type=str, required=False,
                        help="path to test labels")
    parser.add_argument('--valid_labels', type=str, required=False, default=None,
                        help="path to val labels. if this is not provided, we sample val from train.")
    parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
                        help="input type of data")
    parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
                        help="output type of data")
    parser.add_argument('--input_size', type=int, required=True,
                        help="dimenion of features of each frame")
    parser.add_argument('--output_size', type=int, required=True,
                        help="dimension of output of each frame (usually equal to num_labels")
    parser.add_argument('--num_labels', type=int, required=True,
                        help="number of class labels")

    parser.add_argument('--neural_units', type=int, required=False, default=100,
                        help="max number of hidden units for neural programs")
    

    # Args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="min number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.01,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")

    # Args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=0.8,
                        help="split training set for validation." +
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50,
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=6,
                        help="training epochs for symbolic programs")
    parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits"],
                        help="loss function for training")
    parser.add_argument('--class_weights', type=str, required=False, default=None,
                        help="weights for each class in the loss function, comma separated floats")
    parser.add_argument('--num_iter', type=int, required=False, default=10,
                        help="number of iterations for propel")
    parser.add_argument('--num_f_epochs', type=int, required=False, default=100,
                        help="length of training for the neural model")
    # Args for algorithms
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=["mc-sampling", "enumeration",
                            "genetic", "astar-near", "iddfs-near", "rnn"],
                        help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR" +
                        " (>1.0 prunes aggressively in this case)")
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")

    # parser.add_argument('--from_saved', type=bool, required=False, default=False,
    #                     help="load model from saved")
    
    parser.add_argument('--exp_id', type=int, required=False, default=None, help="experiment id")
    parser.add_argument('--base_program_name', type=str, required=False, default="program_ite",
                        help="name of original program")
    parser.add_argument('--hole_node_ind', type=int, required=False, default="-1",
                        help="which node to replace")
    parser.add_argument('--eval', type=bool, required=False, default=False,
                        help="only run evaluation")
    parser.add_argument('--neurh', type=bool, required=False, default=False,
                        help="only run neural heuristics")
    # parser.add_argument()
    return parser.parse_args()

class Subtree_search():

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print(self.device)
        else:
            self.device = 'cpu'
        
        self.loss_weight = torch.tensor([float(w) for w in self.class_weights.split(',')]).to(self.device)
        if self.exp_name == 'crim13' or self.exp_name =='bball':
            # load input data
            self.train_data = np.load(self.train_data)
            self.test_data = np.load(self.test_data)
            self.valid_data = None
            self.train_labels = np.load(self.train_labels)
            self.test_labels = np.load(self.test_labels)
            self.valid_labels = None
            assert self.train_data.shape[-1] == self.test_data.shape[-1] == self.input_size
            if self.valid_data is not None and self.valid_labels is not None:
                self.valid_data = np.load(self.valid_data)
                self.valid_labels = np.load(self.valid_labels)
                assert valid_data.shape[-1] == self.input_size

            self.batched_trainset, self.validset, self.testset = prepare_datasets(self.train_data, self.valid_data, self.test_data, self.train_labels, self.valid_labels, 
            self.test_labels, normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)
        elif self.exp_name == 'mars_an':
            #### start mars
            
            train_datasets = self.train_data.split(",")
            train_raw_features = []
            train_raw_annotations = []
            for fname in train_datasets:
                data = np.load(fname, allow_pickle=True)
                train_raw_features.extend(data["features"])
                train_raw_annotations.extend(data["annotations"])
            test_data = np.load(self.test_data, allow_pickle=True)

            test_raw_features = test_data["features"]
            test_raw_annotations = test_data["annotations"]
            valid_raw_features = None
            valid_raw_annotations = None
            valid_labels = None
            # Check the # of features of the first frame of the first video
            assert len(train_raw_features[0][0]) == len(test_raw_features[0][0]) == self.input_size

            if self.valid_data is not None:
                valid_data = np.load(self.valid_data, allow_pickle=True)
                valid_raw_features = valid_data["features"]
                valid_raw_annotations = valid_data["annotations"]
                assert len(valid_raw_features[0][0]) == self.input_size

            behave_dict = read_into_dict('../near_code_7keypoints/data/MARS_data/behavior_assignments_3class.txt')
            # Reshape the data to trajectories of length 100
            train_features, train_labels = preprocess(train_raw_features, train_raw_annotations, self.train_labels, behave_dict)
            test_features, test_labels = preprocess(test_raw_features, test_raw_annotations, self.train_labels, behave_dict)
            if valid_raw_features is not None and valid_raw_annotations is not None:
                valid_features, valid_labels = preprocess(valid_raw_features, valid_raw_annotations, self.train_labels, behave_dict)
            self.batched_trainset, self.validset, self.testset  = prepare_datasets(train_features, valid_features, test_features,
                                            train_labels, valid_labels, test_labels,
                                    normalize=self.normalize, train_valid_split=self.train_valid_split, batch_size=self.batch_size)

                            ##### END MARS
        else:
            log_and_print('bad experiment name')
            return
        
        
        # self.fix()

        # add subprogram in
        # if self.device == 'cpu':
        #     self.base_program = CPU_Unpickler(open("%s/subprogram.p" % self.base_program_name, "rb")).load()
        # else:
        #     self.base_program = pickle.load(open("%s/subprogram.p" % self.base_program_name, "rb"))
        if self.device == 'cpu':
            self.base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            self.base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))
            # ['program']
        print(self.base_program)
        
        base_folder = os.path.dirname(self.base_program_name)
        # self.weights_dict = np.load(os.path.join(base_folder,'weights.npy'), allow_pickle=True).item()
        
        
        data = self.base_program.submodules
        l = []
        traverse(data,l)
        log_and_print(l)
        # if self.hole_node_ind < 0:
            # self.hole_node_ind = len(l) + self.hole_node_ind
        #if negative, make it positive
        self.hole_node_ind %= len(l)

        self.hole_node = l[self.hole_node_ind]
        

        #for near on subtree
        self.curr_iter = 0
        self.program_path = None 


        if self.exp_id is not None:
            self.trial = self.exp_id
        if self.eval:
            self.evaluate()
        else:
            now = datetime.now()
            self.timestamp = str(datetime.timestamp(now)).split('.')[0][4:]
            log_and_print(self.timestamp)
            full_exp_name = "{}_{}_{}_{}".format(
            self.exp_name, self.algorithm, self.trial, self.timestamp) #unique timestamp for each near run
            self.save_path = os.path.join(self.save_dir, full_exp_name)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            init_logging(self.save_path)
            if self.neurh:

                log_and_print(self.base_program_name)
                self.neural_h()
            else:
                self.run_near()
                self.evaluate_final()

    def fix(self):
        self.real_base =  'results/mars_an_astar-near_1_1605057595/fullprogram'
        if self.device == 'cpu':
            base_program = CPU_Unpickler(open("%s.p" % self.real_base, "rb")).load()
        else:
            base_program = pickle.load(open("%s.p" % self.real_base, "rb"))

        if self.device == 'cpu':
            best_program = CPU_Unpickler(open("%s/subprogram.p" % self.base_program_name, "rb")).load()
        else:
            best_program = pickle.load(open("%s/subprogram.p" % self.base_program_name, "rb"))


        self.full_path = os.path.join(self.base_program_name, "fullprogram.p") #fix this
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(base_program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
        curr_level = 0
        l = []
        traverse(base_program.submodules,l)
        curr_program = base_program.submodules
        change_key(base_program.submodules, [], self.hole_node_ind, best_program.submodules["program"])
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(base_program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
        log_and_print(str(additional_params))
        pickle.dump(base_program, open(self.full_path, "wb"))


        # load results/mars_an_astar-near_1_1605057595/fullprogram
        # load base_1605023425program . subprogram
        # change key

    def evaluate_final(self):
        if self.device == 'cpu':
            program = CPU_Unpickler(open(self.full_path, "rb").load())
        else:
            program = pickle.load(open(self.full_path, "rb"))
        log_and_print(print_program(program, ignore_constants=True))
        l = []
        traverse(program.submodules,l)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}".format(1 - metric))

    def evaluate(self):

        if self.device == 'cpu':
            program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            program = pickle.load(open("%s.p" % self.base_program_name, "rb"))
        # program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        print(print_program(program, ignore_constants=True))
        l = []
        traverse(program.submodules,l)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("F1 score achieved is {:.4f}\n".format(1 - metric))
        log_and_print(str(additional_params))
        
    def evaluate_neurosymb(self, program):
        # if self.device == 'cpu':
        #     program = CPU_Unpickler(open("neursym.p" % self.base_program_name, "rb")).load()
        # else:
        #     program = pickle.load(open("neursym.p" % self.base_program_name, "rb"))
        # # program= CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        print(print_program(program, ignore_constants=True))
        l = []
        traverse(program.submodules,l)
        with torch.no_grad():
            test_input, test_output = map(list, zip(*self.testset))
            true_vals = torch.flatten(torch.stack(test_output)).float().to(self.device)	
            predicted_vals = self.process_batch(program, test_input, self.output_type, self.output_size, self.device)
            
            metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=self.num_labels)
        log_and_print("Test F1 score achieved is {:.4f}\n".format(1 - metric))
        log_and_print(str(additional_params))
        return 1- metric

    def neural_h(self):
        data = self.base_program.submodules
        l = [] #populate AST
        traverse(data,l)
        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(weight=self.loss_weight), #todo
            'evalfxn' : label_correctness,
            'num_labels' : self.num_labels
        }

        scores = [self.base_program_name]
        for hole_node_ind in range(len(l)):

            hole_node = l[hole_node_ind]
            near_input_type = hole_node[0].input_type
            near_output_type = hole_node[0].output_type
            near_input_size = hole_node[0].input_size
            near_output_size = hole_node[0].output_size

            # Initialize program graph starting from trained NN
            program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS, near_input_type, near_output_type, near_input_size, near_output_size,
                self.max_num_units, self.min_num_units, self.max_num_children, 0, self.penalty, ite_beta=self.ite_beta) ## max_depth 0

            # Initialize algorithm
            algorithm = ASTAR_NEAR(frontier_capacity=0)
            score, new_prog, losses = algorithm.run_init(self.timestamp, self.base_program_name, hole_node_ind,
                program_graph, self.batched_trainset, self.validset, train_config, self.device)
            subprogram_str = print_program(hole_node[0])
            test_score = self.evaluate_neurosymb(new_prog)
            stats_arr = [subprogram_str, hole_node[1], score,test_score]
            stats_arr.extend(losses)
            scores.append(stats_arr)
            # scores.append()
            # h_file = os.path.join(self.save_path, "neursym_%d.p"%hole_node_ind)
            # pickle.dump(new_prog, open(h_file, "wb"))

        h_file = os.path.join(self.save_path, "neurh.csv")
        with open(h_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(scores)

    def run_near(self): 
        # print(self.device)
        train_config = {
            'lr' : self.learning_rate,
            'neural_epochs' : self.neural_epochs,
            'symbolic_epochs' : self.symbolic_epochs,
            'optimizer' : optim.Adam,
            'lossfxn' : nn.CrossEntropyLoss(weight=self.loss_weight), #todo
            'evalfxn' : label_correctness,
            'num_labels' : self.num_labels
        }


        near_input_type = self.hole_node[0].input_type
        near_output_type = self.hole_node[0].output_type
        near_input_size = self.hole_node[0].input_size
        print(near_input_size)
        near_output_size = self.hole_node[0].output_size
        

        # Initialize program graph starting from trained NN
        program_graph = ProgramGraph(DSL_DICT, CUSTOM_EDGE_COSTS, near_input_type, near_output_type, near_input_size, near_output_size,
            self.max_num_units, self.min_num_units, self.max_num_children, self.max_depth, self.penalty, ite_beta=self.ite_beta)

        # Initialize algorithm
        algorithm = ASTAR_NEAR(frontier_capacity=self.frontier_capacity)
        best_programs = algorithm.run(self.timestamp, self.base_program_name, self.hole_node_ind,
            program_graph, self.batched_trainset, self.validset, train_config, self.device)
        best_program_str = []
        if self.algorithm == "rnn":
            # special case for RNN baseline
            best_program = best_programs
        else:
            # Print all best programs found
            log_and_print("\n")
            log_and_print("BEST programs found:")
            for item in best_programs:
                program_struct = print_program(item["program"], ignore_constants=True)
                program_info = "struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
                    item["struct_cost"], item["score"], item["path_cost"], item["time"])
                best_program_str.append((program_struct, program_info))
                print_program_dict(item)
            best_program = best_programs[-1]["program"]

        
        # Save best programs
        f = open(os.path.join(self.save_path, "best_programs.txt"),"w")
        f.write( str(best_program_str) )
        f.close()

        self.program_path = os.path.join(self.save_path, "subprogram.p")
        pickle.dump(best_program, open(self.program_path, "wb"))

        self.full_path = os.path.join(self.save_path, "fullprogram.p")

        if self.device == 'cpu':
            base_program = CPU_Unpickler(open("%s.p" % self.base_program_name, "rb")).load()
        else:
            base_program = pickle.load(open("%s.p" % self.base_program_name, "rb"))

        curr_level = 0
        l = []
        traverse(base_program.submodules,l)
        curr_program = base_program.submodules
        change_key(base_program.submodules, [], self.hole_node_ind, best_program.submodules["program"])
        pickle.dump(base_program, open(self.full_path, "wb"))


        # Save parameters
        f = open(os.path.join(self.save_path, "parameters.txt"),"w")

        parameters = ['input_type', 'output_type', 'input_size', 'output_size', 'num_labels', 'neural_units', 'max_num_units', 
            'min_num_units', 'max_num_children', 'max_depth', 'penalty', 'ite_beta', 'train_valid_split', 'normalize', 'batch_size', 
            'learning_rate', 'neural_epochs', 'symbolic_epochs', 'lossfxn', 'class_weights', 'num_iter', 'num_f_epochs', 'algorithm', 
            'frontier_capacity', 'initial_depth', 'performance_multiplier', 'depth_bias', 'exponent_bias', 'num_mc_samples', 'max_num_programs', 
            'population_size', 'selection_size', 'num_gens', 'total_eval', 'mutation_prob', 'max_enum_depth', 'exp_id', 'base_program_name', 'hole_node_ind']
        for p in parameters:
            f.write( p + ': ' + str(self.__dict__[p]) + '\n' )
        f.close()

    def process_batch(self, program, batch, output_type, output_size, device='cpu'):
        # batch_input = [torch.tensor(traj) for traj in batch]
        batch_input = torch.tensor(batch)
        batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
        batch_padded = batch_padded.to(device)
        # out_padded = program(batch_padded)
        out_padded = program.execute_on_batch(batch_padded, batch_lens)
        if output_type == "list":
            out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(program.output_type=='atom'))
        else:
            out_unpadded = out_padded
        if output_size == 1 or output_type == "list":
            return flatten_tensor(out_unpadded).squeeze()
        else:
            if isinstance(out_unpadded, list):
                out_unpadded = torch.cat(out_unpadded, dim=0).to(device)          
            return out_unpadded

def preprocess(features, annotations, action, dct):
    new_features = []
    new_labels = []
    # Each feature array is (# of frames, # of features) dimensional
    for feat_arr, annot_arr in zip(features, annotations):
        length = len(feat_arr) - len(feat_arr) % 100
        # Instead of taking all the features, we only select the ones that are included in at least one feature subset
        feat_arr = np.take(feat_arr[:length,:], MARS_INDICES, axis = 1)
        feat_arr = feat_arr.reshape(-1, 100, len(MARS_INDICES))
        # if (np.isnan(np.sum(feat_arr))):
        #     print("F")
        feat_arr = np.where(np.logical_or(np.isnan(feat_arr), feat_arr == inf), 0, feat_arr).astype('float64')
        annot_arr = annot_arr[:length]

        label_arr = [1 if dct[x] == action else 0 for x in annot_arr]
        # print(label_arr)
        label_arr = np.asarray(label_arr).reshape(-1, 100).astype('float64')
        new_features.append(feat_arr)
        new_labels.append(label_arr)
    return np.concatenate(new_features, axis=0), np.concatenate(new_labels, axis=0)

if __name__ == '__main__':
    args = parse_args()
    if args.exp_name == 'crim13':
        from dsl_crim13 import DSL_DICT, CUSTOM_EDGE_COSTS #todo change this import based on type

    elif 'bball' in args.exp_name:
        from dsl_bball_47 import DSL_DICT, CUSTOM_EDGE_COSTS
    
    elif args.exp_name == 'mars_an':
        from dsl_mars import DSL_DICT, CUSTOM_EDGE_COSTS
        from dsl.mars import MARS_INDICES
        from mars_search import read_into_dict
    search_instance = Subtree_search(**vars(args))
