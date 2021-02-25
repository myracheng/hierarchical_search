python3.8 train.py \
--algorithm astar-near \
--exp_name ballscreen_og \
--trial 1 \
--train_data data/helpers/allskip5/train_raw_trajs.npy \
--valid_data data/helpers/allskip5/test_raw_trajs.npy \
--test_data data/helpers/allskip5/test_raw_trajs.npy \
--train_labels data/helpers/allskip5/train_ballscreens.npy \
--valid_labels data/helpers/allskip5/test_ballscreens.npy \
--test_labels data/helpers/allskip5/test_ballscreens.npy \
--input_type "list" \
--output_type "list" \
--input_size 22 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 8 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 6 \
--symbolic_epochs 15 \
--penalty 0.01 \
--max_num_units 16 \
--min_num_units 4 \
--class_weights "0.1,0.9"

# ballhandler
python3.8 train.py \
--algorithm astar-near \
--exp_name ballhandler \
--trial 1 \
--train_data data/helpers/allskip5/train_raw_trajs.npy \
--valid_data data/helpers/allskip5/test_raw_trajs.npy \
--test_data data/helpers/allskip5/test_raw_trajs.npy \
--train_labels data/helpers/allskip5/train_ballhandlers.npy \
--valid_labels data/helpers/allskip5/test_ballhandlers.npy \
--test_labels data/helpers/allskip5/test_ballhandlers.npy \
--input_type "list" \
--output_type "list" \
--input_size 22 \
--output_size 6 \
--num_labels 6 \
--lossfxn "crossentropy" \
--max_depth 8 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 6 \
--symbolic_epochs 15 \
--penalty 0.01 \
--max_num_units 16 \
--min_num_units 4

python3.8 train.py \
--algorithm astar-near \
--exp_name basketball \
--trial 1 \
--train_data data/helpers/allskip5/train_raw_trajs.npy \
--valid_data data/helpers/allskip5/test_raw_trajs.npy \
--test_data data/helpers/allskip5/test_raw_trajs.npy \
--train_labels data/helpers/allskip5/train_ballscreens.npy \
--valid_labels data/helpers/allskip5/test_ballscreens.npy \
--test_labels data/helpers/allskip5/test_ballscreens.npy \
--input_type "list" \
--output_type "list" \
--input_size 22 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 10 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 4 \
--symbolic_epochs 4 \
--class_weights "1.0,1.5"

python3 train.py \
--algorithm iddfs-near \
--exp_name crim13 \
--trial 1 \
--train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 10 \
--frontier_capacity 8 \
--initial_depth 5 \
--performance_multiplier 0.975 \
--depth_bias 0.95 \
--learning_rate 0.001 \
--neural_epochs 6 \
--symbolic_epochs 15 \
--class_weights "1.0,1.5"

python3 train.py \
--algorithm mc-sampling \
--exp_name crim13 \
--trial 1 \
--train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 10 \
--num_mc_samples 50 \
--learning_rate 0.001 \
--symbolic_epochs 15 \
--class_weights "1.0,1.5"

python3 train.py \
--algorithm enumeration \
--exp_name crim13 \
--trial 1 \
--train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 5 \
--max_num_programs 300 \
--learning_rate 0.001 \
--symbolic_epochs 15 \
--class_weights "1.0,1.5"

python3 train.py \
--algorithm enumeration \
--exp_name crim13 \
--trial 1 \
--train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 5 \
--max_num_programs 300 \
--learning_rate 0.001 \
--symbolic_epochs 15 \
--class_weights "1.0,1.5"

python3 train.py \
--algorithm genetic \
--exp_name crim13 \
--trial 1 \
--train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 10 \
--population_size 15 \
--selection_size 8 \
--num_gens 20 \
--total_eval 100 \
--mutation_prob 0.1 \
--learning_rate 0.001 \
--max_enum_depth 5 \
--symbolic_epochs 15 \
--class_weights "1.0,1.5"

python3 train.py \
--algorithm rnn \
--exp_name crim13 \
--trial 1 \
--train_data data/crim13_processed/train_crim13_data.npy \
--valid_data data/crim13_processed/val_crim13_data.npy \
--test_data data/crim13_processed/test_crim13_data.npy \
--train_labels data/crim13_processed/train_crim13_labels.npy \
--valid_labels data/crim13_processed/val_crim13_labels.npy \
--test_labels data/crim13_processed/test_crim13_labels.npy \
--input_type "list" \
--output_type "list" \
--input_size 19 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--neural_epochs 50 \
--learning_rate 0.001 \
--max_num_units 100 \
--class_weights "1.0,1.5"
