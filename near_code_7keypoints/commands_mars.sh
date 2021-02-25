python3.8 train_mars.py \
--algorithm enumeration \
--exp_name mars_baby_an \
--trial 1 \
--train_data data/MARS_data/mars_all_features_train_1.npz,data/MARS_data/mars_all_features_train_2.npz \
--valid_data data/MARS_data/mars_all_features_val.npz \
--test_data data/MARS_data/mars_all_features_test.npz \
--label "sniff" \
--input_type "list" \
--output_type "list" \
--input_size 316 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 14 \
--max_num_units 16 \
--min_num_units 6 \
--max_num_children 6 \
--learning_rate 0.001 \
--neural_epochs 8 \
--symbolic_epochs 10 \
--class_weights "0.3,0.7" \
--max_num_programs 3 \
--batch_size 128 \
--penalty 0

python3.8 train_mars.py \
--algorithm astar-near \
--exp_name mars_an \
--trial 2 \
--train_data data/MARS_data/mars_all_features_train_1.npz,data/MARS_data/mars_all_features_train_2.npz \
--valid_data data/MARS_data/mars_all_features_val.npz \
--test_data data/MARS_data/mars_all_features_test.npz \
--label "sniff" \
--input_type "list" \
--output_type "list" \
--input_size 316 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 6 \
--max_num_units 16 \
--min_num_units 6 \
--learning_rate 0.0005 \
--neural_epochs 8 \
--symbolic_epochs 15 \
--class_weights "1.0,1.0"

python3.8 train_mars.py \
--algorithm astar-near \
--exp_name mars_an \
--trial 3 \
--train_data data/MARS_data/mars_all_features_train_1.npz,data/MARS_data/mars_all_features_train_2.npz \
--valid_data data/MARS_data/mars_all_features_val.npz \
--test_data data/MARS_data/mars_all_features_test.npz \
--label "sniff" \
--input_type "list" \
--output_type "list" \
--input_size 316 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 6 \
--max_num_units 16 \
--min_num_units 6 \
--learning_rate 0.0005 \
--neural_epochs 8 \
--symbolic_epochs 15 \
--class_weights "1.0,1.0"

#python3.8 train_mars.py \
#--algorithm astar-near \
#--exp_name mars_e \
#--trial 2 \
#--train_data data/MARS_data/mars_all_features_train_1.npz,data/MARS_data/mars_all_features_train_2.npz \
#--valid_data data/MARS_data/mars_all_features_val.npz \
#--test_data data/MARS_data/mars_all_features_test.npz \
#--label "sniff" \
#--input_type "list" \
#--output_type "list" \
#--input_size 316 \
#--output_size 2 \
#--num_labels 1 \
#--lossfxn "crossentropy" \
#--normalize \
#--max_depth 10 \
#--frontier_capacity 8 \
#--max_num_units 16 \
#--min_num_units 6 \
#--max_num_children 6 \
#--learning_rate 0.0005 \
#--neural_epochs 10 \
#--symbolic_epochs 15 \
#--class_weights "1.0,1.0"

#python3.8 train_mars.py \
#--algorithm iddfs-near \
#--exp_name mars_e \
#--trial 1 \
#--train_data data/MARS_data/mars_all_features_train_1.npz,data/MARS_data/mars_all_features_train_2.npz \
#--valid_data data/MARS_data/mars_all_features_val.npz \
#--test_data data/MARS_data/mars_all_features_test.npz \
#--label "sniff" \
#--input_type "list" \
#--output_type "list" \
#--input_size 316 \
#--output_size 2 \
#--num_labels 1 \
#--lossfxn "crossentropy" \
#--normalize \
#--max_depth 8 \
#--frontier_capacity 8 \
#--max_num_units 16 \
#--min_num_units 6 \
#--max_num_children 6 \
#--initial_depth 5 \
#--performance_multiplier 0.975 \
#--depth_bias 0.95 \
#--learning_rate 0.001 \
#--neural_epochs 6 \
#--symbolic_epochs 15 \
#--class_weights "1.0,1.0"

#python3.8 train_mars.py \
#--algorithm rnn \
#--exp_name mars \
#--trial 1 \
#--train_data data/MARS_data/mars_all_features_train_1.npz,data/MARS_data/mars_all_features_train_2.npz \
#--valid_data data/MARS_data/mars_all_features_val.npz \
#--test_data data/MARS_data/mars_all_features_test.npz \
#--label "sniff" \
#--input_type "list" \
#--output_type "list" \
#--normalize \
#--input_size 316 \
#--output_size 2 \
#--num_labels 1 \
#--lossfxn "crossentropy" \
#--neural_epochs 50 \
#--learning_rate 0.001 \
#--max_num_units 100 \
#--class_weights "1.0,1.0"

python3 mars_search.py \
--algorithm astar-near \
--exp_name mars_an \
--trial 1 \
--train_data ../near_code_7keypoints/data/MARS_data/mars_all_features_train_1.npz,../near_code_7keypoints/data/MARS_data/mars_all_features_train_2.npz \
--valid_data ../near_code_7keypoints/data/MARS_data/mars_all_features_val.npz \
--test_data ../near_code_7keypoints/data/MARS_data/mars_all_features_test.npz \
--label "sniff" \
--input_type "list" \
--output_type "list" \
--input_size 316 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--normalize \
--max_depth 6 \
--max_num_units 16 \
--min_num_units 6 \
--max_num_children 6 \
--learning_rate 0.0005 \
--neural_epochs 8 \
--symbolic_epochs 15 \
--class_weights "1.0,1.0"
