# Toward Interpretable Program Repair under Resource Constraints

## Learning a model of programs using graph embeddings
Construct graphs from programs in `pro_near/to_graph_withholes.py`.
Then run `python3.8 prgr_duvenaud.py --datasetPath=../../pronear/pro_near/trees_holes/near_programs/510994`
Different sets of programs are in the divide_datasets function

## near_code
original near code for CRIM 13 and ballscreen

## near_code_7keypoints
near code for MARS

## pro_near
- Iterative hierarchical search in `hierarchical_search.py` 
(or if the experiment is ballscreen, then it will first test out the "ground truth" program)
- Random search or picking a node in `random_search.py`
- 100progs contains 100 depth-3 programs generated by NEAR to be used in elarning the model
- symb_trained pkl
- two_iter_other.p : result of program after running two iterations of hand-tuned algo with labels "other"