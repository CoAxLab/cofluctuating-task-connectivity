#!/bin/bash
#
# This bash script runs different experiments, where things like parcellations and
# other methodological choices are changed. It calls the  python script "run_experiment.py",
# intended to run one particular experiment given an atlas and a config file.

case_file="../data/experiments/pipeline_main.yaml"
python run_experiment.py --config_file $case_file

for i in {2..9}
do
	case_file="../data/experiments/pipeline${i}.yaml"
	python run_experiment.py --config_file $case_file
done	

# Case including task events in connectivity matrices
case_file="../data/experiments/pipeline_edge_w_tasks.yaml"
python run_experiment.py --config_file $case_file

# Main case, but with different atlases
case_file="../data/experiments/pipeline_main.yaml"
python run_experiment.py --config_file $case_file --atlas schaefer
python run_experiment.py --config_file $case_file --atlas craddock

