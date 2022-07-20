#!/bin/bash

echo "computing first-level activation maps at ROI level"
python 02-first_level/021-first_level_node_roi.py

echo "computing first-level connectivity maps"
python 02-first_level/022-first_level_edge.py

echo "computing first-level connectivity maps without gsr in edge time series"
python 02-first_level/023-first_level_edge_nogsr.py

echo "computing first-level connectivity maps with task effects included in edge time series"
python 02-first_level/024-first_level_edge_w_tasks.py

echo "computing first-level connectivity maps with for different parcellations"
python 02-first_level/025-first_level_other_atlas.py


