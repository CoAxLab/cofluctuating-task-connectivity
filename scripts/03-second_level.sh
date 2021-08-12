#!/bin/bash

echo "Estimating group-level for activations maps at ROI-level"
python 03-second_level/031-second_level_node_roi.py

echo "Estimating group-level for activations maps at voxel-level"
python 03-second_level/032-second_level_node_voxel.py

echo "Estimating group-level for connectivity maps"
python 03-second_level/033-second_level_edge_atlas.py

