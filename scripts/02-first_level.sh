#!/bin/bash

echo "computing first-level activation maps at voxel level"
python 02-first_level/021-first_level_node_voxel.py

echo "computing first-level activation maps at ROI level"
python 02-first_level/022-first_level_node_roi.py

echo "computing first-level connectivity maps"
python 02-first_level/023-first_level_edge.py

