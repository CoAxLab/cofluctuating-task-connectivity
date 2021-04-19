#!/bin/bash

echo "Estimating group-level for Node (aka activation) maps WITHOUT gsr"
python 03-second_level/031-second_level_node_nogsr.py

echo "Estimating group-level for Node (aka activation) maps WITH gsr"
python 03-second_level/032-second_level_node_gsr.py

echo "Estimating group-level for edge/link maps WITHOUT gsr"
python 03-second_level/033-second_level_edge_atlas_nogsr.py

echo "Estimating group-level for edge/link maps WITH gsr"
python 03-second_level/034-second_level_edge_atlas_gsr.py
