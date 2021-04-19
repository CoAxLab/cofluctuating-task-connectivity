#!/bin/bash

echo "computing first-level activation maps WITHOUT gsr"
python 02-first_level/021-first_level_node_nogsr.py

echo "computing first-level activation maps WITH gsr"
python 02-first_level/022-first_level_node_gsr.py

echo "computing first-level link maps WITHOUT gsr"
python 02-first_level/023-first_level_edge_atlas_nogsr.py

echo "computing first-level link maps WITH gsr"
python 02-first_level/024-first_level_edge_atlas_gsr.py

