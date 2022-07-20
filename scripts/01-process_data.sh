#!/bin/bash

echo "creating edge time series for Shen atlas with gsr"
python 01-process_data/011-create_edge_imgs_atlas.py gsr

echo "creating edge time series for Shen atlas without gsr"
python 01-process_data/011-create_edge_imgs_atlas.py nogsr
