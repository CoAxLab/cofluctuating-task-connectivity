#!/bin/bash

echo "creating edge time series for atlas clase WITHOUT gsr"
python 01-process_data/011-create_edge_imgs_atlas_nogsr.py

echo "creating edge time series for atlas clase WITH gsr"
python 01-process_data/012-create_edge_imgs_atlas_gsr.py
