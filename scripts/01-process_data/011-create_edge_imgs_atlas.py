#!/usr/bin/env python
# coding: utf-8

# Plot to compute the first level maps

import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from pathlib import Path
import sys
#from cofluctuate_bold_glm import NiftiEdgeAtlas
#from cofluctuate_bold import NiftiEdgeAtlas


def compute_edge_img(run_img, event_file, confounds, atlas_file, denoise_opts, output_dir):
    """
    Auxiliary function to compute and save the edge time series
    to be used in parallel.
    """
    edge_atlas =  NiftiEdgeAtlas(atlas_file = atlas_file, **denoise_opts)
    edge_ts_img = edge_atlas.fit_transform(run_img = run_img, events = event_file, confounds = confounds)
    filename = Path(run_img).name.replace("desc-preproc_bold", "desc-edges_bold")

    np.save(opj(output_dir, "denoised_roi_time_series", Path(run_img).name.replace("desc-preproc_bold.nii.gz", "desc-conf_roi.npy")),
            edge_atlas.atlas_roi_denoised_)  # JAVI: remove this for earlier version
    np.save(opj(output_dir, "denoising_mats", Path(run_img).name.replace("desc-preproc_bold.nii.gz", "desc-denoise_mat.npy")),
            edge_atlas.denoise_mat_)  # JAVI: remove this for earlier version
    edge_ts_img.to_filename(opj(output_dir, filename))


# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)
from src import get_denoise_opts
from src.input_data import get_bold_files, get_confounders_df
from src.cofluctuate_bold import NiftiEdgeAtlas
# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

# Shen Atlas
atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
print("atlas file: ", atlas_file)

confounders_regex = "trans|rot|white_matter$|csf$|global_signal$"
print("nuisance covariates: ", confounders_regex)

# Get denoise options
denoise_opts = get_denoise_opts()
print("denoise options: ", denoise_opts)

# Number of jobs to use
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit", "rest"]:

    print("computing edge imgs for task %s" % task_id)

    # Get preprocessed bold images
    bold_dir = opj(data_dir, "preproc_bold", "task-%s" % task_id)
    run_imgs = get_bold_files(task_id = task_id,
                              bold_dir = bold_dir,
                              subjects = final_subjects)

    # Get confounders files
    confounders_dir = opj(data_dir, "confounders", "task-%s" % task_id)
    conf_dfs = get_confounders_df(task_id = task_id,
                                  confounders_dir = confounders_dir,
                                  subjects = final_subjects,
                                  confounders_regex = confounders_regex)
    # Define events file
    if task_id == "rest":
        event_file = None
    else:
        event_file = opj(data_dir, "task-%s_events.tsv" % task_id)

    output_dir = opj(project_dir, "results/edge_imgs_gsr/shen/task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    # This is to save intermediate files
    Path(opj(output_dir, "denoised_roi_time_series")).mkdir(exist_ok = True, parents = True)
    Path(opj(output_dir, "denoising_mats")).mkdir(exist_ok = True, parents = True) # JAVI: remove this for earlier version

    parallel = Parallel(n_jobs = n_jobs)

    parallel(delayed(compute_edge_img)(run_img = run_img,
                                       event_file = event_file,
                                       confounds = conf_df,
                                       atlas_file = atlas_file,
                                       denoise_opts = denoise_opts,
                                       output_dir = output_dir) for run_img, conf_df in tqdm(zip(run_imgs, conf_dfs))
            )

    del parallel
    _ = gc.collect()



