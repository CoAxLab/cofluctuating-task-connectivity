#!/usr/bin/env python
# coding: utf-8

# Plot to compute the first level maps
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from pathlib import Path
import sys
from cofluctuate_bold_glm import NiftiEdgeSeed
import json

def compute_edge_img(run_img,
                     event_file,
                     confounds,
                     seed,
                     radius,
                     mask_img,
                     smoothing_fwhm,
                     denoise_opts,
                     output_dir):
    """
    Auxiliary function to compute and save the edge time series
    to be used in parallel.
    """
    edge_atlas =   NiftiEdgeSeed(seed = seed,
                                 radius = radius,
                                 mask_img = mask_img,
                                 smoothing_fwhm = smoothing_fwhm,
                                 **denoise_opts)
    edge_ts_img = edge_atlas.fit_transform(run_img = run_img, events = event_file, confounds = confounds)
    filename = Path(run_img).name.replace("desc-preproc_bold", "desc-edges_bold")
    edge_ts_img.to_filename(opj(output_dir, filename))


# Project directory
project_dir = "/home/javi/Documentos/connectivity-phenotype/pip"
sys.path.append(project_dir)
from src import get_denoise_opts
from src.input_data import get_bold_files, get_confounders_df

# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

# first-level mask img to restrict seed edge time series to this 
mask_img = opj(data_dir, "masks", "grey_mask_motion_035.nii.gz")
print("mask file: ", mask_img)

confounders_regex = "trans|rot|white_matter$|csf$"
print("nuisance covariates: ", confounders_regex)

# Get denoise options
denoise_opts = get_denoise_opts()
print("denoise options: ", denoise_opts)

# Number of jobs to use
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit"]:

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

    event_file = opj(data_dir, "task-%s_events.tsv" % task_id)

    # open peak coordinates
    peaks_dir = opj(project_dir, "results/second-level/node/task-%s/Incongruent-Congruent/" % task_id)
    with open(opj(peaks_dir, "peaks.json"), "r") as f:
        peaks_dict = json.load(f)

    for peak_type in ["positive", "negative"]:

        peak_coords = tuple(list(peaks_dict[peak_type]))

        print("computing edge imgs for task %s and seed type %s and coordinates" % (task_id, peak_type), peak_coords)

        output_dir = opj(project_dir, "results/edge_imgs/seed/task-%s" % task_id, peak_type)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        parallel = Parallel(n_jobs = n_jobs)

        parallel(delayed(compute_edge_img)(run_img = run_img,
                                           event_file = event_file,
                                           confounds = conf_df,
                                           seed = peak_coords,
                                           radius = 8.0, # 8 mm of radius, to have some something beyond 6 mm for 
                                           mask_img = mask_img,
                                           smoothing_fwhm = 6.0,
                                           denoise_opts = denoise_opts,
                                           output_dir = output_dir) for run_img, conf_df in tqdm(zip(run_imgs, conf_dfs))
                )

        del parallel
        _ = gc.collect()
