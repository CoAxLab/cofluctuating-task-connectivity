#!/usr/bin/env python
# coding: utf-8

# Compute the first-level activation maps
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as opj
from joblib import Parallel, delayed
from pathlib import Path
import sys
import gc

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src import get_first_level_node_opts
from src.input_data import get_bold_files, get_confounders_df
from src.first_level import get_contrasts


def save_first_level(fmri_glm, output_dir, contrasts):
    """
    Function just to save first level results for a set of 
    contrasts
    """

    for contrast in contrasts:
        contrast_res_dir = opj(output_dir, contrast)
        Path(contrast_res_dir).mkdir(exist_ok=True, parents=True)
        res_dict = fmri_glm.compute_contrast(contrast_def=contrast, output_type="all")
        for name_res, res_img in res_dict.items():
            res_img.to_filename(opj(contrast_res_dir, name_res + ".nii.gz"))

def run_first_level(run_img,
                    design_matrix,
                    first_level_opts,
                    subject_id,
                    output_dir,
                    contrasts):
    """
    Function to run a first level from and edge time series image
    and a design matrix.

    """
    # Define and fit a first level object
    fmri_glm = FirstLevelModel(mask_img = False, **first_level_opts)
    fmri_glm.fit(run_imgs = run_img, design_matrices = design_matrix)

    # Save to disk
    subject_dir = opj(output_dir, "sub-%d" % subject_id)
    Path(subject_dir).mkdir(exist_ok = True, parents=True)

    save_first_level(fmri_glm = fmri_glm, output_dir = subject_dir, contrasts = contrasts)



# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

confounders_regex = "trans|rot|white_matter$|csf$|global_signal$"
print("nuisance covariates: ", confounders_regex)

# Get first level options
first_level_node_opts = get_first_level_node_opts()
n_task_scans = 280 # Stroop, MSIT
frame_times = np.arange(n_task_scans)*first_level_node_opts['t_r']

# Number of jobs to use 
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit"]:

    print("computing first-level node activation maps for task %s" % task_id)

    # Define output directory
    output_dir = opj(project_dir, "results/first-level/node_gsr/task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Get preprocessed bold images
    bold_dir = opj(data_dir, "preproc_bold", "task-%s" % task_id)
    bold_imgs = get_bold_files(task_id = task_id,
                              bold_dir = bold_dir,
                              subjects = final_subjects)

    # Get confounders files
    confounders_dir = opj(data_dir, "confounders", "task-%s" % task_id)
    confounders = get_confounders_df(task_id = task_id,
                                  confounders_dir = confounders_dir,
                                  subjects = final_subjects,
                                  confounders_regex = confounders_regex)
    # build design matrices and get contrasts
    events_file = opj(data_dir, "task-%s_events.tsv" % task_id)
    events = pd.read_csv(events_file, sep="\t")
    task_reg_df =  make_first_level_design_matrix(frame_times = frame_times,
                                                    events = events,
                                                    hrf_model=first_level_node_opts['hrf_model'],
                                                    drift_model = first_level_node_opts['drift_model'],
                                                    high_pass = first_level_node_opts['high_pass']
                                                    )
    task_reg_df.index = np.arange(n_task_scans) # I had to redefine index to be able to concatenate with conf

    print("task regressors + cosines matrix dimensions: ", task_reg_df.shape)
    print("task regressors + cosines: ", task_reg_df.columns)
    # Save design matrix without the confounders plot
    plot_design_matrix(task_reg_df, output_file = opj(output_dir, "task_reg_cosines.png"))

    # Create first-level design matrices concatenating hrf task events, cosines and the confounders
    design_matrices = [pd.concat([task_reg_df, conf_df], axis=1) for conf_df in confounders]
    print("full design matrix dimensions: ", design_matrices[0].shape)
    print("columns ",  design_matrices[0].columns)

    # Get contrasts
    contrasts = get_contrasts(intercept_only=False)
    print("contrasts: ", contrasts)

    parallel = Parallel(n_jobs = n_jobs)

    parallel(delayed(run_first_level)(run_img = run_img,
                                      design_matrix = design_matrix,
                                      first_level_opts = first_level_node_opts,
                                      subject_id = subject_id,
                                      output_dir = output_dir,
                                      contrasts = contrasts) \
            for run_img, design_matrix, subject_id in tqdm(zip(bold_imgs, design_matrices, final_subjects))
            )

    del parallel
    _ = gc.collect()
