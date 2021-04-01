#!/usr/bin/env python
# coding: utf-8

# Compute the first level maps for edge time series
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

from nilearn.glm.first_level import make_first_level_design_matrix

# Project directory
project_dir = "/home/javi/Documentos/connectivity-phenotype/pip"
sys.path.append(project_dir)

from src.config import get_first_level_edge_opts
from src.input_data import get_edge_files
from src.utils import create_edge_mask_from_atlas
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

def edge_img_first_level(run_img,
                         design_matrix,
                         first_level_opts,
                         subject_id,
                         output_dir,
                         contrasts,
                         mask_img = None):
    """
    Function to run a first level from and edge time series image
    and a design matrix.

    """
    from nilearn.glm.first_level import FirstLevelModel

    # Define and fit a first level object
    fmri_glm = FirstLevelModel(mask_img = mask_img, **first_level_opts)
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

# first-level mask img to restrict seed edge time series to this 
mask_img = opj(data_dir, "masks", "grey_mask_motion_035.nii.gz")
print("mask file: ", mask_img)

# Get first level options
first_level_opts =  get_first_level_edge_opts()
n_task_scans = 280 # Stroop, MSIT
frame_times = np.arange(n_task_scans)*first_level_opts['t_r']

# Number of jobs to use 
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit"]:

    for peak_type in ["positive", "negative"]:

        print("computing first-level edge maps for task %s, peak %s" % (task_id, peak_type))

        # Get preprocessed bold images
        edge_imgs_dir = opj(project_dir, "results/edge_imgs/seed", "task-%s" % task_id, peak_type)
        edge_bold_imgs = get_edge_files(task_id = task_id,
                                         edges_bold_dir = edge_imgs_dir,
                                         subjects = final_subjects)

        # build design matrices and get contrasts
        events_file = opj(data_dir, "task-%s_events.tsv" % task_id)
        events = pd.read_csv(events_file, sep="\t")
        design_matrix =  make_first_level_design_matrix(frame_times = frame_times,
                                                        events = events,
                                                        hrf_model=first_level_opts['hrf_model'],
                                                        drift_model = first_level_opts['drift_model']
                                                        )
        contrasts = get_contrasts(intercept_only=False)

        print("design matrix dimensions: ", design_matrix.shape)
        print("regressors: ", design_matrix.columns)
        print("contrasts: ", contrasts)

        output_dir = opj(project_dir, "results/first-level/edge/seed/task-%s" % task_id, peak_type)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        parallel = Parallel(n_jobs = n_jobs)

        parallel(delayed(edge_img_first_level)(run_img = run_img,
                                               design_matrix = design_matrix,
                                               first_level_opts = first_level_opts,
                                               subject_id = subject_id,
                                               output_dir = output_dir,
                                               contrasts = contrasts,
                                               mask_img = mask_img) \
                for run_img, subject_id in tqdm(zip(edge_bold_imgs, final_subjects))
                )

        del parallel
        _ = gc.collect()
