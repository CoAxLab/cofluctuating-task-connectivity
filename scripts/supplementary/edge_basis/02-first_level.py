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
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src import get_first_level_edge_opts
from src.input_data import get_edge_files, get_confounders_df
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

# Shen Atlas
atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
print("atlas file: ", atlas_file)

# Create a first-level mask for this atlas
mask_img = create_edge_mask_from_atlas(atlas_file)

# Get first level options
first_level_opts =  get_first_level_edge_opts()
n_task_scans = 280 # Stroop, MSIT
frame_times = np.arange(n_task_scans)*first_level_opts['t_r']
n_rest_scans = 150 # Resting

# Number of jobs to use 
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit"]:

    print("computing first-level edge maps for task %s" % task_id)

    # Get preprocessed bold images
    edge_imgs_dir = opj(project_dir, "results/supplementary/edge_basis_denoise/shen/edge_imgs", "task-%s" % task_id)
    edge_bold_imgs = get_edge_files(task_id = task_id,
                                     edges_bold_dir = edge_imgs_dir,
                                     subjects = final_subjects)
    # Get motion outliers confounders
    confounders_dir = opj(project_dir, "data/confounders", "task-%s" % task_id)
    confounders_df = get_confounders_df(task_id = task_id,
                                        confounders_dir = confounders_dir,
                                        subjects = final_subjects,
                                        confounders_regex='motion_outlier')

   
    events_file = opj(data_dir, "task-%s_events.tsv" % task_id)
    events = pd.read_csv(events_file, sep="\t")
    design_matrices = [make_first_level_design_matrix(frame_times = frame_times,
                                                          events = events,
                                                          hrf_model=first_level_opts['hrf_model'],
                                                          drift_model = first_level_opts['drift_model'],
                                                          add_regs = motion_df) for motion_df in confounders_df]
    contrasts = get_contrasts(intercept_only=False)

    print("design matrix dimensions: ", [dm.shape for dm in design_matrices])
    print("regressors: ", [dm.columns for dm in design_matrices])
    print("contrasts: ", contrasts)

    output_dir = opj(project_dir, "results/supplementary/edge_basis_denoise/shen/first-level", "task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Save design matrices
    dm_output_dir = opj(output_dir, "design_matrices")
    Path(dm_output_dir).mkdir(exist_ok=True, parents=True)
    for dm, subj in zip(design_matrices, final_subjects):
        dm.to_csv(opj(dm_output_dir, "sub-%d_dm.csv" % subj), index = False)

    parallel = Parallel(n_jobs = n_jobs)
    parallel(delayed(edge_img_first_level)(run_img = run_img,
                                           design_matrix = dm,
                                           first_level_opts = first_level_opts,
                                           subject_id = subject_id,
                                           output_dir = output_dir,
                                           contrasts = contrasts,
                                           mask_img = mask_img) \
            for run_img, dm, subject_id in tqdm(zip(edge_bold_imgs, design_matrices, final_subjects))
            )

    del parallel
    _ = gc.collect()
