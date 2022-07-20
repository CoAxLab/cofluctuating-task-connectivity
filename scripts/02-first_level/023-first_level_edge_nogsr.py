#!/usr/bin/env python
# coding: utf-8

# Compute the first level maps for edge time series that did not include gsr.
# The independent variables are just the task regressors.

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


# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src import get_first_level_edge_opts, get_denoise_opts
from src.input_data import (get_bold_files, get_confounders_df, 
                            get_brainmask_files)
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

def compute_edge_img(run_img,
                     event_file, 
                     confounds, 
                     atlas_file, 
                     mask_img, 
                     denoise_opts):
    """
    Auxiliary function to compute and save the edge time series
    to be used in parallel.
    """
    from src.cofluctuate_bold import NiftiEdgeAtlas

    edge_atlas =  NiftiEdgeAtlas(atlas_file=atlas_file,
                                 mask_img=mask_img,
                                 denoise_task=True,
                                 **denoise_opts)
    edge_ts_img = edge_atlas.fit_transform(run_img = run_img,
                                           events = event_file,
                                           confounds = confounds)
    return edge_ts_img, edge_atlas.atlas_roi_denoised_


def edge_img_first_level(bold_img,
                         events_file,
                         confounds,
                         atlas_file,
                         mask_img,
                         denoise_opts,
                         first_level_opts,
                         subject_id,
                         output_dir):
    """
    Function to run a first level from and edge time series image
    and a design matrix.

    """
    import pandas as pd
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.glm.first_level import make_first_level_design_matrix
    from src.utils import create_edge_mask_from_atlas

    # Compute edge time series (with task conditions in)
    edge_ts_img, atlas_roi_denoised = compute_edge_img(run_img = bold_img,
                                   event_file = events_file,
                                   confounds = confounds,
                                   atlas_file = atlas_file,
                                   mask_img = mask_img,
                                   denoise_opts = denoise_opts
                                   )
    # Remove bold image to save memory
    del bold_img
    
    # Use this edge time series image to compute first level estimations
    # First, extract design matrix
    
    events = pd.read_csv(events_file, sep="\t")

    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=first_level_opts['hrf_model'],
        drift_model=first_level_opts['drift_model']
        )
    
    print(max(np.abs([np.corrcoef(ts, design_matrix.loc[:, "Incongruent"].to_numpy())[0,1] \
     for ts in atlas_roi_denoised.T])))
    print(max(np.abs([np.corrcoef(ts, design_matrix.loc[:, "Congruent"].to_numpy())[0,1] \
     for ts in atlas_roi_denoised.T])))
        
    # Contrasts
    contrasts = ['Incongruent-Congruent'] #get_contrasts(intercept_only=False)
        
    # Create a first-level mask for this atlas
    edge_mask_img = create_edge_mask_from_atlas(atlas_file)


    # Define and fit a first level object
    fmri_glm = FirstLevelModel(mask_img = edge_mask_img, **first_level_opts)
    fmri_glm.fit(run_imgs = edge_ts_img, design_matrices = design_matrix)
    
    # Save to disk
    subject_dir = opj(output_dir, "sub-%d" % subject_id)
    Path(subject_dir).mkdir(exist_ok = True, parents=True)

    save_first_level(fmri_glm = fmri_glm,
                     output_dir = subject_dir,
                     contrasts = contrasts)



# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

# Shen Atlas
atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
print("atlas file: ", atlas_file)

# NO GSR case
confounders_regex = "trans|rot|white_matter$|csf$"
print("nuisance covariates: ", confounders_regex)

# Get denoise options
denoise_opts = get_denoise_opts()

# denoise_opts['hrf_model'] = 'glover + derivative + dispersion'
print("denoise options: ", denoise_opts)

# Get first level options
first_level_opts =  get_first_level_edge_opts()
print("first-level options: ", first_level_opts)

n_task_scans = 280 # Stroop, MSIT
frame_times = np.arange(n_task_scans)*first_level_opts['t_r']
n_rest_scans = 150 # Resting

# Number of jobs to use 
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit"]:

    print("computing first-level edge maps for task %s" % task_id)

    # Get preprocessed bold images
    bold_dir = opj(data_dir, "preproc_bold", "task-%s" % task_id)
    bold_imgs = get_bold_files(task_id = task_id,
                              bold_dir = bold_dir,
                              subjects = final_subjects)

    # Get brainmasks bold images
    mask_dir = opj(data_dir, "brainmasks", "task-%s" % task_id)
    mask_imgs = get_brainmask_files(task_id = task_id,
                              mask_dir = mask_dir,
                              subjects = final_subjects)

    # Get confounders files
    confounders_dir = opj(data_dir, "confounders", "task-%s" % task_id)
    conf_dfs = get_confounders_df(task_id = task_id,
                                  confounders_dir = confounders_dir,
                                  subjects = final_subjects,
                                  confounders_regex = confounders_regex)

    events_file = opj(data_dir, "task-%s_events.tsv" % task_id)

    output_dir = opj(project_dir, "results/first-level/nogsr/edge/shen",
                     "task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    parallel = Parallel(n_jobs = n_jobs)
    parallel(delayed(edge_img_first_level)(
        bold_img,
        events_file,
        conf_df,
        atlas_file,
        mask_img,
        denoise_opts,
        first_level_opts,
        subject_id,
        output_dir) for bold_img, conf_df, mask_img, subject_id 
        in tqdm(zip(bold_imgs, conf_dfs, mask_imgs, final_subjects))
        )

    del parallel
    _ = gc.collect()
