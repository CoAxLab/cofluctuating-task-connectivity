#!/usr/bin/env python
# coding: utf-8

# Compute the first level maps for edge time series in different parcellations.
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

from nilearn.glm.first_level import make_first_level_design_matrix

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src import (get_first_level_edge_opts, get_denoise_opts, 
                 get_first_level_node_opts)
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
        
        
    # Create a first-level mask for this atlas
    edge_mask_img = create_edge_mask_from_atlas(atlas_file)

    # Define and fit a first level object
    fmri_glm = FirstLevelModel(mask_img = edge_mask_img, **first_level_opts)
    fmri_glm.fit(run_imgs = edge_ts_img, design_matrices = design_matrix)
    
    # Save to disk
    subject_dir = opj(output_dir, "sub-%d" % subject_id)
    Path(subject_dir).mkdir(exist_ok = True, parents=True)
    
    # Contrasts
    contrasts = ['Incongruent-Congruent']

    save_first_level(fmri_glm = fmri_glm,
                     output_dir = subject_dir,
                     contrasts = contrasts)

def node_img_first_level(run_img,
                         design_matrix,
                         subject_id,
                         output_dir):
    """
    Function to run a first level from and edge time series image
    and a design matrix.

    """
    from nilearn.glm.first_level import FirstLevelModel

    # Define and fit a first level object
    fmri_glm = FirstLevelModel(mask_img = False,
                               t_r = 2.0,
                               hrf_model = "glover + derivative + dispersion",
                               drift_model = 'cosine',
                               high_pass = 1/187.,
                               smoothing_fwhm = None) # No smoothin. We are doing at ROI level
    fmri_glm.fit(run_imgs = run_img, design_matrices = design_matrix)

    # Save to disk
    subject_dir = opj(output_dir, "sub-%d" % subject_id)
    Path(subject_dir).mkdir(exist_ok = True, parents=True)
    
    # Get contrasts
    contrasts = ["Incongruent-Congruent"] #  for this case we don't need more
    
    save_first_level(fmri_glm = fmri_glm, 
                     output_dir = subject_dir, 
                     contrasts = contrasts)
    
    return fmri_glm


def compute_roi_imgs(run_img, atlas_file, brain_mask):
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.image import new_img_like

    label_masker = NiftiLabelsMasker(labels_img = atlas_file, mask_img=brain_mask)
    roi_ts = label_masker.fit_transform(imgs=run_img)
    # Create fake NIFTI img
    roi_ts = roi_ts.T # Time as the second dimension
    roi_ts_4d = roi_ts[:,None,None,:] # Pad two new dimensions
    roi_img = new_img_like(run_img, roi_ts_4d, affine = np.eye(4)) # Add fake affine (old was:run_img.affine)
    return roi_img


# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

# Atlases
atlas_dict = {'craddock':opj(data_dir, 
                             "atlases", 
                             "CPAC200_space-MNI152NLin6_res-2x2x2.nii.gz"),
              'schaefer':opj(data_dir, 
                             "atlases", 
                             "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
              }

# GSR case
confounders_regex = "trans|rot|white_matter$|csf$|global_signal$"
print("nuisance covariates: ", confounders_regex)

# Get denoise options
denoise_opts = get_denoise_opts()

# denoise_opts['hrf_model'] = 'glover + derivative + dispersion'
print("denoise options: ", denoise_opts)

# Get first level options
first_level_edge_opts =  get_first_level_edge_opts()
print("first-level EDGE options: ", first_level_edge_opts)

first_level_node_opts = get_first_level_node_opts()
print("first-level NODE options: ", first_level_node_opts)

n_task_scans = 280 # Stroop, MSIT
frame_times = np.arange(n_task_scans)*2.0

# Number of jobs to use 
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

 # EDGE PART  
 print("starting EDGE part")
 for atlas_name, atlas_file in atlas_dict.items():
    
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

         output_dir = opj(project_dir,
                          f"results/first-level/gsr/edge/{atlas_name}",
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
             first_level_edge_opts,
             subject_id,
             output_dir) for bold_img, conf_df, mask_img, subject_id
             in tqdm(zip(bold_imgs, conf_dfs, mask_imgs, final_subjects))
             )

         del parallel
         _ = gc.collect()

# NODE PART
print("starting NODE part")
for atlas_name, atlas_file in atlas_dict.items():

    for task_id in ["stroop", "msit"]:

        parallel = Parallel(n_jobs = n_jobs)

         # Get preprocessed bold images
        bold_dir = opj(data_dir, "preproc_bold", "task-%s" % task_id)
        run_imgs = get_bold_files(task_id = task_id,
                                  bold_dir = bold_dir,
                                  subjects = final_subjects)

        # Get brainmasks bold images
        mask_dir = opj(data_dir, "brainmasks", "task-%s" % task_id)
        mask_imgs = get_brainmask_files(task_id = task_id,
                                  mask_dir = mask_dir,
                                  subjects = final_subjects)
    
    
        # Compute ROI time series imgs
        roi_imgs = parallel(delayed(compute_roi_imgs)(bold_img, atlas_file, brain_mask) \
            for bold_img, brain_mask in zip(run_imgs, mask_imgs))
    
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
                                                      hrf_model= first_level_node_opts['hrf_model'],
                                                      drift_model = first_level_node_opts['drift_model'],
                                                      high_pass = first_level_node_opts['high_pass']
                                                      )
        task_reg_df.index = np.arange(n_task_scans) # I had to redefine index to be able to concatenate with conf
    
        # Create first-level design matrices concatenating hrf task events, cosines and the confounders
        design_matrices = [pd.concat([task_reg_df, conf_df], axis=1) for conf_df in confounders]
        print("full design matrix dimensions: ", design_matrices[0].shape)
        print("columns ",  design_matrices[0].columns)
    
    
        # Define output directory for first-level results
        output_dir = opj(project_dir, 
                         f"results/first-level/gsr/node_roi/{atlas_name}",
                         "task-%s" % task_id)
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    
        parallel(delayed(node_img_first_level)(
            run_img = run_img, design_matrix = design_matrix,
            subject_id = subject_id, output_dir = output_dir) 
            for run_img, design_matrix, subject_id in tqdm(zip(roi_imgs, 
                                                               design_matrices, 
                                                               final_subjects)
                                                           )
            )
    
        del parallel
        _ = gc.collect()
