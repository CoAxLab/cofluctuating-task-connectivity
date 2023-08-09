#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script computes the BOLD time series at the Region level.
This saves time when conducting the experiments with different
methodological choices.

"""

import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from pathlib import Path
import sys

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src.input_data import get_bold_files, get_brainmask_files

def compute_save_roi_imgs(run_img, atlas_file, brain_mask, output_dir):
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.image import new_img_like

    label_masker = NiftiLabelsMasker(labels_img = atlas_file, mask_img=brain_mask)
    roi_ts = label_masker.fit_transform(imgs=run_img)
    # Create fake NIFTI img
    roi_ts = roi_ts.T # Time as the second dimension
    roi_ts_4d = roi_ts[:,None,None,:] # Pad two new dimensions
    roi_img = new_img_like(run_img, roi_ts_4d, affine = np.eye(4)) # Add fake affine (old was:run_img.affine)
    
    # Create ROI filename by adding a field in compliance to BIDS
    filename = Path(run_img).name.replace("desc-preproc_bold", 
                                          "desc-preproc_res-ROI_bold")
    roi_img.to_filename(opj(output_dir, filename))


# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

n_subjects = len(final_subjects)
print("number of subjects: ", n_subjects)

# Define atlases (Main: Shen)
atlas_dict = {'shen': opj(data_dir, 
                          "atlases", 
                          "shen_2mm_268_parcellation.nii.gz"),
              'craddock':opj(data_dir, 
                             "atlases", 
                             "CPAC200_space-MNI152NLin6_res-2x2x2.nii.gz"),
              'schaefer':opj(data_dir, 
                             "atlases", 
                             "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
              }

# Number of jobs to use
n_jobs = 10
print("number of parallel jobs to run = %d" % n_jobs)

for atlas_name, atlas_file in  atlas_dict.items():

    print(f"doing atlas: {atlas_name}, file: {atlas_file}")    
    
    task_ids = ["stroop", "msit"]
    if atlas_name == "shen":
        task_ids += ["rest"] # Add resting for main analysis
    
    for task_id in task_ids:
    
        parallel = Parallel(n_jobs = n_jobs)
    
         # Get preprocessed bold images
        bold_dir = opj("/media/javi/ExtraDrive21/cofluctuating-task-connectivity/data", 
                       "preproc_bold", "task-%s" % task_id)
        run_imgs = get_bold_files(task_id = task_id,
                                  bold_dir = bold_dir,
                                  subjects = final_subjects)
    
        # Get brainmasks bold images
        mask_dir = opj(data_dir, "brainmasks", "task-%s" % task_id)
        brainmask_imgs = get_brainmask_files(task_id = task_id,
                                             mask_dir = mask_dir,
                                             subjects = final_subjects)
    
        output_dir = opj(project_dir, 
                         "results/bold_roi268_imgs",
                         f"{atlas_name}/task-{task_id}"
                         )
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        # Compute ROI time series imgs
        parallel(
            delayed(compute_save_roi_imgs)(
                bold_img, atlas_file, brain_mask, output_dir) \
                 for bold_img, brain_mask in zip(run_imgs, brainmask_imgs)
                 )
