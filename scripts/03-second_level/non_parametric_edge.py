#!/usr/bin/env python
# coding: utf-8

# Compute the first level maps for edge time series
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as opj
from pathlib import Path
import sys
from nilearn.glm.second_level import non_parametric_inference

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)
from src.first_level import get_contrasts
from src.utils import create_edge_mask_from_atlas


def get_first_level_files(first_level_dir,
                          subjects,
                          contrast):

    first_level_pattern = opj(first_level_dir, "sub-%d", contrast, "effect_size.nii.gz")
    first_level_imgs = [first_level_pattern % subj for subj in subjects]

    return first_level_imgs


# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
n_subjects = len(final_subjects)
print("number of subjects: ", n_subjects)

# Shen Atlas (this is just for the mask)
atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
print("atlas file: ", atlas_file)
# Create a first-level mask for this atlas
mask_img = create_edge_mask_from_atlas(atlas_file)

# Design matrix for just one-sample test across subjects
design_matrix = pd.DataFrame({'constant': [1]*n_subjects})

contrast="Incongruent-Congruent"
n_perms=50000
random_state=0
for task_id in ["stroop", "msit"]:

    print("doing non-parametric inference for task %s and contrast %s" % (task_id, contrast))

    # Get first-level effect sizes
    first_level_dir = opj(project_dir, "results/first-level/edge_gsr/shen", "task-%s" % task_id)
    first_level_imgs =  get_first_level_files(first_level_dir = first_level_dir,
                                              subjects = final_subjects,
                                              contrast = contrast)
    # Fit this list to a constant to compute one-same t-test
    neg_log_corrected_pvals_img = non_parametric_inference(second_level_input=first_level_imgs,
                                                           mask=mask_img,
                                                           design_matrix = design_matrix,
                                                           n_perm=n_perms,
                                                           random_state=random_state,
                                                           n_jobs=-1,
                                                           verbose=1)

    # Save this
    output_dir = opj(project_dir, "results/second-level/edge_gsr/shen/task-%s" % task_id, contrast)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    neg_log_corrected_pvals_img.to_filename(opj(output_dir, "neg_log_corrected_pvals.nii.gz"))