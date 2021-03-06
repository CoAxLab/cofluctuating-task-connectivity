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

from src import get_first_level_node_opts
from src.input_data import get_bold_files, get_confounders_df
from src.utils import create_edge_mask_from_atlas
from src.first_level import get_contrasts


def compute_ppi(bold_img, atlas_file,
                task_df,
                confounders_df,
                subject_id,
                output_dir):

    """
    Function to run a first level from and edge time series image
    and a design matrix.

    """
    import nibabel as nib
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.glm.first_level import FirstLevelModel

    nifti_label_masker = NiftiLabelsMasker(labels_img=atlas_file)
    roi_mat = nifti_label_masker.fit_transform(bold_img)
    con_ts = np.squeeze(task_df.loc[:, "Congruent"].to_numpy())
    inc_ts = np.squeeze(task_df.loc[:, "Incongruent"].to_numpy())

    first_level = FirstLevelModel(mask_img=False, high_pass=None)

    ppi_con_mat =  np.zeros((268, 268))
    ppi_inc_mat =  np.zeros((268, 268))
    ppi_contrast_mat = np.zeros((268, 268))

    task_cols = list(task_df.columns)
    conf_cols = list(confounders_df.columns)
    for ii in range(268):
        x_region = roi_mat[:, ii]
        y_brain = roi_mat[:, np.arange(268)!=ii].T
        ppi_df = pd.DataFrame({'ppi_inc':inc_ts*x_region, 'ppi_con':con_ts*x_region})
        dm = np.column_stack((task_df.to_numpy(), x_region.reshape(-1,1), ppi_df.to_numpy(), confounders_df.to_numpy()))
        dm = pd.DataFrame(dm, columns = task_cols + ["seed"] + list(ppi_df.columns) + conf_cols)

        Y_img = nib.Nifti1Image(y_brain[:,None,None,:], affine=np.eye(4))
        first_level.fit(run_imgs=Y_img, design_matrices=dm)

        ppi_con = np.squeeze(first_level.compute_contrast(contrast_def='ppi_con',
                                                               output_type='effect_size').get_fdata()
                                 )
        ppi_con_mat[ii,:]=np.insert(ppi_con, ii, 0)

        ppi_inc = np.squeeze(first_level.compute_contrast(contrast_def='ppi_inc',
                                                               output_type='effect_size').get_fdata()
                                 )
        ppi_inc_mat[ii,:]=np.insert(ppi_inc, ii, 0)

        ppi_contrast = np.squeeze(first_level.compute_contrast(contrast_def='ppi_inc-ppi_con',
                                                               output_type='effect_size').get_fdata()
                                 )
        ppi_contrast_mat[ii,:]=np.insert(ppi_contrast, ii, 0)

    # Save to disk
    subject_dir = opj(output_dir, "sub-%d" % subject_id)
    Path(subject_dir).mkdir(exist_ok = True, parents=True)

    dm.to_csv(opj(subject_dir, "ppi_design_example.csv"), index=False)
    np.save(opj(subject_dir, "congruent_ppi.npy"), ppi_con_mat)
    np.save(opj(subject_dir, "incongruent_ppi.npy"), ppi_inc_mat)
    np.save(opj(subject_dir, "incongruent_vs_congruent_ppi.npy"), ppi_contrast_mat)


# Data directory
data_dir = opj(project_dir, "data")

# Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

# Shen Atlas
atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
print("atlas file: ", atlas_file)

# Create a first-level mask for this atlas
mask_img = create_edge_mask_from_atlas(atlas_file)

confounders_regex = "trans|rot|white_matter$|csf$|global_signal$"
print("nuisance covariates: ", confounders_regex)

# Get first level options
first_level_node_opts = get_first_level_node_opts()
n_task_scans = 280 # Stroop, MSIT
frame_times = np.arange(n_task_scans)*first_level_node_opts['t_r']

# Number of jobs to use
n_jobs = 15
print("number of parallel jobs to run = %d" % n_jobs)

for task_id in ["stroop", "msit"]:

    print("computing first-level edge maps for task %s" % task_id)

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
                                                    high_pass = first_level_node_opts['high_pass'])
    print(task_reg_df)
    print("task design matrix dimensions: ", task_reg_df.shape)
    print("confounders design matrix dimensions ", confounders[0].shape)

    output_dir = opj(project_dir, "results/supplementary/ppi/shen/first-level", "task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    parallel = Parallel(n_jobs = n_jobs)

    parallel(delayed(compute_ppi)(bold_img = bold_img,
                                  atlas_file = atlas_file,
                                  task_df = task_reg_df,
                                  confounders_df=conf_df,
                                  subject_id = subject_id,
                                  output_dir=output_dir) for bold_img, conf_df, subject_id in tqdm(zip(bold_imgs, confounders, final_subjects)))

    del parallel
    _ = gc.collect()
