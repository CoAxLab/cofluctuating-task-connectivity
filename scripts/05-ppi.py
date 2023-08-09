"""

This script computes the PPI-based correlation matrices for our
main experiment.

"""

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
import argparse
import shutil

from nilearn.image import load_img
from nilearn.glm.first_level import make_first_level_design_matrix

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src.input_data import (get_bold_roi_files, get_confounders_df)
from src.utils import load_config_file

from sklearn.preprocessing import scale

def compute_ppi(bold_roi_img,
                task_df,
                confounders_df,
                subject_id,
                output_dir, 
                case_opts):

    """
    Function to run a PPI model for each region as a seed

    """
    import nibabel as nib
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.glm.first_level import FirstLevelModel
    
    bold_roi_img = load_img(bold_roi_img)
    roi_mat = np.squeeze(bold_roi_img.get_fdata()) # ROI x Time

    # Demean task conditions, for later constructing PPI terms
    con_ts = scale(np.squeeze(task_df.loc[:, "Congruent"].to_numpy()), 
                   with_std=False)
    
    inc_ts = scale(np.squeeze(task_df.loc[:, "Incongruent"].to_numpy()),
                   with_std=False)
                   
    # confounders: task main effects, covariates, cosines. 
    # Then center everything
    confounders_mat = np.column_stack(
        (task_df.to_numpy(), confounders_df.to_numpy())
        )
    confounders_mat = scale(confounders_mat, with_std=False)
    
    task_cols = list(task_df.columns)
    conf_cols = list(confounders_df.columns)
    
    common_opts = {'t_r':2.0,
                   'standardize': case_opts['standardize'],
                   'smoothing_fwhm': None,
                   'noise_model': case_opts['noise_model'],
                   'signal_scaling': False 
                   }
    first_level = FirstLevelModel(mask_img=False, **common_opts)

    ppi_con_mat =  np.zeros((268, 268))
    ppi_inc_mat =  np.zeros((268, 268))
    ppi_contrast_mat = np.zeros((268, 268))
  
    for ii in range(268):
        constant = np.array([1]*280) # This is for modelling the intercept
        # These are the outcome regions, all but the seed one
        Y_brain = roi_mat[np.arange(268)!=ii, :]
        # Seed region time course, demeaned
        x_region = scale(roi_mat[ii, :], with_std=False)
        # PPI for incongruent, demeaned
        ppi_inc = scale(inc_ts*x_region, with_std=False)
        # PPI for congruent, demeaned
        ppi_con = scale(con_ts*x_region, with_std=False)
        # PPI matrix
        ppi_df = pd.DataFrame({'ppi_inc':ppi_inc, 'ppi_con':ppi_con})
        
        # DESIGN MATRIX: PPI terms + covariates (seed region, tasks, others)
        dm = np.column_stack(
            (constant.reshape(-1,1), ppi_df.to_numpy(), 
             x_region.reshape(-1,1), confounders_mat)
            )
        dm = pd.DataFrame(dm, 
                          columns = ['constant'] + list(ppi_df.columns) + \
                              ["seed"] + task_cols + conf_cols)

        Y_img = nib.Nifti1Image(Y_brain[:, None,None,:], affine=np.eye(4))
        first_level.fit(run_imgs=Y_img, design_matrices=dm)

        ppi_contrast = np.squeeze(first_level.compute_contrast(
            contrast_def='ppi_inc-ppi_con',
            output_type='effect_size').get_fdata()
                                 )
        ppi_contrast_mat[ii,:]=np.insert(ppi_contrast, ii, 0)

    # Save to disk
    subject_dir = opj(output_dir, "sub-%d" % subject_id)
    Path(subject_dir).mkdir(exist_ok = True, parents=True)

    dm.to_csv(opj(subject_dir, "ppi_design_example.csv"), index=False)
    np.save(opj(subject_dir, "incongruent_vs_congruent_ppi.npy"), ppi_contrast_mat)

def main():

    parser = argparse.ArgumentParser(
        description='Compute PPI correlation maps for main experiment'
        )
    
    parser.add_argument('--output_dir',
                        type=Path,
                        help="Name for the output directory"
                        )

    opts = parser.parse_args()
    # Load config file    
    config_file = opj(project_dir, "data", "experiments", "pipeline_main.yaml")
    case_opts = load_config_file(config_file)
    print(case_opts)    
    
    if opts.output_dir:
        output_dir = Path(opts.output_dir).resolve().as_posix()
    else:
        output_dir = opj(project_dir, 
                         "results", 
                         "ppi", 
                         "shen")

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Copy config file to output_dir
    shutil.copy(config_file, output_dir)
    
    # Data directory
    data_dir = opj(project_dir, "data")
    
    #Subject to use
    final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
    print("first 10 subjects: ", final_subjects[:10])
    
    # Grab atlas
    atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
    print("atlas file: ", atlas_file)
    
    confounders_regex = "trans|rot|white_matter$|csf$"
    if case_opts['gsr']:
        confounders_regex = confounders_regex + "|global_signal$"
    print("nuisance covariates: ", confounders_regex)
    
    n_jobs = 10
    roi_labels = np.sort(np.unique(load_img(atlas_file).get_fdata()))[1:]
    
    global frame_times 
    frame_times = np.arange(280)*2.0

    n_task_scans = 280 # Stroop, MSIT
    frame_times = np.arange(n_task_scans)*2.0
    
    # Number of jobs to use
    n_jobs = 15
    print("number of parallel jobs to run = %d" % n_jobs)
    
    for task_id in ["stroop", "msit"]:
        
        # Get preprocessed bold images in ROI resolution
        bold_roi_dir = opj(project_dir, "results/bold_roi268_imgs", "shen", 
                           "task-%s" % task_id)
        bold_roi_imgs = get_bold_roi_files(task_id = task_id,
                                           bold_roi_dir = bold_roi_dir,
                                           subjects = final_subjects)
        
        # Get confounders files
        confounders_dir = opj(
            "/media/javi/ExtraDrive21/cofluctuating-task-connectivity/data", 
            "confounders", "task-%s" % task_id)
        confounders = get_confounders_df(task_id = task_id,
                                         confounders_dir = confounders_dir,
                                         subjects = final_subjects,
                                         confounders_regex = confounders_regex)
        
        events_file = opj(data_dir, "task-%s_events.tsv" % task_id)
        events = pd.read_csv(events_file, sep="\t")                         
        task_reg_df = make_first_level_design_matrix(
            frame_times = frame_times,
            events = events,
            hrf_model= case_opts['hrf_model'],
            drift_model = 'cosine',
            high_pass = case_opts['high_pass']
            )
        task_reg_df.index = np.arange(280)
        # Remove constant term. It will be included later manually
        task_reg_df = task_reg_df.drop(columns='constant')
        print(task_reg_df.head())
        print("task design matrix dimensions: ", task_reg_df.shape)
        print(confounders[0].head())
        print("confounders design matrix dimensions ", confounders[0].shape)
        
        task_output_dir = opj(output_dir, "task-%s" % task_id)
        Path(task_output_dir).mkdir(exist_ok=True, parents=True)
    
        parallel = Parallel(n_jobs = n_jobs)
    
        parallel(delayed(compute_ppi)(bold_roi_img=bold_img,
                                      task_df=task_reg_df,
                                      confounders_df=conf_df,
                                      subject_id=subject_id,
                                      output_dir=task_output_dir,
                                      case_opts = case_opts) 
                 for bold_img, conf_df, subject_id in tqdm(zip(bold_roi_imgs, 
                                                               confounders, 
                                                               final_subjects)
                                                           )
                 )
    
        del parallel
        _ = gc.collect()

if __name__ == "__main__":
    sys.exit(main())
