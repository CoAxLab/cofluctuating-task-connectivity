import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from pathlib import Path
import sys

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)

from src.input_data import get_bold_files, get_confounders_df
from src import get_first_level_node_opts
from src.first_level import get_contrasts


def map_to_atlas(img, atlas_file):

    from nilearn.image import load_img, new_img_like

    atlas_img = load_img(atlas_file)
    atlas_img_data = atlas_img.get_fdata()

    n_rois = np.max(atlas_img_data)-1

    #Map data onto Shen Parcellation file
    img_data = img.get_fdata()
    img_data = np.squeeze(img_data) # A vector of 268 values

    img_atlas_data = np.zeros_like(atlas_img_data)
    for ii in np.arange(1, n_rois+1):
        img_atlas_data[atlas_img_data==ii] = img_data[int(ii-1)]

    img_atlas = new_img_like(atlas_img, img_atlas_data)
    return img_atlas


def save_first_level(fmri_glm, output_dir, contrasts, atlas_file):
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

            res_img_shen = map_to_atlas(res_img, atlas_file)
            res_img_shen.to_filename(opj(output_dir, name_res + "_roi.nii.gz"))


def run_first_level(run_img,
                design_matrix,
                subject_id,
                output_dir,
                contrasts,
                atlas_file):
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

    save_first_level(fmri_glm = fmri_glm, output_dir = subject_dir, contrasts = contrasts, atlas_file = atlas_file)
    return fmri_glm


def compute_roi_imgs(run_img, atlas_file):
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.image import new_img_like

    label_masker = NiftiLabelsMasker(labels_img = atlas_file)
    roi_ts = label_masker.fit_transform(imgs=run_img)
    # Create fake NIFTI img
    roi_ts = roi_ts.T # Time as the second dimension
    roi_ts_4d = roi_ts[:,None,None,:] # Pad two new dimensions
    roi_img = new_img_like(run_img, roi_ts_4d, affine = np.eye(4)) # Add fake affine (old was:run_img.affine)
    return roi_img


def save_second_level(second_level, contrast, atlas_file, output_dir):
    """
    Function just to save first level results for a set of
    contrasts
    """
    from nilearn.image import new_img_like

    res_dict = second_level.compute_contrast(first_level_contrast=contrast, output_type="all")

    for name_res, res_img in res_dict.items():
        res_img.to_filename(opj(output_dir, name_res + ".nii.gz"))

        res_img_shen = map_to_atlas(res_img, atlas_file)
        res_img_shen.to_filename(opj(output_dir, name_res + "_shen.nii.gz"))


# Data directory
data_dir = opj(project_dir, "data")

#Subject to use
final_subjects = np.loadtxt(opj(data_dir, "subjects_intersect_motion_035.txt"))
print("first 10 subjects: ", final_subjects[:10])

n_subjects = len(final_subjects)
print("number of subjects: ", n_subjects)

# Shen Atlas
atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
print("atlas file: ", atlas_file)

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

    parallel = Parallel(n_jobs = n_jobs)

     # Get preprocessed bold images
    bold_dir = opj(data_dir, "preproc_bold", "task-%s" % task_id)
    run_imgs = get_bold_files(task_id = task_id,
                              bold_dir = bold_dir,
                              subjects = final_subjects)

    # Compute ROI time series imgs
    roi_imgs = parallel(delayed(compute_roi_imgs)(img, atlas_file) for img in run_imgs)

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

    # Get contrasts
    contrasts = get_contrasts(intercept_only=False)
    print("contrasts: ", contrasts)

    # Define output directory for first-level results
    output_dir = opj(project_dir, "results/first-level/node_gsr_roi/task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    list_first_levels = parallel(delayed(run_first_level)(run_img = run_img,
                                      design_matrix = design_matrix,
                                      subject_id = subject_id,
                                      output_dir = output_dir,
                                      contrasts = contrasts,
                                      atlas_file=atlas_file) \
            for run_img, design_matrix, subject_id in tqdm(zip(roi_imgs, design_matrices, final_subjects))
            )

    del parallel
    _ = gc.collect()

