from os.path import join as opj

def get_bold_files(task_id, bold_dir, subjects):
    """
    Function to load the fMRIPrep preprocessed BOLD images
    """

    bold_pattern = opj(bold_dir, "sub-%d_ses-01" + "_task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % task_id)
    run_imgs = [bold_pattern % subj for subj in subjects]
 
    return run_imgs

def get_brainmask_files(task_id, mask_dir, subjects):
    """
    Function to load the fMRIPrep preprocessed BOLD images
    """

    mask_pattern = opj(mask_dir, "sub-%d_ses-01" + "_task-%s_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz" % task_id)
    mask_imgs = [mask_pattern % subj for subj in subjects]
 
    return mask_imgs


def get_confounders_df(task_id, confounders_dir, subjects, confounders_regex="trans|rot|white_matter$|csf$"):
    """
    Function to load the fMRIPrep confounders files, with a default set of
    confounders including the 24 motion paramters and the mean WM and CSF signals
    """
    
    import pandas as pd
    
    conf_pattern = opj(confounders_dir, "sub-%d_ses-01" + "_task-%s_desc-confounds_regressors.tsv" % task_id)
    conf_files = [conf_pattern % subj for subj in subjects]
    conf_dfs = [pd.read_csv(file, sep="\t").filter(regex=confounders_regex).fillna(0) \
                for file in conf_files]
        
    return conf_dfs

def get_edge_files(task_id, edges_bold_dir, subjects):
    """
    Function to load the computed edge time series images
    """

    edges_pattern = opj(edges_bold_dir, "sub-%d_ses-01" + "_task-%s_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz" % task_id)
    run_imgs = [edges_pattern % subj for subj in subjects]
 
    return run_imgs

def get_bold_roi_files(task_id, bold_roi_dir, subjects):
    """
    Function to load the computed edge time series images
    """

    file_pattern = opj(
        bold_roi_dir, 
        "sub-%d_ses-01" + f"_task-{task_id}_space-MNI152NLin2009cAsym_desc-preproc_res-ROI_bold.nii.gz"
        )
    bold_roi_imgs = [file_pattern % subj for subj in subjects]
 
    return bold_roi_imgs
