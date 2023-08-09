"""

Script that runs a experiment with a particular methodological pipeline,
provided by a config file, and a particular parcellation.

"""

import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
from pathlib import Path
import sys
import shutil
import yaml
from fractions import Fraction

def fraction_constructor(loader, node):
    value = loader.construct_scalar(node)
    return float(Fraction(value))

yaml.add_constructor('!fraction', fraction_constructor)

from nilearn.image import load_img, new_img_like
from nilearn.glm.first_level import make_first_level_design_matrix

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)
from src.cofluctuate_bold import NiftiEdgeAtlas
from src.input_data import (get_bold_roi_files, get_confounders_df)
from src.utils import create_edge_mask_from_atlas


def compute_edge_img(bold_roi_img, events_file, confounds, atlas_file,
                     denoise_opts, output_dir=None):
    """
    Auxiliary function to compute and save the edge time series
    to be used in parallel.
    """

    edge_atlas =  NiftiEdgeAtlas(atlas_file = atlas_file,
                                 detrend = denoise_opts['detrend'],
                                 hrf_model= denoise_opts['hrf_model'],
                                 high_pass = denoise_opts['high_pass'],
                                 fir_delays = denoise_opts['fir_delays'],
                                 t_r = 2.0, 
                                 denoise_task = denoise_opts['denoise_task']) 
    edge_ts_img = edge_atlas.fit_transform(run_img = bold_roi_img, 
                                           events = events_file, 
                                           confounds = confounds)
    if output_dir:
        filename = Path(bold_roi_img).name.replace("desc-preproc", 
                                                   "desc-edges_bold")
        np.save(opj(output_dir, "denoising_mats", 
                    Path(bold_roi_img).name.replace("desc-preproc_res-ROI_bold.nii.gz", 
                                                    "desc-denoise_mat.npy")),
                edge_atlas.denoise_mat_)  # JAVI: remove this for earlier version
        edge_ts_img.to_filename(opj(output_dir, filename))
    return edge_ts_img


def save_first_level(fmri_glm, output_dir, contrasts):
    """
    Function just to save first level results for a set of 
    contrasts
    """

    for contrast in contrasts:
        contrast_res_dir = opj(output_dir, contrast)
        Path(contrast_res_dir).mkdir(exist_ok=True, parents=True)
        res_dict = fmri_glm.compute_contrast(contrast_def=contrast, 
                                             output_type="all")
        for name_res, res_img in res_dict.items():
            res_img.to_filename(opj(contrast_res_dir, name_res + ".nii.gz"))

def first_level_pipeline(bold_roi_img,
                         events_file,
                         confounds,
                         atlas_file,
                         subject_id,
                         output_dir,
                         case_opts
                         ):

    from nilearn.glm.first_level import FirstLevelModel


    # Generate edge and ROI time series as niftis
    edge_img = compute_edge_img(bold_roi_img = bold_roi_img, 
                                events_file = events_file,
                                confounds = confounds, 
                                atlas_file = atlas_file, 
                                denoise_opts = case_opts['edge_denoise_opts'])
    node_img =  bold_roi_img

    common_opts = {'t_r':2.0,
                   'standardize': case_opts['standardize'],
                   'smoothing_fwhm': None,
                   'noise_model': case_opts['noise_model'],
                   'signal_scaling': False 
                   }
    # Define and fit a first level object
    edge_glm = FirstLevelModel(mask_img = edge_first_mask,
                              **common_opts)
                              
    node_glm = FirstLevelModel(mask_img = False,
                              **common_opts)
    
    contrasts = ["Incongruent-Congruent"] #  for this analysis we don't need more
    
    frame_times = np.arange(280)*2.0     
    events = pd.read_csv(events_file, sep="\t")                         
    dm_node = make_first_level_design_matrix(frame_times = frame_times,
                                             events = events,
                                             hrf_model= case_opts['hrf_model'],
                                             drift_model = 'cosine',
                                             high_pass = case_opts['high_pass']
                                             )
    dm_node.index = np.arange(280) # I had to redefine index to be able to concatenate with conf
    
    # Create first-level design matrices concatenating hrf task events, cosines and the confounders
    dm_node = pd.concat([dm_node, confounds], axis=1)
    dm_node.to_csv(opj(output_dir, "dm_node.csv"), index=False)
    
    node_glm.fit(run_imgs = node_img, design_matrices = dm_node)
    # Save to disk
    node_out_dir = opj(output_dir, "node", "sub-%d" % subject_id)
    Path(node_out_dir).mkdir(exist_ok = True, parents=True)

    save_first_level(fmri_glm = node_glm, 
                     output_dir = node_out_dir, 
                     contrasts = contrasts)
    
    dm_edge = make_first_level_design_matrix(frame_times = frame_times,
                                             events = events,
                                             hrf_model= case_opts['hrf_model'],
                                             drift_model = None)
    dm_edge.to_csv(opj(output_dir, "dm_edge.csv"), index=False)
    
    edge_glm.fit(run_imgs = edge_img, design_matrices = dm_edge)
    
    # Save to disk
    edge_out_dir = opj(output_dir, "edge", "sub-%d" % subject_id)
    Path(edge_out_dir).mkdir(exist_ok = True, parents=True)
    
    save_first_level(fmri_glm = edge_glm, 
                     output_dir = edge_out_dir, 
                     contrasts = contrasts)

def load_config_file(yaml_file):    
    with open(yaml_file, 'r') as file:   
        config_dat = yaml.load(file, Loader=yaml.FullLoader)
    
    pass_tests = True
    keys = ['edge_denoise_opts', 'hrf_model', 
            'noise_model', 'high_pass', 
            'drift_model', 'standardize', 
            'gsr']
    
    for key in keys:
        if key not in list(config_dat.keys()):
            pass_tests = False
            
    if pass_tests:
        if "denoise_task" not in list(config_dat['edge_denoise_opts'].keys()):
            pass_tests = False
            
    assert pass_tests
    
    # Convert any "None" to None
    for key, value in config_dat.items():
        if value == "None":
            config_dat[key] = None
        
    for key, value in config_dat['edge_denoise_opts'].items():
        if value == "None":
            config_dat['edge_denoise_opts'][key] = None
            
    # If hrf model in edge_denoise_opts is equal to "fir", set the delays to account for a modelling response of up to 24 secs 
    if config_dat['edge_denoise_opts']['hrf_model'] == 'fir':
    	config_dat['edge_denoise_opts']['fir_delays'] = list(range(1, 13))
    else:
    	config_dat['edge_denoise_opts']['fir_delays'] = [0]
    return config_dat

def create_case_outdir(case_opts):
    
    output_dir = "pipeline"
    output_dir += "_events-" + str(int(case_opts['edge_denoise_opts']['denoise_task']))
    
    if case_opts['edge_denoise_opts']['hrf_model'] == case_opts['hrf_model']:
        output_dir += "_denoise-" + "Node"
    else:
        output_dir += "_denoise-" + case_opts['edge_denoise_opts']['hrf_model']
        
    output_dir += "_whiten-" + case_opts['noise_model']
    
    if case_opts['hrf_model'] is None:
        output_dir += "_hrf-0"
    else:
        output_dir += "_hrf-1"
        
    output_dir += "_zscore-" + str(int(case_opts['standardize']))
    
    output_dir += "_gsr-" + str(int(case_opts['gsr']))
        
    return output_dir

def main():
    
    parser = argparse.ArgumentParser(description='Run a particular experiment')
    
    parser.add_argument('--config_file',
                        required=True,
                        type=Path,
                        help="Config file scenario")
    
    parser.add_argument('--output_dir',
                        type=Path,
                        help="Name for the output directory")
    
    parser.add_argument('--atlas',
                        type=str,
                        choices = ["shen", "schaefer", "craddock"],
                        default="shen",
                        help="Name of parcellation (default: shen)")
    
    opts = parser.parse_args()
    
    # Load config file
    config_file = Path(opts.config_file).resolve().as_posix()
    
    case_opts = load_config_file(config_file)
    
    print(case_opts)    
    if opts.output_dir:
        output_dir = Path(opts.output_dir).resolve().as_posix()
    else:
        output_dir = create_case_outdir(case_opts)
        output_dir = opj(project_dir, "results/experiments", 
                         output_dir, opts.atlas)
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    # Copy config file to case dir             
    shutil.copy(config_file, output_dir)
    
    # Data directory
    data_dir = opj(project_dir, "data")
    
    #Subject to use
    final_subjects = np.loadtxt(opj(data_dir, 
                                    "subjects_intersect_motion_035.txt"))
    print("first 10 subjects: ", final_subjects[:10])
    
    # Define atlases (Main: Shen)
    atlas_dict = {
        'shen': opj(
            data_dir, 
            "atlases", 
            "shen_2mm_268_parcellation.nii.gz"),
        
        'craddock':opj(
            data_dir, 
            "atlases", 
            "CPAC200_space-MNI152NLin6_res-2x2x2.nii.gz"),
        
        'schaefer':opj(
            data_dir, 
            "atlases", 
            "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
                  }
    
    # Grab atlas
    atlas_file = atlas_dict[opts.atlas]
    print("atlas file: ", atlas_file)
    
    global edge_first_mask
    edge_first_mask = create_edge_mask_from_atlas(atlas_file)
    
    confounders_regex = "trans|rot|white_matter$|csf$"
    if case_opts['gsr']:
        confounders_regex = confounders_regex + "|global_signal$"
    print("nuisance covariates: ", confounders_regex)
    
    n_jobs = 10
    roi_labels = np.sort(np.unique(load_img(atlas_file).get_fdata()))[1:]
    
    for task_id in ["stroop", "msit"]:
    
        # Get preprocessed bold images in ROI resolution
        bold_roi_dir = opj(project_dir, "results/bold_roi268_imgs", opts.atlas, 
                           "task-%s" % task_id)
        bold_roi_imgs = get_bold_roi_files(task_id = task_id,
                                           bold_roi_dir = bold_roi_dir,
                                           subjects = final_subjects)
        
        # This is fake label image to compute the edge time series
        # using directly the BOLD images at the ROI level.
        fake_label_img = new_img_like(bold_roi_imgs[0], 
                                      roi_labels[:,np.newaxis,np.newaxis])
        
        # Get confounders files
        confounders_dir = opj(
            "/media/javi/ExtraDrive21/cofluctuating-task-connectivity/data", 
            "confounders", "task-%s" % task_id)
        confounders = get_confounders_df(task_id = task_id,
                                         confounders_dir = confounders_dir,
                                         subjects = final_subjects,
                                         confounders_regex = confounders_regex)
        
        # build design matrices and get contrasts
        events_file = opj(data_dir, "task-%s_events.tsv" % task_id)
        print(events_file)
        task_output_dir = opj(output_dir, "task-%s" % task_id)
        Path(task_output_dir).mkdir(exist_ok=True, parents=True)
        
        parallel = Parallel(n_jobs = n_jobs)
        
        parallel(
            delayed(first_level_pipeline)(
                bold_roi_img = bold_roi_img, 
                events_file = events_file,
                confounds = conf_df,
                atlas_file = fake_label_img,
                subject_id = subject_id,
                output_dir = task_output_dir,
                case_opts = case_opts)
            for (bold_roi_img, 
                 conf_df, 
                 subject_id) in tqdm(zip(bold_roi_imgs, 
                                         confounders,
                                         final_subjects)
                                         )
                                         )
                                               
                                             
        
if __name__ == "__main__":
    sys.exit(main())
