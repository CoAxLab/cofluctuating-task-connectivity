"""

This script generates the correlation matrices for our main experiment.

"""

import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed
import gc
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
from nilearn.glm.second_level import SecondLevelModel

# Project directory
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)
from src import get_denoise_opts
from src.cofluctuate_bold import NiftiEdgeAtlas
from src.input_data import (get_bold_roi_files, get_confounders_df)
from src.utils import create_edge_mask_from_atlas
from src.first_level import get_contrasts


def compute_edge_img(bold_roi_img, events_file, confounds, atlas_file,
                     denoise_opts):
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
   
    return edge_ts_img
    
def fit_first_level(bold_roi_img, 
                    events_file, 
                    confounds, 
                    atlas_file,
                    case_opts
                    ):
    
    from nilearn.glm.first_level import FirstLevelModel
    
    # Generate edge and ROI time series as niftis
    edge_img = compute_edge_img(bold_roi_img = bold_roi_img, 
                                events_file = events_file,
                                confounds = confounds, 
                                atlas_file = atlas_file, 
                                denoise_opts = case_opts['edge_denoise_opts'])
                       
    common_opts = {'t_r':2.0,
                   'standardize': case_opts['standardize'],
                   'smoothing_fwhm': None,
                   'noise_model': case_opts['noise_model'],
                   'signal_scaling': False 
                   }
    # Define and fit a first level object
    edge_glm = FirstLevelModel(mask_img = edge_first_mask,
                              **common_opts)
                                  
    if events_file:
        events = pd.read_csv(events_file, sep="\t")
    else:
        events = None
    
    dm_edge = make_first_level_design_matrix(frame_times = frame_times,
                                             events = events,
                                             hrf_model= case_opts['hrf_model'],
                                             drift_model = None)
    
    edge_glm.fit(run_imgs = edge_img, design_matrices = dm_edge)
    return edge_glm

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

def main():
    
    parser = argparse.ArgumentParser(
        description='Compute connectivity stats at the group level'
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
                         "correlation_matrices", 
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
    
    global edge_first_mask
    edge_first_mask = create_edge_mask_from_atlas(atlas_file)
    
    confounders_regex = "trans|rot|white_matter$|csf$"
    if case_opts['gsr']:
        confounders_regex = confounders_regex + "|global_signal$"
    print("nuisance covariates: ", confounders_regex)
    
    n_jobs = 10
    roi_labels = np.sort(np.unique(load_img(atlas_file).get_fdata()))[1:]
    
    global frame_times 
    frame_times = np.arange(280)*2.0
    
    for task_id in ["stroop", "msit", "rest"]:
        
        if task_id == "rest":
            frame_times = np.arange(150)*2.0
            
        # Get preprocessed bold images in ROI resolution
        bold_roi_dir = opj(project_dir, "results/bold_roi268_imgs", "shen", 
                           "task-%s" % task_id)
        bold_roi_imgs = get_bold_roi_files(task_id = task_id,
                                           bold_roi_dir = bold_roi_dir,
                                           subjects = final_subjects)
        
        # This is fake label image to compute the edge time series
        # using directly the BOLD images at the ROI level.
        fake_label_img = new_img_like(bold_roi_imgs[0], 
                                      roi_labels[:, np.newaxis, np.newaxis])
        
        # Get confounders files
        confounders_dir = opj(
            "/media/javi/ExtraDrive21/cofluctuating-task-connectivity/data", 
            "confounders", "task-%s" % task_id)
        confounders = get_confounders_df(task_id = task_id,
                                         confounders_dir = confounders_dir,
                                         subjects = final_subjects,
                                         confounders_regex = confounders_regex)
        
        events_file = opj(data_dir, "task-%s_events.tsv" % task_id)
        if task_id == "rest":
            events_file = None
        
        task_output_dir = opj(output_dir, "task-%s" % task_id)
        Path(task_output_dir).mkdir(exist_ok=True, parents=True)
        
        parallel = Parallel(n_jobs = n_jobs)
        first_level_fits = parallel(
            delayed(fit_first_level)(
                bold_roi_img = bold_roi_img, 
                events_file = events_file,
                confounds = conf_df,
                atlas_file = fake_label_img,
                case_opts = case_opts)
            
            for (bold_roi_img, 
                 conf_df, 
                 subject_id) in tqdm(zip(bold_roi_imgs, 
                                         confounders,
                                         final_subjects)
                                         )
                                         )
        contrasts = ["constant", "Incongruent", "Congruent"]
        if task_id == "rest":
            contrasts = ["constant"]
            
        for contrast in contrasts:
            ef_contrasts = [edge_glm.compute_contrast(contrast,
                                                      output_type='effect_size'
                                                      )
                           for edge_glm in first_level_fits]
            
            second_level = SecondLevelModel(mask_img = edge_first_mask)
            dm = pd.DataFrame({'constant': [1]*len(ef_contrasts)})
            second_level.fit(ef_contrasts, design_matrix = dm)
            
            t_stat = second_level.compute_contrast('constant', 
                                                   output_type = 'stat'
                                                   )
            contrast_output_dir = opj(task_output_dir, contrast)
            Path(contrast_output_dir).mkdir(exist_ok=True, parents=True)
            t_stat.to_filename(opj(contrast_output_dir, 'stat.nii.gz'))
                                                            
                                             
        
if __name__ == "__main__":
    sys.exit(main())
