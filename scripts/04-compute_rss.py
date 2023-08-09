"""

This script computes the root sum of squares from the BOLD time series
at the region level (in the Shen Atlas).

"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from os.path import join as opj
from tqdm import tqdm
import argparse

from scipy.spatial.distance import squareform
from nilearn.image import load_img, new_img_like
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)
from src.cofluctuate_bold import NiftiEdgeAtlas
from src.input_data import (get_bold_roi_files, get_confounders_df)

def extract_edge_ts(img):
    edge_ts_img = load_img(img)
    edge_ts_data = np.squeeze(edge_ts_img.get_fdata())
    n_vols = edge_ts_data.shape[-1]

    edge_ts_data = np.array([squareform(edge_ts_data[:,:,ii],
                                        checks=False) for ii in range(n_vols)])
    return edge_ts_data

def compute_rss(bold_roi_img, conf_df, atlas_file):
    run_img = load_img(bold_roi_img)
    
    # This is the same for same experiment, but keeping task events
    edge_atlas =  NiftiEdgeAtlas(atlas_file=atlas_file,
                                 detrend=False,
                                 high_pass=1/187.,
                                 t_r = 2.0,
                                 denoise_task = False)

    edge_img = edge_atlas.fit_transform(run_img=run_img, confounds=conf_df)
    edge_ts = extract_edge_ts(edge_img)
    rss = np.sqrt(np.sum(edge_ts**2, axis=1))
    return rss


def main():

    parser = argparse.ArgumentParser(description='Compute RSS time series')
    
    parser.add_argument('--output_dir',
                        type=Path,
                        help="Name for the output directory"
                        )
    
    opts = parser.parse_args()

    if opts.output_dir:
        output_dir = Path(opts.output_dir).resolve().as_posix()
    else:
        output_dir = opj(project_dir,
                         "results",
                         "rss_w_task",
                         "shen")
        
    # Data directory
    data_dir = opj(project_dir, "data")
    
    #Subjects to use
    final_subjects = np.loadtxt(opj(data_dir,
                                    "subjects_intersect_motion_035.txt")
                                )
    print("first 10 subjects: ", final_subjects[:10])
     
    confounders_regex = "trans|rot|white_matter$|csf$|global_signal$"
    print("nuisance covariates: ", confounders_regex)
    
    # Grab atlas
    atlas_file = opj(data_dir, "atlases", "shen_2mm_268_parcellation.nii.gz")
    print("atlas file: ", atlas_file)
    roi_labels = np.sort(np.unique(load_img(atlas_file).get_fdata()))[1:]

    for task_id in ["stroop", "msit", "rest"]:
        print(f"doing task: {task_id}")
        # Get preprocessed bold images in ROI resolution
        bold_roi_dir = opj(project_dir, "results/bold_roi268_imgs", "shen", 
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
    
        task_out_dir = opj(output_dir, "task-%s" % task_id)
        Path(task_out_dir).mkdir(exist_ok=True, parents=True)
    
        for subj, bold_roi_img, conf_df in tqdm(zip(final_subjects,
                                                    bold_roi_imgs,
                                                    confounders
                                                    )):
            rss = compute_rss(bold_roi_img=bold_roi_img, 
                              conf_df=conf_df,
                              atlas_file=fake_label_img)
    
            filename = Path(bold_roi_img).name
            filename = filename.replace("preproc_res-ROI_bold.nii.gz", 
                                        "rss.npy")
            np.save(opj(task_out_dir, filename), rss)
    
if __name__ == "__main__":
    sys.exit(main())
    
