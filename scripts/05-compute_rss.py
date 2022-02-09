import numpy as np
import pandas as pd
import sys
from pathlib import Path
from os.path import join as opj
from tqdm import tqdm

from scipy.spatial.distance import squareform
from nilearn.image import load_img
project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"
sys.path.append(project_dir)
from src.cofluctuate_bold import NiftiEdgeAtlas


def extract_edge_ts(img):
    edge_ts_img = load_img(img)
    edge_ts_data = np.squeeze(edge_ts_img.get_fdata())
    n_vols = edge_ts_data.shape[-1]

    edge_ts_data = np.array([squareform(edge_ts_data[:,:, ii], checks=False) for ii in range(n_vols)])
    return edge_ts_data

def compute_rss(bold_img_file, conf_file, atlas_file):
    run_img = load_img(bold_img_file)

    regex_conf = "trans|rot|white_matter$|csf$|global_signal$"
    conf_df = pd.read_csv(conf_file, sep="\t").filter(regex=regex_conf).fillna(0)

    edge_atlas = NiftiEdgeAtlas(atlas_file = atlas_file,
                           high_pass = 1/187., t_r = 2.0)

    edge_img = edge_atlas.fit_transform(run_img=run_img, confounds=conf_df)
    edge_ts = extract_edge_ts(edge_img)
    rss = np.sqrt(np.sum(edge_ts**2, axis=1))
    return rss

final_subjects = np.loadtxt(opj(project_dir, "data/subjects_intersect_motion_035.txt"))
print("the number of subjects is ", final_subjects.shape)

atlas_file = opj(project_dir, "data/atlases/shen_2mm_268_parcellation.nii.gz")

for task_id in ["stroop", "msit", "rest"]:
    bold_pattern = opj(project_dir, "data/preproc_bold/task-%s" % task_id + "/" + \
    "sub-%d_ses-01_" + "task-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % task_id)

    conf_pattern = opj(project_dir, "data/confounders/task-%s" % task_id + "/" + \
    "sub-%d_ses-01_" + "task-%s_desc-confounds_regressors.tsv" % task_id)

    output_dir = opj(project_dir, "results", "rss_w_task", "task-%s" % task_id)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for subj in tqdm(final_subjects):
        rss = compute_rss(bold_img_file = bold_pattern % subj, conf_file = conf_pattern % subj,
                          atlas_file = atlas_file)

        filename = Path( bold_pattern % subj).name
        filename = filename.replace("preproc_bold.nii.gz", "rss.npy")
        np.save(opj(output_dir, filename), rss)


