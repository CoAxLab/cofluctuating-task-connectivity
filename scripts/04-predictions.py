#!/bin/python

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from os.path import join as opj

from scipy.spatial.distance import squareform
from nilearn.glm.first_level import compute_regressor
from nilearn.image import load_img

def load_data(task_id):
    """
    Function to load the edge imgs and the motion outliers as covariates in
    estimating the group-level effect sizes.
    """

    base_dir = opj(project_dir, "results/edge_imgs_gsr/shen/task-%s" % task_id)
    filename = "sub-%d" + "_ses-01_task-%s_space-MNI152NLin2009cAsym_desc-edges_bold.nii.gz" % task_id

    pattern = opj(base_dir, filename)

    edge_imgs_2d = []
    for subj in tqdm(final_subjects):
        edge_img = load_img(pattern % subj)
        edge_img_data = np.squeeze(edge_img.get_fdata()) # get data and drop third dummy dimensions
        edge_data_2d = np.row_stack([squareform(edge_img_data[:,:,ii], checks=False) for ii in range(n_scans)])
        edge_imgs_2d.append(edge_data_2d)

    base_dir = opj(project_dir, "data/confounders/task-%s" % task_id)
    filename = "sub-%d_ses-01" + "_task-%s_desc-confounds_regressors.tsv" % task_id
    pattern = opj(base_dir, filename)
    motion_df = [pd.read_csv(pattern % subj, sep="\t").filter(regex="motion_outlier").to_numpy() \
                  for subj in final_subjects]

    return edge_imgs_2d, motion_df


def estimate_model(idxs, edge_imgs_2d, list_confs):

    from sklearn.linear_model import LinearRegression
    import numpy as np

    linReg = LinearRegression()

    first_level_betas = []
    first_level_intercepts = []

    for ix in idxs:
        # Edge time series
        Y = edge_imgs_2d[ix,:,:]
        C = list_confs[ix]

        linReg.fit(np.column_stack((X,C)), Y)
        first_level_betas.append(linReg.coef_[:,:6])
        first_level_intercepts.append(linReg.intercept_)

    betas_avg = np.array(first_level_betas).mean(axis=0)
    intercepts_avg = np.array(first_level_intercepts).mean(axis=0)

    return intercepts_avg, betas_avg

def prediction(intercepts_avg, betas_avg):
    return intercepts_avg + X.dot(betas_avg.T)

def _compute_score_cv(edge_imgs_2d, list_confs, n_splits, seed):

    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros(shape=(n_subjects, n_subjects))

    n_links = edge_imgs_2d.shape[-1]

    rsquare_scores = np.zeros(shape=(n_splits, n_links))
    pearson_scores = np.zeros(shape=(n_splits, n_links))

    for i_split, (train_idxs, test_idxs) in enumerate(cv.split(dummy_X)):
        # Estimate model
        intercepts_avg, betas_avg = estimate_model(train_idxs, edge_imgs_2d, list_confs)

        # Prediction using this estimation
        Y_pred = prediction(intercepts_avg, betas_avg)

        # test
        Y_test_avg = edge_imgs_2d[test_idxs,:,:].mean(0)

        rsquare_scores[i_split,:] = [r2_score(Y_test_avg[:,ii], Y_pred[:,ii]) for ii in range(n_links)]
        pearson_scores[i_split,:] = [np.corrcoef(Y_test_avg[:,ii], Y_pred[:,ii])[0,1] for ii in range(n_links)]

    return rsquare_scores.mean(axis=0), pearson_scores.mean(axis=0)

def compute_scores(task_id, n_splits = 3, n_shuffles=20, n_jobs=1):

    import shutil
    import tempfile
    import gc
    from joblib import delayed, Parallel

    edge_imgs_2d, list_confs = load_data(task_id=task_id)
    edge_imgs_2d = np.array(edge_imgs_2d)

    temp_dir = tempfile.mkdtemp()

    edge_imgs_2d_mem = np.memmap(temp_dir + "/" + "edge_imgs_2d.npy",
                                 dtype = edge_imgs_2d.dtype,
                                 shape = edge_imgs_2d.shape,
                                 mode='w+')

    edge_imgs_2d_mem[:] = edge_imgs_2d[:]
    del edge_imgs_2d

    parallel = Parallel(n_jobs=n_jobs)
    res = parallel(delayed(_compute_score_cv)(edge_imgs_2d_mem,
                                                   list_confs,
                                                   n_splits,
                                                   seed)
                   for seed in tqdm(range(n_shuffles))
                  )

    r2_scores_edges, pearson_scores_edges = zip(*res)

    shutil.rmtree(temp_dir)
    del parallel
    _ = gc.collect()

    return np.array(r2_scores_edges), np.array(pearson_scores_edges)



project_dir = "/home/javi/Documentos/cofluctuating-task-connectivity"

final_subjects = np.loadtxt(opj(project_dir, "data/subjects_intersect_motion_035.txt"))
n_subjects = len(final_subjects)
print("the number of subjects is", n_subjects)

n_scans = 280
print("the number of scans is ", n_scans)

t_r = 2.0
print("Repetition time", t_r)

frame_times = np.arange(n_scans)*t_r

## Compute input matrix for predictions, i.e., the task effects
task_events = pd.read_csv(opj(project_dir, "data/task-stroop_events.tsv"), sep="\t")

inc_cond = task_events[task_events.loc[:,"trial_type"]=="Incongruent"].to_numpy().T
inc_cond[2,:] = [1,1,1,1]
con_cond = task_events[task_events.loc[:,"trial_type"]=="Congruent"].to_numpy().T
con_cond[2,:] = [1,1,1,1]

inc_regressor, _ = compute_regressor(inc_cond,
                                     hrf_model = "glover + derivative + dispersion",
                                     frame_times=frame_times)
inc_regressor = np.squeeze(inc_regressor)

con_regressor, _ = compute_regressor(con_cond,
                                     hrf_model = "glover + derivative + dispersion",
                                     frame_times=frame_times)
con_regressor = np.squeeze(con_regressor)

X = np.column_stack((inc_regressor, con_regressor))

output_dir = Path(opj(project_dir, "results/generalizability/gsr"))
output_dir.mkdir(exist_ok=True, parents=True)
output_dir = output_dir.resolve().as_posix()

print("computing scores for stroop task")
r2_scores_edges_stroop, pearson_scores_edges_stroop = compute_scores(task_id = "stroop",
                                                                     n_splits = 3,
                                                                     n_shuffles=20,
                                                                     n_jobs=10)

np.savez(opj(output_dir, "scores_stroop.npz"),
         r2_scores_edges = r2_scores_edges_stroop,
         pearson_scores_edges = pearson_scores_edges_stroop)


print("computing scores for MSIT task")
r2_scores_edges_msit, pearson_scores_edges_msit = compute_scores(task_id = "msit",
                                                                     n_splits = 3,
                                                                     n_shuffles=20,
                                                                     n_jobs=10)
np.savez(opj(output_dir, "scores_msit.npz"),
         r2_scores_edges = r2_scores_edges_msit,
         pearson_scores_edges = pearson_scores_edges_msit)
